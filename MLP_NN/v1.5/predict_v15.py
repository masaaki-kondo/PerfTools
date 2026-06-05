"""v1.5 cross-GPU kernel performance estimator — CLI (self-contained).

Predicts a kernel's metrics on a TARGET GPU from its NCU profile measured on a
SOURCE GPU.  Fully offline.  Depends only on numpy / pandas / torch + v15_core.

INPUTS
  kernel stats (NCU, source GPU) — either a batch CSV (--kernel-stats) or a
  single kernel via --set "Col=Val".  GPU specs come from the input row's spec
  columns if present (the example carries them), else from the static spec file
  (data/gpu_specs.csv) looked up by --src / --tgt (or per-row src_gpu/tgt_gpu).

OUTPUT
  --out pred.csv   estimated metrics (+ roofline quantities) per kernel.
  --log run.log    plain-text run summary (also to stderr).

EXAMPLES
  python MLP_NN/v1.5/predict_v15.py --kernel-stats MLP_NN/examples/kernel_stats_example.csv \
         --out pred.csv --log run.log
  python MLP_NN/v1.5/predict_v15.py --kernel-stats MLP_NN/examples/kernel_stats_example.csv \
         --row 3 --out row3.csv
  python MLP_NN/v1.5/predict_v15.py --src A100 --tgt GB200 --out one.csv \
         --set "Block Size=256" --set "Execution Time=5000" ...
"""
import os, sys, csv, json, pickle, argparse, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import torch
import v15_core as core

ROOT = os.path.dirname(os.path.dirname(HERE))            # repo root
SPEC_FILE = os.path.join(ROOT, "data", "gpu_specs.csv")
SPEC = core.GPU_SPEC_FEATURES
SRC_PFX, TGT_PFX = "SRC ", "TGT "                        # spec columns in all-column CSVs
OUT_COLS = ["Execution Time [ns]", "Memory Throughput [%]", "Achieved Occupancy",
            "brk_memory", "brk_pipeline_contention", "brk_sync",
            "brk_scheduling_overhead", "t_mem_ns", "t_comp_ns", "t_roof_ns", "efficiency_eta"]


def load_source_model(art, src):
    ck = torch.load(f"{art}/model_{src}.pt", map_location="cpu", weights_only=False)
    m = core.MultiBranchMLP(branch_dims=ck["branch_dims"], shared_branch_indices=ck["shared"],
                            branch_hidden=64, shared_hidden=128, n_shared_layers=2,
                            regression_outputs=ck["reg_out"], breakdown_outputs=ck["brk_out"], dropout=0.1)
    m.load_state_dict(ck["state_dict"]); m.eval()
    with open(f"{art}/stats_{src}.pkl", "rb") as f:
        stats = pickle.load(f)
    return m, stats, ck


def load_spec_file(path):
    specs = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            g = row["gpu"].strip().lower()
            specs[g] = {"vec": np.array([float(row[c]) for c in SPEC]),
                        "peak_fp32": float(row["peak_fp32"]), "peak_fp64": float(row["peak_fp64"]),
                        "dram_bw": float(row["dram_bw"])}
    return specs


def resolve_specs(row, src_gpu, tgt_gpu, spec_file):
    """Specs from the row's spec columns (all-column CSV) if present, else the spec file."""
    if (SRC_PFX + SPEC[0]) in row and row.get(SRC_PFX + SPEC[0]) == row.get(SRC_PFX + SPEC[0]):
        src_vec = np.array([float(row[SRC_PFX + c]) for c in SPEC])
        tgt_vec = np.array([float(row[TGT_PFX + c]) for c in SPEC])
        peaks = {"peak_fp32": float(row[TGT_PFX + "peak_fp32"]),
                 "peak_fp64": float(row[TGT_PFX + "peak_fp64"]), "dram_bw": float(row[TGT_PFX + "dram_bw"])}
        return src_vec, tgt_vec, peaks
    s, t = src_gpu.lower(), tgt_gpu.lower()
    return spec_file[s]["vec"], spec_file[t]["vec"], spec_file[t]


def build_sample(srow, src_gpu, tgt_gpu, src_vec, tgt_vec, peaks, reg_out, brk_out):
    s = pd.Series(srow)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = tgt_vec / np.where(np.abs(src_vec) > 1e-8, src_vec, 1e-8)

    def g0(k):
        try:
            v = float(s.get(k)); return v if v == v else 0.0
        except (TypeError, ValueError):
            return 0.0
    cyc = g0("Elapsed Cycles [cycle]")
    fp32 = (2 * g0("Predicated-On FFMA Operations Per Cycle [inst]")
            + g0("Predicated-On FADD Thread Instructions Executed Per Cycle [inst/cycle]")
            + g0("Predicated-On FMUL Thread Instructions Executed Per Cycle [inst/cycle]")) * cyc
    fp64 = (2 * g0("Predicated-On DFMA Operations Per Cycle [inst]")
            + g0("Predicated-On DADD Thread Instructions Executed Per Cycle [inst/cycle]")
            + g0("Predicated-On DMUL Thread Instructions Executed Per Cycle [inst/cycle]")) * cyc
    bytes_ = g0("Memory Throughput [byte/s]") * g0("Execution Time") * 1e-9
    return {
        "kernel_config": core.extract_features(s, core.KERNEL_CONFIG_FEATURES),
        "workload": np.concatenate([core.extract_features(s, core.WORKLOAD_PROFILE_FEATURES),
                                    core.extract_features(s, core.STALL_FEATURES), core.compute_breakdown(s)]),
        "source_specs": np.where(np.isnan(src_vec), np.nan, np.maximum(src_vec, 0.0)),
        "target_specs": ratio.astype(np.float64),
        "derived": core.compute_derived_features(s, tgt_vec),
        "target_regression": np.full(reg_out, np.nan), "target_breakdown": np.full(brk_out, np.nan),
        "kernel_name": str(srow.get("Kernel Name", "")), "src_gpu": src_gpu, "tgt_gpu": tgt_gpu,
        "roofline_bytes": bytes_, "roofline_fp32_flops": fp32, "roofline_fp64_flops": fp64,
        "target_dram_bw": peaks["dram_bw"], "target_peak_fp32": peaks["peak_fp32"],
        "target_peak_fp64": peaks["peak_fp64"]}


def to_rows(P, samples):
    et = P["et_p"]
    b = np.array([s["roofline_bytes"] for s in samples]); bw = np.array([s["target_dram_bw"] for s in samples])
    f32 = np.array([s["roofline_fp32_flops"] for s in samples]); f64 = np.array([s["roofline_fp64_flops"] for s in samples])
    p32 = np.array([s["target_peak_fp32"] for s in samples]); p64 = np.array([s["target_peak_fp64"] for s in samples])
    with np.errstate(divide="ignore", invalid="ignore"):
        t_mem = b / bw * 1e9; t_comp = np.fmax(f32 / p32, f64 / p64) * 1e9
        t_roof = np.fmax(t_mem, t_comp); eta = t_roof / et
    out = []
    for i, s in enumerate(samples):
        out.append({"kernel_name": s["kernel_name"], "src_gpu": s["src_gpu"], "tgt_gpu": s["tgt_gpu"],
                    "Execution Time [ns]": et[i], "Memory Throughput [%]": P["pr"][i, core.MEMPCT],
                    "Achieved Occupancy": P["pr"][i, core.OCC],
                    **{f"brk_{core.BRK[j]}": P["pb"][i, j] for j in range(len(core.BRK))},
                    "t_mem_ns": t_mem[i], "t_comp_ns": t_comp[i], "t_roof_ns": t_roof[i], "efficiency_eta": eta[i]})
    return out


PROVENANCE = [
    "# v1.5 cross-GPU kernel estimator — output",
    "# INPUT SOURCES: kernel stats = NCU on the SOURCE GPU; GPU specs = spec sheet (data/);",
    "#   roofline terms COMPUTED in-tool (bytes=MemThr[byte/s]*ExecTime; flops=(2*FFMA+FADD+FMUL)*cycles).",
    "# OUTPUTS: ExecTime/Mem%/Occupancy/brk_* = v1.5 predictions; t_*_ns + efficiency_eta = roofline.",
]


def main():
    ap = argparse.ArgumentParser(description="v1.5 cross-GPU kernel estimator")
    ap.add_argument("--specs", default=SPEC_FILE)
    ap.add_argument("--artifact", default=f"{HERE}/v15_artifact")
    ap.add_argument("--kernel-stats", help="batch mode: CSV of source-GPU kernel stats")
    ap.add_argument("--row", type=int, help="batch mode: predict only this 1-based data row")
    ap.add_argument("--set", action="append", default=[], metavar="COL=VAL", help="single mode (repeatable)")
    ap.add_argument("--src"); ap.add_argument("--tgt")
    ap.add_argument("--out", required=True); ap.add_argument("--log")
    ap.add_argument("--no-comments", action="store_true")
    args = ap.parse_args()

    meta = json.load(open(f"{args.artifact}/meta.json"))
    spec_file = load_spec_file(args.specs)
    reg_out, brk_out, known = meta["reg_out"], meta["brk_out"], set(meta["sources"])
    cache = {}

    def model_for(src):
        s = src.lower()
        if s not in cache:
            cache[s] = load_source_model(args.artifact, s)
        return cache[s]

    if args.kernel_stats:
        df = pd.read_csv(args.kernel_stats)
        if args.row is not None:
            if not (1 <= args.row <= len(df)):
                ap.error(f"--row {args.row} out of range (1..{len(df)})")
            df = df.iloc[[args.row - 1]]
        sc, tc = "src_gpu" in df.columns, "tgt_gpu" in df.columns
        rows_in = [(r, str(r["src_gpu"]) if sc else args.src, str(r["tgt_gpu"]) if tc else args.tgt)
                   for r in df.to_dict("records")]
    else:
        if not args.set:
            ap.error("provide either --kernel-stats (batch) or --set (single)")
        d = {}
        for kv in args.set:
            k, v = kv.split("=", 1)
            try:
                d[k.strip()] = float(v)
            except ValueError:
                d[k.strip()] = v
        rows_in = [(d, args.src, args.tgt)]

    for _, sg, tg in rows_in:
        for nm, gpu in [("--src", sg), ("--tgt", tg)]:
            if gpu is None:
                ap.error(f"{nm} GPU not given (and not a column in the CSV)")
        if str(sg).lower() not in known:
            ap.error(f"no per-source model for src '{sg}' (have: {sorted(known)})")

    out_rows = [None] * len(rows_in)
    by_src = {}
    for i, (_, sg, _) in enumerate(rows_in):
        by_src.setdefault(sg.lower(), []).append(i)
    for src, idxs in by_src.items():
        model, stats, _ = model_for(src)
        sub = []
        for i in idxs:
            row, sg, tg = rows_in[i]
            sv, tv, pk = resolve_specs(row, sg, tg, spec_file)
            sub.append(build_sample(row, sg, tg, sv, tv, pk, reg_out, brk_out))
        for i, r in zip(idxs, to_rows(core.predict(model, stats, sub), sub)):
            out_rows[i] = r

    cols = ["kernel_name", "src_gpu", "tgt_gpu"] + OUT_COLS
    with open(args.out, "w", newline="") as f:
        if not args.no_comments:
            f.write("\n".join(PROVENANCE) + "\n")
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in out_rows:
            w.writerow({c: (f"{r[c]:.6g}" if isinstance(r[c], float) else r[c]) for c in cols})

    log = [f"v1.5 cross-GPU estimator  run @ {datetime.datetime.now().isoformat(timespec='seconds')}",
           f"model    : {meta['model']}", f"specs    : {args.specs}  (GPUs: {sorted(spec_file)})",
           f"mode     : {'batch' if args.kernel_stats else 'single'}" + (f", row {args.row}" if args.row else ""),
           f"kernels  : {len(rows_in)}   src->tgt: "
           + ", ".join(f"{sg}->{tg}" for _, sg, tg in rows_in[:4]) + (" ..." if len(rows_in) > 4 else ""),
           f"mean predicted ExecTime: {np.nanmean([r['Execution Time [ns]'] for r in out_rows]):.1f} ns",
           f"output   : {args.out}"]
    text = "\n".join(log); sys.stderr.write(text + "\n")
    if args.log:
        open(args.log, "w").write(text + "\n")


if __name__ == "__main__":
    main()
