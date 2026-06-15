"""v1.5 cross-GPU kernel performance estimator — CLI (self-contained).

Predicts a kernel's metrics on a TARGET GPU from its NCU profile measured on a
SOURCE GPU.  Fully offline.  Depends only on numpy / pandas / torch + v15_core.

TWO INPUT MODES (every input is supplied; no hidden lookups):
  1. full input via CLI options — one prediction, every column is a flag.
  2. --csv FILE --row {N|all} — a CSV (NCU + SRC/TGT spec columns per row);
     --row is REQUIRED: a 1-based data row, or 'all' for every row.
  Build a CSV from raw NCU + the spec sheet with MLP_NN/examples/prepare_data.py.

OUTPUT
  --out pred.csv   estimated metrics (+ roofline quantities) per kernel.
  --log run.log    plain-text run summary (also to stderr).

EXAMPLES
  python MLP_NN/v1.5/predict_v15.py \
         --csv MLP_NN/examples/example_inputs.csv --row all --out pred.csv
  python MLP_NN/v1.5/predict_v15.py \
         --csv MLP_NN/examples/example_inputs.csv --row 3 --out row3.csv
"""
import os, sys, csv, re, json, pickle, argparse, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import torch
import v15_core as core

ROOT = os.path.dirname(os.path.dirname(HERE))            # repo root
SPEC = core.GPU_SPEC_FEATURES
SRC_PFX, TGT_PFX = "SRC ", "TGT "                        # spec columns (every input is IN the CSV)
OUT_COLS = ["Execution Time [ns]", "Memory Throughput [%]", "Achieved Occupancy",
            "brk_memory", "brk_pipeline_contention", "brk_sync",
            "brk_scheduling_overhead", "t_mem_ns", "t_comp_ns", "t_roof_ns", "efficiency_eta"]

# ---- the full set of input columns (every input can be a CLI option) ----
FP_COUNTERS = [
    "Predicated-On FFMA Operations Per Cycle [inst]",
    "Predicated-On FADD Thread Instructions Executed Per Cycle [inst/cycle]",
    "Predicated-On FMUL Thread Instructions Executed Per Cycle [inst/cycle]",
    "Predicated-On DFMA Operations Per Cycle [inst]",
    "Predicated-On DADD Thread Instructions Executed Per Cycle [inst/cycle]",
    "Predicated-On DMUL Thread Instructions Executed Per Cycle [inst/cycle]",
    "Elapsed Cycles [cycle]"]
NCU_COLS = (core.KERNEL_CONFIG_FEATURES + core.WORKLOAD_PROFILE_FEATURES
            + core.STALL_FEATURES + ["Execution Time"] + FP_COUNTERS)
SPEC_COLS = ([f"{SRC_PFX}{c}" for c in SPEC] + [f"{TGT_PFX}{c}" for c in SPEC]
             + ["TGT peak_fp32", "TGT peak_fp64", "TGT dram_bw"])
INPUT_COLS = ["src_gpu", "tgt_gpu"] + NCU_COLS + SPEC_COLS      # every input column

_flag = lambda c: "--" + re.sub(r"[^a-z0-9]+", "-", c.lower()).strip("-")
_dest = lambda c: re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_")
DEST2COL = {_dest(c): c for c in INPUT_COLS}


def load_source_model(art, src):
    ck = torch.load(f"{art}/model_{src}.pt", map_location="cpu", weights_only=False)
    m = core.MultiBranchMLP(branch_dims=ck["branch_dims"], shared_branch_indices=ck["shared"],
                            branch_hidden=64, shared_hidden=128, n_shared_layers=2,
                            regression_outputs=ck["reg_out"], breakdown_outputs=ck["brk_out"], dropout=0.1)
    m.load_state_dict(ck["state_dict"]); m.eval()
    with open(f"{art}/stats_{src}.pkl", "rb") as f:
        stats = pickle.load(f)
    return m, stats, ck


def resolve_specs(row):
    """Every input is IN the CSV row — read the GPU spec columns directly.
    Returns None if the spec columns are missing (so the caller can error)."""
    if (SRC_PFX + SPEC[0]) not in row or row.get(SRC_PFX + SPEC[0]) != row.get(SRC_PFX + SPEC[0]):
        return None
    src_vec = np.array([float(row[SRC_PFX + c]) for c in SPEC])
    tgt_vec = np.array([float(row[TGT_PFX + c]) for c in SPEC])
    peaks = {"peak_fp32": float(row[TGT_PFX + "peak_fp32"]),
             "peak_fp64": float(row[TGT_PFX + "peak_fp64"]), "dram_bw": float(row[TGT_PFX + "dram_bw"])}
    return src_vec, tgt_vec, peaks


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


def _to_num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def main():
    ap = argparse.ArgumentParser(
        description="v1.5 cross-GPU kernel estimator. Two input modes: "
                    "(1) full input via CLI options, or (2) --csv FILE [--row N].")
    ap.add_argument("--artifact", default=f"{HERE}/v15_artifact")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log")
    ap.add_argument("--no-comments", action="store_true")
    # mode 2: CSV
    m2 = ap.add_argument_group("Mode 2: CSV input")
    m2.add_argument("--csv", "--kernel-stats", dest="csv", help="input CSV (all columns)")
    m2.add_argument("--row", help="REQUIRED with --csv: a 1-based data row number, "
                                  "or 'all' to predict every data row (the header is metadata)")
    # mode 1: every input column as its own CLI option
    m1 = ap.add_argument_group("Mode 1: full input via CLI options (one prediction)")
    for c in INPUT_COLS:
        m1.add_argument(_flag(c), dest=_dest(c), metavar="V", help=c)
    args = ap.parse_args()

    meta = json.load(open(f"{args.artifact}/meta.json"))
    reg_out, brk_out, known = meta["reg_out"], meta["brk_out"], set(meta["sources"])
    cache = {}

    def model_for(src):
        s = src.lower()
        if s not in cache:
            cache[s] = load_source_model(args.artifact, s)
        return cache[s]

    cli_given = {DEST2COL[d]: getattr(args, d) for d in DEST2COL if getattr(args, d) is not None}
    if args.csv:
        df = pd.read_csv(args.csv)
        if args.row is None:
            ap.error("--row is required with --csv: a 1-based row number, or 'all'")
        if str(args.row).lower() != "all":
            try:
                n = int(args.row)
            except ValueError:
                ap.error("--row must be an integer or 'all'")
            if not (1 <= n <= len(df)):
                ap.error(f"--row {n} out of range (1..{len(df)})")
            df = df.iloc[[n - 1]]
        rows_in = [(r, str(r.get("src_gpu")), str(r.get("tgt_gpu"))) for r in df.to_dict("records")]
    elif cli_given:                       # mode 1: full input from CLI options
        row = {k: _to_num(v) for k, v in cli_given.items()}
        rows_in = [(row, str(row.get("src_gpu")), str(row.get("tgt_gpu")))]
    else:
        ap.error("provide inputs via --csv FILE (mode 2) or the per-column CLI options (mode 1)")

    for _, sg, tg in rows_in:
        for nm, gpu in [("src_gpu", sg), ("tgt_gpu", tg)]:
            if gpu in (None, "None", ""):
                ap.error(f"{nm} not given")
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
            spec = resolve_specs(row)
            if spec is None:
                ap.error("input is missing the GPU spec columns (SRC ... / TGT ... / TGT peak_*). "
                         "Every input must be supplied — in the CSV or as CLI options. "
                         "Build a CSV with MLP_NN/examples/prepare_data.py.")
            sv, tv, pk = spec
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
           f"model    : {meta['model']}", "specs    : read from the input (SRC/TGT columns/options)",
           f"mode     : {'CSV' if args.csv else 'CLI'}" + (f", row {args.row}" if args.row else ""),
           f"kernels  : {len(rows_in)}   src->tgt: "
           + ", ".join(f"{sg}->{tg}" for _, sg, tg in rows_in[:4]) + (" ..." if len(rows_in) > 4 else ""),
           f"mean predicted ExecTime: {np.nanmean([r['Execution Time [ns]'] for r in out_rows]):.1f} ns",
           f"output   : {args.out}"]
    text = "\n".join(log); sys.stderr.write(text + "\n")
    if args.log:
        open(args.log, "w").write(text + "\n")


if __name__ == "__main__":
    main()
