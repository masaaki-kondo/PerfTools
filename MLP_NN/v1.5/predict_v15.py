"""v1.5 cross-GPU kernel performance estimator — CLI.

Predicts a kernel's metrics on a TARGET GPU from its profile measured on a
SOURCE GPU. Fully offline. Two input modes, CSV output + plain-text log.

INPUTS
  --specs  specs.csv         STATIC per-GPU spec file (ships with the tool).
  --src / --tgt  GPU names    (rows in specs.csv, e.g. A100, H100, GB200).
  kernel stats (source-GPU profile of the kernel), in ONE of two modes:
    batch : --kernel-stats kernels.csv   (one row per kernel; may carry its own
                                          src_gpu / tgt_gpu columns)
            optionally --row N to pick just the N-th data row (1-based).
    single: --set "Block Size=256" --set "Execution Time=5000" ...   (repeatable)

OUTPUT
  --out pred.csv   estimated metrics (+ roofline quantities) per kernel.
  --log run.log    plain-text run summary (also echoed to stderr).

EXAMPLES
  python MLP_NN/v1.5/predict_v15.py --kernel-stats MLP_NN/examples/kernel_stats_example.csv \
         --out pred.csv --log run.log
  python MLP_NN/v1.5/predict_v15.py --kernel-stats MLP_NN/examples/kernel_stats_example.csv \
         --row 3 --out row3.csv
  python MLP_NN/v1.5/predict_v15.py --src A100 --tgt GB200 --out one.csv \
         --set "Block Size=256" --set "Execution Time=5000" ...
"""
import os, sys, csv, json, pickle, argparse, datetime
os.environ.setdefault("ROOFLINE_INPUTS", "1")          # v1.5 input features
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))           # repo root (.../MLP_NN/v1.5 -> root)
sys.path.insert(0, ROOT)
import numpy as np
import pandas as pd
import torch

import MLP_NN.eval_ensemble_cv as E
from MLP_NN.mlp import MultiBranchMLP
from MLP_NN.test_per_source_model_nogb10 import ET, MEMPCT, OCC, BRK
from MLP_NN.data_pipeline_v2 import (
    GPU_SPEC_FEATURES, KERNEL_CONFIG_FEATURES, WORKLOAD_PROFILE_FEATURES,
    STALL_FEATURES, compute_breakdown, compute_derived_features, extract_features)

OUT_COLS = ["Execution Time [ns]", "Memory Throughput [%]", "Achieved Occupancy",
            "brk_memory", "brk_pipeline_contention", "brk_sync",
            "brk_scheduling_overhead", "t_mem_ns", "t_comp_ns", "t_roof_ns",
            "efficiency_eta"]
IMPUTE_KEYS = ["kernel_config", "workload", "source_specs", "target_specs"]


def load_source_model(art, src):
    ck = torch.load(f"{art}/model_{src}.pt", map_location="cpu", weights_only=False)
    model = MultiBranchMLP(branch_dims=ck["branch_dims"], shared_branch_indices=ck["shared"],
                           branch_hidden=64, shared_hidden=128, n_shared_layers=2,
                           regression_outputs=ck["reg_out"], breakdown_outputs=ck["brk_out"],
                           dropout=0.1)
    model.load_state_dict(ck["state_dict"]); model.eval()
    with open(f"{art}/stats_{src}.pkl", "rb") as f:
        stats = pickle.load(f)
    return model, stats, ck


def load_specs(path):
    specs = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            g = row["gpu"].strip().lower()
            specs[g] = {"vec": np.array([float(row[c]) for c in GPU_SPEC_FEATURES]),
                        "peak_fp32": float(row["peak_fp32"]), "peak_fp64": float(row["peak_fp64"]),
                        "dram_bw": float(row["dram_bw"])}
    return specs


def build_sample(srow, src_gpu, tgt_gpu, specs, stats, reg_out, brk_out):
    """srow: dict of source-GPU kernel stats. Returns a model-ready sample.
    Missing inputs are imputed with the model's TRAINING means (in `stats`) so a
    single row never blows up (per-batch zero-fill would)."""
    s = pd.Series(srow)
    sg, tg = src_gpu.lower(), tgt_gpu.lower()
    src_vec, tgt_vec = specs[sg]["vec"], specs[tg]["vec"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = tgt_vec / np.where(np.abs(src_vec) > 1e-8, src_vec, 1e-8)

    def g0(k):                          # missing -> 0 (for roofline counters)
        try:
            v = float(s.get(k));  return v if v == v else 0.0
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
    samp = {
        "kernel_config": extract_features(s, KERNEL_CONFIG_FEATURES),
        "workload": np.concatenate([extract_features(s, WORKLOAD_PROFILE_FEATURES),
                                    extract_features(s, STALL_FEATURES), compute_breakdown(s)]),
        "source_specs": np.where(np.isnan(src_vec), np.nan, np.maximum(src_vec, 0.0)),
        "target_specs": ratio.astype(np.float64),
        "derived": compute_derived_features(s, tgt_vec),
        "target_regression": np.full(reg_out, np.nan), "target_breakdown": np.full(brk_out, np.nan),
        "kernel_name": str(srow.get("Kernel Name", "")), "benchmark": "infer",
        "src_gpu": src_gpu, "tgt_gpu": tgt_gpu,
        "roofline_bytes": bytes_, "roofline_fp32_flops": fp32, "roofline_fp64_flops": fp64,
        "target_dram_bw": specs[tg]["dram_bw"], "target_peak_fp32": specs[tg]["peak_fp32"],
        "target_peak_fp64": specs[tg]["peak_fp64"],
    }
    # impute NaN inputs with training means (prevents single-row blow-up)
    for k in IMPUTE_KEYS:
        mean = stats[k][0]
        samp[k] = np.where(np.isnan(samp[k]), mean, samp[k])
    samp["derived"] = np.where(np.isnan(samp["derived"]), 0.0, samp["derived"])
    return samp


def predict(model, stats, samples):
    P = E.predict(model, stats, samples)
    et = P["et_p"]
    b = np.array([s["roofline_bytes"] for s in samples]); bw = np.array([s["target_dram_bw"] for s in samples])
    f32 = np.array([s["roofline_fp32_flops"] for s in samples]); f64 = np.array([s["roofline_fp64_flops"] for s in samples])
    p32 = np.array([s["target_peak_fp32"] for s in samples]); p64 = np.array([s["target_peak_fp64"] for s in samples])
    with np.errstate(divide="ignore", invalid="ignore"):
        t_mem = b / bw * 1e9
        t_comp = np.fmax(f32 / p32, f64 / p64) * 1e9
        t_roof = np.fmax(t_mem, t_comp)
        eta = t_roof / et
    rows = []
    for i, s in enumerate(samples):
        rows.append({"kernel_name": s["kernel_name"], "src_gpu": s["src_gpu"], "tgt_gpu": s["tgt_gpu"],
                     "Execution Time [ns]": et[i], "Memory Throughput [%]": P["pr"][i, MEMPCT],
                     "Achieved Occupancy": P["pr"][i, OCC],
                     **{f"brk_{BRK[j]}": P["pb"][i, j] for j in range(len(BRK))},
                     "t_mem_ns": t_mem[i], "t_comp_ns": t_comp[i], "t_roof_ns": t_roof[i],
                     "efficiency_eta": eta[i]})
    return rows


PROVENANCE = [
    "# v1.5 cross-GPU kernel estimator — output",
    "# INPUT SOURCES:",
    "#   kernel stats  : measured by NCU on the SOURCE GPU (user-supplied, per kernel)",
    "#   GPU specs     : static specs.csv, looked up by --src / --tgt GPU name",
    "#   roofline terms: COMPUTED in-tool (bytes=MemThr[byte/s]*ExecTime;",
    "#                   flops=(2*FFMA+FADD+FMUL)*cycles; t_roof=max(bytes/BW,flops/peak))",
    "# OUTPUT COLUMNS  : kernel_name/src_gpu/tgt_gpu = identifiers;",
    "#   Execution Time [ns], Memory Throughput [%], Achieved Occupancy, brk_* = v1.5 predictions;",
    "#   t_mem/t_comp/t_roof_ns, efficiency_eta(=t_roof/ExecTime) = roofline quantities.",
]


def main():
    ap = argparse.ArgumentParser(description="v1.5 cross-GPU kernel estimator")
    ap.add_argument("--specs", default=f"{HERE}/specs.csv")
    ap.add_argument("--artifact", default=f"{HERE}/v15_artifact")
    ap.add_argument("--kernel-stats", help="batch mode: CSV of source-GPU kernel stats")
    ap.add_argument("--row", type=int, help="batch mode: predict only this 1-based data row")
    ap.add_argument("--set", action="append", default=[], metavar="COL=VAL",
                    help="single mode: one kernel-stat column (repeatable)")
    ap.add_argument("--src", help="source GPU name (row in specs.csv)")
    ap.add_argument("--tgt", help="target GPU name (row in specs.csv)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", help="plain-text run log (default: stderr only)")
    ap.add_argument("--no-comments", action="store_true", help="omit the # provenance header")
    args = ap.parse_args()

    meta = json.load(open(f"{args.artifact}/meta.json"))
    specs = load_specs(args.specs)
    reg_out, brk_out, known_src = meta["reg_out"], meta["brk_out"], set(meta["sources"])
    _cache = {}

    def model_for(src):
        s = src.lower()
        if s not in _cache:
            _cache[s] = load_source_model(args.artifact, s)
        return _cache[s]

    if args.kernel_stats:
        df = pd.read_csv(args.kernel_stats)
        if args.row is not None:
            if not (1 <= args.row <= len(df)):
                ap.error(f"--row {args.row} out of range (1..{len(df)})")
            df = df.iloc[[args.row - 1]]
        src_col, tgt_col = "src_gpu" in df.columns, "tgt_gpu" in df.columns
        rows_in = [(r, str(r["src_gpu"]) if src_col else args.src,
                    str(r["tgt_gpu"]) if tgt_col else args.tgt) for r in df.to_dict("records")]
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
            if str(gpu).lower() not in specs:
                ap.error(f"GPU '{gpu}' not in specs.csv (known: {sorted(specs)})")
        if str(sg).lower() not in known_src:
            ap.error(f"no per-source model for src '{sg}' (have: {sorted(known_src)})")

    # group rows by source GPU, predict each group with its model + stats
    out_rows = [None] * len(rows_in)
    by_src = {}
    for i, (_, sg, _) in enumerate(rows_in):
        by_src.setdefault(sg.lower(), []).append(i)
    for src, idxs in by_src.items():
        model, stats, _ = model_for(src)
        sub = [build_sample(rows_in[i][0], rows_in[i][1], rows_in[i][2], specs, stats, reg_out, brk_out)
               for i in idxs]
        for i, r in zip(idxs, predict(model, stats, sub)):
            out_rows[i] = r

    cols = ["kernel_name", "src_gpu", "tgt_gpu"] + OUT_COLS
    with open(args.out, "w", newline="") as f:
        if not args.no_comments:
            f.write("\n".join(PROVENANCE) + "\n")
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in out_rows:
            w.writerow({c: (f"{r[c]:.6g}" if isinstance(r[c], float) else r[c]) for c in cols})

    log = [
        f"v1.5 cross-GPU estimator  run @ {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"model    : {meta['model']}",
        f"specs    : {args.specs}  (GPUs: {sorted(specs)})",
        f"mode     : {'batch' if args.kernel_stats else 'single'}"
        + (f", row {args.row}" if args.row else ""),
        f"kernels  : {len(rows_in)}   src->tgt: "
        + ", ".join(f"{sg}->{tg}" for _, sg, tg in rows_in[:4]) + (" ..." if len(rows_in) > 4 else ""),
        f"mean predicted ExecTime: {np.nanmean([r['Execution Time [ns]'] for r in out_rows]):.1f} ns",
        f"output   : {args.out}",
    ]
    text = "\n".join(log)
    sys.stderr.write(text + "\n")
    if args.log:
        open(args.log, "w").write(text + "\n")


if __name__ == "__main__":
    main()
