"""v4.0 cross-GPU kernel performance estimator -- CLI (self-contained).

v4.0 = the released NO-ET pure NN (no Execution Time as a predictive input; no
analytical formula at runtime -- a plain neural-net forward pass). Same CLI and
same I/O as v1.5 (predict_v15.py): a drop-in replacement. Fully offline.

TWO INPUT MODES (every input is supplied; no hidden lookups except the SOURCE GPU's
datasheet bw/peak, looked up by src_gpu name for the roofline terms):
  1. full input via CLI options -- one prediction, every column is a flag.
  2. --csv FILE --row {N|all} -- a CSV (NCU + SRC/TGT spec columns per row).
  Build a CSV from raw NCU + the spec sheet with MLP_NN/examples/prepare_data.py.

OUTPUT
  --out pred.csv   estimated metrics (+ roofline quantities) per kernel.
  --log run.log    plain-text run summary (also to stderr).

EXAMPLES
  python MLP_NN/v40/predict_v40.py \
         --csv MLP_NN/examples/example_input_mixed-src_20kernels.csv --row all --out pred.csv
  python MLP_NN/v40/predict_v40.py \
         --csv MLP_NN/examples/example_input_mixed-src_20kernels.csv --row 3 --out row3.csv
"""
import os, sys, csv, re, pickle, argparse, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "v1.5")); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import numpy as np
import pandas as pd
import torch
import v15_core as core                    # feature extractors / MultiBranchMLP (shared)
import v40_core                            # v4.0 no-ET inference

SPEC = core.GPU_SPEC_FEATURES
SRC_PFX, TGT_PFX = "SRC ", "TGT "
OUT_COLS = ["Execution Time [ns]", "Memory Throughput [%]", "Achieved Occupancy",
            "brk_memory", "brk_pipeline_contention", "brk_sync",
            "brk_scheduling_overhead", "t_mem_ns", "t_comp_ns", "t_roof_ns", "efficiency_eta"]
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
INPUT_COLS = ["src_gpu", "tgt_gpu"] + NCU_COLS + SPEC_COLS
_flag = lambda c: "--" + re.sub(r"[^a-z0-9]+", "-", c.lower()).strip("-")
_dest = lambda c: re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_")
DEST2COL = {_dest(c): c for c in INPUT_COLS}


def resolve_specs(row):
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
                    "Execution Time [ns]": float(et[i]), "Memory Throughput [%]": float(P["mem"][i]),
                    "Achieved Occupancy": float(P["occ"][i]),
                    **{f"brk_{core.BRK[j]}": float(P["brk"][i, j]) for j in range(len(core.BRK))},
                    "t_mem_ns": float(t_mem[i]), "t_comp_ns": float(t_comp[i]),
                    "t_roof_ns": float(t_roof[i]), "efficiency_eta": float(eta[i])})
    return out


PROVENANCE = [
    "# v4.0 cross-GPU kernel estimator (no-ET pure NN) -- output",
    "# INPUT SOURCES: kernel stats = NCU on the SOURCE GPU; GPU specs = spec sheet (SRC/TGT cols);",
    "#   roofline terms COMPUTED in-tool (bytes=MemThr[byte/s]*ExecTime; flops=(2*FFMA+FADD+FMUL)*cycles).",
    "# MODEL: v4.0 -- Execution Time is NOT a predictive input; inference is a plain NN forward pass.",
    "# OUTPUTS: ExecTime/Mem%/Occupancy/brk_* = v4.0 predictions; t_*_ns + efficiency_eta = roofline.",
]


def _to_num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def main():
    ap = argparse.ArgumentParser(
        description="v4.0 cross-GPU kernel estimator (no-ET pure NN). Two input modes: "
                    "(1) full input via CLI options, or (2) --csv FILE [--row N].")
    ap.add_argument("--artifact", default=f"{HERE}/v40_artifact")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log")
    ap.add_argument("--no-comments", action="store_true")
    m2 = ap.add_argument_group("Mode 2: CSV input")
    m2.add_argument("--csv", "--kernel-stats", dest="csv", help="input CSV (all columns)")
    m2.add_argument("--row", help="REQUIRED with --csv: a 1-based data row number, "
                                  "or 'all' to predict every data row (the header is metadata)")
    m1 = ap.add_argument_group("Mode 1: full input via CLI options (one prediction)")
    for c in INPUT_COLS:
        m1.add_argument(_flag(c), dest=_dest(c), metavar="V", help=c)
    args = ap.parse_args()

    models, meta = v40_core.load(args.artifact)
    reg_out, brk_out = meta["reg_out"], meta["brk_out"]
    known = {"a100", "h100", "gb200"}        # sources with datasheet bw/peak for the roofline

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
    elif cli_given:
        row = {k: _to_num(v) for k, v in cli_given.items()}
        rows_in = [(row, str(row.get("src_gpu")), str(row.get("tgt_gpu")))]
    else:
        ap.error("provide inputs via --csv FILE (mode 2) or the per-column CLI options (mode 1)")

    for _, sg, tg in rows_in:
        for nm, gpu in [("src_gpu", sg), ("tgt_gpu", tg)]:
            if gpu in (None, "None", ""):
                ap.error(f"{nm} not given")
        if str(sg).lower() not in known:
            ap.error(f"unknown src_gpu '{sg}' (need datasheet bw/peak; have: {sorted(known)})")

    samples = []
    for row, sg, tg in rows_in:
        spec = resolve_specs(row)
        if spec is None:
            ap.error("input is missing the GPU spec columns (SRC ... / TGT ... / TGT peak_*). "
                     "Every input must be supplied -- in the CSV or as CLI options. "
                     "Build a CSV with MLP_NN/examples/prepare_data.py.")
        sv, tv, pk = spec
        samples.append(build_sample(row, sg, tg, sv, tv, pk, reg_out, brk_out))
    out_rows = to_rows(v40_core.predict(models, meta, samples), samples)

    cols = ["kernel_name", "src_gpu", "tgt_gpu"] + OUT_COLS
    with open(args.out, "w", newline="") as f:
        if not args.no_comments:
            f.write("\n".join(PROVENANCE) + "\n")
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in out_rows:
            w.writerow({c: (f"{r[c]:.6g}" if isinstance(r[c], float) else r[c]) for c in cols})

    log = [f"v4.0 cross-GPU estimator (no-ET pure NN)  run @ {datetime.datetime.now().isoformat(timespec='seconds')}",
           f"model    : v4.0  ({meta['seeds']}-seed ensemble)", "specs    : read from the input (SRC/TGT columns/options)",
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
