"""Build an all-columns example input CSV for predict_v15.py from Yoshida-san's
raw NCU export + our spec sheet — both expected in /data.

EVERY input the estimator needs is written INTO the output CSV:
  - NCU columns                          (from the raw NCU export, source GPU)
  - "SRC <spec>" / "TGT <spec>" + "TGT peak_*"   (GPU specs, from the spec sheet)
so the CSV is fully self-describing — the estimator reads nothing else at runtime.

The raw NCU data is large and NOT shipped; place it in data/ to regenerate.
This is a dev/build tool: it uses the project's data pipeline (which merges the
spec sheet) to compute the per-GPU specs exactly as the model was trained on.

USAGE (from repo root, with raw NCU data present in data/)
  python MLP_NN/examples/prepare_example.py --raw data/20260522_wide.zip --src A100 \
         --n 20 --out MLP_NN/examples/kernel_stats_example.csv
"""
import os, sys, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "MLP_NN", "v1.5"))
import numpy as np
import pandas as pd
import v15_core as core
try:
    from MLP_NN.data_pipeline_v2 import (
        extract_gpu_data, GPU_SPEC_FEATURES, GPU_PEAK_SPECS, GPU_HW_FALLBACK)
except ImportError:
    sys.exit("prepare_example needs the project data pipeline (MLP_NN/data_pipeline_v2.py) "
             "and the raw NCU data in data/ — neither is part of the shipped deliverable.")

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
ALL_GPUS = ["A100", "H100", "GB200"]


def gpu_specs(full, g):
    """Per-GPU spec vector (median of the 13 spec features) + absolute peaks/BW,
    exactly as training built them (spec-sheet-merged)."""
    rows = [core.extract_features(r, GPU_SPEC_FEATURES)
            for df in full[g].values() for _, r in df.iterrows()]
    vec = np.nanmedian(np.array(rows, float), axis=0)
    lc = g.lower()
    return vec, (GPU_PEAK_SPECS[lc]["peak_fp32"], GPU_PEAK_SPECS[lc]["peak_fp64"],
                 GPU_HW_FALLBACK[lc]["dram_bw"])


def main():
    ap = argparse.ArgumentParser(description="build all-columns example from raw NCU + spec sheet")
    ap.add_argument("--raw", default="data/20260522_wide.zip")
    ap.add_argument("--src", required=True)
    ap.add_argument("--tgt", help="target GPU (default: round-robin over the others)")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not os.path.exists(args.raw):
        sys.exit(f"raw NCU data not found: {args.raw} (not shipped — place it in data/)")

    full = {g: extract_gpu_data(args.raw, g.lower()) for g in ALL_GPUS}
    specs = {g: gpu_specs(full, g) for g in ALL_GPUS}
    sg = args.src
    others = [g for g in ALL_GPUS if g.lower() != sg.lower()]

    rows = []
    for df in full[sg].values():
        for _, r in df.iterrows():
            if any(pd.isna(pd.to_numeric(r.get(c), errors="coerce"))
                   for c in ("Execution Time", "Block Size", "Memory Throughput [byte/s]")):
                continue
            rows.append(r)
            if len(rows) >= args.n:
                break
        if len(rows) >= args.n:
            break
    if not rows:
        sys.exit(f"no complete kernel rows for source GPU {sg} in {args.raw}")

    sp_cols = ([f"SRC {c}" for c in core.GPU_SPEC_FEATURES]
               + [f"TGT {c}" for c in core.GPU_SPEC_FEATURES]
               + ["TGT peak_fp32", "TGT peak_fp64", "TGT dram_bw"])
    header = ["Kernel Name", "src_gpu", "tgt_gpu"] + NCU_COLS + sp_cols
    svec, _ = specs[sg]
    recs = []
    for i, r in enumerate(rows):
        tg = args.tgt or others[i % len(others)]
        tvec, (tp32, tp64, tbw) = specs[tg]
        rec = {"Kernel Name": r.get("Kernel Name", f"kernel_{i}"), "src_gpu": sg, "tgt_gpu": tg}
        for c in NCU_COLS:
            rec[c] = r.get(c)
        for j, c in enumerate(core.GPU_SPEC_FEATURES):
            rec[f"SRC {c}"] = repr(float(svec[j])); rec[f"TGT {c}"] = repr(float(tvec[j]))
        rec["TGT peak_fp32"] = repr(float(tp32)); rec["TGT peak_fp64"] = repr(float(tp64))
        rec["TGT dram_bw"] = repr(float(tbw))
        recs.append(rec)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(recs)[header].to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(recs)} kernels, {len(header)} columns "
          f"({len(NCU_COLS)} NCU + {len(sp_cols)} spec-sheet)")


if __name__ == "__main__":
    main()
