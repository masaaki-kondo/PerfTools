"""Build an all-columns example input CSV for predict_v15.py from Yoshida-san's
raw NCU export + our spec sheet — both expected in /data.

The raw NCU data is large and NOT shipped; place it in data/ to regenerate.
The output row carries EVERY input column the estimator consumes:
  - NCU columns      (from the raw NCU export, source GPU)
  - "SRC <spec>" / "TGT <spec>" + "TGT peak_*"   (from the spec sheet, data/gpu_specs.csv)
so the example is fully self-describing even though the NCU data isn't in the repo.

USAGE (from repo root)
  python MLP_NN/examples/prepare_example.py --raw data/20260522_wide.zip --src A100 \
         --n 20 --out MLP_NN/examples/kernel_stats_example.csv
"""
import os, sys, csv, zipfile, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "MLP_NN", "v1.5"))
import pandas as pd
import v15_core as core

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


def load_specs(path):
    specs = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            specs[row["gpu"].strip().lower()] = row
    return specs


def main():
    ap = argparse.ArgumentParser(description="build all-columns example from raw NCU + spec sheet")
    ap.add_argument("--raw", default="data/20260522_wide.zip")
    ap.add_argument("--specs", default="data/gpu_specs.csv")
    ap.add_argument("--src", required=True)
    ap.add_argument("--tgt", help="target GPU (default: round-robin over the others)")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    for p in (args.raw, args.specs):
        if not os.path.exists(p):
            sys.exit(f"missing input: {p}  (raw NCU data is not shipped — place it in data/)")

    specs = load_specs(args.specs)
    sg = args.src.lower()
    if sg not in specs:
        sys.exit(f"source GPU {args.src} not in {args.specs}")
    others = [g for g in ALL_GPUS if g.lower() != sg]

    # read raw NCU rows for the source GPU from the wide zip (entries: "<gpu>/<bench>__<hash>.csv")
    rows = []
    with zipfile.ZipFile(args.raw) as z:
        for name in z.namelist():
            if not name.lower().startswith(sg + "/") or not name.endswith(".csv"):
                continue
            df = pd.read_csv(z.open(name), low_memory=False)
            df = df.rename(columns={                       # match the training extractor
                "Duration [ns]": "Execution Time",
                "Achieved Occupancy [%]": "Achieved Occupancy",
                "Achieved Active Warps Per SM": "Achieved Active Warps Per SM [warp]"})
            if "Threads" not in df.columns and {"Block Size", "Grid Size"} <= set(df.columns):
                df["Threads"] = (pd.to_numeric(df["Block Size"], errors="coerce")
                                 * pd.to_numeric(df["Grid Size"], errors="coerce"))
            for _, r in df.iterrows():
                # skip incomplete profiles (some kernels in the export lack counters)
                if any(pd.isna(pd.to_numeric(r.get(c), errors="coerce"))
                       for c in ("Execution Time", "Block Size", "Memory Throughput [byte/s]")):
                    continue
                rows.append(r)
                if len(rows) >= args.n:
                    break
            if len(rows) >= args.n:
                break
    if not rows:
        sys.exit(f"no rows for source GPU {args.src} in {args.raw}")

    # spec-column headers (from spec sheet)
    sp_cols = ([f"SRC {c}" for c in core.GPU_SPEC_FEATURES]
               + [f"TGT {c}" for c in core.GPU_SPEC_FEATURES]
               + ["TGT peak_fp32", "TGT peak_fp64", "TGT dram_bw"])
    header = ["Kernel Name", "src_gpu", "tgt_gpu"] + NCU_COLS + sp_cols

    recs = []
    for i, r in enumerate(rows):
        tg = (args.tgt or others[i % len(others)])
        tgr = specs[tg.lower()]; srr = specs[sg]
        rec = {"Kernel Name": r.get("Kernel Name", f"kernel_{i}"), "src_gpu": args.src, "tgt_gpu": tg}
        for c in NCU_COLS:
            rec[c] = r.get(c)
        for c in core.GPU_SPEC_FEATURES:
            rec[f"SRC {c}"] = srr[c]; rec[f"TGT {c}"] = tgr[c]
        rec["TGT peak_fp32"] = tgr["peak_fp32"]; rec["TGT peak_fp64"] = tgr["peak_fp64"]
        rec["TGT dram_bw"] = tgr["dram_bw"]
        recs.append(rec)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(recs)[header].to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(recs)} kernels, {len(header)} columns "
          f"({len(NCU_COLS)} NCU + {len(sp_cols)} spec-sheet)")


if __name__ == "__main__":
    main()
