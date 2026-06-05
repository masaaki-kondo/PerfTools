"""Prepare a kernel-stats input CSV for predict_v15.py from raw NCU data + the
spec sheet (Yoshida-san's profiling export + data/gpu_microarch_specs.csv).

The raw NCU export is read with the same extractor the trainer uses (which also
pulls per-GPU values from the spec sheet).  We then write ONLY the columns the
estimator needs (meta.json -> kernel_stats_columns), one row per kernel, with
src_gpu / tgt_gpu attached.

USAGE (from repo root)
  python MLP_NN/v1.5/prepare_input.py --raw data/20260522_wide.zip \
         --src A100 --tgt H100 --n 20 --out MLP_NN/examples/kernel_stats_example.csv

  # mixed targets (round-robin over the other GPUs):
  python MLP_NN/v1.5/prepare_input.py --raw data/20260522_wide.zip --src A100 \
         --n 20 --out kernels.csv
"""
import os, sys, json, argparse
sys.path.insert(0, ".")
import pandas as pd
from MLP_NN.data_pipeline_v2 import extract_gpu_data

HERE = "MLP_NN/v1.5"
SPEC_SHEET = "data/gpu_microarch_specs.csv"      # the spec sheet lives in /data
ALL = ["A100", "H100", "GB200"]


def main():
    ap = argparse.ArgumentParser(description="build kernel-stats CSV from raw NCU + spec sheet")
    ap.add_argument("--raw", default="data/20260522_wide.zip", help="raw NCU wide-zip export")
    ap.add_argument("--src", required=True, help="source GPU the kernels were profiled on")
    ap.add_argument("--tgt", help="target GPU (default: round-robin over the other GPUs)")
    ap.add_argument("--n", type=int, default=20, help="number of kernel rows to emit")
    ap.add_argument("--out", required=True)
    ap.add_argument("--spec-sheet", default=SPEC_SHEET, help="(read automatically by the extractor)")
    args = ap.parse_args()

    if not os.path.exists(args.spec_sheet):
        sys.exit(f"spec sheet not found at {args.spec_sheet}")
    meta = json.load(open(f"{HERE}/v15_artifact/meta.json"))
    kstat = meta["kernel_stats_columns"]

    data = extract_gpu_data(args.raw, args.src.lower())   # {benchmark: df}, specs attached
    rows = [r for df in data.values() for _, r in df.iterrows()]
    if not rows:
        sys.exit(f"no kernels found for source GPU {args.src} in {args.raw}")
    rows = rows[:args.n]
    others = [g for g in ALL if g.lower() != args.src.lower()]
    recs = []
    for i, r in enumerate(rows):
        tg = args.tgt if args.tgt else others[i % len(others)]
        rec = {"Kernel Name": r.get("Kernel Name", f"kernel_{i}"),
               "src_gpu": args.src, "tgt_gpu": tg}
        for c in kstat:
            rec[c] = r.get(c)
        recs.append(rec)
    cols = ["Kernel Name", "src_gpu", "tgt_gpu"] + kstat
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(recs)[cols].to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(recs)} kernels, {len(cols)} columns "
          f"(src={args.src}, tgt={args.tgt or 'round-robin ' + '/'.join(others)})")


if __name__ == "__main__":
    main()
