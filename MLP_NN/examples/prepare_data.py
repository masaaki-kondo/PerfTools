"""Build an all-columns input CSV for predict_v15.py from raw NCU data + the
spec sheet.  Self-contained: depends only on numpy / pandas + v15_core (no other
project code).

REQUIRES Yoshida-san's raw NCU export (the per-GPU wide zip).  That data is large
and is NOT included in /data — place it there (e.g. data/20260522_wide.zip) to
regenerate the example.  The spec sheet (data/gpu_microarch_specs.csv) IS shipped.

EVERY input the estimator needs is written INTO the output CSV, one row per kernel:
  - NCU columns                          (from the raw NCU export, source GPU)
  - "SRC <spec>" / "TGT <spec>" + "TGT peak_*"   (GPU specs, from the spec sheet)
so the CSV is fully self-describing — the estimator reads nothing else.

USAGE (from repo root, with the raw NCU zip placed in data/)
  python MLP_NN/examples/prepare_data.py --raw data/20260522_wide.zip --n 20
"""
import os, sys, csv, zipfile, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "MLP_NN", "v1.5"))
import numpy as np
import pandas as pd
import v15_core as core

SPEC_SHEET = os.path.join(ROOT, "data", "gpu_microarch_specs.csv")
ALL_GPUS = ["A100", "H100", "GB200"]
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
SPEC = core.GPU_SPEC_FEATURES                # 13: [0:5] from NCU, [5:13] from the sheet


def load_sheet(path):
    """Per-GPU raw quantities from the spec sheet (unit-converted as in training)."""
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.setdefault(r["spec"], r)
    def g(spec, col):
        v = str(rows.get(spec, {}).get(col, "")).strip()
        return float(v) if v not in ("", "nan", "None") else np.nan
    out = {}
    for col in ALL_GPUS:
        out[col] = {
            "peak_fp32": g("peak_fp32", col) * 1e12, "peak_fp64": g("peak_fp64", col) * 1e12,
            "peak_tensor": g("peak_fp16_tensor", col) * 1e12, "l2": g("l2_cache_total", col) * 1e6,
            "dram_bw": g("dram_bandwidth", col) * 1e9, "n_sms": g("num_sms", col),
            "lat": g("dram_latency_model_recommended", col), "cpu_bw": g("cpu_gpu_bandwidth", col) * 1e9}
    return out


def read_ncu(raw, gpu):
    """All kernel rows for a GPU from the wide zip, with the training-time renames."""
    frames = []
    with zipfile.ZipFile(raw) as z:
        for name in z.namelist():
            if not name.lower().startswith(gpu.lower() + "/") or not name.endswith(".csv"):
                continue
            df = pd.read_csv(z.open(name), low_memory=False).rename(columns={
                "Duration [ns]": "Execution Time", "Achieved Occupancy [%]": "Achieved Occupancy",
                "Achieved Active Warps Per SM": "Achieved Active Warps Per SM [warp]"})
            if "Threads" not in df.columns and {"Block Size", "Grid Size"} <= set(df.columns):
                df["Threads"] = (pd.to_numeric(df["Block Size"], errors="coerce")
                                 * pd.to_numeric(df["Grid Size"], errors="coerce"))
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def spec_vector(ncu_df, sh):
    """The 13 GPU spec features: [0:5] median of NCU-architectural columns,
    [5:13] derived from the spec sheet (exactly as the training pipeline)."""
    vec = np.full(13, np.nan)
    for j in range(5):                          # NCU-architectural (median over rows)
        c = SPEC[j]
        if c in ncu_df.columns:
            vec[j] = np.nanmedian(pd.to_numeric(ncu_df[c], errors="coerce"))
    nsm = np.nan
    if "# SMs [SM]" in ncu_df.columns:
        s = pd.to_numeric(ncu_df["# SMs [SM]"], errors="coerce").dropna()
        nsm = float(s.iloc[0]) if len(s) else np.nan
    if not np.isfinite(nsm):
        nsm = sh["n_sms"]
    derived = [sh["peak_fp32"] / sh["peak_fp64"], sh["dram_bw"] / nsm, sh["l2"] / nsm,
               sh["peak_fp32"] / sh["dram_bw"], sh["peak_tensor"], sh["peak_tensor"] / sh["peak_fp32"],
               sh["lat"], sh["cpu_bw"]]
    vec[5:13] = derived
    return vec, (sh["peak_fp32"], sh["peak_fp64"], sh["dram_bw"])


def main():
    ap = argparse.ArgumentParser(description="build all-columns example from raw NCU + spec sheet")
    ap.add_argument("--raw", default="data/20260522_wide.zip",
                    help="raw NCU wide zip (Yoshida-san's export; NOT shipped — place in data/)")
    ap.add_argument("--src", default="A100,H100,GB200",
                    help="source GPU(s), comma-separated (default: all three, mixed)")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", help="output CSV (default: a meaningful name in MLP_NN/examples/)")
    args = ap.parse_args()
    if not os.path.exists(args.raw):
        sys.exit(f"raw NCU data not found: {args.raw}\n"
                 "  Yoshida-san's NCU export is large and is NOT included in /data — "
                 "place the wide zip there to (re)build the example.")
    if not os.path.exists(SPEC_SHEET):
        sys.exit(f"spec sheet not found: {SPEC_SHEET}")
    sources = [s.strip() for s in args.src.split(",") if s.strip()]
    if not args.out:
        tag = sources[0] if len(sources) == 1 else "mixed-src"
        args.out = os.path.join(HERE, f"example_input_{tag}_{args.n}kernels.csv")

    sheet = load_sheet(SPEC_SHEET)
    ncu = {g: read_ncu(args.raw, g) for g in ALL_GPUS}
    specs = {g: spec_vector(ncu[g], sheet[g]) for g in ALL_GPUS}

    def complete_rows(g):
        df = ncu[g]
        if df.empty:
            return []
        ok = ~pd.concat([pd.to_numeric(df.get(c), errors="coerce").isna()
                         for c in ("Execution Time", "Block Size", "Memory Throughput [byte/s]")],
                        axis=1).any(axis=1)
        return [r for _, r in df[ok].iterrows()]
    pools = {g: complete_rows(g) for g in sources}
    if not any(pools.values()):
        sys.exit(f"no complete kernel rows for sources {sources} in {args.raw}")

    sp_cols = ([f"SRC {c}" for c in SPEC] + [f"TGT {c}" for c in SPEC]
               + ["TGT peak_fp32", "TGT peak_fp64", "TGT dram_bw"])
    header = ["Kernel Name", "src_gpu", "tgt_gpu"] + NCU_COLS + sp_cols

    recs, ptr, i = [], {g: 0 for g in sources}, 0
    while len(recs) < args.n:
        advanced = False
        for sg in sources:
            if len(recs) >= args.n or ptr[sg] >= len(pools[sg]):
                continue
            r = pools[sg][ptr[sg]]; ptr[sg] += 1; advanced = True
            tg = [g for g in ALL_GPUS if g.lower() != sg.lower()][i % 2]; i += 1
            svec, _ = specs[sg]; tvec, (tp32, tp64, tbw) = specs[tg]
            rec = {"Kernel Name": r.get("Kernel Name", f"kernel_{len(recs)}"), "src_gpu": sg, "tgt_gpu": tg}
            for c in NCU_COLS:
                rec[c] = r.get(c)
            for j, c in enumerate(SPEC):
                rec[f"SRC {c}"] = repr(float(svec[j])); rec[f"TGT {c}"] = repr(float(tvec[j]))
            rec["TGT peak_fp32"] = repr(float(tp32)); rec["TGT peak_fp64"] = repr(float(tp64))
            rec["TGT dram_bw"] = repr(float(tbw))
            recs.append(rec)
        if not advanced:
            break

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(recs)[header].to_csv(args.out, index=False)
    by_src = {g: sum(1 for r in recs if r["src_gpu"] == g) for g in sources}
    print(f"wrote {args.out}: {len(recs)} kernels, {len(header)} columns "
          f"({len(NCU_COLS)} NCU + {len(sp_cols)} spec-sheet); by source: {by_src}")


if __name__ == "__main__":
    main()
