#!/usr/bin/env python3
"""Build the model input CSV (one row per kernel and target GPU) from ncu profiles on a SOURCE GPU and the spec sheet.

The predictors (MLP_NN/mlp-nn.py, PINN/pinn.py) read every input from this CSV:
    ncu metrics of the kernel on the source GPU   +   SRC <spec> / TGT <spec> columns   +   TGT peak_* columns.
The target GPU is never profiled: it is described by its specs only (known GPU = a column of the spec sheet; unknown or
future GPU = a column of a file you pass with --target-specs).

Usage
    # Yoshida-san's raw ncu-full files of one or more runs (files or folders), source H200, targets H100 and GB200
    python examples/prepare_data.py --ncu <HeCBench>/H200/ncu-full/ace-cuda__f899139d.csv \
        --src H200 --tgt H100,GB200 --out inputs.csv
    # a GPU that is not in the spec sheet: its specs at runtime, same layout (spec,unit,<GPU column>)
    python examples/prepare_data.py --ncu <files> --src H200 --tgt MyGPU --target-specs my_gpu.csv --out inputs.csv
    python MLP_NN/mlp-nn.py --csv inputs.csv --row all --out pred.csv

ncu input: the long-form `ncu --csv` export (one row per kernel x metric; Yoshida-san's ncu-full files), or a wide CSV
(one row per kernel). One row per kernel name is used: for a long-form
file the FIRST value ncu reports for each (kernel, metric), as in the training data.

Spec features (SRC and TGT), 13 per GPU
    from the spec sheet (data/v4/gpu_microarch_specs.csv; rows with model_input = yes):
        Peak FP32 / Peak FP64, DRAM BW per SM, L2 Size per SM, Peak FP32 / DRAM BW, Peak FP16 Tensor, Peak Tensor / Peak FP32,
        DRAM Latency (runtime input), CPU-GPU BW
        GPU Maximum Warps Per Scheduler and Theoretical Active Warps Per Scheduler = max_warps_per_scheduler (hardware)
    representative values of a typical kernel on that SM design (the ncu medians the models are evaluated with):
        Theoretical Active Warps per SM = max_warps_per_sm (full occupancy)
        Block Limit Warps               = max_warps_per_sm / 8 (256-thread blocks)
        Shared Memory Configuration Size = the most common carveout: 16 KB on compute capability 8.9 / 12.x, else 32 KB
                                           (--smem-config to set it, e.g. for a new SM design)
GPUs the predictors know (the training GPUs) get the values their training rows carried: spec sheet + the model's
runtime_inputs_training.csv, peaks at the boost clock (the predictors rescale to the measured clock themselves). Any
other GPU gets exactly the values of the sheet / --target-specs / --runtime-inputs; give sm_clock_operating there to
state the clock it will run at (peaks are scaled to it; default: the boost clock).
"""
import argparse, glob, io, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# flags that select the model's feature lists; must be set before network is imported
os.environ.update(HITRATES="1", LOG_SPECS="0", DROP_C2C="0", LAT_ANCHOR="1", INST_ANCHOR="2")
import numpy as np
import pandas as pd

SPEC_SHEET = os.path.join(ROOT, "data", "v4", "gpu_microarch_specs.csv")
# GPUs whose specs the predictors carry in their own tables (training GPUs); every other target is described only by
# this CSV, so it also gets a TGT n_sms column. GH200_144GB is a spec-sheet column but NOT one of them.
PREDICTOR_GPUS = {"A100", "H100", "H200", "GH200", "GB200", "GB10", "L40S", "B300", "H100_MIG_1g", "H100_MIG_2g", "H100_MIG_4g"}
GB, TF, MB = 1e9, 1e12, 1e6                     # sheet units -> SI, as the training pipeline converts them
FP_COUNTERS = ["Predicated-On FFMA Operations Per Cycle [inst]", "Predicated-On FADD Thread Instructions Executed Per Cycle [inst/cycle]",
               "Predicated-On FMUL Thread Instructions Executed Per Cycle [inst/cycle]", "Predicated-On DFMA Operations Per Cycle [inst]",
               "Predicated-On DADD Thread Instructions Executed Per Cycle [inst/cycle]", "Predicated-On DMUL Thread Instructions Executed Per Cycle [inst/cycle]",
               "Elapsed Cycles [cycle]", "Executed Instructions [inst]", "Issued Instructions [inst]"]


# ------------------------------------------------------------------------------------------------------------------- #
# ncu profiles                                                                                                         #
# ------------------------------------------------------------------------------------------------------------------- #
NA_TOKENS = {"", "n/a", "N/A", "—", "nan", "NaN", "None", "none"}
IDENTITY_COLS = ("Block Size", "Grid Size", "Device", "CC", "Context", "Stream", "Process Name", "Host Name")


def _value(v):
    """ncu 'Metric Value' -> float; NaN when empty / n/a / not a number (a missing metric is never read as 0)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v).strip()
    if s in NA_TOKENS:
        return np.nan
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return np.nan


def read_ncu(path):
    """One row per kernel name of one ncu file (long or wide form), with the column names the models were trained on."""
    raw = open(path, "rb").read()
    lines = raw.decode("utf-8", "replace").splitlines()
    hdr = next((i for i, l in enumerate(lines) if l.startswith('"ID","Process ID"')), None)    # long form
    if hdr is not None:
        df = pd.read_csv(io.BytesIO(raw), skiprows=hdr, low_memory=False, na_values=list(NA_TOKENS), keep_default_na=True)
        df = df[df["Kernel Name"].notna() & df["Metric Name"].notna()].reset_index(drop=True)
        unit = df["Metric Unit"].fillna("").astype(str).str.strip() if "Metric Unit" in df.columns else pd.Series("", index=df.index)
        col = pd.Series(np.where(unit != "", df["Metric Name"].astype(str) + " [" + unit + "]", df["Metric Name"].astype(str)), index=df.index)
        val = df["Metric Value"].map(_value)
        rows = []
        for kname, grp in df.groupby("Kernel Name", sort=False):
            row = {"Kernel Name": kname, **{k: grp.iloc[0].get(k) for k in IDENTITY_COLS if k in df.columns}}
            seen = {}
            for c, v in zip(col.loc[grp.index], val.loc[grp.index]):
                seen.setdefault(c, v)                  # first value ncu reports
            row.update(seen)                           # metrics win over same-named identity columns ('Block Size')
            rows.append(row)
        text = pd.DataFrame(rows).to_csv(index=False)
    else:                                              # wide form (one row per kernel already)
        start = next((i for i, l in enumerate(lines) if not l.startswith("#")), 0)
        text = "\n".join(lines[start:])
    w = pd.read_csv(io.StringIO(text), low_memory=False)
    w = w.rename(columns={"Duration [ns]": "Execution Time", "Achieved Occupancy [%]": "Achieved Occupancy"})
    if "Threads" not in w.columns and {"Block Size", "Grid Size"} <= set(w.columns):
        w["Threads"] = pd.to_numeric(w["Block Size"], errors="coerce") * pd.to_numeric(w["Grid Size"], errors="coerce")
    return w.drop_duplicates("Kernel Name")


def ncu_files(paths):
    out = []
    for p in paths:
        out += sorted(glob.glob(os.path.join(p, "*.csv"))) if os.path.isdir(p) else [p]
    return [f for f in out if not f.endswith(".stderr.txt")]


# ------------------------------------------------------------------------------------------------------------------- #
# GPU specs                                                                                                            #
# ------------------------------------------------------------------------------------------------------------------- #
def read_specs(paths):
    """{GPU column: {spec: value}} from sheets in the spec,unit,<GPU columns> layout; later files override earlier ones
    cell by cell (empty cells ignored). A spec listed in several units (latency in cycles and ns) is kept in ns."""
    out = {}
    for p in paths:
        S = pd.read_csv(p, dtype=str, keep_default_na=False)
        gpus = [c for c in S.columns if c not in ("spec", "unit", "model_input", "confidence_notes", "source", "notes")]
        for _, r in S.iterrows():
            if str(r.get("unit", "")).strip() == "cycles":
                continue
            for g in gpus:
                v = str(r[g]).strip()
                if v:
                    out.setdefault(g, {})[r["spec"].strip()] = v
    return out


def spec_features(SPEC, d, gpu, smem_override=None):
    """The 13 spec features of one GPU (SPEC order) + the 4 absolute peaks the predictors read (TGT peak_*)."""
    def f(k, scale=1.0, required=True):
        try:
            return float(d[k]) * scale
        except (KeyError, ValueError):
            if required:
                sys.exit(f"spec '{k}' missing for {gpu} (spec sheet or --target-specs / --runtime-inputs)")
            return np.nan
    boost = f("sm_clock_boost", GB)
    clk = f("sm_clock_operating", GB, required=False)              # unknown GPU: the clock it will run at, if given
    cs = clk / boost if clk == clk else 1.0
    sms, bw, l2 = f("num_sms"), f("dram_bandwidth", GB), f("l2_cache_total", MB)
    f32, f64, ten = f("peak_fp32", TF) * cs, f("peak_fp64", TF) * cs, f("peak_fp16_tensor", TF) * cs
    wps, wsm = f("max_warps_per_scheduler"), f("max_warps_per_sm")
    cc = str(d.get("compute_capability", ""))
    smem = smem_override or (16384.0 if (cc.startswith("8.9") or cc.startswith("12")) else 32768.0)
    v = {"GPU Maximum Warps Per Scheduler [warp]": wps, "Theoretical Active Warps Per Scheduler [warp]": wps,   # hardware
         "Theoretical Active Warps per SM [warp]": wsm, "Block Limit Warps [block]": wsm / 8.0,                 # typical kernel
         "Shared Memory Configuration Size [byte]": smem,
         "Peak FP32 / Peak FP64": f32 / f64, "DRAM BW per SM [byte/s/SM]": bw / sms, "L2 Size per SM [byte/SM]": l2 / sms,
         "Peak FP32 / DRAM BW [flop/byte]": f32 / bw, "Peak FP16 Tensor [flop/s]": ten, "Peak Tensor / Peak FP32": ten / f32,
         "DRAM Latency [ns]": f("dram_latency_model_recommended"), "CPU-GPU BW [byte/s]": f("cpu_gpu_bandwidth", GB)}
    missing = [c for c in SPEC if c not in v]
    if missing:
        sys.exit(f"spec features not handled: {missing}")
    peaks = {"peak_fp32": f32, "peak_fp64": f64, "dram_bw": bw, "peak_inst": sms * 4.0 * (clk if clk == clk else boost)}
    return [v[c] for c in SPEC], peaks, sms


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ncu", nargs="+", required=True, help="ncu CSV files or folders of them (profiled on --src)")
    ap.add_argument("--src", required=True, help="GPU the profiles come from (a column of the spec sheet)")
    ap.add_argument("--tgt", required=True, help="target GPU(s), comma-separated (spec-sheet or --target-specs columns)")
    ap.add_argument("--target-specs", help="spec file for GPUs not in the spec sheet (same layout)")
    ap.add_argument("--runtime-inputs", help="extra runtime values (same layout), e.g. a target's DRAM latency")
    ap.add_argument("--model-dir", default=os.path.join(ROOT, "MLP_NN"),
                    help="model folder: its network.py (feature lists) is used")
    ap.add_argument("--weights-dir", default=os.path.join(ROOT, "MLP_NN", "v4.55"),
                    help="weights folder: its runtime_inputs_training.csv (actual values of the training GPUs) is used")
    ap.add_argument("--smem-config", type=float, help="Shared Memory Configuration Size [byte] for the TARGET (new SM design)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sys.path.insert(0, a.model_dir)
    import network as core
    SPEC = core.GPU_SPEC_FEATURES
    NCU_COLS = core.KERNEL_CONFIG_FEATURES + core.WORKLOAD_PROFILE_FEATURES + core.STALL_FEATURES + ["Execution Time"] + FP_COUNTERS

    files = [SPEC_SHEET] + [p for p in (os.path.join(a.weights_dir, "runtime_inputs_training.csv"), a.target_specs, a.runtime_inputs) if p]
    specs = read_specs(files)                   # sheet + runtime values (DRAM latency, measured overrides) + user files
    for g in [a.src] + a.tgt.split(","):
        if g not in specs:
            sys.exit(f"no specs for {g}: not a column of {SPEC_SHEET}; pass it with --target-specs")

    def gpu_specs(g):
        if g in PREDICTOR_GPUS:
            # a GPU the predictors know: the values its training rows carried = spec sheet + runtime_inputs_training.csv
            # (e.g. GB200 148 SMs, 6580 GB/s), peaks at the boost clock; the predictor rescales them to the clock the
            # labels were measured at itself, so sm_clock_operating must NOT be applied here
            return {k: v for k, v in specs[g].items() if k != "sm_clock_operating"}
        return specs[g]                         # any other GPU: exactly what the sheet / --target-specs / --runtime-inputs say

    svec, _, _ = spec_features(SPEC, gpu_specs(a.src), a.src)
    tg = {g: spec_features(SPEC, gpu_specs(g), g, a.smem_config) + (g in PREDICTOR_GPUS,) for g in a.tgt.split(",")}

    rows = []
    for f in ncu_files(a.ncu):
        run = os.path.basename(f)[:-4]
        for _, r in read_ncu(f).iterrows():
            for g, (tvec, pk, sms, is_known) in tg.items():
                rec = {"run": run, "Kernel Name": r["Kernel Name"], "src_gpu": a.src, "tgt_gpu": g}
                rec.update({c: r.get(c) for c in NCU_COLS})
                rec.update({f"SRC {c}": svec[j] for j, c in enumerate(SPEC)}); rec.update({f"TGT {c}": tvec[j] for j, c in enumerate(SPEC)})
                rec.update({f"TGT {k}": v for k, v in pk.items()})
                if not is_known:
                    rec["TGT n_sms"] = sms      # a GPU the predictors do not know needs its SM count
                rows.append(rec)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(rows)} rows ({len(rows) // max(len(tg), 1)} kernels x {len(tg)} targets)")


if __name__ == "__main__":
    main()
