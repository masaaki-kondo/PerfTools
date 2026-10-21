#!/usr/bin/env python3
"""Predict ONE profiled application run on ONE GPU described by a spec sheet (e.g. a dummy GPU).

What it does
    1. Reads the run's ncu profile on a real BASE GPU (Yoshida-san's long-form `ncu --csv` file) and turns it into
       one row per kernel, with the same conversion as the training data.
    2. Reads the same run's nsys kernel-time file for the launch count and the measured time of every kernel.
    3. Predicts every kernel twice with the published model: on the base GPU itself (1x reference) and on the
       target GPU taken from the spec sheet (absolute values, one column of the sheet).
    4. Sums kernels x launches into the application time and reports the speedup target vs base.

    app time   = sum over kernels of (predicted time per launch x number of launches)
    speedup    = predicted base time / predicted target time
    anchored   = measured base time / speedup   (measured level, model speedup: the column to compare)

Kernels without an nsys launch count, or without a finite prediction, are left out of every total (never counted as
zero) and are listed on stderr. Gaps between launches, host code and copies are not modelled.

Usage (from anywhere; --repo is the PerfTools checkout that holds the models, default: the parent of this file's folder)
    python examples/predict_app.py \
        --ncu   <HeCBench>/H200/ncu-full/ace-cuda__f899139d.csv \
        --nsys  <HeCBench>/H200/nsys-kernel-time/ace-cuda__f899139d.csv \
        --base  H200 \
        --specs data/dummy_gpu_specs_ship5.csv --gpu G1_2X \
        --model mlpnn --out results.csv

Spec sheet: rows `spec,unit,<GPU columns>...` (the layout of data/dummy_gpu_specs_ship5.csv). Rows read:
    num_sms [count], dram_bandwidth [GB/s], l2_cache_total [MB], peak_fp32 / peak_fp64 / peak_fp16_tensor [TFLOPS],
    dram_latency_model_recommended (or _measured / _paper) [ns], cpu_gpu_bandwidth [GB/s], sm_clock_boost (or sm_clock) [GHz].
Other rows (architecture, notes, ...) are ignored. Values are taken as absolute specs of the target GPU.

Output: one line per call on stdout, and the same as one row appended to --out (header written when the file is new).

Error and confidence (one `status` per call):
    known         --target-nsys given (the run was also measured on the target): the ACTUAL error, model_err_target_%.
    interpolated  target inside the trained range: likely_target_ms_lo / _hi = the measured time is expected in this range
                  for 8 in 10 apps (10th-90th percentile of predicted / measured over 1,223 app runs on held-out GPUs,
                  i.e. GPUs the model never saw; BAND below).
    extrapolated  target outside the trained range (more SMs, bandwidth or FLOPs than any trained GPU, or outside the
                  model's own trust envelope): no calibrated range exists; only the seed spread is given.
    Seed spread (every status): *_seed_min / *_seed_max = the same prediction from each ensemble member alone. It shows
    how much the model is unsure of itself, not how far it is from the truth.
"""
import argparse, csv, io, os, re, subprocess, sys, tempfile

# ----------------------------------------------------------------------------------------------------------------- #
# Pipeline flags. They select the feature lists of the input CSV and must be set BEFORE the repo modules are         #
# imported. The predictors set their own model flags (LOG_SPECS, WAVE_ANCHOR, MEM_CAP, ...) themselves.              #
# ----------------------------------------------------------------------------------------------------------------- #
os.environ.update(HITRATES="1", LOG_SPECS="0", DROP_C2C="0", LAT_ANCHOR="1", INST_ANCHOR="2", SPEC_CLOCK="operating")

import numpy as np
import pandas as pd

# Published models: (predictor script, artifact) relative to the repo root.
MODELS = {"mlpnn": ("MLP_NN/mlp-nn.py", "MLP_NN/v4.55"),   # MLP-NN v4.55
          "pinn":  ("PINN/pinn.py", "PINN/v3.25")}         # PINN v3.25

# Base GPUs: the GPUs with ncu training profiles (B300 has nsys only, so it cannot be a base).
# ARCH = the 5 NCU-architectural spec inputs (warps per scheduler, active warps per SM / per scheduler, shared-memory
# configuration size, block limit warps) at their representative values: the per-GPU medians over the GPU's ncu
# profiles, which every reported accuracy uses (= max_warps_per_scheduler, max_warps_per_sm, 32 KB / 16 KB, warps / 8;
# see examples/kernel_inputs/prepare_data.py). Training itself uses each kernel's own values.
ARCH = {"A100": (16, 64, 16, 32768, 8), "H100": (16, 64, 16, 32768, 8), "H200": (16, 64, 16, 32768, 8),
        "GH200": (16, 64, 16, 32768, 8), "GB200": (16, 64, 16, 32768, 8), "L40S": (12, 48, 12, 16384, 6)}

GB, TF, MB = 1e9, 1e12, 1024 ** 2          # sheet units -> SI (MB as the pipeline uses it)

# App-level held-out error: 10th / 90th percentile of predicted / measured app time over the 7 leave-one-GPU-out folds
# (1,223 app runs; app time = kernel time x launches, as here). Measured time likely in [pred / p90, pred / p10].
BAND = {"mlpnn": (0.7172, 1.2869), "pinn": (0.6540, 1.2677)}

# GPUs the models were trained on (full GPUs; MIG cells down to 16 SMs were also in training).
TRAINED = ["a100", "h100", "h200", "gh200", "gb200", "l40s", "b300"]
MIN_TRAINED_SMS = 16


# ----------------------------------------------------------------------------------------------------------------- #
# 1. ncu profile: long form (one row per kernel launch x metric) -> one row per kernel                               #
#    Same rules as the training-data conversion: the FIRST value seen for a (kernel, metric) is kept, and a        #
#    missing / non-numeric value stays NaN (never 0).                                                                #
# ----------------------------------------------------------------------------------------------------------------- #
NA_TOKENS = {"", "n/a", "N/A", "—", "nan", "NaN", "None", "none"}
IDENTITY_COLS = ("Block Size", "Grid Size", "Device", "CC", "Context", "Stream", "Process Name", "Host Name")


def _parse_value(v):
    """ncu 'Metric Value' -> float; NaN for empty / n/a / non-numeric (thousands separators removed)."""
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
    """Kernel table of one ncu-full file, with the column names the models were trained on."""
    raw = open(path, "rb").read()
    lines = raw.decode("utf-8", "replace").splitlines()
    # the header is the line starting '"ID","Process ID"'; program output and '#' comments can precede it
    hdr = next((i for i, l in enumerate(lines) if l.startswith('"ID","Process ID"')), None)
    if hdr is None:
        hdr = next((i for i, l in enumerate(lines[:80]) if l.startswith('"')), None)
    if hdr is None:
        sys.exit(f"{path}: no ncu CSV header found")
    df = pd.read_csv(io.BytesIO(raw), skiprows=hdr, low_memory=False, na_values=list(NA_TOKENS), keep_default_na=True)
    if not {"Kernel Name", "Metric Name", "Metric Value"} <= set(df.columns):
        sys.exit(f"{path}: not a long-form ncu CSV (needs Kernel Name / Metric Name / Metric Value)")
    df = df[df["Kernel Name"].notna() & df["Metric Name"].notna()].reset_index(drop=True)
    unit = df["Metric Unit"].fillna("").astype(str).str.strip() if "Metric Unit" in df.columns else pd.Series("", index=df.index)
    col = pd.Series(np.where(unit != "", df["Metric Name"].astype(str) + " [" + unit + "]", df["Metric Name"].astype(str)), index=df.index)
    val = df["Metric Value"].map(_parse_value)
    rows = []
    for kname, grp in df.groupby("Kernel Name", sort=False):
        row = {"Kernel Name": kname}
        for k in IDENTITY_COLS:
            if k in df.columns:
                row[k] = grp.iloc[0].get(k, np.nan)
        seen = {}
        for c, v in zip(col.loc[grp.index], val.loc[grp.index]):
            seen.setdefault(c, v)                             # first value seen wins
        row.update(seen)    # metrics overwrite same-named identity columns (Block Size (256, 1, 1) -> metric 256)
        rows.append(row)
    # round-trip through CSV text so dtypes are what the pipeline's CSV reader produces
    wide = pd.read_csv(io.StringIO(pd.DataFrame(rows).to_csv(index=False)), low_memory=False)
    wide = wide.rename(columns={"Duration [ns]": "Execution Time", "Achieved Occupancy [%]": "Achieved Occupancy"})
    if "Threads" not in wide.columns and {"Block Size", "Grid Size"} <= set(wide.columns):
        wide["Threads"] = pd.to_numeric(wide["Block Size"], errors="coerce") * pd.to_numeric(wide["Grid Size"], errors="coerce")
    return wide


def read_header(path):
    """'# key: value' lines at the top of a HeCBench profile (benchmark, args, args_source, ...)."""
    h = {}
    for l in open(path, errors="ignore"):
        if not l.startswith("#"):
            break
        if ":" in l:
            k, v = l[1:].split(":", 1); h[k.strip()] = v.strip()
    return h


# ----------------------------------------------------------------------------------------------------------------- #
# 2. nsys kernel times: one line per launch -> per kernel: launch count and mean time                                #
# ----------------------------------------------------------------------------------------------------------------- #
_CAST = re.compile(r"\((?:unsigned |signed )?(?:int|bool|long long|long|char|short|unsigned|size_t)\)")


def norm_kernel(k):
    """nsys writes template arguments with C casts ('<(int)1024, (bool)0>'), ncu without ('<1024, 0>'): drop the casts
    so the two name the same kernel the same way."""
    return _CAST.sub("", str(k))


def read_nsys(path):
    """{normalised kernel name: (launches, mean ns)}; launches whose names normalise alike are pooled."""
    d = {}
    for l in open(path, errors="ignore"):
        if l.startswith("#") or l.startswith("kernel_name"):
            continue
        try:
            r = next(csv.reader([l])); d.setdefault(norm_kernel(r[0]), []).append(float(r[1]))
        except Exception:
            pass
    return {k: (len(v), float(np.mean(v))) for k, v in d.items()}


# ----------------------------------------------------------------------------------------------------------------- #
# 3. GPU specs -> model input columns                                                                                #
# ----------------------------------------------------------------------------------------------------------------- #
def base_specs(dp, gpu, ncu):
    """SRC spec columns of the base GPU (also the TGT columns of the 1x reference), from the model's own GPU tables.
    Mirrors spec_features in examples/kernel_inputs/prepare_data.py."""
    g = gpu.lower(); hw, pk, lb = dp.GPU_HW_FALLBACK[g], dp.GPU_PEAK_SPECS[g], dp.GPU_LATENCY_BW_SPECS[g]
    nsm = pd.to_numeric(ncu.get("# SMs [SM]", pd.Series(dtype=float)), errors="coerce").dropna()
    nsm = float(nsm.iloc[0]) if len(nsm) else hw["n_sms"]     # SMs as the profile saw them, else the table's
    spec = dict(zip(SPEC[:5], ARCH[gpu]))
    spec.update({"Peak FP32 / Peak FP64": pk["peak_fp32"] / pk["peak_fp64"], "DRAM BW per SM [byte/s/SM]": hw["dram_bw"] / nsm,
                 "L2 Size per SM [byte/SM]": pk["l2_size"] / nsm, "Peak FP32 / DRAM BW [flop/byte]": pk["peak_fp32"] / hw["dram_bw"],
                 "Peak FP16 Tensor [flop/s]": pk["peak_tensor_fp16"], "Peak Tensor / Peak FP32": pk["peak_tensor_fp16"] / pk["peak_fp32"],
                 "DRAM Latency [ns]": lb["dram_latency_ns"], "CPU-GPU BW [byte/s]": lb["cpu_gpu_bw"]})
    peaks = {"peak_fp32": pk["peak_fp32"], "peak_fp64": pk["peak_fp64"], "dram_bw": hw["dram_bw"], "peak_inst": hw.get("peak_inst", np.nan)}
    return spec, peaks


def read_sheet(path, gpu):
    """One GPU column of the spec sheet, in SI units. A spec that appears in several units (e.g. latency in cycles and
    in ns) is taken in the unit listed here; DRAM latency falls back from the model-recommended value to the measured
    and then to the paper value (the real-GPU sheet data/v4/gpu_microarch_specs.csv has only the latter)."""
    S = pd.read_excel(path, dtype=object) if path.endswith((".xlsx", ".xls")) else pd.read_csv(path, dtype=object)
    S["spec"] = S["spec"].astype(str).str.replace(r"^\s*\d+\.\s*", "", regex=True).str.strip()
    S["unit"] = S["unit"].astype(str).str.strip() if "unit" in S.columns else ""
    if gpu not in S.columns:
        sys.exit(f"{path}: no column {gpu!r} (columns: {[c for c in S.columns if c not in ('spec', 'unit', 'confidence_notes', 'source')]})")
    def val(names, unit, required=True):
        for n in names:
            m = S[(S["spec"] == n) & (S["unit"].str.lower() == unit.lower())]
            if len(m) == 0:
                m = S[S["spec"] == n]
            v = pd.to_numeric(m[gpu], errors="coerce").dropna() if len(m) else []
            if len(v):                                     # first name that has a value for this GPU
                return float(v.iloc[0])
        if required:
            sys.exit(f"{path}: missing row {names[0]} [{unit}] for {gpu}")
        return np.nan
    return {"sms": val(["num_sms"], "count"), "bw": val(["dram_bandwidth"], "GB/s") * GB, "l2": val(["l2_cache_total"], "MB") * MB,
            "f32": val(["peak_fp32"], "TFLOPS") * TF, "f64": val(["peak_fp64"], "TFLOPS") * TF, "ten": val(["peak_fp16_tensor"], "TFLOPS") * TF,
            "lat": val(["dram_latency_model_recommended", "dram_latency_measured", "dram_latency_paper"], "ns", required=False),
            "cpu": val(["cpu_gpu_bandwidth"], "GB/s") * GB, "clk": val(["sm_clock_boost", "sm_clock"], "GHz") * 1e9}


def target_columns(t):
    """TGT columns of a sheet GPU (same convention as the model's own spec columns, + TGT n_sms). The 5 NCU-architectural
    entries stay those of the base GPU (same SM design)."""
    return {"TGT Peak FP32 / Peak FP64": t["f32"] / t["f64"], "TGT DRAM BW per SM [byte/s/SM]": t["bw"] / t["sms"],
            "TGT L2 Size per SM [byte/SM]": t["l2"] / t["sms"], "TGT Peak FP32 / DRAM BW [flop/byte]": t["f32"] / t["bw"],
            "TGT Peak FP16 Tensor [flop/s]": t["ten"], "TGT Peak Tensor / Peak FP32": t["ten"] / t["f32"],
            "TGT DRAM Latency [ns]": t["lat"], "TGT CPU-GPU BW [byte/s]": t["cpu"],
            "TGT peak_fp32": t["f32"], "TGT peak_fp64": t["f64"], "TGT dram_bw": t["bw"],
            "TGT peak_inst": t["sms"] * 4.0 * t["clk"],           # 4 schedulers / SM, 1 warp instruction / cycle each
            "TGT n_sms": t["sms"]}


# ----------------------------------------------------------------------------------------------------------------- #
# 4. Prediction: the published predictor script, unchanged, on a temporary input CSV                                 #
# ----------------------------------------------------------------------------------------------------------------- #
def predict(repo, model, rows, base):
    script, art = (os.path.join(repo, p) for p in MODELS[model])
    extra = ["--no-route"] if model == "pinn" else []          # PINN: every target from the same weights
    with tempfile.TemporaryDirectory() as td:
        fi, fo = os.path.join(td, "in.csv"), os.path.join(td, "out.csv")
        rows.to_csv(fi, index=False)
        r = subprocess.run([sys.executable, script, "--csv", fi, "--row", "all", "--artifact", art] + extra +
                           ["--out", fo, "--no-comments"], cwd=repo, capture_output=True, text=True,
                           env=dict(os.environ, AT_BASE=base.lower(), PRED_SEED_TIMES="1"))
        # AT_BASE: the real GPU a sheet GPU is built from; PRED_SEED_TIMES: also write each ensemble member's time
        if r.returncode:
            sys.exit(r.stderr[-2000:])
        P = pd.read_csv(fo)
    assert len(P) == len(rows)
    num = lambda c: pd.to_numeric(P[c], errors="coerce").to_numpy()
    seeds = np.column_stack([num(c) for c in P.columns if c.startswith("et_seed")])   # kernels x seeds [ns]
    return {"et": num("Execution Time [ns]"), "seeds": seeds, "trust": num("trust_w"),
            "limited": pd.to_numeric(P.get("issue_limited", pd.Series(0, index=P.index)), errors="coerce").fillna(0).to_numpy()}


def out_of_range(dp, t):
    """Which absolute specs of the sheet GPU lie outside the GPUs the model was trained on (list of names)."""
    hw = {g: dp.GPU_HW_FALLBACK[g] for g in TRAINED}; pk = {g: dp.GPU_PEAK_SPECS[g] for g in TRAINED}
    rng = {"num_sms": (MIN_TRAINED_SMS, max(h["n_sms"] for h in hw.values()), t["sms"]),
           "dram_bandwidth": (min(h["dram_bw"] for h in hw.values()), max(h["dram_bw"] for h in hw.values()), t["bw"]),
           "peak_fp32": (min(p["peak_fp32"] for p in pk.values()), max(p["peak_fp32"] for p in pk.values()), t["f32"]),
           "peak_fp64": (min(p["peak_fp64"] for p in pk.values()), max(p["peak_fp64"] for p in pk.values()), t["f64"]),
           "peak_fp16_tensor": (min(p["peak_tensor_fp16"] for p in pk.values()), max(p["peak_tensor_fp16"] for p in pk.values()), t["ten"])}
    # 10% slack: the sheet holds datasheet (boost-clock) peaks, the tables the clock the labels were measured at
    return [k for k, (lo, hi, v) in rng.items() if not (lo / 1.1 <= v <= hi * 1.1)]


def main():
    global SPEC
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ncu", required=True, help="long-form ncu-full CSV of the run on the base GPU")
    ap.add_argument("--nsys", required=True, help="nsys kernel-time CSV of the same run on the base GPU")
    ap.add_argument("--base", required=True, choices=sorted(ARCH), help="GPU the run was profiled on")
    ap.add_argument("--specs", required=True, help="spec sheet (csv/xlsx) holding the target GPU")
    ap.add_argument("--gpu", required=True, help="target column of the spec sheet")
    ap.add_argument("--model", default="mlpnn", choices=sorted(MODELS))
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), help="PerfTools checkout with the models")
    ap.add_argument("--target-nsys", help="optional: nsys kernel-time CSV of the same run measured ON THE TARGET GPU "
                                          "(a real GPU) -> status 'known' and the actual error")
    ap.add_argument("--out", help="CSV to append the result row to")
    a = ap.parse_args()

    # repo modules: the model's GPU tables and the feature lists (read-only use)
    mdir = os.path.dirname(os.path.join(a.repo, MODELS[a.model][0]))
    # same spec sheet + runtime inputs as the predictor (its GPU tables: base GPU specs at the measured clock)
    os.environ.setdefault("GPU_SPEC_SHEET", os.path.join(a.repo, "data", "v4", "gpu_microarch_specs.csv"))
    os.environ.setdefault("GPU_RUNTIME_INPUTS", os.path.join(a.repo, MODELS[a.model][1], "runtime_inputs_training.csv"))
    sys.path[:0] = [a.repo, mdir]
    import network as core
    import data_pipeline_v2 as dp
    SPEC = core.GPU_SPEC_FEATURES
    NCU_COLS = (core.KERNEL_CONFIG_FEATURES + core.WORKLOAD_PROFILE_FEATURES + core.STALL_FEATURES + ["Execution Time"] +
                ["Predicated-On FFMA Operations Per Cycle [inst]", "Predicated-On FADD Thread Instructions Executed Per Cycle [inst/cycle]",
                 "Predicated-On FMUL Thread Instructions Executed Per Cycle [inst/cycle]", "Predicated-On DFMA Operations Per Cycle [inst]",
                 "Predicated-On DADD Thread Instructions Executed Per Cycle [inst/cycle]", "Predicated-On DMUL Thread Instructions Executed Per Cycle [inst/cycle]",
                 "Elapsed Cycles [cycle]", "Executed Instructions [inst]", "Issued Instructions [inst]"])

    ncu, ns, hd = read_ncu(a.ncu), read_nsys(a.nsys), read_header(a.ncu)
    run = os.path.basename(a.ncu)[:-4]
    spec, peaks = base_specs(dp, a.base, ncu)

    # one input row per kernel; SRC = base GPU, TGT = base GPU (1x reference) -- the sheet GPU is set below
    rows, meas, launches = [], [], []
    for _, r in ncu.drop_duplicates("Kernel Name").sort_values("Kernel Name").iterrows():
        rec = {"Kernel Name": r["Kernel Name"], "src_gpu": a.base, "tgt_gpu": a.base}
        rec.update({c: r.get(c) for c in NCU_COLS})
        rec.update({f"SRC {c}": float(spec[c]) for c in SPEC}); rec.update({f"TGT {c}": float(spec[c]) for c in SPEC})
        rec.update({f"TGT {k}": float(v) for k, v in peaks.items()})
        n, t = ns.get(norm_kernel(r["Kernel Name"]), (np.nan, np.nan))
        rows.append(rec); launches.append(n); meas.append(t)
    X = pd.DataFrame(rows); launches, meas = np.array(launches, float), np.array(meas, float)

    B = predict(a.repo, a.model, X, a.base)                                       # 1x: the base GPU itself
    tspec = read_sheet(a.specs, a.gpu)
    if not tspec["lat"] == tspec["lat"]:                                          # no DRAM latency in the sheet:
        if a.gpu.lower() not in dp.GPU_LATENCY_BW_SPECS:                          # only a real GPU can fall back
            sys.exit(f"{a.specs}: no DRAM latency for {a.gpu}")
        tspec["lat"] = dp.GPU_LATENCY_BW_SPECS[a.gpu.lower()]["dram_latency_ns"]  # the value the model uses for that GPU
    Xt = X.assign(**target_columns(tspec)); Xt["tgt_gpu"] = a.gpu
    T = predict(a.repo, a.model, Xt, a.base)                                      # the sheet GPU
    p_base, p_tgt, lim = B["et"], T["et"], T["limited"]

    ok = np.isfinite(launches) & np.isfinite(meas) & np.isfinite(p_base) & (p_base > 0) & np.isfinite(p_tgt) & (p_tgt > 0)
    for k in X.loc[~ok, "Kernel Name"]:
        print(f"left out (no nsys launch count or no prediction): {k[:100]}", file=sys.stderr)
    ms = lambda t: float((t[ok] * launches[ok]).sum() / 1e6)                      # per-launch ns x launches -> ms
    res = {"run": run, "app": hd.get("benchmark", run.split("__")[0]), "args": hd.get("args", ""),
           "base_gpu": a.base, "target_gpu": a.gpu, "model": a.model, "kernels": int(ok.sum()), "kernels_left_out": int((~ok).sum()),
           "launches": int(launches[ok].sum()), "measured_base_ms": ms(meas), "predicted_base_ms": ms(p_base),
           "predicted_target_ms": ms(p_tgt), "issue_limited_kernels": int(lim[ok].sum())}
    res["model_err_base_%"] = (res["predicted_base_ms"] / res["measured_base_ms"] - 1) * 100   # actual error, base GPU
    res["speedup"] = res["predicted_base_ms"] / res["predicted_target_ms"]
    res["anchored_target_ms"] = res["measured_base_ms"] / res["speedup"]

    # seed spread: the app total from each ensemble member alone (same kernels, same launches)
    tgt_s = np.array([ms(T["seeds"][:, j]) for j in range(T["seeds"].shape[1])])
    spd_s = np.array([ms(B["seeds"][:, j]) for j in range(B["seeds"].shape[1])]) / tgt_s
    res.update({"predicted_target_ms_seed_min": float(tgt_s.min()), "predicted_target_ms_seed_max": float(tgt_s.max()),
                "speedup_seed_min": float(spd_s.min()), "speedup_seed_max": float(spd_s.max())})

    # status: known (measured on the target) / interpolated / extrapolated
    oor = out_of_range(dp, tspec)
    w = (p_tgt * launches)[ok]; untrusted = float(w[T["trust"][ok] < 1].sum() / w.sum())   # time share outside the trust envelope
    res["untrusted_time_share"] = untrusted
    res["out_of_trained_range"] = ";".join(oor)
    res["status"] = "extrapolated" if (oor or untrusted > 0) else "interpolated"
    lo, hi = BAND[a.model]
    res["likely_target_ms_lo"] = res["predicted_target_ms"] / hi if res["status"] == "interpolated" else np.nan
    res["likely_target_ms_hi"] = res["predicted_target_ms"] / lo if res["status"] == "interpolated" else np.nan
    res["measured_target_ms"] = res["model_err_target_%"] = np.nan
    if a.target_nsys:                                                             # measured on the target: actual error
        nt = read_nsys(a.target_nsys)
        mt = np.array([nt.get(norm_kernel(k), (np.nan, np.nan)) for k in X["Kernel Name"]], float)   # (launches, mean ns)
        m2 = ok & np.isfinite(mt[:, 0]) & np.isfinite(mt[:, 1])
        res["measured_target_ms"] = float((mt[m2, 0] * mt[m2, 1]).sum() / 1e6)
        res["model_err_target_%"] = (float((p_tgt[m2] * launches[m2]).sum() / 1e6) / res["measured_target_ms"] - 1) * 100
        res["status"] = "known"

    print(", ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items()))
    if a.out:
        new = not os.path.exists(a.out)
        pd.DataFrame([res]).to_csv(a.out, mode="a", header=new, index=False)


if __name__ == "__main__":
    main()
