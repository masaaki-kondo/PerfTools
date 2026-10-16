"""v3 cross-GPU kernel performance estimator -- CLI (self-contained).

v3 = v4.1 (no-ET pure NN) + per-prediction CONFIDENCE and error breakdown.
v4.1 was the released NO-ET pure NN (no Execution Time as a predictive input; no
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
  python MLP_NN/v40/predict_v41.py \
         --csv MLP_NN/examples/example_inputs.csv --row all --out pred.csv
  python MLP_NN/v40/predict_v41.py \
         --csv MLP_NN/examples/example_inputs.csv --row 3 --out row3.csv
"""
# RELEASE 2026-10-16 (public): weights in v322_artifact/; specs from data/gpu_microarch_specs_v3.csv; actual values
# from runtime_inputs_training.csv (+ --runtime-inputs). See README.md in this folder.
import os, sys, csv, re, pickle, argparse, datetime
# v3.1 adds Metro's L1/L2 cache HIT RATES to the workload features. The flag must be
# set BEFORE v15_core is imported, because the feature list is built at import time.
os.environ["HITRATES"] = "1"
os.environ["LAT_ANCHOR"] = "1"
# v3.22 (2026-10-15) = v3.19 recipe + WAVE_ANCHOR=1 (one-wave floor: t >= ET_src / max(waves_src, 1) x per-SM ratio,
# inside t_roof for the physics loss, the collocation grid and inference) + PINN_GRID extended to x256 (the x128
# far-field turn-up of v3.19 was the x32 grid edge). Artifacts: v322_artifact (dummy / unseen) + v322_inrange_artifact.
os.environ["INST_ANCHOR"] = "2"
# v3.22 (2026-10-15) = v3.20 recipe (wave floor C=0.5, grid x256) trained on ALL store data (data/20261015_store_wide.zip:
# every full ncu run of A100/H200/GH200/L40S + MIG 16/32/64-SM cells, USE_MIG_4G=1) + B300 as an nsys-only TARGET
# (INCLUDE_B300=1; see data_pipeline_v2.nsys_only_target). LOGO: 7 folds (h200 h100 a100 gh200 l40s gb200 b300).
# v3.22 (2026-10-15) = v3.21 + ROOFLINE_SLOPES=1 (training only: per-axis slope targets = roofline asymptotes instead of the
# confounded fit). Inference identical to v3.21. Single-axis sweeps (scale_gpu_sweep AXIS=bw|comp|sm) are the acceptance test.
os.environ["WAVE_ANCHOR"] = "1"            # one-wave floor inside t_roof (trained with it)
os.environ["WAVE_ANCHOR_C"] = "0.5"        # trained at C=0.5 (data-supported: 3.4% of real rows below it, vs 22% at C=1)
os.environ["SPEC_CLOCK"] = "operating"      # peaks at the clock the labels were measured at (2026-10-15)
os.environ["MEM_CAP"] = "1"                 # memory-efficiency cap at inference (L40S 58.3 -> 46.9; see data_pipeline_v2.t_memcap_of)
os.environ.setdefault("COMP_CAP", "1")        # compute-saturation cap at inference (data_pipeline_v2.t_compcap_of), 2026-10-16
os.environ.setdefault("AXIS_TRANSPORT", "1")          # single-axis extrapolation fix (data_pipeline_v2.axis_weight / REF_BUILDER), 2026-10-16
# v3.18 (2026-10-07) = v3.17 trained with SPEC_DROPOUT=0.2 (spec-feature dropout, training only) and scored
# with the out-of-envelope residual cap (TRUST_MODE=cap: see v322_core.predict_seeds). LOGO vs v3.17, native
# truth: H200 27.2 -> 19.5, H100 27.9 -> 22.5, A100 51.6 -> 34.2, GH200 31.0 -> 32.3, L40S 47.9 -> 47.9,
# GB200 54.7 -> 46.8; in-sample 9.8 -> 16.7 (use v3.17 for targets that are in the training set).
# v3.22 (2026-10-15) = v3.18 recipe retrained on the NAME-FIXED labels (2026-10-09 nsys/ncu kernel-name join fix;
# pre-fix artifacts stay under v3.18/). Weights = v3.18/v318nf_artifact (dummy / unseen targets, spec dropout 0.2 +
# trust cap) and v3.18/v317nf_inrange_artifact (in-range targets, v3.17 recipe), promoted here as v322_artifact and
# v322_inrange_artifact. LOGO native truth: H200 18.5, H100 23.7, A100 34.4, GH200 31.0, L40S 47.9, GB200 48.9
# (6-fold mean 34.1); in-sample 11.4 (routed); 16x far field COMM/COMP/LAT .052/.058/.038 (ideal .062).
os.environ.setdefault("TRUST_Z", "6,12")
os.environ.setdefault("TRUST_MODE", "cap")
# v3.18: v3.15 (PINN, native labels) + latency floor + instruction floor on t_roof (the far-field blow-up is driven by zero-flop kernels)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..")); sys.path.insert(0, HERE)   # repo root (shared MLP_NN package) + own v15_core copy
# Spec sheet (specs only) + runtime inputs (actual values: training defaults, then the user's --runtime-inputs files).
os.environ.setdefault("GPU_SPEC_SHEET", os.path.join(HERE, "..", "..", "data", "gpu_microarch_specs_v3.csv"))
_ri = [os.path.join(HERE, "runtime_inputs_training.csv")]
for _i, _a in enumerate(sys.argv):
    if _a == "--runtime-inputs" and _i + 1 < len(sys.argv):
        _ri.append(os.path.abspath(sys.argv[_i + 1]))
os.environ["GPU_RUNTIME_INPUTS"] = os.pathsep.join(_ri)
import numpy as np
import pandas as pd
import torch
import v15_core as core
from MLP_NN import data_pipeline_v2 as dp   # spec_delta / GPU dicts (single source of truth)                    # feature extractors / MultiBranchMLP (shared)
import v322_core                            # v3 no-ET inference + confidence

SPEC = core.GPU_SPEC_FEATURES
SRC_PFX, TGT_PFX = "SRC ", "TGT "
OUT_COLS = ["Execution Time [ns]", "Memory Throughput [%]", "Achieved Occupancy",
            "brk_memory", "brk_pipeline_contention", "brk_sync",
            "brk_scheduling_overhead", "t_mem_ns", "t_comp_ns", "t_roof_ns", "efficiency_eta", "trust_w"]
FP_COUNTERS = [
    "Predicated-On FFMA Operations Per Cycle [inst]",
    "Predicated-On FADD Thread Instructions Executed Per Cycle [inst/cycle]",
    "Predicated-On FMUL Thread Instructions Executed Per Cycle [inst/cycle]",
    "Predicated-On DFMA Operations Per Cycle [inst]",
    "Predicated-On DADD Thread Instructions Executed Per Cycle [inst/cycle]",
    "Predicated-On DMUL Thread Instructions Executed Per Cycle [inst/cycle]",
    "Elapsed Cycles [cycle]",
    "Executed Instructions [inst]"]        # INST_ANCHOR: compute anchor for zero-flop kernels
NCU_COLS = (core.KERNEL_CONFIG_FEATURES + core.WORKLOAD_PROFILE_FEATURES
            + core.STALL_FEATURES + ["Execution Time"] + FP_COUNTERS)
SPEC_COLS = ([f"{SRC_PFX}{c}" for c in SPEC] + [f"{TGT_PFX}{c}" for c in SPEC]
             + ["TGT peak_fp32", "TGT peak_fp64", "TGT dram_bw"])
INPUT_COLS = ["src_gpu", "tgt_gpu"] + NCU_COLS + SPEC_COLS
_flag = lambda c: "--" + re.sub(r"[^a-z0-9]+", "-", c.lower()).strip("-")
_dest = lambda c: re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_")
DEST2COL = {_dest(c): c for c in INPUT_COLS}



def _table_specs(gpu):
    """Under SPEC_CLOCK=operating the 8 sheet-derived spec entries and the peaks of a REAL GPU are taken from the
    pipeline tables (operating-clock scaled) instead of the CSV, which was written at the datasheet boost clock.
    The 5 NCU-architectural entries and any GPU unknown to the tables (dummies) keep the CSV values."""
    g = str(gpu).lower()
    if os.environ.get("SPEC_CLOCK", "boost") != "operating" or g not in dp.GPU_PEAK_SPECS or g not in dp.GPU_HW_FALLBACK:
        return None
    pk, hw, lb = dp.GPU_PEAK_SPECS[g], dp.GPU_HW_FALLBACK[g], dp.GPU_LATENCY_BW_SPECS.get(g, {})
    n = hw["n_sms"]
    d = {"Peak FP32 / Peak FP64": pk["peak_fp32"] / pk["peak_fp64"], "DRAM BW per SM [byte/s/SM]": hw["dram_bw"] / n,
         "L2 Size per SM [byte/SM]": pk["l2_size"] / n, "Peak FP32 / DRAM BW [flop/byte]": pk["peak_fp32"] / hw["dram_bw"],
         "Peak FP16 Tensor [flop/s]": pk["peak_tensor_fp16"], "Peak Tensor / Peak FP32": pk["peak_tensor_fp16"] / pk["peak_fp32"],
         "DRAM Latency [ns]": lb.get("dram_latency_ns", float("nan")), "CPU-GPU BW [byte/s]": lb.get("cpu_gpu_bw", float("nan")),
         "Host Link Coherent [0/1]": lb.get("host_coherent", float("nan")), "Host PCIe Gen": lb.get("host_pcie_gen", float("nan"))}
    return d, {"peak_fp32": pk["peak_fp32"], "peak_fp64": pk["peak_fp64"], "dram_bw": hw["dram_bw"], "peak_inst": hw.get("peak_inst", float("nan"))}


def _is_unscaled(row, gpu, pfx):
    """True when the row describes the REAL GPU, not a scaled variant such as scale_gpu_sweep's 'f x H200' (same
    tgt_gpu name, some capacity columns x f). Sources are never scaled. For targets every absolute column that a
    single-axis sweep can touch must match the datasheet: 'TGT dram_bw' (bandwidth axis), 'TGT peak_fp32' against the
    BOOST-clock peak (compute axis; the table holds the operating-clock value), 'TGT n_sms' when present (SM axis).
    Per-SM columns are NOT used (real rows can carry a different SM count: GB200 152 vs 148, B300 132 vs 160 -- those
    rows NEED the table override). 2026-10-15."""
    if pfx == SRC_PFX:
        return True
    g = str(gpu).lower(); hw = dp.GPU_HW_FALLBACK.get(g); pk = dp.GPU_PEAK_SPECS.get(g)
    if not hw or not pk:
        return False
    def near(col, ref):
        try:
            v = float(row.get(col, float("nan")))
        except (TypeError, ValueError):
            return True                      # column absent -> cannot be scaled through it
        return (v != v) or (ref > 0 and abs(v / ref - 1.0) < 0.02)
    scale = dp.GPU_CLOCKS.get(g, {}).get("scale") or 1.0
    if (near(TGT_PFX + "dram_bw", hw["dram_bw"]) and near(TGT_PFX + "peak_fp32", pk["peak_fp32"] / scale)
            and near(TGT_PFX + "n_sms", hw["n_sms"])):
        return True
    # 2026-10-16: also the REAL GPU when the row carries the values of a published spec sheet for it (inputs built
    # from data/gpu_microarch_specs*.csv, e.g. the public example_inputs.csv: GB200 8000 GB/s vs the training 6580).
    return any(near(TGT_PFX + "dram_bw", bw) and near(TGT_PFX + "peak_fp32", f32) and near(TGT_PFX + "n_sms", sms)
               for bw, f32, sms in _sheet_refs().get(g, []))


_SHEET_REFS = None
def _sheet_refs():
    """{gpu: [(dram_bw B/s, boost peak_fp32 FLOP/s, n_sms), ...]} from every spec sheet shipped next to the code."""
    global _SHEET_REFS
    if _SHEET_REFS is None:
        import csv, glob
        _SHEET_REFS = {}
        root = os.path.join(HERE, "..", "..", "data")
        paths = sorted(set(glob.glob(os.path.join(root, "gpu_microarch_specs*.csv")) + [os.environ.get("GPU_SPEC_SHEET", "")]) - {""})
        for p in paths:
            try:
                rows = {r["spec"]: r for r in csv.DictReader(open(p))}
            except (OSError, KeyError):
                continue
            for col in rows.get("dram_bandwidth", {}):
                if col in ("spec", "unit", "confidence_notes", "source", "notes"):
                    continue
                try:
                    t = (float(rows["dram_bandwidth"][col]) * 1e9, float(rows["peak_fp32"][col]) * 1e12, float(rows["num_sms"][col]))
                except (KeyError, TypeError, ValueError):
                    continue
                _SHEET_REFS.setdefault(col.lower(), []).append(t)
    return _SHEET_REFS


def _apply_table_specs(row, src_vec, tgt_vec, peaks):
    for gpu, vec, pfx in ((row.get("src_gpu"), src_vec, SRC_PFX), (row.get("tgt_gpu"), tgt_vec, TGT_PFX)):
        ts = _table_specs(gpu)
        if ts is None or not _is_unscaled(row, gpu, pfx):
            continue
        for j, c in enumerate(SPEC):
            if c in ts[0]:
                vec[j] = ts[0][c]
        if vec is tgt_vec:
            peaks.update(ts[1])
    return src_vec, tgt_vec, peaks

def _ref_sample(sample, gpu):
    """AXIS_TRANSPORT reference: the same kernel and source, target replaced by the REAL GPU `gpu` (table specs at the
    operating clock; the 5 NCU-architectural entries are kept from the requested target)."""
    ts = _table_specs(gpu)
    if ts is None or "_row" not in sample:
        return None
    tv = np.array(sample["_tgt_vec"], float).copy()
    for j, c in enumerate(SPEC):
        if c in ts[0]:
            tv[j] = ts[0][c]
    row = dict(sample["_row"]); hw = dp.GPU_HW_FALLBACK[str(gpu).lower()]; lb = dp.GPU_LATENCY_BW_SPECS.get(str(gpu).lower(), {})
    for j, c in enumerate(SPEC):
        row[TGT_PFX + c] = float(tv[j])
    row[TGT_PFX + "n_sms"] = float(hw["n_sms"]); row[TGT_PFX + "dram_bw"] = ts[1]["dram_bw"]
    row[TGT_PFX + "peak_fp32"] = ts[1]["peak_fp32"]; row[TGT_PFX + "peak_fp64"] = ts[1]["peak_fp64"]; row[TGT_PFX + "peak_inst"] = ts[1]["peak_inst"]
    row["tgt_gpu"] = gpu
    return _build_sample(row, sample["src_gpu"], gpu, sample["_src_vec"], tv, ts[1], sample["_ro"], sample["_bo"])


dp.REF_BUILDER = _ref_sample


def resolve_specs(row):
    if (SRC_PFX + SPEC[0]) not in row or row.get(SRC_PFX + SPEC[0]) != row.get(SRC_PFX + SPEC[0]):
        return None
    src_vec = np.array([float(row[SRC_PFX + c]) for c in SPEC])
    tgt_vec = np.array([float(row[TGT_PFX + c]) for c in SPEC])
    peaks = {"peak_fp32": float(row[TGT_PFX + "peak_fp32"]),
             "peak_fp64": float(row[TGT_PFX + "peak_fp64"]), "dram_bw": float(row[TGT_PFX + "dram_bw"])}
    try:
        peaks["peak_inst"] = float(row[TGT_PFX + "peak_inst"])
    except (KeyError, TypeError, ValueError):
        peaks["peak_inst"] = float("nan")
    return _apply_table_specs(row, src_vec, tgt_vec, peaks)


def build_sample(srow, src_gpu, tgt_gpu, src_vec, tgt_vec, peaks, reg_out, brk_out):
    _out = _build_sample(srow, src_gpu, tgt_gpu, src_vec, tgt_vec, peaks, reg_out, brk_out)
    _out.update(_row=srow, _src_vec=np.array(src_vec, float), _tgt_vec=np.array(tgt_vec, float), _peaks=dict(peaks), _ro=reg_out, _bo=brk_out)
    return _out


def _build_sample(srow, src_gpu, tgt_gpu, src_vec, tgt_vec, peaks, reg_out, brk_out):
    s = pd.Series(srow)
    ratio = dp.spec_delta(tgt_vec, src_vec)          # ratio / log-ratio / categorical diff -- must match training

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
    inst_ = g0("Executed Instructions [inst]")
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
        "roofline_inst": inst_, "target_peak_inst": peaks.get("peak_inst", float("nan")),
        # LAT_ANCHOR ingredients (no-ops unless the artifact was trained with it)
        # NaN-preserving on purpose: the trainer keeps NaN here, and a NaN Mem% must NOT become
        # 0 (= "latency class") -- that re-classified 6 degenerate GB200 kernels and blew them up 333x.
        "src_dur_ns": float(pd.to_numeric(s.get("Execution Time"), errors="coerce")),
        "src_mem_pct": float(pd.to_numeric(s.get("Memory Throughput [%]"), errors="coerce")),
        "src_waves": float(pd.to_numeric(s.get("Waves Per SM"), errors="coerce")),      # WAVE_ANCHOR / WAVE_CLAMP
        "src_dram_latency": dp.GPU_LATENCY_BW_SPECS.get(str(src_gpu).lower(), {}).get("dram_latency_ns", float("nan")),
        "target_dram_latency": float(s.get(TGT_PFX + "DRAM Latency [ns]", float("nan")) or float("nan")),
        "src_n_sms": dp.GPU_HW_FALLBACK.get(str(src_gpu).lower(), {}).get("n_sms", float("nan")),
        "target_n_sms": (float(s[TGT_PFX + "n_sms"]) if (TGT_PFX + "n_sms") in s.index and s[TGT_PFX + "n_sms"] == s[TGT_PFX + "n_sms"]
                         else (peaks["dram_bw"] / float(s.get(TGT_PFX + "DRAM BW per SM [byte/s/SM]", float("nan")))
                               if float(s.get(TGT_PFX + "DRAM BW per SM [byte/s/SM]", 0) or 0) > 0 else float("nan"))),
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
                    "t_roof_ns": float(t_roof[i]), "efficiency_eta": float(eta[i]),
                    "trust_w": float(P.get("trust_w", np.ones(len(samples)))[i])})
    return out


PROVENANCE = [
    "# v3 cross-GPU kernel estimator (no-ET NN + confidence) -- output",
    "# INPUT SOURCES: kernel stats = NCU on the SOURCE GPU; GPU specs = spec sheet (SRC/TGT cols);",
    "#   roofline terms COMPUTED in-tool (bytes=MemThr[byte/s]*ExecTime; flops=(2*FFMA+FADD+FMUL)*cycles).",
    "# MODEL: v3 -- Execution Time is NOT a predictive input; inference is a plain NN forward pass.",
    "# CONFIDENCE (--confidence): conf_* from 5-seed ensemble spread + roofline consistency; exp_err_* = measured typical error for that bucket.",
    "# OUTPUTS: ExecTime/Mem%/Occupancy/brk_* = v3 predictions; t_*_ns + efficiency_eta = roofline.",
]


def _to_num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


# ROUTER (2026-10-08): v3.17 and v3.18 sit at two ends of one trade-off -- spec dropout (v3.18) buys
# extrapolation to GPUs the model never saw (LOGO H200 27.2 -> 19.5, A100 51.6 -> 34.2) at the cost of
# accuracy on GPUs it did see (in-sample 9.8 -> 16.7). One set of weights cannot have both (the dropout-rate
# sweep moved along that line, never off it), so the merged estimator picks per ROW by the TARGET GPU:
#   target in the training set  -> in-range artifact (v3.17 recipe), no trust cap
#   anything else (dummy / new) -> v3.18 artifact, out-of-envelope residual cap on
TRAINED_TARGETS = {"a100", "h100", "gb200", "h200", "gh200", "l40s", "b300",      # b300: trained target from v3.22
                   "h100_mig_1g", "h100_mig_2g", "h100_mig_3g"}      # production pool (USE_MIG_4G=0)
_PER_ROW = ("trust_w", "et_p", "mem", "occ", "brk", "troof")
_PER_SEED = ("et_seeds", "mem_seeds", "occ_seeds", "brk_seeds", "log_et_seeds")


def _is_trained(tgt):
    return str(tgt).strip().lower().replace("-", "_").replace(" ", "_") in TRAINED_TARGETS


def route_predict(models, meta, samples, args):
    n = len(samples)
    inr = None; ia = args.inrange_artifact
    if ia is None and os.path.realpath(args.artifact) == os.path.realpath(os.path.join(HERE, "v322_artifact")):
        ia = None   # release 2026-10-16: in-range routing OFF until a v3.21 in-range model is tested (pass --inrange-artifact to opt in)
    if not args.no_route and ia and os.path.exists(os.path.join(ia, "meta.pkl")):
        inr = v322_core.load(ia)
    known = np.array([inr is not None and _is_trained(s["tgt_gpu"]) for s in samples], dtype=bool)
    if not known.any():
        return v322_core.predict_seeds(models, meta, samples), ["v3.22"] * n
    parts = {}
    saved = {k: os.environ.pop(k, None) for k in ("TRUST_Z", "TRUST_MODE")}   # in-range: no cap
    try:
        parts[True] = v322_core.predict_seeds(inr[0], inr[1], [s for s, k in zip(samples, known) if k])
    finally:
        for k, v in saved.items():
            if v is not None: os.environ[k] = v
    if (~known).any():
        parts[False] = v322_core.predict_seeds(models, meta, [s for s, k in zip(samples, known) if not k])
    R = {}
    for key in parts[True]:
        a = np.asarray(parts[True][key])
        if key in _PER_ROW:
            out = np.empty((n,) + a.shape[1:], dtype=a.dtype); out[known] = a
            if False in parts: out[~known] = np.asarray(parts[False][key])
        elif key in _PER_SEED:
            if False in parts and np.asarray(parts[False][key]).shape[0] != a.shape[0]:
                R[key] = None; continue          # different seed counts: per-seed outputs not mergeable
            out = np.empty((a.shape[0], n) + a.shape[2:], dtype=a.dtype); out[:, known] = a
            if False in parts: out[:, ~known] = np.asarray(parts[False][key])
        else:
            raise KeyError(f"router: unknown predict_seeds output {key!r}")
        R[key] = out
    return R, ["v3.22-inrange" if k else "v3.22" for k in known]


def main():
    ap = argparse.ArgumentParser(
        description="v3 cross-GPU kernel estimator (no-ET NN + confidence). Two input modes: "
                    "(1) full input via CLI options, or (2) --csv FILE [--row N].")
    ap.add_argument("--artifact", default=os.path.join(HERE, "v322_artifact"))   # release 2026-10-16: v3.21 production weights (3 seeds)
    ap.add_argument("--inrange-artifact", default=None,
                    help="v3.17-recipe artifact (no spec dropout) used for TARGET GPUs that are in the "
                         "training set. Default: v322_inrange_artifact, but ONLY when --artifact is the "
                         "production v322_artifact -- a LOGO fold artifact is never routed (the in-range "
                         "model has seen the held-out GPU, which would leak). See ROUTER below.")
    ap.add_argument("--no-route", action="store_true", help="use --artifact for every row")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runtime-inputs", metavar="CSV", action="append",
                    help="your actual GPU values (spec,unit,<GPU columns>) applied on top of runtime_inputs_training.csv; repeatable")
    ap.add_argument("--log")
    ap.add_argument("--no-comments", action="store_true")
    ap.add_argument("--confidence", action="store_true",
                    help="add per-output confidence (ensemble spread + roofline consistency): "
                         "conf_* label, exp_err_* expected error, and the raw signals")
    ap.add_argument("--truth", metavar="PATH",
                    help="row-matched ground-truth CSV (meta-/S- schema); prints an error "
                         "breakdown of the predictions against it and adds error columns")
    ap.add_argument("--dump-records", metavar="PATH",
                    help="also write a per-kernel profiling record (io_signal schema: "
                         "meta-/I-/O-; PREDICTIONS ONLY, no truth) to PATH")
    m2 = ap.add_argument_group("Mode 2: CSV input")
    m2.add_argument("--csv", "--kernel-stats", dest="csv", help="input CSV (all columns)")
    m2.add_argument("--row", help="REQUIRED with --csv: a 1-based data row number, "
                                  "or 'all' to predict every data row (the header is metadata)")
    m1 = ap.add_argument_group("Mode 1: full input via CLI options (one prediction)")
    for c in INPUT_COLS:
        m1.add_argument(_flag(c), dest=_dest(c), metavar="V", help=c)
    args = ap.parse_args()

    models, meta = v322_core.load(args.artifact)
    reg_out, brk_out = meta["reg_out"], meta["brk_out"]
    # sources the pipeline has datasheet values for (src SM count / DRAM latency for the latency floor).
    # v3.18 trains on H200 / GH200 / L40S as sources too, so they are valid here (2026-10-09).
    known = set(dp.GPU_HW_FALLBACK) & set(dp.GPU_LATENCY_BW_SPECS) & set(dp.GPU_PEAK_SPECS)

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
    R, routed = route_predict(models, meta, samples, args)
    out_rows = to_rows(R, samples)
    for r, how in zip(out_rows, routed): r["model_used"] = how
    cols_extra = []
    if args.confidence or args.truth:
        C = v322_core.confidence(R, samples)
        for i, r in enumerate(out_rows):
            for k in ("ExecTime", "Mem%", "Occupancy", "breakdown"):
                r[f"conf_{k}"] = C[f"conf_{k}"][i]
                r[f"exp_err_{k}"] = float(C[f"exp_err_{k}"][i])
            for k in ("ens_cv_et", "ens_sd_mem", "ens_sd_occ", "ens_sd_brk",
                      "eta", "eta_violation", "log_dist_roof", "mem_consistency"):
                r[k] = float(C[k][i])
        cols_extra = ([f"conf_{k}" for k in ("ExecTime", "Mem%", "Occupancy", "breakdown")]
                      + [f"exp_err_{k}" for k in ("ExecTime", "Mem%", "Occupancy", "breakdown")]
                      + ["ens_cv_et", "ens_sd_mem", "ens_sd_occ", "ens_sd_brk",
                         "eta", "eta_violation", "log_dist_roof", "mem_consistency"])
    if args.truth:
        cols_extra += v322_core.attach_errors(out_rows, args.truth)
    if args.confidence:
        v322_core.print_confidence_legend()
        import collections as _c
        sys.stderr.write("\nconfidence of THIS run:\n")
        for _k in ("ExecTime", "Mem%", "Occupancy", "breakdown"):
            _d = _c.Counter(r[f"conf_{_k}"] for r in out_rows)
            _n = max(sum(_d.values()), 1)
            sys.stderr.write(f"  {_k:<11}" + "  ".join(
                f"{_l}={_d.get(_l,0):>5} ({100*_d.get(_l,0)/_n:4.1f}%)"
                for _l in ("high", "medium", "low")) + "\n")

    cols = ["kernel_name", "src_gpu", "tgt_gpu"] + OUT_COLS + ["model_used"] + cols_extra
    with open(args.out, "w", newline="") as f:
        if not args.no_comments:
            f.write("\n".join(PROVENANCE) + "\n")
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in out_rows:
            w.writerow({c: (f"{r[c]:.6g}" if isinstance(r[c], float) else r[c]) for c in cols})

    if args.dump_records:
        n = core.dump_records(args.dump_records, "v3", samples, out_rows,
                              v322_core._features, core.DERIVED_NAMES_V40, args.no_comments)
        sys.stderr.write(f"records  : {n} rows -> {args.dump_records}\n")

    log = [f"v3 cross-GPU estimator (no-ET NN + confidence)  run @ {datetime.datetime.now().isoformat(timespec='seconds')}",
           f"model    : v3  ({meta['seeds']}-seed ensemble)", "specs    : read from the input (SRC/TGT columns/options)",
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
