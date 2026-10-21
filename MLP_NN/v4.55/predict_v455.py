"""MLP-NN v4.55 -- cross-GPU kernel performance estimator. Self-contained.

Predicts, for CUDA kernels profiled with ncu on a SOURCE GPU, the execution time, memory throughput, occupancy and
stall breakdown on a TARGET GPU (a real GPU, or a hypothetical one described by its specs). See README.md.

Lineage: MLP-NN v4.55 = the v4.46 inference code with the weights of the v4.55 production run (5 seeds).

INPUT   --csv FILE --row {N|all}: one row per kernel (ncu metrics + SRC/TGT spec columns), built with
        MLP_NN/examples/prepare_data.py; or every column as a CLI option (one prediction).
        Specs come from data/v4/gpu_microarch_specs.csv; actual values (operating clocks, DRAM latency, ...) from
        runtime_inputs_training.csv in this folder, plus any --runtime-inputs FILE.
OUTPUT  --out pred.csv: predicted metrics (+ roofline quantities) per kernel. --confidence adds confidence levels.

    python MLP_NN/v4.55/predict_v455.py --csv MLP_NN/examples/example_inputs.csv --row all --out pred.csv
"""
import os, sys, csv, re, pickle, argparse, datetime
# v4.5 adds Metro's L1/L2 cache HIT RATES to the workload features. The flag must be
# set BEFORE v15_core is imported, because the feature list is built at import time.
os.environ["HITRATES"] = "1"
# v4.42: v4.41 (fixed nsys labels) + the PINN floors: latency anchor and instruction anchor,
# so the latency class stays pinned out of the trained envelope (v4.41 16x LAT 0.792, non-monotone).
# v4.46 (2026-10-15) = v4.42 recipe (v4.41 + LAT_ANCHOR=1 INST_ANCHOR=2) with the artifact trained on the NAME-FIXED
# labels (v4.42/v442nf_artifact) promoted as the version default v446_artifact. LOGO native truth: H200 18.5, H100 23.9,
# A100 36.0, GH200 30.2, L40S 50.1, GB200 47.2 (6-fold mean 34.3); in-sample 18.9; 16x far field .128/.146/.138.
os.environ["LOG_SPECS"] = "1"
os.environ["LAT_ANCHOR"] = "1"
# v4.46 (2026-10-15) = v4.43 recipe + WAVE_ANCHOR=1 (one-wave floor inside t_roof, see data_pipeline_v2.t_wave_of).
os.environ["INST_ANCHOR"] = "2"
# v4.46 (2026-10-15) = v4.44 recipe (wave floor C=0.5) trained on ALL store data (data/20261015_store_wide.zip incl. MIG
# 64-SM cell, USE_MIG_4G=1) + B300 as an nsys-only target (INCLUDE_B300=1). LOGO: 7 folds incl. b300.
# RELEASE: default weights = v455_artifact in this folder (the v4.55 production run, 5 seeds) run through this
# inference code (v4.46 line: AXIS_TRANSPORT rev 9, COMP_CAP, MEM_CAP roofline floor, issue limit in the core).
# v4.46 (2026-10-16) = v4.45 + distillation ON with roofline single-axis labels (ROOFLINE_SLOPES=1, NO_DISTILL=0,
# SYN_AXIS to 32x). Training-only change; inference identical to v4.45.
os.environ["WAVE_ANCHOR"] = "1"            # one-wave floor inside t_roof (trained with it)
os.environ["WAVE_ANCHOR_C"] = "0.5"        # trained at C=0.5 (data-supported: 3.4% of real rows below it, vs 22% at C=1)
os.environ["SPEC_CLOCK"] = "operating"      # peaks at the clock the labels were measured at (2026-10-15)
os.environ["MEM_CAP"] = "1"                 # memory-efficiency cap at inference (L40S 58.3 -> 46.9; see data_pipeline_v2.t_memcap_of)
os.environ.setdefault("COMP_CAP", "1")        # compute-saturation cap at inference (data_pipeline_v2.t_compcap_of), 2026-10-16
os.environ.setdefault("AT_BUSY_SMS", "1")             # per-SM bandwidth over busy SMs, hypothetical targets only (data_pipeline_v2.busy_on), 2026-10-18
os.environ.setdefault("AXIS_TRANSPORT", "1")          # single-axis extrapolation fix (data_pipeline_v2.axis_weight / REF_BUILDER), 2026-10-16
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..")); sys.path.insert(0, HERE)   # repo root (shared MLP_NN package) + own v15_core copy
# Spec sheet (specs only) + runtime inputs (actual values: training defaults, then the user's --runtime-inputs files).
os.environ.setdefault("GPU_SPEC_SHEET", os.path.join(HERE, "..", "..", "data", "v4", "gpu_microarch_specs.csv"))
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
import v455_core                            # v4.5 no-ET inference + confidence

SPEC = core.GPU_SPEC_FEATURES
SRC_PFX, TGT_PFX = "SRC ", "TGT "
# ISSUE_FLOOR (instruction-issue limit, 0 = off) is read by the core, v*_core.predict_seeds (2026-10-21)
OUT_COLS = ["Execution Time [ns]", "Memory Throughput [%]", "Achieved Occupancy",
            "brk_memory", "brk_pipeline_contention", "brk_sync",
            "brk_scheduling_overhead", "running_cycles", "exec_cycles", "cyc_running", "cyc_memory", "cyc_pipeline_contention",
            "cyc_sync", "cyc_scheduling_overhead", "t_mem_ns", "t_comp_ns", "t_roof_ns", "efficiency_eta", "trust_w", "issue_limited"]
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
        root = os.path.join(HERE, "..", "..", "data", "v4")
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
        "src_grid": float(pd.to_numeric(s.get("Grid Size"), errors="coerce")),          # AT_BUSY_SMS (2026-10-18)
        "src_issued_inst": float(pd.to_numeric(s.get("Issued Instructions [inst]"), errors="coerce")),   # running_cycles output (2026-10-18)
        "src_dram_latency": dp.GPU_LATENCY_BW_SPECS.get(str(src_gpu).lower(), {}).get("dram_latency_ns", float("nan")),
        "target_dram_latency": float(s.get(TGT_PFX + "DRAM Latency [ns]", float("nan")) or float("nan")),
        "src_n_sms": dp.GPU_HW_FALLBACK.get(str(src_gpu).lower(), {}).get("n_sms", float("nan")),
        "src_clock_ncu": (dp.GPU_CLOCKS.get(str(src_gpu).lower(), {}).get("ncu") or float("nan")),       # LAT_CLOCK only (2026-10-17)
        "target_clock_op": (dp.GPU_CLOCKS.get(str(tgt_gpu).lower(), {}).get("operating") or float("nan")),
        "target_n_sms": (float(s[TGT_PFX + "n_sms"]) if (TGT_PFX + "n_sms") in s.index and s[TGT_PFX + "n_sms"] == s[TGT_PFX + "n_sms"]
                         else (peaks["dram_bw"] / float(s.get(TGT_PFX + "DRAM BW per SM [byte/s/SM]", float("nan")))
                               if float(s.get(TGT_PFX + "DRAM BW per SM [byte/s/SM]", 0) or 0) > 0 else float("nan"))),
        "target_dram_bw": peaks["dram_bw"], "target_peak_fp32": peaks["peak_fp32"],
        "target_peak_fp64": peaks["peak_fp64"]}


def to_rows(P, samples):
    et = np.array(P["et_p"], float); mem = np.array(P["mem"], float)
    # instruction-issue limit: applied in the core (predict_seeds, 2026-10-21); et/mem arrive already limited
    hit = np.asarray(P.get("issue_limited", np.zeros(len(et), bool)), bool)
    pinst = np.array([s.get("target_peak_inst", np.nan) for s in samples], float)
    iss = np.array([s.get("src_issued_inst", np.nan) for s in samples], float)
    b = np.array([s["roofline_bytes"] for s in samples]); bw = np.array([s["target_dram_bw"] for s in samples])
    f32 = np.array([s["roofline_fp32_flops"] for s in samples]); f64 = np.array([s["roofline_fp64_flops"] for s in samples])
    p32 = np.array([s["target_peak_fp32"] for s in samples]); p64 = np.array([s["target_peak_fp64"] for s in samples])
    with np.errstate(divide="ignore", invalid="ignore"):
        t_mem = b / bw * 1e9; t_comp = np.fmax(f32 / p32, f64 / p64) * 1e9
        t_roof = np.fmax(t_mem, t_comp); eta = t_roof / et
    # execution-cycle breakdown (2026-10-19): running + stall = 1; see PROVENANCE
    nsm = np.array([s.get("target_n_sms", np.nan) for s in samples], float)
    stall = [j for j, b in enumerate(core.BRK) if b != "running"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ecyc = et * 1e-9 * pinst / (4.0 * nsm)
        run = np.minimum(iss / (et * 1e-9 * pinst), 1.0)
        sh = P["brk"][:, stall] / P["brk"][:, stall].sum(axis=1, keepdims=True)
    out = []
    for i, s in enumerate(samples):
        out.append({"kernel_name": s["kernel_name"], "src_gpu": s["src_gpu"], "tgt_gpu": s["tgt_gpu"],
                    "Execution Time [ns]": float(et[i]), "Memory Throughput [%]": float(mem[i]),
                    "Achieved Occupancy": float(P["occ"][i]),
                    **{f"brk_{core.BRK[j]}": float(P["brk"][i, j]) for j in range(len(core.BRK))},
                    "running_cycles": float(s.get("src_issued_inst", float("nan"))),
                    "exec_cycles": float(ecyc[i]), "cyc_running": float(run[i]),
                    **{f"cyc_{core.BRK[j]}": float((1.0 - run[i]) * sh[i, k]) for k, j in enumerate(stall)},
                    "t_mem_ns": float(t_mem[i]), "t_comp_ns": float(t_comp[i]),
                    "t_roof_ns": float(t_roof[i]), "efficiency_eta": float(eta[i]),
                    "trust_w": float(P.get("trust_w", np.ones(len(samples)))[i]), "issue_limited": int(hit[i])})
    return out


PROVENANCE = [
    "# MLP-NN v4.55 cross-GPU kernel estimator (no-ET NN + confidence) -- output",
    "# INPUT SOURCES: kernel stats = NCU on the SOURCE GPU; GPU specs = spec sheet (SRC/TGT cols);",
    "#   roofline terms COMPUTED in-tool (bytes=MemThr[byte/s]*ExecTime; flops=(2*FFMA+FADD+FMUL)*cycles).",
    "# MODEL: v4.5 -- Execution Time is NOT a predictive input; inference is a plain NN forward pass.",
    "# CONFIDENCE (--confidence): conf_* from 5-seed ensemble spread + roofline consistency; exp_err_* = measured typical error for that bucket.",
    "# OUTPUTS: ExecTime/Mem%/Occupancy/brk_* = v4.5 predictions; t_*_ns + efficiency_eta = roofline.",
    "# running_cycles: warp-cycles in ncu's Selected (issuing) state = issued warp instructions of the SOURCE run (1 cycle each);"
    "#   exact for a target with the same SM architecture, NaN when the input has no 'Issued Instructions [inst]'.",
    "# issue_limited: 1 = predicted time raised to the instruction-issue limit, issued / (4 x SMs x clock); Mem% scaled to match.",
    "# exec_cycles / cyc_*: execution-cycle breakdown, running + stall = 1 (fractions of exec_cycles). Per scheduler every cycle"
    "#   either issues (running) or not (stall): cyc_running = issued warp inst / (time x peak issue rate = 4 schedulers x SMs x"
    "#   clock), = ncu issued / Total SMSP Elapsed Cycles (median ratio 1.000-1.003 on 6,445 ncu runs, 9 GPUs); the other"
    "#   cyc_* = (1 - cyc_running) x the predicted stall shares. exec_cycles = time x clock (clock = target peak_inst / 4 / SMs)."
    "#   NaN without an Issued Instructions [inst] input; exact issued count only for a same-architecture target.",
    "# trust_w: 1 = target GPU inside the trained spec envelope; <1 = outside (TRUST_Z), prediction blended toward the physics anchor.",
]


def _to_num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def main():
    ap = argparse.ArgumentParser(
        description="v4.5 cross-GPU kernel estimator (no-ET NN + confidence). Two input modes: "
                    "(1) full input via CLI options, or (2) --csv FILE [--row N].")
    ap.add_argument("--runtime-inputs", metavar="CSV", action="append",
                    help="your actual GPU values (spec,unit,<GPU columns>) applied on top of runtime_inputs_training.csv; repeatable")
    ap.add_argument("--artifact", default=os.path.join(HERE, "v455_artifact"))   # MLP-NN v4.55: the v4.46 inference code with the weights of the v4.55 production run (5 seeds)
    ap.add_argument("--out", required=True)
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
        m1.add_argument(_flag(c), dest=_dest(c), metavar="V", help=c.replace("%", "%%"))   # argparse: a literal % must be doubled
    args = ap.parse_args()

    models, meta = v455_core.load(args.artifact)
    reg_out, brk_out = meta["reg_out"], meta["brk_out"]
    if meta.get("busy_sms_train"):            # trained with the busy-SM floor on every target (AT_BUSY_SMS=2), 2026-10-18
        dp.AT_BUSY_MODE, dp.AT_BUSY_SMS = "2", True
    for _k, _a in (("busy_form", "AT_BUSY_FORM"), ("busy_beta", "AT_BUSY_BETA")):   # rule variant the weights were trained with (2026-10-19)
        if _k in meta: setattr(dp, _a, meta[_k])
    if "busy_use" in meta: dp.AT_BUSY_USE = set(meta["busy_use"])
    if "lat_clock_alpha" in meta and "LAT_CLOCK" not in os.environ:   # clock-aware latency floor the weights were trained with (2026-10-20)
        dp.LAT_CLOCK, dp.LAT_CLOCK_ALPHA = True, float(meta["lat_clock_alpha"])
    if meta.get("busy_feat"):                 # rule enters as inputs; real targets keep the plain floor they were trained on
        dp.BUSY_FEAT = True
        if dp.AT_BUSY_MODE == "2" or os.environ.get("BUSY_FEAT_FLOOR", "1") != "1": dp.AT_BUSY_MODE, dp.AT_BUSY_SMS = "0", False
        # BUSY_FEAT_FLOOR=1 (default): hypothetical targets still get the busy-SM floor (AT_BUSY_SMS=1)
    if brk_out == len(core.BRK) + 1:          # 5th breakdown bucket "running" (BRK_RUNNING=1 artifacts)
        core.BRK = list(core.BRK) + ["running"]
        dp.BRK_RUNNING = True; dp.STALL_BREAKDOWN_GROUPS.setdefault("running", [])
        core.compute_breakdown = dp.compute_breakdown   # source breakdown input also has the 5th bucket (= 1 per issued inst)
        OUT_COLS.insert(OUT_COLS.index("brk_scheduling_overhead") + 1, "brk_running")
    # sources the pipeline has datasheet values for. v4.42/v4.46 train on H200 / GH200 / L40S as sources too
    # (INCLUDE_H200=1 INCLUDE_L40S=1), so any pipeline-known source is valid (ported from predict_v318, 2026-10-15).
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
    R = v455_core.predict_seeds(models, meta, samples)
    out_rows = to_rows(R, samples)
    cols_extra = []
    if args.confidence or args.truth:
        C = v455_core.confidence(R, samples)
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
    if os.environ.get("PRED_SEED_TIMES") == "1" and R.get("et_seeds") is not None:
        # per-seed ExecTime [ns] (2026-10-21, opt-in): the spread of the ensemble, e.g. for an app-level seed range.
        # Same instruction-issue floor as the ensemble answer (applied in the core on the mean only).
        _es = np.asarray(R["et_seeds"], float)
        if os.environ.get("ISSUE_FLOOR", "1") == "1":
            with np.errstate(divide="ignore", invalid="ignore"):
                _ti = np.array([s.get("src_issued_inst", np.nan) / s.get("target_peak_inst", np.nan) * 1e9 for s in samples], float)
            _es = np.where(np.isfinite(_ti)[None, :] & (_ti[None, :] > _es), _ti[None, :], _es)
        for i, r in enumerate(out_rows):
            r.update({f"et_seed{j}": float(_es[j, i]) for j in range(_es.shape[0])})
        cols_extra += [f"et_seed{j}" for j in range(_es.shape[0])]
    if args.truth:
        cols_extra += v455_core.attach_errors(out_rows, args.truth)
    if args.confidence:
        v455_core.print_confidence_legend()
        import collections as _c
        sys.stderr.write("\nconfidence of THIS run:\n")
        for _k in ("ExecTime", "Mem%", "Occupancy", "breakdown"):
            _d = _c.Counter(r[f"conf_{_k}"] for r in out_rows)
            _n = max(sum(_d.values()), 1)
            sys.stderr.write(f"  {_k:<11}" + "  ".join(
                f"{_l}={_d.get(_l,0):>5} ({100*_d.get(_l,0)/_n:4.1f}%)"
                for _l in ("high", "medium", "low")) + "\n")

    cols = ["kernel_name", "src_gpu", "tgt_gpu"] + OUT_COLS + cols_extra
    with open(args.out, "w", newline="") as f:
        if not args.no_comments:
            f.write("\n".join(PROVENANCE) + "\n")
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in out_rows:
            w.writerow({c: (f"{r[c]:.6g}" if isinstance(r[c], float) else r[c]) for c in cols})

    if args.dump_records:
        n = core.dump_records(args.dump_records, "v4.5", samples, out_rows,
                              v455_core._features, core.DERIVED_NAMES_V40, args.no_comments)
        sys.stderr.write(f"records  : {n} rows -> {args.dump_records}\n")

    log = [f"v4.5 cross-GPU estimator (no-ET NN + confidence)  run @ {datetime.datetime.now().isoformat(timespec='seconds')}",
           f"model    : v4.5  ({meta['seeds']}-seed ensemble)", "specs    : read from the input (SRC/TGT columns/options)",
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
