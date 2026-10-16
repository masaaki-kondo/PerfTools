"""
Pipeline v2 — reads data/20260522_wide.zip (the long-to-wide-converted
ncu output for 4 GPUs: A100, H100, GB10, GB200).

API mirrors data_pipeline.py (same exported symbols & shapes) so the
existing experiment scripts can switch with a one-line import.

Invariants:
  - Missing metric values stay NaN (never 0). Imputation, if any, is
    explicit (column mean) and applied at the stack/normalize boundary,
    NOT silently inside extract_features.
  - Cross-GPU pairing key is (benchmark, hash) — the same ncu run
    (same source code + same args) measured on different GPUs.
"""

from __future__ import annotations
import io
import os
import re
import zipfile
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Feature definitions — adapted to the new wide-form column names (unit
# suffixes included). Renames `Duration [ns]`->`Execution Time` and
# `Achieved Occupancy [%]`->`Achieved Occupancy` happen in extract_gpu_data.

KERNEL_CONFIG_FEATURES = [
    "Block Size",
    "Grid Size",
    "Threads",                       # derived = Block Size * Grid Size
    "Registers Per Thread [register/thread]",
    "Static Shared Memory Per Block [byte/block]",
    "Dynamic Shared Memory Per Block [byte/block]",
    "Shared Memory Per Block [byte/block]",
]

WORKLOAD_PROFILE_FEATURES = [
    "Achieved Occupancy",
    "Achieved Active Warps Per SM [warp]",
    "Compute (SM) Throughput [%]",
    "Memory Throughput [%]",
    "L1/TEX Cache Throughput [%]",
    "L2 Cache Throughput [%]",
    "Memory Throughput [byte/s]",     # absent on GB10 → NaN, never 0
    "Eligible Warps Per Scheduler [warp]",
    "Executed Ipc Active [inst/cycle]",
    "Warp Cycles Per Executed Instruction [cycle/inst]",
    "Theoretical Active Warps per SM [warp]",
    "Waves Per SM",
    "Block Limit Registers [block]",
    "Block Limit Warps [block]",
    "Block Limit SM [block]",
    "Block Limit Shared Mem [block]",
]

# --- Metro-parity option (v3.1 / v4.5): L1 and L2 CACHE HIT RATES -------------------
# Metro feeds L2 Hit Rate and L1/TEX Hit Rate; we historically fed only the cache
# THROUGHPUT percentages, which are a different quantity (utilisation vs. hit fraction)
# and only loosely correlated with them (R2 0.55 / 0.71 from our other features).
# Both are measured by NCU and present on every GPU incl. the MIG configs (70-85%
# populated), so this costs no new profiling. Gated by HITRATES=1 so that EXISTING
# artifacts keep their input dimensions -- turning it on changes the feature vector.
if os.environ.get("HITRATES", "0") == "1":
    WORKLOAD_PROFILE_FEATURES = WORKLOAD_PROFILE_FEATURES + [
        "L2 Hit Rate [%]",
        "L1/TEX Hit Rate [%]",
    ]

# 16 common-on-all-4-GPUs stall categories
# (GMMA = H100 only; IMC Miss = A100/H100 only — both dropped here).
STALL_FEATURES = [
    "Stall Barrier [inst]",
    "Stall Branch Resolving [inst]",
    "Stall Dispatch Stall [inst]",
    "Stall Drain [inst]",
    "Stall LG Throttle [inst]",
    "Stall Long Scoreboard [inst]",
    "Stall MIO Throttle [inst]",
    "Stall Math Pipe Throttle [inst]",
    "Stall Membar [inst]",
    "Stall Misc [inst]",
    "Stall No Instruction [inst]",
    "Stall Not Selected [inst]",
    "Stall Short Scoreboard [inst]",
    "Stall Sleeping [inst]",
    "Stall Tex Throttle [inst]",
    "Stall Wait [inst]",
]

GPU_SPEC_FEATURES = [
    # CC dropped — it's a label, not physics; doesn't transfer to
    # unseen GPUs. Replaced with peak tensor throughput + the
    # tensor/FP32 ratio (an architectural-specialization fingerprint).
    "GPU Maximum Warps Per Scheduler [warp]",
    "Theoretical Active Warps per SM [warp]",
    "Theoretical Active Warps Per Scheduler [warp]",
    # "Max Cluster Size [block]" and "Block Limit Barriers [block]" removed
    # 2026-06-03: absent in A100 NCU (Hopper-only / unreported) -> imputed
    # placeholders (0 per-source, fabricated mean in combined); ablation
    # showed dropping them is net-neutral-to-positive (e.g. A100 brk_sync
    # 0.68->0.76). Provenance is cleaner without them.
    "Shared Memory Configuration Size [byte]",
    "Block Limit Warps [block]",
    # Derived architectural ratios (constant per GPU). Capture
    # cross-architecture character — most importantly the FP32/FP64
    # ratio (separates consumer GB10 from HPC) and the Tensor/FP32
    # ratio (separates AI-specialized from general-purpose).
    "Peak FP32 / Peak FP64",
    "DRAM BW per SM [byte/s/SM]",
    "L2 Size per SM [byte/SM]",
    "Peak FP32 / DRAM BW [flop/byte]",
    "Peak FP16 Tensor [flop/s]",
    "Peak Tensor / Peak FP32",
    # Added 2026-06-01 — memory-latency + host↔device bandwidth.
    # DRAM Latency: model-recommended ns from data/gpu_microarch_specs.csv.
    # CPU-GPU BW: aggregate bidir bytes/s — PCIe Gen4/5 for A100/H100,
    # NVLink-C2C for the Grace-integrated GB200/GB10.
    "DRAM Latency [ns]",
    "CPU-GPU BW [byte/s]",
]

# DROP_C2C: remove "CPU-GPU BW [byte/s]" from the target spec vector.
# LOGO diagnosis 2026-09-17: only ONE training GPU (GB200) has NVLink-C2C (900 GB/s), so
# the network learned the value as a proxy for "this is GB200" rather than as physics.
# GH200 also has C2C, and the model applied GB200-like scaling to a Hopper die: held-out
# GH200 scored MAPE 69.7% / R2 -0.887 with the feature, 27.8% / R2 0.751 with it set to
# H100's value. The feature was never physically justified for this target anyway --
# host<->device bandwidth governs data transfer, not how long a kernel runs ON the GPU.
# It is the LAST entry, so removing it leaves every other feature index unchanged.
# Gated: existing artifacts were trained with 13 spec features and must keep them.
if os.environ.get("DROP_C2C", "0") == "1":
    GPU_SPEC_FEATURES = [f for f in GPU_SPEC_FEATURES if f != "CPU-GPU BW [byte/s]"]

# HOST_CAT: the host link as two CATEGORIES instead of one bandwidth number.
# The bandwidth number was doing double duty: 900 GB/s meant "GB200" to the network (only
# one training GPU had it -> GH200 mis-scaled 70%), while 64 GB/s tagged the two PCIe-4
# parts (A100, L40S) -- dropping the feature outright (v4.17) lost that tag and L40S blew
# up to 316%. Two flags keep the useful part and remove the identity leak:
#     Host Link Coherent [0/1]   1 for Grace-attached (NVLink-C2C), else 0
#     Host PCIe Gen              4 / 5 / 6
# Categorical -> they enter the target vector as tgt - src, never as a (log) ratio.
HOST_CAT = os.environ.get("HOST_CAT", "0") == "1"
HOST_CAT_FEATURES = ["Host Link Coherent [0/1]", "Host PCIe Gen"]
if HOST_CAT:
    GPU_SPEC_FEATURES = [f for f in GPU_SPEC_FEATURES if f != "CPU-GPU BW [byte/s]"] + HOST_CAT_FEATURES
CATEGORICAL_SPECS = set(HOST_CAT_FEATURES)
SPEC_CAT_MASK = np.array([f in CATEGORICAL_SPECS for f in GPU_SPEC_FEATURES], dtype=bool)


def spec_delta(tgt, src):
    """Target-spec input vector: ratio (or log-ratio under LOG_SPECS) for physical specs,
    plain difference for categorical ones. The single place this rule lives."""
    tgt = np.asarray(tgt, float); src = np.asarray(src, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = tgt / np.where(np.abs(src) > 1e-8, src, 1e-8)
        if LOG_SPECS:
            r = np.log(np.maximum(r, 1e-12))
    return np.where(SPEC_CAT_MASK, tgt - src, r)


def host_link_flags(interconnect: str):
    """('NVLink-C2C' | 'PCIe 5.0 x16' | ...) -> (coherent 0/1, pcie generation)."""
    t = str(interconnect or "").lower()
    coh = 1.0 if ("c2c" in t or "nvlink" in t) else 0.0
    m = re.search(r"pcie\s*(\d)", t)
    gen = float(m.group(1)) if m else (5.0 if coh else np.nan)   # Grace parts also expose PCIe 5
    return coh, gen

# Per-GPU vendor specs (hardcoded — verify against datasheets).
# Used only to derive the 4 ratio features above; raw absolute peaks
# are intentionally NOT exposed as features (already indirectly present
# in per-kernel utilization metrics).
GPU_PEAK_SPECS = {
    # peak_tensor_fp16 = FP16 dense tensor-core peak (FLOPS).
    # Verify against vendor sheets if used in publication.
    "a100":  {"peak_fp32": 19.5e12,  "peak_fp64": 9.7e12,
              "peak_tensor_fp16": 312e12,  "l2_size": 40e6},
    "h100":  {"peak_fp32": 67.0e12,  "peak_fp64": 34.0e12,
              "peak_tensor_fp16": 989e12,  "l2_size": 50e6},
    "gb200": {"peak_fp32": 80.0e12,  "peak_fp64": 40.0e12,
              "peak_tensor_fp16": 2250e12, "l2_size": 60e6},  # estimate
    "gb10":  {"peak_fp32": 31.0e12,  "peak_fp64": 0.97e12,
              "peak_tensor_fp16": 125e12,  "l2_size": 16e6},  # estimate; consumer
}

# Fallback DRAM BW + SM count if not readable from the data (e.g., GB10
# has no DRAM Bandwidth column).
GPU_HW_FALLBACK = {
    "a100":  {"dram_bw": 1.555e12, "n_sms": 108},
    "h100":  {"dram_bw": 3.000e12, "n_sms": 132},
    "gb200": {"dram_bw": 8.000e12, "n_sms": 148},  # estimate
    "gb10":  {"dram_bw": 5.12e11,  "n_sms": 80},   # estimate
}

# Memory latency (ns) and CPU↔GPU interconnect bandwidth (bytes/s).
# Values from data/gpu_microarch_specs.csv (cross-checked 2026-05-29):
#   - DRAM latency = model-recommended ns (A100/H100 from pointer-chase
#     measurement; GB200 = estimate higher than Hopper per dual-die finding;
#     GB10 = LPDDR5X analog estimate).
#   - CPU-GPU BW = aggregate bidirectional. A100=PCIe Gen4 x16, H100=PCIe
#     Gen5 x16, GB200=NVLink-C2C, GB10=NVLink-C2C (~600 GB/s).
GPU_LATENCY_BW_SPECS = {
    "a100":  {"dram_latency_ns": 395.0, "cpu_gpu_bw":  64.0e9, "host_coherent": 0.0, "host_pcie_gen": 4.0},
    "h100":  {"dram_latency_ns": 370.0, "cpu_gpu_bw": 128.0e9, "host_coherent": 0.0, "host_pcie_gen": 5.0},
    "gb200": {"dram_latency_ns": 440.0, "cpu_gpu_bw": 900.0e9, "host_coherent": 1.0, "host_pcie_gen": 5.0},
    "gb10":  {"dram_latency_ns": 210.0, "cpu_gpu_bw": 600.0e9, "host_coherent": 1.0, "host_pcie_gen": 5.0},
}


# LOG_SPECS: feed the target/source spec RATIO in log space.
# Diagnosed 2026-09-12: with raw ratios, whole-GPU scaling walks the inputs exponentially
# out of distribution -- at 16x the swept features sit at z-scores of 20.6 / 34.8 / 12.4
# (anything past ~3 is already outside training). The network saturates there, its output
# stops responding to input, and NO loss term can repair that: the PINN slope constraint
# is violated even on its own collocation points (log-residual 4.0 at 16x) because there
# is no gradient path left. log(ratio) turns 16x into log(16)=2.77, a z-score around 2, so
# scaling becomes a LINEAR SHIFT in input space instead of an exponential excursion --
# the geometry a network can actually extrapolate along.
# Gated: existing artifacts were trained on raw ratios and must keep them.
LOG_SPECS = os.environ.get("LOG_SPECS", "0") == "1"


# INST_ANCHOR: compute anchor for ZERO-FLOP kernels.
# 56.6% of compute-bound kernels have f32 = f64 = 0 (integer / LSU / atomics work that
# the six FP counters never see), so t_comp collapses to the 1e-30 floor and t_roof =
# t_mem: the model is told they are memory-bound. Those kernels are the worst error class
# (10.7% vs 4.1%) and the one every model shares. `Executed Instructions [inst]` is
# populated on 95% of them; divided by the target's peak issue rate it gives a compute
# time a kernel cannot beat. Used ONLY where FP flops are zero, so FP kernels are untouched.
# Modes:  0 off | 1 anchor inside t_comp (v4.16/v3.8: shifts the derived-feature
# distribution for 389 kernels and BROKE COMM at 16x) | 2 anchor the ROOFLINE FLOOR ONLY
# (t_roof = max(t_mem, t_comp, t_inst) for zero-flop kernels; every derived feature keeps
# the plain FP t_comp, so the input distribution is untouched).
INST_ANCHOR_MODE = os.environ.get("INST_ANCHOR", "0")
INST_ANCHOR = INST_ANCHOR_MODE == "1"

# LAT_ANCHOR: a floor for the LATENCY class -- zero-flop kernels that do NOT saturate DRAM
# (source Mem% < 50: histogram / bsearch / conversions, ~30% of the set). They have no
# roofline at all (t_comp is the 1e-30 floor and t_mem is tiny), so the model gets no
# physics for them and their far-field projection is the worst column. The floor:
#     t_lat = C * ET_src * (lat_tgt / lat_src) * (SMs_src / SMs_tgt)
# i.e. the source's dependent-latency chain, re-timed at the target's DRAM latency and
# spread perfectly over the target's SMs. Even so it is NOT a bound at C=1: on the 697
# measured LATENCY-class pairs (2026-10-01) the true target ET is below t_lat for 50% of
# cross-GPU pairs (median ET/t_lat 0.65 at p10, 0.29 at p1) -- those kernels gain more
# than the SM ratio, e.g. from cache/L2 effects the chain model ignores. C (LAT_ANCHOR_C)
# is the safety factor that turns it into one: violations 4.6% at C=0.5, 1.1% at C=0.3,
# <1% at C=0.25 (the roofline floor itself is violated by 0.3%). Default 0.25.
# Applied to the FLOOR ONLY, never to the derived features (same lesson as INST_ANCHOR).
LAT_ANCHOR = os.environ.get("LAT_ANCHOR", "0") == "1"
LAT_ANCHOR_C = float(os.environ.get("LAT_ANCHOR_C", "0.25"))
# WAVE_ANCHOR (2026-10-15): one-wave floor. A kernel with a FIXED grid cannot finish faster than one wave of its
# own blocks on the target, however many SMs the target has (strong scaling). 38% of the 17-app kernels have < 2
# waves on H200 and 49% < 32, so the x16-x32 dummy projections were above the physical ceiling for them.
#   WAVE_ANCHOR=1 : the floor goes INTO t_roof (training floor, collocation grid, inference) -> retrain.
#   WAVE_CLAMP=1  : post-hoc clamp ET >= t_wave at inference only, for artifacts trained WITHOUT the floor
#                   (putting it into t_roof of such an artifact would inflate ET multiplicatively instead).
#   WAVE_ANCHOR_C : scale (1.0 = the full one-wave time; < 1 softens it).
WAVE_ANCHOR = os.environ.get("WAVE_ANCHOR", "0") == "1"
WAVE_CLAMP = os.environ.get("WAVE_CLAMP", "0") == "1"
WAVE_ANCHOR_C = float(os.environ.get("WAVE_ANCHOR_C", "1.0"))
# MEM_CAP (2026-10-15): memory-efficiency CAP, the mirror of the wave floor. A kernel that is memory-bound on its
# source (Mem% >= 60) cannot be LESS efficient on a target whose DRAM bandwidth per SM is <= the source's (easier
# to saturate) than it was on the source: ET <= MEM_CAP_MARGIN * bytes / (BW_tgt * eta_src), eta_src = achieved DRAM
# fraction on the source. Motivation: L40S (6 GB/s/SM) memory-bound kernels achieve 75% of peak, the net assumed 43%
# (bias 1.8). Truth exceeds the cap at margin 1.25 in <= 6% of eligible rows on every fold (0% H200/GH200/H100).
# Applied at inference BEFORE the floors (floors win). Off by default.
MEM_CAP = os.environ.get("MEM_CAP", "0") == "1"
MEM_CAP_MARGIN = float(os.environ.get("MEM_CAP_MARGIN", "1.25"))
MEM_CAP_MINPCT = float(os.environ.get("MEM_CAP_MINPCT", "60"))
# COMP_CAP (2026-10-16): compute-saturation cap at inference, the compute twin of MEM_CAP. See t_compcap_of.
COMP_CAP = os.environ.get("COMP_CAP", "0") == "1"
COMP_CAP_MARGIN = float(os.environ.get("COMP_CAP_MARGIN", "1.25"))


def t_lat_of(s, fp=1.0):
    """Latency floor [s] for one sample dict (0 when the sample is not LATENCY-class or
    LAT_ANCHOR is off). `fp` = compute-peak scale factor of a collocation/synthetic point;
    peak scaling is treated as SM scaling, so the floor divides by it."""
    if not LAT_ANCHOR:
        return 0.0
    f = float(s.get("roofline_fp32_flops", 0.0) or 0.0) + float(s.get("roofline_fp64_flops", 0.0) or 0.0)
    mp = s.get("src_mem_pct", np.nan)
    if f > 0 or not (mp == mp) or mp >= 50.0:
        return 0.0
    et, ls, lt = s.get("src_dur_ns", np.nan), s.get("src_dram_latency", np.nan), s.get("target_dram_latency", np.nan)
    ss, st = s.get("src_n_sms", np.nan), s.get("target_n_sms", np.nan)
    v = [et, ls, lt, ss, st]
    if not all(x == x and x > 0 for x in v):
        return 0.0
    return LAT_ANCHOR_C * et * 1e-9 * (lt / ls) * (ss / st) / fp


def t_wave_of(s):
    """One-wave floor [s] for one sample dict (0 unless WAVE_ANCHOR or WAVE_CLAMP is on, or an ingredient is
    missing). t_wave = C * ET_src / max(waves_src, 1) * r. One wave of the kernel's own blocks can speed up at
    most by the PER-SM resource improvement, so r = min(src/tgt per-SM issue rate, src/tgt per-SM DRAM bandwidth);
    each ratio falls back to 1 (same SM) when the target value is unknown. Per-SM rates come from the pipeline
    tables for real GPUs and from the TGT peak_inst / dram_bw / n_sms columns for dummies, so along the PINN
    collocation grid and the whole-GPU dummies (per-SM constant) r = 1 and the floor is a CONSTANT.
    waves_src < 1 (grid smaller than the source GPU) means the whole kernel is one wave."""
    if not (WAVE_ANCHOR or WAVE_CLAMP):
        return 0.0
    et, w = s.get("src_dur_ns", np.nan), s.get("src_waves", np.nan)
    if not (et == et and w == w and np.isfinite(et) and np.isfinite(w) and et > 0 and w >= 0):
        return 0.0
    sg, tg = str(s.get("src_gpu", "")).lower(), str(s.get("tgt_gpu", "")).lower()
    hs, ht = GPU_HW_FALLBACK.get(sg, {}), GPU_HW_FALLBACK.get(tg, {})
    ss = s.get("src_n_sms", np.nan); ss = ss if ss == ss else hs.get("n_sms", np.nan)
    st = s.get("target_n_sms", np.nan); st = st if st == st else ht.get("n_sms", np.nan)
    pis = hs.get("peak_inst", np.nan)
    pit = s.get("target_peak_inst", np.nan); pit = pit if pit == pit else ht.get("peak_inst", np.nan)
    bws = hs.get("dram_bw", np.nan)
    bwt = s.get("target_dram_bw", np.nan); bwt = bwt if bwt == bwt else ht.get("dram_bw", np.nan)
    def ratio(a, b):   # (src per-SM) / (tgt per-SM); 1 when anything is unknown
        v = [a, b, ss, st]
        if not all(x == x and np.isfinite(x) and x > 0 for x in v):
            return 1.0
        return (a / ss) / (b / st)
    r = min(ratio(pis, pit), ratio(bws, bwt))
    return WAVE_ANCHOR_C * et * 1e-9 / max(float(w), 1.0) * r


def t_memcap_of(s):
    """Memory-efficiency cap [s] for one sample dict (inf when MEM_CAP is off, the row is not memory-bound on the
    source, the target's DRAM BW per SM is above the source's, or an ingredient is missing). See MEM_CAP above."""
    if not MEM_CAP:
        return np.inf
    mp = s.get("src_mem_pct", np.nan); b = s.get("roofline_bytes", np.nan); et = s.get("src_dur_ns", np.nan)
    sg = str(s.get("src_gpu", "")).lower(); hs = GPU_HW_FALLBACK.get(sg, {})
    bws, ns = hs.get("dram_bw", np.nan), hs.get("n_sms", np.nan)
    bwt, nt = s.get("target_dram_bw", np.nan), s.get("target_n_sms", np.nan)
    v = [mp, b, et, bws, ns, bwt, nt]
    if not all(x == x and np.isfinite(x) and x > 0 for x in v) or mp < MEM_CAP_MINPCT:
        return np.inf
    if bwt / nt > 1.05 * bws / ns:
        return np.inf
    eta = (b / (et * 1e-9)) / bws                     # achieved DRAM fraction on the source
    if not (0.05 < eta <= 1.5):
        return np.inf
    return MEM_CAP_MARGIN * b / (bwt * min(eta, 1.0))


def t_compcap_of(s):
    """Compute-saturation cap [s] for one sample dict (inf when COMP_CAP is off, the kernel is not compute-bound on
    the source, the target has at least the source's binding FP peak, or an ingredient is missing).
    The median compute-bound kernel uses 24% of its binding FP peak, so less FP throughput costs time only once the
    pipe fills:  sat = min( ET_src * r,  max( ET_src * f_w,  t_fp_tgt / 0.9 ) ),  cap = COMP_CAP_MARGIN * max(sat, ET_src * f_w)
    r = t_fp_tgt / t_fp_src (binding FP32/FP64 peak),  f_w = max(w_tgt,1)/max(w_src,1) the wave-count growth when the
    target has fewer SMs. On 157 measured pairs with >1x less binding FP peak (H100/GB200 -> A100, A100/H100/GB200 ->
    L40S): MAPE 49.8 -> 25.7, bias 0.98 (margin 1.25 = the MEM_CAP margin, not tuned separately)."""
    if not COMP_CAP:
        return np.inf
    try:
        et = float(s.get("src_dur_ns", np.nan)); sg = str(s.get("src_gpu", "")).lower()
        hs, ps = GPU_HW_FALLBACK.get(sg, {}), GPU_PEAK_SPECS.get(sg, {})
        f32 = float(s.get("roofline_fp32_flops", 0) or 0); f64 = float(s.get("roofline_fp64_flops", 0) or 0); b = float(s.get("roofline_bytes", np.nan))
        tcs = max(f32 / ps["peak_fp32"], f64 / ps["peak_fp64"]); tct = max(f32 / float(s["target_peak_fp32"]), f64 / float(s["target_peak_fp64"]))
        tms = b / hs["dram_bw"]
        if not (et == et and et > 0 and tcs > 0 and np.isfinite(tct) and tcs > tms):
            return np.inf
        r = tct / tcs
        if not r > 1.0:
            return np.inf
        ns, nt = float(hs.get("n_sms", np.nan)), float(s.get("target_n_sms", np.nan)); w = float(s.get("src_waves", np.nan))
        fw = (max(w * ns / nt, 1.0) / max(w, 1.0)) if all(x == x and x > 0 for x in (ns, nt)) and w == w and w >= 0 else 1.0
        e = et * 1e-9
        sat = min(e * r, max(e * fw, tct / 0.9))
        return COMP_CAP_MARGIN * max(sat, e * fw)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return np.inf


def troof_floor(t_mem, t_comp, inst=None, peak_inst=None, f32=None, f64=None, t_lat=0.0, t_wave=0.0):
    """The execution-time lower bound [s] the model predicts a residual against.
    max(t_mem, t_comp) always; plus the instruction-issue floor for zero-flop kernels
    under INST_ANCHOR=2; plus the latency floor under LAT_ANCHOR. Array-safe."""
    t = np.fmax(np.asarray(t_mem, float), np.asarray(t_comp, float))
    if INST_ANCHOR_MODE == "2" and inst is not None and peak_inst is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            ti = np.asarray(inst, float) / np.asarray(peak_inst, float)
        no_fp = (np.nan_to_num(np.asarray(f32, float)) + np.nan_to_num(np.asarray(f64, float))) <= 0
        t = np.where(no_fp & np.isfinite(ti) & (ti > 0), np.fmax(t, ti), t)
    if LAT_ANCHOR:
        t = np.fmax(t, np.nan_to_num(np.asarray(t_lat, float), nan=0.0))
    if WAVE_ANCHOR:
        t = np.fmax(t, np.nan_to_num(np.asarray(t_wave, float), nan=0.0))
    return t


def tcomp_anchor(f32, f64, inst, p32, p64, peak_inst, floor=1e-30):
    """Binding compute time. Falls back to instruction-issue time for zero-flop kernels."""
    f32 = np.asarray(f32, float); f64 = np.asarray(f64, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        tc = np.fmax(f32 / p32, f64 / p64)
        if INST_ANCHOR:
            t_inst = np.asarray(inst, float) / np.asarray(peak_inst, float)
            no_fp = (np.nan_to_num(f32) + np.nan_to_num(f64)) <= 0
            tc = np.where(no_fp & np.isfinite(t_inst) & (t_inst > 0), t_inst, tc)
    return np.fmax(np.nan_to_num(tc, nan=floor), floor)


SPEC_CLOCK = os.environ.get("SPEC_CLOCK", "boost")     # "operating": see _load_spec_sheet_overrides
GPU_CLOCKS = {}


def _load_spec_sheet_overrides():
    """Single source of truth: override the hardcoded GPU spec dicts above with
    values from data/gpu_microarch_specs.csv. This keeps everything the model
    consumes traceable to NCU data or the spec sheet (no hidden constants).
    Unit conversions: GB/s->byte/s (1e9), TFLOPS->FLOPS (1e12), MB->byte (1e6)."""
    import csv
    import os
    # GPU_SPEC_SHEET (2026-10-16): path of the spec sheet (default: data/gpu_microarch_specs.csv, unchanged).
    # GPU_RUNTIME_INPUTS: os.pathsep-separated CSVs in the same layout (spec,unit,<GPU columns>,...) applied ON TOP of
    # the sheet, in order, cell by cell (empty cells ignored). They carry ACTUAL values that are not specs -- operating
    # clocks, achievable bandwidths, latency estimates -- so the spec sheet itself holds datasheet values only.
    path = os.environ.get("GPU_SPEC_SHEET") or os.path.join(os.path.dirname(__file__), os.pardir,
                                                            "data", "gpu_microarch_specs_v3.csv")
    if not os.path.exists(path):
        return  # keep hardcoded fallback if the sheet is unavailable
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.setdefault(r["spec"], r)   # first row wins for duplicate keys
    for _ri in [x for x in os.environ.get("GPU_RUNTIME_INPUTS", "").split(os.pathsep) if x]:
        if not os.path.exists(_ri):
            raise FileNotFoundError(f"GPU_RUNTIME_INPUTS: {_ri} not found")
        with open(_ri) as f:
            for r in csv.DictReader(f):
                tgt = rows.setdefault(r["spec"], {"spec": r["spec"]})
                for k, v in r.items():
                    if k not in ("spec", "unit", "confidence_notes", "source", "notes") and str(v or "").strip() != "":
                        tgt[k] = v

    def g(spec, col):
        v = str(rows.get(spec, {}).get(col, "")).strip()
        return float(v) if v not in ("", "nan", "None") else None

    # GH200 / L40S added 2026-08-17 from Yoshida-san's new ncu-full profiles.
    # A column is only used if the sheet actually carries it (see the guard below).
    for lc, col in {"a100": "A100", "h100": "H100", "gb200": "GB200", "gb10": "GB10",
                    "gh200": "GH200", "l40s": "L40S",
                    "h200": "H200",
                    "h100_mig_1g": "H100_MIG_1g", "h100_mig_2g": "H100_MIG_2g",
                    "h100_mig_4g": "H100_MIG_4g",
                    "b300": "B300"}.items():          # B300: nsys-only target (2026-10-15)
        if g("peak_fp32", col) is None or g("dram_bandwidth", col) is None:
            continue                      # sheet has no such GPU -> leave dicts untouched
        # SPEC_CLOCK=operating (2026-10-15): the datasheet peaks assume the BOOST clock, but the labels were measured
        # at the clock the node actually ran (sheet row sm_clock_operating: H100 1.08 of 1.755, GB200 1.09 of 1.965,
        # MIG-2g 1.08 of 1.785; the rest ~boost). Scale every compute-side peak by operating/boost so the spec
        # vector describes the machine the labels came from (fixes the H100 0.78 / GB200 0.47 fold biases at the
        # root). Default "boost" = the old behaviour; every version before v3.21/v4.45 was trained with it.
        _boost = g("sm_clock_boost", col); _opr = g("sm_clock_operating", col)
        _cs = (_opr / _boost) if (SPEC_CLOCK == "operating" and _boost and _opr) else 1.0
        _clk = _opr if (SPEC_CLOCK == "operating" and _opr) else _boost
        GPU_CLOCKS[lc] = {"boost": _boost, "ncu": g("sm_clock_ncu", col), "operating": _opr, "scale": _cs}
        GPU_PEAK_SPECS[lc] = {
            "peak_fp32": g("peak_fp32", col) * 1e12 * _cs,
            "peak_fp64": g("peak_fp64", col) * 1e12 * _cs,
            "peak_tensor_fp16": g("peak_fp16_tensor", col) * 1e12 * _cs,
            "l2_size": g("l2_cache_total", col) * 1e6,
        }
        GPU_HW_FALLBACK[lc] = {
            "dram_bw": g("dram_bandwidth", col) * 1e9,
            "n_sms": g("num_sms", col),
            # peak warp-instruction issue rate: 4 sub-partitions per SM, 1 warp-inst/cycle
            # each. Used by the INST_ANCHOR fallback for kernels with zero FP flops.
            "peak_inst": (g("num_sms", col) or 0) * 4 * (_clk or 0) * 1e9,
        }
        _coh, _gen = host_link_flags(str(rows.get("cpu_gpu_interconnect", {}).get(col, "")))
        GPU_LATENCY_BW_SPECS[lc] = {
            "dram_latency_ns": g("dram_latency_model_recommended", col),
            "cpu_gpu_bw": g("cpu_gpu_bandwidth", col) * 1e9,
            "host_coherent": _coh, "host_pcie_gen": _gen,
        }


_load_spec_sheet_overrides()

OUTPUT_REGRESSION = [
    "Execution Time",            # renamed from "Duration [ns]"
    "Memory Throughput [%]",     # byte/s no longer predicted — derived
    "Achieved Occupancy",        # post-hoc as Mem% * peak_BW(target)/100.
]

# Architectural fallback peak DRAM bandwidth per GPU (bytes / sec), used
# when the calibration from data is unavailable (e.g., GB10 has no
# byte/s column at all). Source: vendor specs.
PEAK_BW_FALLBACK = {
    "A100":  1.555e12,    # ~1.555 TB/s (A100 80GB HBM2e)
    "H100":  3.000e12,    # ~3.0 TB/s   (H100 NVL HBM3)
    "GB10":  5.12e11,     # ~512 GB/s   (Grace Blackwell GB10 LPDDR5X)
    "GB200": 8.000e12,    # ~8.0 TB/s   (GB200 Blackwell HBM3e)
}


def compute_peak_bandwidths(gpu_data: dict[str, dict[str, pd.DataFrame]]) -> dict[str, float]:
    """Per-GPU peak DRAM bandwidth (bytes / sec).

    Calibrated from kernel rows where both Memory Throughput [byte/s] and
    Memory Throughput [%] are present: peak = byte/s / (% / 100). The
    median across such rows is the stable peak constant for that GPU.

    For GPUs without a byte/s column (e.g., GB10), falls back to the
    architectural value in PEAK_BW_FALLBACK.
    """
    peaks: dict[str, float] = {}
    for gpu, benches in gpu_data.items():
        ratios: list[float] = []
        for df in benches.values():
            if ("Memory Throughput [byte/s]" not in df.columns
                    or "Memory Throughput [%]" not in df.columns):
                continue
            bs = pd.to_numeric(df["Memory Throughput [byte/s]"],
                               errors="coerce")
            pct = pd.to_numeric(df["Memory Throughput [%]"],
                                 errors="coerce")
            valid = bs.notna() & pct.notna() & (pct > 1.0)  # avoid /~0
            if valid.any():
                ratios.extend((bs[valid] / (pct[valid] / 100.0)).tolist())
        if ratios:
            peaks[gpu] = float(np.median(ratios))
        else:
            peaks[gpu] = float(PEAK_BW_FALLBACK.get(gpu, np.nan))
    return peaks

# 4-bucket stall breakdown using the 16 common stalls.
# Bucket names chosen so each is unambiguous:
#   memory     — stalls caused by memory subsystem (scoreboards, throttles, drain)
#   pipeline_contention — compute-pipeline contention (math throttle, dispatch, wait, branch)
#   sync                — synchronization stalls (barrier, membar)
#   scheduling_overhead — warp-scheduling overhead (not-selected, sleeping, misc)
STALL_BREAKDOWN_GROUPS = {
    "memory": [
        "Stall Long Scoreboard [inst]",
        "Stall Short Scoreboard [inst]",
        "Stall LG Throttle [inst]",
        "Stall MIO Throttle [inst]",
        "Stall Tex Throttle [inst]",
        "Stall Drain [inst]",
    ],
    "pipeline_contention": [
        "Stall Math Pipe Throttle [inst]",
        "Stall Wait [inst]",
        "Stall No Instruction [inst]",
        "Stall Branch Resolving [inst]",
        "Stall Dispatch Stall [inst]",
    ],
    "sync": [
        "Stall Barrier [inst]",
        "Stall Membar [inst]",
    ],
    "scheduling_overhead": [
        "Stall Not Selected [inst]",
        "Stall Sleeping [inst]",
        "Stall Misc [inst]",
    ],
}

# Features that should be log1p-transformed where used (heavy-tailed).
LOG_FEATURE_SUBSTRINGS = (
    "Throughput [byte/s]", "Frequency [hz]", "Bandwidth",
    "Cycles [cycle]", "Threads", "Grid Size", "Block Size",
)


def should_log(name: str) -> bool:
    return any(s in name for s in LOG_FEATURE_SUBSTRINGS)


def apply_log_mask(x: np.ndarray, names: list[str]) -> np.ndarray:
    out = x.astype(np.float64, copy=True)
    for i, n in enumerate(names):
        if should_log(n):
            out[..., i] = np.log1p(np.where(np.isnan(out[..., i]), 0.0,
                                            np.maximum(out[..., i], 0.0)))
            # Preserve NaN where input was NaN
            out[..., i] = np.where(np.isnan(x[..., i]), np.nan,
                                    out[..., i])
    return out


# ---------------------------------------------------------------------------
# Reading the wide zip

_BENCH_HASH_RE = re.compile(r"^(?P<bench>.+?)__(?P<hash>[0-9a-fA-F]+)$")


def extract_gpu_data(wide_zip_path: str, gpu_dir: str) -> dict[str, pd.DataFrame]:
    """Load all wide CSVs for a single GPU from data/20260522_wide.zip.

    Returns: {benchmark_key: wide_df}, where benchmark_key = "<bench>__<hash>".
    Each df is one row per kernel; columns include identity, regression
    outputs, workload/stall metrics, and (per row) GPU specs.

    Renames `Duration [ns]` -> `Execution Time` and
    `Achieved Occupancy [%]` -> `Achieved Occupancy` to match
    OUTPUT_REGRESSION names.

    Drops rows where Execution Time is NaN (empty/failed kernel rows).
    """
    out: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(wide_zip_path) as zf:
        members = [n for n in zf.namelist()
                   if n.startswith(f"{gpu_dir}/") and n.endswith(".csv")
                   and n != "_manifest.csv"]
        for name in members:
            fname = name.split("/")[-1].removesuffix(".csv")
            m = _BENCH_HASH_RE.match(fname)
            if not m:
                continue
            key = f"{m.group('bench')}__{m.group('hash')}"
            with zf.open(name) as f:
                df = pd.read_csv(f, low_memory=False)
            # Renames so OUTPUT_REGRESSION names work
            renames = {}
            if "Duration [ns]" in df.columns:
                renames["Duration [ns]"] = "Execution Time"
            if "Achieved Occupancy [%]" in df.columns:
                renames["Achieved Occupancy [%]"] = "Achieved Occupancy"
            if "Achieved Active Warps Per SM" in df.columns:
                renames["Achieved Active Warps Per SM"] = (
                    "Achieved Active Warps Per SM [warp]"
                )
            if renames:
                df = df.rename(columns=renames)
            # Derived: Threads = Block Size * Grid Size (when both numeric)
            if "Threads" not in df.columns and \
               "Block Size" in df.columns and "Grid Size" in df.columns:
                df["Threads"] = pd.to_numeric(df["Block Size"], errors="coerce") \
                    * pd.to_numeric(df["Grid Size"], errors="coerce")
            # Filter empty kernel rows: Execution Time must be present.
            if "Execution Time" in df.columns:
                df = df[pd.to_numeric(df["Execution Time"],
                                      errors="coerce").notna()].copy()
            if df.empty:
                continue
            _attach_derived_specs(df, gpu_dir)
            out[key] = df
    return out


def _attach_derived_specs(df: pd.DataFrame, gpu_dir: str) -> None:
    """In-place attach 4 derived architectural ratio columns to a wide df.

    The values are constants for the GPU (same across all kernels of that
    GPU). DRAM BW and SM count are read from the data when available
    (`DRAM Bandwidth [byte/s]`, `# SMs [SM]`); fall back to vendor specs
    in GPU_HW_FALLBACK otherwise (e.g., GB10 has no DRAM Bandwidth col).
    FP32/FP64 peaks and L2 size come from GPU_PEAK_SPECS (hardcoded).
    """
    g = gpu_dir.lower()
    peak = GPU_PEAK_SPECS.get(g, {})
    fb = GPU_HW_FALLBACK.get(g, {})

    def _first_nonnull(col: str, fallback):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s):
                return float(s.iloc[0])
        return fallback

    # # SMs in the data is architectural (same per kernel) — trust it.
    n_sms = _first_nonnull("# SMs [SM]", fb.get("n_sms", np.nan))
    # DRAM Bandwidth in the data is per-kernel ACHIEVED bandwidth, not
    # the architectural peak — using it here would give different values
    # per kernel and underestimate the GPU's peak. Always use the vendor
    # fallback (peak DRAM BW) for the derived ratios.
    dram_bw = fb.get("dram_bw", np.nan)
    fp32 = peak.get("peak_fp32", np.nan)
    fp64 = peak.get("peak_fp64", np.nan)
    l2_sz = peak.get("l2_size", np.nan)

    def _safe_div(a, b):
        if a is None or b is None:
            return np.nan
        if not (np.isfinite(a) and np.isfinite(b)) or b == 0:
            return np.nan
        return a / b

    df["Peak FP32 / Peak FP64"] = _safe_div(fp32, fp64)
    df["DRAM BW per SM [byte/s/SM]"] = _safe_div(dram_bw, n_sms)
    df["L2 Size per SM [byte/SM]"] = _safe_div(l2_sz, n_sms)
    df["Peak FP32 / DRAM BW [flop/byte]"] = _safe_div(fp32, dram_bw)
    tensor = peak.get("peak_tensor_fp16", np.nan)
    df["Peak FP16 Tensor [flop/s]"] = tensor
    df["Peak Tensor / Peak FP32"] = _safe_div(tensor, fp32)
    lb = GPU_LATENCY_BW_SPECS.get(g, {})
    df["DRAM Latency [ns]"] = lb.get("dram_latency_ns", np.nan)
    df["CPU-GPU BW [byte/s]"] = lb.get("cpu_gpu_bw", np.nan)
    df["Host Link Coherent [0/1]"] = lb.get("host_coherent", np.nan)
    df["Host PCIe Gen"] = lb.get("host_pcie_gen", np.nan)


_WCPI = os.environ.get("WCPI", "0") == "1"
_WCPI_WANT = "Warp Cycles Per Executed Instruction [cycle/inst]"
_WCPI_HAVE = "Warp Cycles Per Executed Instruction [cycle]"


def extract_features(row: pd.Series, names: list[str]) -> np.ndarray:
    """Pull a list of named features from a row. NaN stays NaN."""
    vals = np.empty(len(names), dtype=np.float64)
    for i, n in enumerate(names):
        if n not in row.index:
            # WCPI=1 (2026-10-04): ncu names this metric "... [cycle]"; the feature list has asked for
            # "... [cycle/inst]" since v1.5, so it was all-NaN (a constant) in every model. Gated:
            # existing artifacts were trained with it constant and must keep seeing it that way.
            if _WCPI and n == _WCPI_WANT and _WCPI_HAVE in row.index:
                n = _WCPI_HAVE
            else:
                vals[i] = np.nan
                continue
        v = row[n]
        try:
            vals[i] = float(v)
        except (TypeError, ValueError):
            vals[i] = np.nan
    # ncu emits 'inf' for zero-denominator ratios (seen: Waves Per SM, 2026-10-01). A non-finite
    # value is a MISSING value: keep it NaN (training-mean imputed), never let inf reach a std.
    vals[~np.isfinite(vals)] = np.nan
    return vals


# ---------------------------------------------------------------------------
# Derived features (kept compatible with old pipeline's compute_derived_features)

def compute_derived_features(src_row: pd.Series,
                              target_specs_raw: np.ndarray) -> np.ndarray:
    """8 derived features (same shape/order as v1):
       0 grid_to_sm_ratio
       1 threads_to_sm_ratio
       2 log_grid
       3 fills_gpu (Block Limit SM)
       4 block_size_anchor
       5 mem_activity_norm
       6 is_memory_active
       7 src_log_exec_time
    """
    def _g(k):
        try:
            v = src_row.get(k, np.nan)
            return float(v) if v is not None and not (isinstance(v, float)
                                                       and np.isnan(v)) \
                   else np.nan
        except Exception:
            return np.nan

    block = _g("Block Size")
    grid = _g("Grid Size")
    threads = block * grid if block == block and grid == grid else np.nan
    # # SMs proxy: target spec's "Block Limit SM" is per-SM, not SM count.
    # Use a proxy: target_specs[0] (CC) as a weak proxy is poor; better to use
    # Theoretical Active Warps per SM scaled. Use Block Limit SM column.
    sms_proxy = _g("Block Limit SM [block]")
    if not (sms_proxy == sms_proxy) or sms_proxy <= 0:
        sms_proxy = 1.0
    grid_to_sm = grid / sms_proxy if grid == grid else np.nan
    threads_to_sm = threads / sms_proxy if threads == threads else np.nan
    log_grid = np.log1p(max(grid, 0.0)) if grid == grid else np.nan
    fills_gpu = sms_proxy
    block_size_anchor = block if block == block else np.nan
    mem_pct = _g("Memory Throughput [%]")
    mem_activity_norm = (mem_pct / 100.0) if mem_pct == mem_pct else np.nan
    is_memory_active = (1.0 if mem_pct == mem_pct and mem_pct > 5.0
                        else 0.0)
    src_exec = _g("Execution Time")
    src_log_exec_time = (np.log1p(max(src_exec, 0.0))
                         if src_exec == src_exec else np.nan)
    return np.array([grid_to_sm, threads_to_sm, log_grid, fills_gpu,
                     block_size_anchor, mem_activity_norm,
                     is_memory_active, src_log_exec_time], dtype=np.float64)


# ---------------------------------------------------------------------------
# Stall breakdown — normalize to sum-to-1 with NaN passthrough.

def compute_breakdown(row: pd.Series) -> np.ndarray:
    cats = list(STALL_BREAKDOWN_GROUPS.keys())
    sums = np.empty(len(cats), dtype=np.float64)
    for i, cat in enumerate(cats):
        vals = []
        for n in STALL_BREAKDOWN_GROUPS[cat]:
            try:
                v = float(row.get(n, np.nan))
                if v == v:                # not NaN
                    vals.append(v)
            except Exception:
                pass
        sums[i] = float(np.sum(vals)) if vals else np.nan
    total = np.nansum(sums)
    if not (total > 0):
        return np.full(len(cats), np.nan, dtype=np.float64)
    return sums / total


# ---------------------------------------------------------------------------
# Training pairs — paired by (benchmark__hash, kernel name) across GPUs.

def _roofline_terms(src_row, src_gpu, tgt_gpu):
    """Roofline ingredients (v2). bytes & REAL flops are kernel-intrinsic
    (from the SOURCE measurement); target peak BW/FP32/FP64 from the spec sheet.
    FP32 flops = (2*FFMA + FADD + FMUL) ops/cyc x Elapsed Cycles; FP64 likewise
    with DFMA/DADD/DMUL. (Validated vs Compute%: e.g. adam-cuda ~16 TFLOP/s.)"""
    def g(c):
        v = pd.to_numeric(src_row.get(c), errors="coerce")
        return float(v) if v == v else np.nan
    dur_ns = g("Execution Time")                       # = Duration
    cyc = g("Elapsed Cycles [cycle]")
    mem_bps = g("Memory Throughput [byte/s]")
    fp32 = (2 * g("Predicated-On FFMA Operations Per Cycle [inst]")
            + g("Predicated-On FADD Thread Instructions Executed Per Cycle [inst/cycle]")
            + g("Predicated-On FMUL Thread Instructions Executed Per Cycle [inst/cycle]")) * cyc
    fp64 = (2 * g("Predicated-On DFMA Operations Per Cycle [inst]")
            + g("Predicated-On DADD Thread Instructions Executed Per Cycle [inst/cycle]")
            + g("Predicated-On DMUL Thread Instructions Executed Per Cycle [inst/cycle]")) * cyc
    inst = g("Executed Instructions [inst]")           # warp-level, for the INST_ANCHOR fallback
    t = tgt_gpu.lower(); sg = src_gpu.lower()
    return {
        # LAT_ANCHOR ingredients (see t_lat_of); NaN stays NaN -> the floor is simply 0
        "src_dur_ns": dur_ns, "src_mem_pct": g("Memory Throughput [%]"),
        "src_waves": g("Waves Per SM"),                            # WAVE_ANCHOR ingredient (see t_wave_of)
        "src_dram_latency": GPU_LATENCY_BW_SPECS.get(sg, {}).get("dram_latency_ns", np.nan),
        "target_dram_latency": GPU_LATENCY_BW_SPECS.get(t, {}).get("dram_latency_ns", np.nan),
        "src_n_sms": GPU_HW_FALLBACK.get(sg, {}).get("n_sms", np.nan),
        "target_n_sms": GPU_HW_FALLBACK.get(t, {}).get("n_sms", np.nan),
        "roofline_inst": inst if inst == inst else 0.0,          # warp-inst
        "target_peak_inst": GPU_HW_FALLBACK.get(t, {}).get("peak_inst", np.nan),
        "roofline_bytes": mem_bps * dur_ns * 1e-9,             # byte
        "roofline_fp32_flops": fp32,                           # flop
        "roofline_fp64_flops": fp64,                           # flop
        "target_dram_bw": GPU_HW_FALLBACK.get(t, {}).get("dram_bw", np.nan),
        "target_peak_fp32": GPU_PEAK_SPECS.get(t, {}).get("peak_fp32", np.nan),
        "target_peak_fp64": GPU_PEAK_SPECS.get(t, {}).get("peak_fp64", np.nan),
    }



# ---------------------------------------------------------------------------
# nsys-only TARGET GPU (B300, 2026-10-15): native kernel times exist (data/nsys_et.csv) but no ncu
# profile. It can be a training/test TARGET: ExecTime label = nsys, every other ncu metric NaN
# (masked in the loss: Mem%, occupancy, breakdown), the 5 NCU-architectural spec features inherited
# from an arch-base GPU (H200: the same kernel's row, else the base median) exactly as
# project_dummy_gpu.py does for a dummy GPU, the sheet-derived spec columns from the tables.
NCU_ARCH_SPEC_COLS = ["GPU Maximum Warps Per Scheduler [warp]", "Theoretical Active Warps per SM [warp]",
                      "Theoretical Active Warps Per Scheduler [warp]", "Shared Memory Configuration Size [byte]",
                      "Block Limit Warps [block]"]


def nsys_only_target(gpu: str, arch_base: dict, verbose: bool = True, prefer: str = "H200") -> dict:
    """{benchmark__hash: df} for a GPU that has nsys times but no ncu. `arch_base` is either {run: df} (one GPU)
    or {GPU: {run: df}} (every loaded GPU; nsys-only ones are ignored). A run is kept when ANY arch-base GPU has it
    (a pair needs a source with the same run, not H200 specifically -- the H200-only base kept 79 of 300 usable
    B300 runs, audit 2026-10-15); the 5 NCU-architectural entries come from `prefer`'s row when it has the run,
    else the first GPU that does."""
    if arch_base and all(isinstance(v, dict) for v in arch_base.values()):
        gpus = [prefer] + [g for g in arch_base if g != prefer]
        merged = {}
        for g in gpus:
            for run, df in (arch_base.get(g) or {}).items():
                if "_nsys_launches" in df.columns:
                    continue
                merged.setdefault(run, df)
        arch_base = merged
    import pandas as _pd
    g = str(gpu).lower()
    a = _pd.read_csv("data/nsys_et.csv"); a = a[a.gpu == g]
    out, n_rows, n_inherit = {}, 0, 0
    for run, grp in a.groupby("run"):
        if run not in arch_base:
            continue
        base = arch_base[run]
        med = {c: (float(_pd.to_numeric(base[c], errors="coerce").median()) if c in base.columns else np.nan) for c in NCU_ARCH_SPEC_COLS}
        recs = []
        for kn, ns, nl in zip(grp.kernel, grp.nsys_ns, grp.n_launches):
            if not (ns == ns and ns > 0):
                continue
            rec = {"Kernel Name": kn, "Execution Time": float(ns), "_nsys_launches": float(nl)}
            b = base[base["Kernel Name"] == kn]
            for c in NCU_ARCH_SPEC_COLS:
                v = _pd.to_numeric(b[c].iloc[0], errors="coerce") if (len(b) and c in b.columns) else np.nan
                rec[c] = float(v) if v == v else med[c]
            n_inherit += int(len(b) > 0)
            recs.append(rec)
        if recs:
            df = _pd.DataFrame(recs); _attach_derived_specs(df, g); out[run] = df; n_rows += len(df)
    if verbose:
        print(f"  nsys-only target {g}: {len(out)} runs, {n_rows} kernels ({n_inherit} with the same kernel in the arch base)", flush=True)
    return out


# ---------------------------------------------------------------------------
# AXIS_TRANSPORT (2026-10-16): single-resource extrapolation fix, inference only, model-agnostic.
# Every real training pair moves bandwidth, compute and SM count together, so both model lines learned "faster GPU"
# as ONE direction and mis-attribute single-axis changes (compute-only x32: memory-bound kernels 2.5x faster).
# Outside the region where training GPUs exist IN BALANCE SPACE (bandwidth/SM, FP32/SM, flop/byte, FP32/FP64), hand
# the residual over to the kernel's MEASURED source efficiency r_src = ln(ET_src / t_roof_src) -- i.e. predict
# "same efficiency as measured, times the target's own roofline" -- which is exact physics on every axis (the
# roofline already picks the binding resource). Whole-GPU scaling keeps the balance of a real GPU (distance 0) and
# therefore keeps the model. w = 1 within AT_D0 (log-distance to the nearest training GPU), 0 beyond AT_D1.
AXIS_TRANSPORT = os.environ.get("AXIS_TRANSPORT", "0") == "1"
AT_D0 = float(os.environ.get("AT_D0", "0.69"))      # ln 2
AT_D1 = float(os.environ.get("AT_D1", "1.39"))      # ln 4


def _balance(bw, p32, p64, n):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.array([np.log(bw / n), np.log(p32 / n), np.log(p32 / bw), np.log(p32 / p64)], float)


REF_BUILDER = None      # set by a predict module: f(sample, gpu_name) -> the same kernel/source pair with the target
                        # replaced by that REAL GPU (used for the efficiency reference of AXIS_TRANSPORT)


AT_MEM_BETA = float(os.environ.get("AT_MEM_BETA", "0.45"))
AT_ETA_MAX = float(os.environ.get("AT_ETA_MAX", "0.9"))


def mem_eff_shift(s_tgt, s_ref):
    """Little's-law correction for AXIS_TRANSPORT (2026-10-16): memory efficiency falls with the bytes each SM must
    keep in flight, D = (DRAM BW / SM) x DRAM latency. Across the 7 real GPUs the median achieved DRAM fraction of
    memory-bound kernels follows eta ~ D^-0.43 (corr -0.78; leave-one-GPU-out log error 0.235 vs 0.321 for constant
    efficiency). Residual shift = AT_MEM_BETA x m x ln(D_tgt / D_ref), m = how memory-bound the kernel is at the target
    (logistic in ln(t_mem / t_other))."""
    try:
        bt, nt, lt = float(s_tgt["target_dram_bw"]), float(s_tgt["target_n_sms"]), float(s_tgt.get("target_dram_latency", np.nan))
        br, nr, lr = float(s_ref["target_dram_bw"]), float(s_ref["target_n_sms"]), float(s_ref.get("target_dram_latency", np.nan))
        if not (lt == lt and lt > 0): lt = lr
        b = float(s_tgt["roofline_bytes"]); f32 = float(s_tgt.get("roofline_fp32_flops", 0) or 0); f64 = float(s_tgt.get("roofline_fp64_flops", 0) or 0)
        inst = float(s_tgt.get("roofline_inst", 0) or 0)
        tm = b / bt
        tc = float(tcomp_anchor(f32, f64, inst, float(s_tgt["target_peak_fp32"]), float(s_tgt["target_peak_fp64"]), float(s_tgt.get("target_peak_inst", np.nan))))
        to = max(tc, float(t_lat_of(s_tgt)), 1e-30)
        m = 1.0 / (1.0 + np.exp(-np.log(max(tm, 1e-30) / to) / 0.5))
        dt, dr = bt / nt * lt, br / nr * lr
        if not (np.isfinite(dt) and np.isfinite(dr) and dt > 0 and dr > 0 and np.isfinite(m)):
            return 0.0
        return AT_MEM_BETA * m * float(np.log(dt / dr))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return 0.0


def far_factors(s_tgt, s_ref, troof_ref_ns):
    """AXIS_TRANSPORT far-field answer as a function of the reference prediction ET_ref (2026-10-16 rev 3):
        ET_far = max(ET_mem, ET_rest)
        ET_mem  = bytes / (BW_tgt x eta_tgt),  eta_tgt = eta_ref x (D_tgt / D_ref)^-AT_MEM_BETA, capped at max(1, eta_ref)
                  eta_ref = bytes / (BW_ref x ET_ref)  (achieved DRAM fraction at the reference),  D = BW/SM x latency
        ET_rest = ET_ref x t_rest_tgt / t_roof_ref,  t_rest = the roofline floor WITHOUT the memory term
    Both are linear in ET_ref, returned as factors (g1, g2, g3, g4):
        ET_far = max( ET_ref*g1, g2, ET_ref*g4 )   [ns]   (rev 4; g3 unused)
    Properties: more bandwidth never slower (ET_mem ~ BW^-(1-beta)), never faster than bytes/BW_peak, continuous at
    the reference (both terms reduce to ET_ref there). None when an ingredient is missing."""
    try:
        bt, nt, lt = float(s_tgt["target_dram_bw"]), float(s_tgt["target_n_sms"]), float(s_tgt.get("target_dram_latency", np.nan))
        br, nr, lr = float(s_ref["target_dram_bw"]), float(s_ref["target_n_sms"]), float(s_ref.get("target_dram_latency", np.nan))
        if not (lr == lr and lr > 0): lr = GPU_LATENCY_BW_SPECS.get(str(s_ref.get("tgt_gpu", "")).lower(), {}).get("dram_latency_ns", np.nan)
        if not (lt == lt and lt > 0): lt = lr
        b = float(s_tgt["roofline_bytes"]); f32 = float(s_tgt.get("roofline_fp32_flops", 0) or 0); f64 = float(s_tgt.get("roofline_fp64_flops", 0) or 0)
        inst = float(s_tgt.get("roofline_inst", 0) or 0); pit = float(s_tgt.get("target_peak_inst", np.nan))
        tc = float(tcomp_anchor(f32, f64, inst, float(s_tgt["target_peak_fp32"]), float(s_tgt["target_peak_fp64"]), pit))
        trest = float(troof_floor(0.0, tc, inst, pit, f32, f64, t_lat_of(s_tgt), t_wave_of(s_tgt))) * 1e9
        dt, dr = bt / nt * lt, br / nr * lr
        v = [bt, br, dt, dr, b, trest, troof_ref_ns]
        if not all(np.isfinite(x) and x > 0 for x in v):
            return None
        # rev 6: the law was fitted on D = 3.6 .. 21.5 KiB in flight per SM (7 real GPUs); use it only from half the
        # smallest to twice the largest measured D, constant efficiency beyond (AT_D_LO / AT_D_HI, bytes).
        lo, hi = float(os.environ.get("AT_D_LO", "1825")), float(os.environ.get("AT_D_HI", "44000"))
        dtc, drc = min(max(dt * 1e-9, lo), hi), min(max(dr * 1e-9, lo), hi)
        k = (dtc / drc) ** (-AT_MEM_BETA)
        tm_ref = b / br * 1e9
        # rev 4: split ET_ref by the reference roofline terms (constant-efficiency shares); only the MEMORY share moves
        # with bandwidth and Little's law, and never below bytes / peak BW. At the reference: max(shares) = ET_ref.
        #   ET_far = max( ET_ref*g1, g2, ET_ref*g4 ) restricted to kernels with a memory share; g3 unused (kept = 0)
        g1 = (tm_ref / troof_ref_ns) * (br / bt) / k
        # rev 5: the memory share may not run above AT_ETA_MAX of peak bandwidth (p75 of achieved DRAM fraction is
        # 0.83-0.89 on every real GPU), unless it already did at the reference (then that efficiency is kept):
        #   ET_mem >= min( bytes / (AT_ETA_MAX x BW_tgt),  ET_ref x g3 ),  g3 = constant-efficiency memory share
        g3 = (tm_ref / troof_ref_ns) * (br / bt)
        # rev 7 (2026-10-16): compute SATURATION instead of constant efficiency when the rest-of-roofline grows. The
        # median compute-bound kernel uses 24% of its binding FP peak, so less FP throughput only costs time once the
        # pipe fills: ET_rest = min( ET_ref*g4,  max( ET_ref*g4r, t_comp_tgt / AT_ETA_C ) ),  g4r = the rest share at
        # the reference. Measured on 157 real pairs with >=1.5x less binding FP peak: |log err| 0.716 -> 0.352 (H100->L40S
        # FP64 x1/15: measured 2.8x slower, constant efficiency says 15x). Speed-ups (g4 <= g4r) are unchanged.
        ir = float(s_ref.get("target_peak_inst", np.nan))
        tcr = float(tcomp_anchor(f32, f64, inst, float(s_ref["target_peak_fp32"]), float(s_ref["target_peak_fp64"]), ir))
        trest_r = float(troof_floor(0.0, tcr, inst, ir, f32, f64, t_lat_of(s_ref), t_wave_of(s_ref))) * 1e9
        g4r = trest_r / troof_ref_ns if (np.isfinite(trest_r) and trest_r > 0) else trest / troof_ref_ns
        # rev 7b: saturation is about FP units PER SM. Fewer SMs also remove issue slots and latency hiding, so the
        # unchanged share grows with the wave count: x max(w_tgt,1)/max(w_ref,1), w = the kernel's waves on that GPU
        # (no SM change -> 1). Without it 1/8 of the SMs gave 1.8x instead of ~8x for compute-bound kernels.
        nt_, nr_ = float(s_tgt["target_n_sms"]), float(s_ref["target_n_sms"])
        ws_ = float(s_tgt.get("src_waves", np.nan)); ns_ = float(GPU_HW_FALLBACK.get(str(s_tgt.get("src_gpu", "")).lower(), {}).get("n_sms", np.nan))
        if np.isfinite(ws_) and ws_ >= 0 and np.isfinite(ns_) and ns_ > 0 and nt_ > 0 and nr_ > 0:
            fs_ = max(ws_ * ns_ / nt_, 1.0) / max(ws_ * ns_ / nr_, 1.0)
        else:
            fs_ = max(nr_ / nt_, 1.0) if (nt_ > 0 and nr_ > 0) else 1.0
        g4r *= fs_
        g5 = tc * 1e9 / float(os.environ.get("AT_ETA_C", "0.9"))
        return (g1, b / bt * 1e9 / AT_ETA_MAX, g3, trest / troof_ref_ns, g4r, g5)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


def axis_weight(samples, exclude=()):
    """(w, nearest) per sample: w as in axis_transport, nearest = the training GPU closest in balance space."""
    ex = {str(e).lower() for e in exclude} | {"gb10"}
    names, pts = [], []
    for g in GPU_HW_FALLBACK:
        if g in GPU_PEAK_SPECS and g not in ex:
            q = _balance(GPU_HW_FALLBACK[g]["dram_bw"], GPU_PEAK_SPECS[g]["peak_fp32"], GPU_PEAK_SPECS[g]["peak_fp64"], GPU_HW_FALLBACK[g]["n_sms"])
            if np.all(np.isfinite(q)):
                names.append(g); pts.append(q)
    pts = np.array(pts); n = len(samples); w = np.ones(n); near = [None] * n
    # rev 8 (2026-10-16): TOTAL SIZE as a fifth coordinate (AT_SIZE=1). Whole-GPU scaling keeps every balance ratio, so
    # a 16x H200 sat at distance 0 and got the raw network, whose residual extrapolates in absolute size (latency
    # kernels 0.037 vs the grid ceiling 0.068 at 16x). Distance = log of how far the SM count lies outside the trained
    # range [min, max] of real SM counts; the same ln2 -> ln4 ramp then hands it to constant efficiency.
    _sms = [float(GPU_HW_FALLBACK[g]["n_sms"]) for g in names]
    _nlo, _nhi = (min(_sms), max(_sms)) if _sms else (np.nan, np.nan)
    _size = os.environ.get("AT_SIZE", "1") == "1"
    for i, s in enumerate(samples):
        try:
            bt = _balance(float(s["target_dram_bw"]), float(s["target_peak_fp32"]), float(s["target_peak_fp64"]), float(s["target_n_sms"]))
        except (KeyError, TypeError, ValueError):
            continue
        if not np.all(np.isfinite(bt)):
            continue
        dd = np.max(np.abs(pts - bt[None, :]), axis=1); k = int(np.argmin(dd)); dk = dd[k]
        if _size:
            try:
                _nt = float(s["target_n_sms"])
                if np.isfinite(_nt) and _nt > 0:
                    dk = max(dk, np.log(_nt / _nhi), np.log(_nlo / _nt))
            except (KeyError, TypeError, ValueError):
                pass
        w[i] = float(np.clip((AT_D1 - dk) / max(AT_D1 - AT_D0, 1e-9), 0.0, 1.0))
        # Efficiency reference (2026-10-16 rev 2): the GPU the configuration is DERIVED from -- the target's own name
        # when it is a real GPU (scale_gpu_sweep keeps "H200" on its scaled rows; for an unmodified real target the
        # reference pass reproduces the prediction, so transport is a no-op), else AT_BASE (dummy sheets declare
        # base_arch_profile = H200), else the nearest training GPU. Nearest-only switched references mid-sweep
        # (32x-bandwidth H200 -> B300 efficiency) and broke continuity.
        tg = str(s.get("tgt_gpu", "")).lower()
        near[i] = tg if (tg in GPU_HW_FALLBACK and tg in GPU_PEAK_SPECS) else (os.environ.get("AT_BASE", "").lower() or names[k])
    return w, near


def axis_transport(samples, exclude=()):
    """(w, r_src) per sample; w = 1 -> use the model, w = 0 -> measured source efficiency x target roofline."""
    ex = {str(e).lower() for e in exclude} | {"gb10"}
    pts = [_balance(GPU_HW_FALLBACK[g]["dram_bw"], GPU_PEAK_SPECS[g]["peak_fp32"], GPU_PEAK_SPECS[g]["peak_fp64"], GPU_HW_FALLBACK[g]["n_sms"])
           for g in GPU_HW_FALLBACK if g in GPU_PEAK_SPECS and g not in ex]
    pts = np.array([q for q in pts if np.all(np.isfinite(q))])
    n = len(samples); w = np.ones(n); r = np.full(n, np.nan)
    for i, s in enumerate(samples):
        try:
            bt = _balance(float(s["target_dram_bw"]), float(s["target_peak_fp32"]), float(s["target_peak_fp64"]), float(s["target_n_sms"]))
        except (KeyError, TypeError, ValueError):
            continue
        if not np.all(np.isfinite(bt)):
            continue
        d = float(np.min(np.max(np.abs(pts - bt[None, :]), axis=1)))           # Chebyshev log-distance
        sg = str(s.get("src_gpu", "")).lower(); hw, pk, lb = GPU_HW_FALLBACK.get(sg), GPU_PEAK_SPECS.get(sg), GPU_LATENCY_BW_SPECS.get(sg, {})
        et = float(s.get("src_dur_ns", np.nan) or np.nan)
        if not hw or not pk or not (et == et and et > 0):
            continue
        ck = GPU_CLOCKS.get(sg, {}); cs = 1.0
        if ck.get("ncu") and ck.get("operating"):
            cs = ck["ncu"] / ck["operating"]           # source ET is the ncu duration -> peaks at the ncu clock
        s2 = dict(s); s2.update(target_dram_bw=hw["dram_bw"], target_peak_fp32=pk["peak_fp32"] * cs, target_peak_fp64=pk["peak_fp64"] * cs,
                                 target_peak_inst=hw.get("peak_inst", np.nan) * cs, target_n_sms=hw["n_sms"],
                                 target_dram_latency=lb.get("dram_latency_ns", np.nan), tgt_gpu=sg)
        b = float(s.get("roofline_bytes", np.nan)); f32 = float(s.get("roofline_fp32_flops", 0) or 0); f64 = float(s.get("roofline_fp64_flops", 0) or 0)
        inst = float(s.get("roofline_inst", 0) or 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            tm = b / s2["target_dram_bw"]
            tc = float(tcomp_anchor(f32, f64, inst, s2["target_peak_fp32"], s2["target_peak_fp64"], s2["target_peak_inst"]))
            tr = float(troof_floor(tm, tc, inst, s2["target_peak_inst"], f32, f64, t_lat_of(s2), t_wave_of(s2)))
        if not (np.isfinite(tr) and tr > 0):
            continue
        r[i] = np.log(et * 1e-9 / tr)
        w[i] = float(np.clip((AT_D1 - d) / max(AT_D1 - AT_D0, 1e-9), 0.0, 1.0))
    return w, r

_NSYS = None
def native_et(gpu, run, kernel, ncu_ns):
    """ET label under ET_LABEL=nsys. Returns (value_ns, source) with source in
    {'nsys', 'ncu/ratio', 'ncu'}. Tables: data/nsys_et.csv, data/ncu_inflation.csv (build_nsys_et.py)."""
    global _NSYS
    if _NSYS is None:
        import pandas as _pd
        a = _pd.read_csv("data/nsys_et.csv"); b = _pd.read_csv("data/ncu_inflation.csv")
        _NSYS = [{(g, r, k): v for g, r, k, v in zip(a.gpu, a.run, a.kernel, a.nsys_ns)},
                 {(r, k): v for r, k, v in zip(b.run, b.kernel, b.ratio)}]
    v = _NSYS[0].get((str(gpu).lower(), run, kernel))
    if v is not None and v > 0:
        return float(v), "nsys"
    r = _NSYS[1].get((run, kernel))
    if r is not None and r > 0 and ncu_ns == ncu_ns:
        return float(ncu_ns) / float(r), "ncu/ratio"
    # ET_LABEL_FALLBACK=run: no ratio for THIS kernel -> median ratio of the other kernels of
    # the same run (same binary and args; the inflation is launch-pattern driven), else the
    # global median. Keeps the 3,050 "plain ncu" pairs from staying inflated.
    if os.environ.get("ET_LABEL_FALLBACK", "") == "run" and ncu_ns == ncu_ns:
        if len(_NSYS) == 2:
            import collections as _c, numpy as _np
            byrun = _c.defaultdict(list)
            for (rr, kk), vv in _NSYS[1].items(): byrun[rr].append(vv)
            _NSYS.append({rr: float(_np.median(vv)) for rr, vv in byrun.items()}); _NSYS.append(float(_np.median(list(_NSYS[1].values()))))
        rr = _NSYS[2].get(run, _NSYS[3])
        return float(ncu_ns) / float(rr), "ncu/run-ratio"
    return float(ncu_ns) if ncu_ns == ncu_ns else np.nan, "ncu"
ET_LABEL = os.environ.get("ET_LABEL", "ncu")


def _relabel_et(tgt_gpu, run, kernel, reg):
    """Swap reg[ET] (ncu Duration) for the native time when ET_LABEL=nsys."""
    if ET_LABEL != "nsys":
        return reg
    reg = np.array(reg, dtype=np.float64)
    v, src = native_et(tgt_gpu, run, kernel, reg[0])
    reg[0] = v; ET_LABEL_STATS[src] += 1; ET_LABEL_SRC[(str(tgt_gpu).lower(), run, kernel)] = src
    return reg


def label_weight(s):
    """LABEL_W="w_nsys,w_ratio,w_ncu" (e.g. "1,0.7,0.5"): sample weight by how the ET label was
    obtained -- native, ratio-corrected, or plain ncu (inflated for a tail of kernels). 1 if unset."""
    spec = os.environ.get("LABEL_W", "")
    if not spec:
        return 1.0
    w = [float(x) for x in spec.split(",")]
    src = ET_LABEL_SRC.get((str(s.get("tgt_gpu", "")).lower(), s.get("benchmark", ""), s.get("kernel_name", "")), "ncu")
    return w[0] if src == "nsys" else (w[1] if src.startswith("ncu/") else w[2])
ET_LABEL_STATS = {"nsys": 0, "ncu/ratio": 0, "ncu/run-ratio": 0, "ncu": 0}
ET_LABEL_SRC = {}          # (tgt_gpu, run, kernel) -> label source, for LABEL_W


def build_training_pairs(
    gpu_benchmarks: dict[str, dict[str, pd.DataFrame]],
) -> list[dict]:
    samples: list[dict] = []
    gpus = list(gpu_benchmarks.keys())

    # Union of benchmark__hash keys across GPUs.
    all_keys: set[str] = set()
    for g in gpus:
        all_keys |= set(gpu_benchmarks[g].keys())

    n_by_dir: dict[str, int] = {}

    for key in sorted(all_keys):
        gpus_with_key = [g for g in gpus if key in gpu_benchmarks[g]]
        if len(gpus_with_key) < 2:
            continue
        for src_gpu in gpus_with_key:
            if "_nsys_launches" in gpu_benchmarks[src_gpu][key].columns:
                continue          # nsys-only GPU (B300): no ncu features -> TARGET only, never a source
            for tgt_gpu in gpus_with_key:
                if src_gpu == tgt_gpu:
                    continue
                src_df = gpu_benchmarks[src_gpu][key]
                tgt_df = gpu_benchmarks[tgt_gpu][key]
                shared_kernels = (set(src_df["Kernel Name"])
                                   & set(tgt_df["Kernel Name"]))
                for kname in sorted(shared_kernels):   # deterministic pair order (a set iterates in hash order -> batch order -> training trajectory differed run to run)
                    src_row = src_df[src_df["Kernel Name"] == kname].iloc[0]
                    tgt_row = tgt_df[tgt_df["Kernel Name"] == kname].iloc[0]
                    src_specs = extract_features(src_row, GPU_SPEC_FEATURES)
                    tgt_specs = extract_features(tgt_row, GPU_SPEC_FEATURES)
                    # RAW source specs (no log1p) — sensei prefers raw values;
                    # the per-feature standardization downstream handles scale.
                    source_specs_raw = np.where(
                        np.isnan(src_specs), np.nan,
                        np.maximum(src_specs, 0.0))
                    # target_specs = ratio target_over_source (mirrors v1).
                    tgt_over_src = spec_delta(tgt_specs, src_specs)   # ratio / log-ratio / categorical diff
                    # Pre-compute the source kernel's breakdown — a strong
                    # prior for the target's stall distribution. Append
                    # to the workload vector (semantically natural; same
                    # branch as the raw stall metrics).
                    src_breakdown = compute_breakdown(src_row)
                    samples.append({
                        "kernel_config": extract_features(
                            src_row, KERNEL_CONFIG_FEATURES),
                        "workload": np.concatenate([
                            extract_features(src_row,
                                              WORKLOAD_PROFILE_FEATURES),
                            extract_features(src_row, STALL_FEATURES),
                            src_breakdown,   # 4 fractions: src GPU's stall split
                        ]),
                        "source_specs": source_specs_raw,
                        "target_specs": tgt_over_src.astype(np.float64),
                        "derived": compute_derived_features(src_row,
                                                             tgt_specs),
                        "target_regression": _relabel_et(tgt_gpu, key, kname, extract_features(
                            tgt_row, OUTPUT_REGRESSION)),
                        "target_breakdown": compute_breakdown(tgt_row),
                        "kernel_name": kname,
                        "benchmark": key,
                        "src_gpu": src_gpu,
                        "tgt_gpu": tgt_gpu,
                        **_roofline_terms(src_row, src_gpu, tgt_gpu),
                    })
                    k = f"{src_gpu}->{tgt_gpu}"
                    n_by_dir[k] = n_by_dir.get(k, 0) + 1

    print(f"Total pairs: {len(samples)}  by direction: {n_by_dir}")
    return samples


# ---------------------------------------------------------------------------
# Stacking + NaN handling.
# NaN policy:
#   - Inputs (kernel_config, workload, source_specs, target_specs, derived):
#       impute with the COLUMN MEAN (over the stack), so NaN never becomes 0.
#   - Targets (target_regression, target_breakdown): NaN kept; the training
#       loop must mask NaN entries from the loss.

def _impute_mean(arr: np.ndarray) -> np.ndarray:
    """Replace NaN per-column with that column's mean (over rows where present).
    Columns that are 100% NaN are filled with 0 explicitly (last resort)."""
    if arr.ndim == 1:
        return arr
    out = arr.copy()
    col_mean = np.nanmean(out, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
    nan_mask = np.isnan(out)
    for j in range(out.shape[1]):
        out[nan_mask[:, j], j] = col_mean[j]
    return out


def stack_samples(samples: list[dict]) -> dict[str, np.ndarray]:
    keys = ["kernel_config", "workload", "source_specs", "target_specs",
            "derived", "target_regression", "target_breakdown"]
    out: dict[str, np.ndarray] = {}
    for k in keys:
        out[k] = np.stack([s[k] for s in samples], axis=0).astype(np.float64)
    # Impute INPUTS only (so 0 is never substituted for missing).
    for k in ("kernel_config", "workload", "source_specs", "target_specs",
              "derived"):
        out[k] = _impute_mean(out[k])
    # Targets keep their NaN; training loop must mask them.
    return out
