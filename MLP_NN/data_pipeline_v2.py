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
    "a100":  {"dram_latency_ns": 395.0, "cpu_gpu_bw":  64.0e9},
    "h100":  {"dram_latency_ns": 370.0, "cpu_gpu_bw": 128.0e9},
    "gb200": {"dram_latency_ns": 440.0, "cpu_gpu_bw": 900.0e9},
    "gb10":  {"dram_latency_ns": 210.0, "cpu_gpu_bw": 600.0e9},
}


def _load_spec_sheet_overrides():
    """Single source of truth: override the hardcoded GPU spec dicts above with
    values from data/gpu_microarch_specs.csv. This keeps everything the model
    consumes traceable to NCU data or the spec sheet (no hidden constants).
    Unit conversions: GB/s->byte/s (1e9), TFLOPS->FLOPS (1e12), MB->byte (1e6)."""
    import csv
    import os
    path = os.path.join(os.path.dirname(__file__), os.pardir,
                        "data", "gpu_microarch_specs.csv")
    if not os.path.exists(path):
        return  # keep hardcoded fallback if the sheet is unavailable
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.setdefault(r["spec"], r)   # first row wins for duplicate keys

    def g(spec, col):
        v = str(rows.get(spec, {}).get(col, "")).strip()
        return float(v) if v not in ("", "nan", "None") else None

    for lc, col in {"a100": "A100", "h100": "H100",
                    "gb200": "GB200", "gb10": "GB10"}.items():
        GPU_PEAK_SPECS[lc] = {
            "peak_fp32": g("peak_fp32", col) * 1e12,
            "peak_fp64": g("peak_fp64", col) * 1e12,
            "peak_tensor_fp16": g("peak_fp16_tensor", col) * 1e12,
            "l2_size": g("l2_cache_total", col) * 1e6,
        }
        GPU_HW_FALLBACK[lc] = {
            "dram_bw": g("dram_bandwidth", col) * 1e9,
            "n_sms": g("num_sms", col),
        }
        GPU_LATENCY_BW_SPECS[lc] = {
            "dram_latency_ns": g("dram_latency_model_recommended", col),
            "cpu_gpu_bw": g("cpu_gpu_bandwidth", col) * 1e9,
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


def extract_features(row: pd.Series, names: list[str]) -> np.ndarray:
    """Pull a list of named features from a row. NaN stays NaN."""
    vals = np.empty(len(names), dtype=np.float64)
    for i, n in enumerate(names):
        if n not in row.index:
            vals[i] = np.nan
            continue
        v = row[n]
        try:
            vals[i] = float(v)
        except (TypeError, ValueError):
            vals[i] = np.nan
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
    t = tgt_gpu.lower()
    return {
        "roofline_bytes": mem_bps * dur_ns * 1e-9,             # byte
        "roofline_fp32_flops": fp32,                           # flop
        "roofline_fp64_flops": fp64,                           # flop
        "target_dram_bw": GPU_HW_FALLBACK.get(t, {}).get("dram_bw", np.nan),
        "target_peak_fp32": GPU_PEAK_SPECS.get(t, {}).get("peak_fp32", np.nan),
        "target_peak_fp64": GPU_PEAK_SPECS.get(t, {}).get("peak_fp64", np.nan),
    }


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
            for tgt_gpu in gpus_with_key:
                if src_gpu == tgt_gpu:
                    continue
                src_df = gpu_benchmarks[src_gpu][key]
                tgt_df = gpu_benchmarks[tgt_gpu][key]
                shared_kernels = (set(src_df["Kernel Name"])
                                   & set(tgt_df["Kernel Name"]))
                for kname in shared_kernels:
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
                    with np.errstate(divide="ignore", invalid="ignore"):
                        tgt_over_src = tgt_specs / np.where(
                            np.abs(src_specs) > 1e-8, src_specs, 1e-8)
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
                        "target_regression": extract_features(
                            tgt_row, OUTPUT_REGRESSION),
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
