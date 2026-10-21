"""Self-contained inference core for the v1.5 cross-GPU estimator.

Contains the model class + the EXACT feature construction used at training time,
copied verbatim from the research code so predictions match.  Depends only on
numpy / pandas / torch — nothing else in MLP_NN/.

v1.5 = MLP + roofline quantities as extra INPUT features (always on here).
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# model (verbatim from MLP_NN/mlp.py)                                          #
# --------------------------------------------------------------------------- #
class MultiBranchMLP(nn.Module):
    def __init__(self, branch_dims, shared_branch_indices=None, branch_hidden=64,
                 shared_hidden=128, n_shared_layers=2, regression_outputs=5,
                 breakdown_outputs=4, dropout=0.1):
        super().__init__()
        self.n_branches = len(branch_dims)
        self.shared_groups = shared_branch_indices or []
        branch_to_encoder, encoders, assigned = {}, [], set()
        for group in self.shared_groups:
            if len({branch_dims[i] for i in group}) != 1:
                raise ValueError(f"Shared branches must have same input dim: {group}")
            ei = len(encoders)
            encoders.append(self._make_branch(branch_dims[group[0]], branch_hidden, dropout))
            for i in group:
                branch_to_encoder[i] = ei; assigned.add(i)
        for i in range(self.n_branches):
            if i not in assigned:
                branch_to_encoder[i] = len(encoders)
                encoders.append(self._make_branch(branch_dims[i], branch_hidden, dropout))
        self.encoders = nn.ModuleList(encoders)
        self.branch_to_encoder = branch_to_encoder
        in_dim = branch_hidden * self.n_branches
        layers = []
        for _ in range(n_shared_layers):
            layers += [nn.Linear(in_dim, shared_hidden), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = shared_hidden
        self.shared = nn.Sequential(*layers)
        self.regression_head = nn.Linear(shared_hidden, regression_outputs)
        self.breakdown_head = nn.Linear(shared_hidden, breakdown_outputs) if breakdown_outputs > 0 else None

    @staticmethod
    def _make_branch(in_dim, hidden, dropout):
        return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))

    # SPEC_DROPOUT (2026-10-02): drop individual SPEC features (branches 2 = source_specs,
    # 3 = target_specs) at random during training, so no single spec value can act as a GPU
    # identity tag (the host-link number became "this is GB200" once GH200 was held out and
    # cost the NN 10-15 points there; the PINN, regularised by its physics loss, does not
    # leak). Inference is untouched (eval mode). Read from the environment so every trainer
    # picks it up without a signature change; 0 = off (every artifact before v4.39).
    SPEC_DROPOUT = float(os.environ.get("SPEC_DROPOUT", "0") or 0)

    def forward(self, *inputs):
        if self.training and self.SPEC_DROPOUT > 0:
            inputs = tuple(F.dropout(x, p=self.SPEC_DROPOUT, training=True) if i in (2, 3) else x
                           for i, x in enumerate(inputs))
        outs = [self.encoders[self.branch_to_encoder[i]](x) for i, x in enumerate(inputs)]
        hidden = self.shared(torch.cat(outs, dim=-1))
        out = {"regression": self.regression_head(hidden)}
        if self.breakdown_head is not None:
            out["breakdown"] = F.softmax(self.breakdown_head(hidden), dim=-1)
        return out


# --------------------------------------------------------------------------- #
# input column lists (verbatim from MLP_NN/data_pipeline_v2.py)                #
# --------------------------------------------------------------------------- #
KERNEL_CONFIG_FEATURES = [
    "Block Size", "Grid Size", "Threads", "Registers Per Thread [register/thread]",
    "Static Shared Memory Per Block [byte/block]", "Dynamic Shared Memory Per Block [byte/block]",
    "Shared Memory Per Block [byte/block]"]
WORKLOAD_PROFILE_FEATURES = [
    "Achieved Occupancy", "Achieved Active Warps Per SM [warp]", "Compute (SM) Throughput [%]",
    "Memory Throughput [%]", "L1/TEX Cache Throughput [%]", "L2 Cache Throughput [%]",
    "Memory Throughput [byte/s]", "Eligible Warps Per Scheduler [warp]",
    "Executed Ipc Active [inst/cycle]", "Warp Cycles Per Executed Instruction [cycle/inst]",
    "Theoretical Active Warps per SM [warp]", "Waves Per SM", "Block Limit Registers [block]",
    "Block Limit Warps [block]", "Block Limit SM [block]", "Block Limit Shared Mem [block]"]
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

STALL_FEATURES = [
    "Stall Barrier [inst]", "Stall Branch Resolving [inst]", "Stall Dispatch Stall [inst]",
    "Stall Drain [inst]", "Stall LG Throttle [inst]", "Stall Long Scoreboard [inst]",
    "Stall MIO Throttle [inst]", "Stall Math Pipe Throttle [inst]", "Stall Membar [inst]",
    "Stall Misc [inst]", "Stall No Instruction [inst]", "Stall Not Selected [inst]",
    "Stall Short Scoreboard [inst]", "Stall Sleeping [inst]", "Stall Tex Throttle [inst]",
    "Stall Wait [inst]"]
GPU_SPEC_FEATURES = [
    "GPU Maximum Warps Per Scheduler [warp]", "Theoretical Active Warps per SM [warp]",
    "Theoretical Active Warps Per Scheduler [warp]", "Shared Memory Configuration Size [byte]",
    "Block Limit Warps [block]", "Peak FP32 / Peak FP64", "DRAM BW per SM [byte/s/SM]",
    "L2 Size per SM [byte/SM]", "Peak FP32 / DRAM BW [flop/byte]", "Peak FP16 Tensor [flop/s]",
    "Peak Tensor / Peak FP32", "DRAM Latency [ns]", "CPU-GPU BW [byte/s]"]

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
# HOST_CAT: host link as two categories (coherent flag, PCIe gen) -- see data_pipeline_v2.
HOST_CAT_FEATURES = ["Host Link Coherent [0/1]", "Host PCIe Gen"]
if os.environ.get("HOST_CAT", "0") == "1":
    GPU_SPEC_FEATURES = [f for f in GPU_SPEC_FEATURES if f != "CPU-GPU BW [byte/s]"] + HOST_CAT_FEATURES
SPEC_CAT_MASK = [f in HOST_CAT_FEATURES for f in GPU_SPEC_FEATURES]
STALL_BREAKDOWN_GROUPS = {
    "memory": ["Stall Long Scoreboard [inst]", "Stall Short Scoreboard [inst]",
               "Stall LG Throttle [inst]", "Stall MIO Throttle [inst]",
               "Stall Tex Throttle [inst]", "Stall Drain [inst]"],
    "pipeline_contention": ["Stall Math Pipe Throttle [inst]", "Stall Wait [inst]",
                            "Stall No Instruction [inst]", "Stall Branch Resolving [inst]",
                            "Stall Dispatch Stall [inst]"],
    "sync": ["Stall Barrier [inst]", "Stall Membar [inst]"],
    "scheduling_overhead": ["Stall Not Selected [inst]", "Stall Sleeping [inst]", "Stall Misc [inst]"]}
OUTPUT_REGRESSION = ["Execution Time", "Memory Throughput [%]", "Achieved Occupancy"]

ET, MEMPCT, OCC = 0, 1, 2
BRK = list(STALL_BREAKDOWN_GROUPS.keys())
KEEP, SRC_LOG_IDX, RATIO_CAP = [0, 1, 2, 3, 4], 7, 50.0
KEYS = ["kernel_config", "workload", "source_specs", "target_specs", "derived", "target_regression"]


# --------------------------------------------------------------------------- #
# feature functions (verbatim)                                                 #
# --------------------------------------------------------------------------- #
def extract_features(row, names):
    vals = np.empty(len(names), dtype=np.float64)
    for i, n in enumerate(names):
        if n not in row.index:
            vals[i] = np.nan; continue
        try:
            vals[i] = float(row[n])
        except (TypeError, ValueError):
            vals[i] = np.nan
    # ncu emits 'inf' for zero-denominator ratios (seen: Waves Per SM, 2026-10-01). A non-finite
    # value is a MISSING value: keep it NaN (training-mean imputed), never let inf reach a std.
    vals[~np.isfinite(vals)] = np.nan
    return vals


def compute_breakdown(row):
    cats = list(STALL_BREAKDOWN_GROUPS.keys())
    sums = np.empty(len(cats), dtype=np.float64)
    for i, cat in enumerate(cats):
        vals = []
        for n in STALL_BREAKDOWN_GROUPS[cat]:
            try:
                v = float(row.get(n, np.nan))
                if v == v:
                    vals.append(v)
            except Exception:
                pass
        sums[i] = float(np.sum(vals)) if vals else np.nan
    total = np.nansum(sums)
    if not (total > 0):
        return np.full(len(cats), np.nan, dtype=np.float64)
    return sums / total


def compute_derived_features(src_row, target_specs_raw):
    def _g(k):
        try:
            v = src_row.get(k, np.nan)
            return float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else np.nan
        except Exception:
            return np.nan
    block, grid = _g("Block Size"), _g("Grid Size")
    threads = block * grid if block == block and grid == grid else np.nan
    sms_proxy = _g("Block Limit SM [block]")
    if not (sms_proxy == sms_proxy) or sms_proxy <= 0:
        sms_proxy = 1.0
    grid_to_sm = grid / sms_proxy if grid == grid else np.nan
    threads_to_sm = threads / sms_proxy if threads == threads else np.nan
    log_grid = np.log1p(max(grid, 0.0)) if grid == grid else np.nan
    mem_pct = _g("Memory Throughput [%]")
    mem_activity_norm = (mem_pct / 100.0) if mem_pct == mem_pct else np.nan
    is_memory_active = 1.0 if mem_pct == mem_pct and mem_pct > 5.0 else 0.0
    src_exec = _g("Execution Time")
    src_log = np.log1p(max(src_exec, 0.0)) if src_exec == src_exec else np.nan
    return np.array([grid_to_sm, threads_to_sm, log_grid, sms_proxy,
                     block if block == block else np.nan, mem_activity_norm,
                     is_memory_active, src_log], dtype=np.float64)


def make_derived_raw(derived):
    base = derived[:, KEEP]
    src_raw = np.expm1(derived[:, SRC_LOG_IDX:SRC_LOG_IDX + 1])
    return np.concatenate([base, src_raw], axis=1)


def recover_et_relative(pr_col, src_ns):
    return np.clip(pr_col, 0.0, RATIO_CAP) * src_ns


# --------------------------------------------------------------------------- #
# inference                                                                    #
# --------------------------------------------------------------------------- #
def _featurize(samples):
    """Stack per-sample branch arrays and append the v1.5 roofline input feats."""
    stk = ["kernel_config", "workload", "source_specs", "target_specs",
           "derived", "target_regression", "target_breakdown"]
    D = {k: np.stack([s[k] for s in samples], 0).astype(np.float64) for k in stk}
    src_ns = np.expm1(D["derived"][:, SRC_LOG_IDX]).clip(min=1.0)
    D["derived"] = make_derived_raw(D["derived"])
    b = np.array([s["roofline_bytes"] for s in samples])
    f32 = np.array([s["roofline_fp32_flops"] for s in samples])
    f64 = np.array([s["roofline_fp64_flops"] for s in samples])
    bw = np.array([s["target_dram_bw"] for s in samples])
    p32 = np.array([s["target_peak_fp32"] for s in samples])
    p64 = np.array([s["target_peak_fp64"] for s in samples])
    with np.errstate(divide="ignore", invalid="ignore"):
        t_mem, t_comp = b / bw, np.fmax(f32 / p32, f64 / p64)
        ai = (f32 + f64) / b
        bott = np.log1p(np.clip(t_mem, 0, None)) - np.log1p(np.clip(t_comp, 0, None))
    L = lambda x: np.log1p(np.clip(x, 0, None))
    feats = np.column_stack([L(b), L(f32), L(f64), L(t_mem), L(t_comp), L(ai), bott])
    feats = np.where(np.isfinite(feats), feats, np.nan)
    D["derived"] = np.concatenate([D["derived"], feats], axis=1)
    return D, src_ns


STD_EPS = 1e-6      # below this a training feature is ~constant -> it carries no signal


def safe_standardize(x, mean, std):
    """Z-score, but ZERO OUT features whose training std ~= 0.

    A feature that was constant in training (std collapses to the fit-time floor, e.g.
    1e-8) was only ever seen by the network as z=0 (x == mean). A nonzero test value
    would otherwise standardize to (x-mean)/1e-8 ~= 1e11 and saturate the network
    (the GENESIS dynamic-shared-memory blow-up). Such a feature has no learned signal,
    so map it to 0 (= the training-constant) instead of exploding."""
    std = np.asarray(std, dtype=float)
    safe = std > STD_EPS
    z = np.where(safe, (np.asarray(x, dtype=float) - mean) / np.where(safe, std, 1.0), 0.0)
    # CLIP_Z (2026-10-02): clip standardized inputs to +-c sigma. An unseen GPU whose spec sits
    # far outside the training range (L40S held out: fp32:fp64 log-ratio +3.5 vs a training
    # spread of +-0.3) otherwise walks the network into a region it never saw -> 50-1000x
    # blow-ups. Applied identically in training (eval_ensemble_cv._norm), so in-range inputs
    # are untouched. 0 = off (every artifact before v4.32).
    c = float(os.environ.get("CLIP_Z", "0") or 0)
    return np.clip(z, -c, c) if c > 0 else z


def trust_weight(x, mean, std, spec=None):
    """TRUST REGION for an unseen target GPU (2026-10-02). Returns a per-row weight w in [0,1]:
    1 inside the training range, falling to 0 as the target-spec vector leaves it.
    zmax = max over features of |unclipped z-score|; w = clip((hi - zmax) / (hi - lo), 0, 1)
    with TRUST_Z="lo,hi" (e.g. "3,6"). The caller blends the network's residual as
    w * r + (1 - w) * r_typical, so a GPU far outside training (L40S held out: several
    axes at 4-10 sigma) gets "typical efficiency x its own roofline" instead of an
    unconstrained extrapolation (which reached 1000x), while a 16x whole-GPU projection
    (z ~ 2-3) is untouched. Off unless TRUST_Z is set."""
    spec = spec if spec is not None else os.environ.get("TRUST_Z", "")
    if not spec:
        return None
    lo, hi = (float(v) for v in spec.split(","))
    std = np.asarray(std, dtype=float); safe = std > STD_EPS
    z = np.where(safe, (np.asarray(x, dtype=float) - mean) / np.where(safe, std, 1.0), 0.0)
    zmax = np.nanmax(np.abs(z), axis=1)
    return np.clip((hi - zmax) / max(hi - lo, 1e-9), 0.0, 1.0)


def predict(model, stats, samples):
    """Return dict(et_p, pr[:, reg], pb[:, brk]).  NaN inputs are imputed with the
    model's TRAINING means (in `stats`) so a single row never blows up."""
    D, src_ns = _featurize(samples)
    N = {}
    for k in KEYS:
        mean, std = stats[k]
        x = np.where(np.isnan(D[k]), mean, D[k])            # training-mean impute
        N[k] = torch.tensor(safe_standardize(x, mean, std), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        po = model(N["kernel_config"], N["workload"], N["source_specs"],
                   N["target_specs"], N["derived"])
    rmn, rsd = stats["target_regression"]
    pr = po["regression"].numpy() * rsd + rmn
    et = recover_et_relative(pr[:, ET], src_ns)
    pb = po["breakdown"].numpy()
    return dict(et_p=et, pr=pr, pb=pb)


# --------------------------------------------------------------------------- #
# profiling records dump (io_signal schema; shared by every deliverable)       #
# --------------------------------------------------------------------------- #
# Per-model I-derived:: column names. The roofline-input feats are appended to
# the derived branch at inference (see _featurize / v40_core._features).
_ROOF7 = ["log_bytes", "log_fp32_flops", "log_fp64_flops", "log_t_mem_target",
          "log_t_comp_target", "log_arith_intensity", "bottleneck_log_tmem_minus_tcomp"]
# v1.5 / v2.1 (use source Execution Time as an input -> keep src_raw_exec_time_ns):
DERIVED_NAMES_V15 = ["grid_to_sm_ratio", "threads_to_sm_ratio", "log_grid", "fills_gpu",
                     "block_size_anchor", "src_raw_exec_time_ns"] + _ROOF7
# v4.0 / v4.1 (no-ET: Execution Time is NOT an input -> src_raw_exec_time_ns REMOVED;
# the no-ET models instead carry 4 extra source-roofline / log-rate inputs):
DERIVED_NAMES_V40 = ["grid_to_sm_ratio", "threads_to_sm_ratio", "log_grid", "fills_gpu",
                     "block_size_anchor"] + _ROOF7 + ["log_src_t_mem", "log_src_t_comp",
                     "log_src_roof", "log_mem_throughput_bps"]


def dump_records(path, model_tag, samples, out_rows, featurize_fn, derived_names, no_comments=False):
    """Write a per-kernel profiling record in the io_signal schema:
        meta-*  identity (+ pair_type = self/cross)
        I-*     model INPUTS, grouped kcfg/wl/srcspec/tgtspec(ratio)/derived -- EXACTLY
                the features THIS model consumes (an excluded input never appears; e.g.
                v4.x has no I-derived::src_raw_exec_time_ns)
        O-*     model PREDICTIONS (the 7 outputs)        aux-roofline::  in-tool roofline
    PREDICTIONS ONLY -- no ground-truth columns. Join a separate truth record on
    meta-(kernel, src_gpu, tgt_gpu) to score. For meta-pair_type=self (src==tgt) the
    input's measured outputs ARE the target truth; for cross they are the source's, so
    the deliverable cannot supply target truth here.
    `featurize_fn(samples)` must return (D, _) with D holding the stacked input branches
    (e.g. core._featurize or v40_core._features); `out_rows` are the to_rows() dicts.
    """
    import csv as _csv
    D = featurize_fn(samples)[0]
    wl_names = list(WORKLOAD_PROFILE_FEATURES) + list(STALL_FEATURES) + [f"src_brk_{b}" for b in BRK]
    spec = [("I-kcfg::", "kernel_config", list(KERNEL_CONFIG_FEATURES)),
            ("I-wl::", "workload", wl_names),
            ("I-srcspec::", "source_specs", list(GPU_SPEC_FEATURES)),
            ("I-tgtspec(ratio)::", "target_specs", list(GPU_SPEC_FEATURES)),
            ("I-derived::", "derived", list(derived_names))]
    groups = []                                   # (prefix, key, effective_names)
    for pfx, key, names in spec:
        w = D[key].shape[1]
        if w != len(names):                       # width mismatch -> generic fallback (never silently drop)
            names = [f"{key}_{i}" for i in range(w)]
        groups.append((pfx, key, names))
    out_keys = ([("O-Execution Time [ns]", "Execution Time [ns]"),
                 ("O-Memory Throughput [%]", "Memory Throughput [%]"),
                 ("O-Achieved Occupancy", "Achieved Occupancy")]
                + [(f"O-brk_{b}", f"brk_{b}") for b in BRK])
    aux_keys = [(f"aux-roofline::{k}", k) for k in ("t_mem_ns", "t_comp_ns", "t_roof_ns", "efficiency_eta")]
    header = (["meta-model", "meta-src_gpu", "meta-tgt_gpu", "meta-kernel", "meta-pair_type"]
              + [pfx + n for pfx, _, names in groups for n in names]
              + [h for h, _ in out_keys] + [h for h, _ in aux_keys])

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return f"{float(v):.8g}" if np.isfinite(v) else ""
        return v
    with open(path, "w", newline="") as f:
        if not no_comments:
            f.write(f"# {model_tag} profiling records -- io_signal schema (meta-/I-/O-); PREDICTIONS ONLY (no truth).\n")
            f.write("# Join a separate truth record on meta-(kernel,src_gpu,tgt_gpu). pair_type=self => src==tgt (input measured == target truth).\n")
        w = _csv.writer(f)
        w.writerow(header)
        for i, s in enumerate(samples):
            r = out_rows[i]
            pt = "self" if str(s["src_gpu"]).lower() == str(s["tgt_gpu"]).lower() else "cross"
            line = [model_tag, s["src_gpu"], s["tgt_gpu"], s.get("kernel_name", ""), pt]
            for _, key, names in groups:
                arr = D[key][i]
                line += [fmt(arr[j]) for j in range(len(names))]
            line += [fmt(r[k]) for _, k in out_keys]
            line += [fmt(r[k]) for _, k in aux_keys]
            w.writerow(line)
    return len(samples)
