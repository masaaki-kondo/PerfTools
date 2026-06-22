"""Self-contained inference core for the v1.5 cross-GPU estimator.

Contains the model class + the EXACT feature construction used at training time,
copied verbatim from the research code so predictions match.  Depends only on
numpy / pandas / torch — nothing else in MLP_NN/.

v1.5 = MLP + roofline quantities as extra INPUT features (always on here).
"""
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

    def forward(self, *inputs):
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

# Training-stat std floor. Features that were CONSTANT during training have their
# std floored at ~1e-8 in the saved stats; dividing a non-zero input deviation by
# that floor blows the z-score to ~1e+10 and saturates the NN to garbage. Inference
# inputs may legitimately differ from training-time-constant features (e.g.
# Dynamic Shared Memory present in this kernel but absent in training set), so we
# detect "floored" features (std <= STD_EPS) and zero their z-score, matching what
# the network actually saw during training. The 18 floored stds sit at 1e-8 with a
# wide gap to the next real feature (>1e-4); STD_EPS=1e-6 catches the dead set only.
STD_EPS = 1e-6
def _safe_z(x, mean, std):
    safe = np.where(std > STD_EPS, std, 1.0)
    z = (x - mean) / safe
    z = np.where(std > STD_EPS, z, 0.0)
    return z


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


def predict(model, stats, samples):
    """Return dict(et_p, pr[:, reg], pb[:, brk]).  NaN inputs are imputed with the
    model's TRAINING means (in `stats`) so a single row never blows up."""
    D, src_ns = _featurize(samples)
    N = {}
    for k in KEYS:
        mean, std = stats[k]
        x = np.where(np.isnan(D[k]), mean, D[k])            # training-mean impute
        N[k] = torch.tensor(_safe_z(x, mean, std), dtype=torch.float32)
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
