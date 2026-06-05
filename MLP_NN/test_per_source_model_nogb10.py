"""
Single-source-GPU model — sensei's directive (2026-05-29 Slack).

Train ONE MLP per source GPU; that one model handles all valid targets
via the existing target-spec branch (target/source ratio of GPU specs).
Per-target metrics are reported by splitting the test set by tgt_gpu.

Fixed config (the headline from BRK_W sweep): relative-ratio target,
raw src-exec input, group-by-kernel 70/15/15 split, MultiBranchMLP with
the shared source/target spec branches (now 15 spec features incl.
DRAM latency + CPU-GPU BW).

CLI:
  --src   a100 | h100 | gb200 | gb10 | all      (all = combined baseline)
  --brkw  float   breakdown loss weight
  --out   path    CSV destination
  --seed  int     (default 42)

Output CSV columns:
  src_filter, tgt_gpu, brkw, split, output,
  R2, MAE, MAE_pp, MAPE, sMAPE, truth_mean, pred_min, pred_max, n
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, ".")
from MLP_NN import MultiBranchMLP
import MLP_NN.data_pipeline_v2 as _dp
from MLP_NN.data_pipeline_v2 import (
    extract_gpu_data, build_training_pairs, stack_samples, OUTPUT_REGRESSION,
)

# Feature ablation toggle: DROP_NEW_FEATS=1 drops the last two GPU spec
# features (DRAM Latency [ns], CPU-GPU BW [byte/s]) at runtime so we can
# compare with- vs without-the-new-inputs. build_training_pairs reads
# _dp.GPU_SPEC_FEATURES at call time, so truncating it here is sufficient.
if os.environ.get("DROP_NEW_FEATS") == "1":
    _dp.GPU_SPEC_FEATURES = _dp.GPU_SPEC_FEATURES[:-2]
    print(f"[ablation] DROP_NEW_FEATS=1 -> GPU_SPEC_FEATURES now "
          f"{len(_dp.GPU_SPEC_FEATURES)} (dropped last 2)")

# Ablation toggle: DROP_CLUSTER_BARRIER=1 drops the two NCU occupancy specs
# that A100 lacks (Hopper-only / unreported) -> tests if they matter.
if os.environ.get("DROP_CLUSTER_BARRIER") == "1":
    _drop = {"Max Cluster Size [block]", "Block Limit Barriers [block]"}
    _dp.GPU_SPEC_FEATURES = [f for f in _dp.GPU_SPEC_FEATURES if f not in _drop]
    print(f"[ablation] DROP_CLUSTER_BARRIER=1 -> GPU_SPEC_FEATURES now "
          f"{len(_dp.GPU_SPEC_FEATURES)} (dropped cluster+barrier)")

SEED_DEFAULT = 42
EPOCHS = 100
LR = 1e-3
BATCH = 32
RATIO_CAP = 50.0
BRK = ["memory", "pipeline_contention", "sync", "scheduling_overhead"]
ET = OUTPUT_REGRESSION.index("Execution Time")
MEMPCT = OUTPUT_REGRESSION.index("Memory Throughput [%]")
OCC = OUTPUT_REGRESSION.index("Achieved Occupancy")
SRC_LOG_IDX = 7
KEEP = [0, 1, 2, 3, 4]

ALL_ZIPS = {
    "A100":  ("data/20260522_wide.zip", "a100"),
    "H100":  ("data/20260522_wide.zip", "h100"),
    "GB200": ("data/20260522_wide.zip", "gb200"),
}


def group_split(subset, seed):
    groups = sorted({(s["benchmark"], s["kernel_name"]) for s in subset})
    rng = np.random.RandomState(seed)
    rng.shuffle(groups)
    ng = len(groups)
    n_tr = int(0.70 * ng)
    n_va = int(0.15 * ng)
    g_tr = set(groups[:n_tr])
    g_va = set(groups[n_tr:n_tr + n_va])
    g_te = set(groups[n_tr + n_va:])
    tr = [s for s in subset if (s["benchmark"], s["kernel_name"]) in g_tr]
    va = [s for s in subset if (s["benchmark"], s["kernel_name"]) in g_va]
    te = [s for s in subset if (s["benchmark"], s["kernel_name"]) in g_te]
    return tr, va, te


def make_derived_raw(derived):
    """Headline 'raw' src-exec input: base derived features + raw src ns."""
    base = derived[:, KEEP]
    src_raw = np.expm1(derived[:, SRC_LOG_IDX:SRC_LOG_IDX + 1])
    return np.concatenate([base, src_raw], axis=1)


def r2(p, t_):
    p = np.asarray(p, dtype=np.float64)
    t_ = np.asarray(t_, dtype=np.float64)
    mask = ~(np.isnan(p) | np.isnan(t_))
    if not mask.any():
        return float("nan")
    p, t_ = p[mask], t_[mask]
    sr = np.sum((t_ - p) ** 2)
    st = np.sum((t_ - t_.mean()) ** 2)
    return float(1 - sr / st) if st > 0 else 0.0


def recover_et_relative(pr_col, src_ns):
    return np.clip(pr_col, 0.0, RATIO_CAP) * src_ns


def train_one(samples, brkw, seed, src_filter_label):
    """Train one model on `samples`; return per-(split, tgt, output) metrics.

    Returns: list of CSV-ready rows.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)

    tr, va, te = group_split(list(samples), seed)
    split_pairs = {"train": tr, "val": va, "test": te}
    names = [n for n in ("train", "val", "test") if split_pairs[n]]
    D = {n: stack_samples(split_pairs[n]) for n in names}

    # Relative-ratio target: predict target_exec / source_exec.
    src_ns = {n: np.expm1(D[n]["derived"][:, SRC_LOG_IDX]).clip(min=1.0)
              for n in names}
    true_et = {n: D[n]["target_regression"][:, ET].copy() for n in names}
    for n in names:
        rg = D[n]["target_regression"] = D[n]["target_regression"].copy()
        rg[:, ET] = true_et[n] / src_ns[n]
        D[n]["derived"] = make_derived_raw(D[n]["derived"])

    keys = ["kernel_config", "workload", "source_specs", "target_specs",
            "derived", "target_regression"]
    stats = {}

    def norm(d, fit):
        o = {}
        for k in keys:
            arr = d[k]
            if fit:
                m = np.nanmean(arr, axis=0)
                s = np.nanstd(arr, axis=0) + 1e-8
                m = np.where(np.isnan(m), 0.0, m)
                s = np.where(np.isnan(s), 1.0, s)
                stats[k] = (m, s)
            else:
                m, s = stats[k]
            o[k] = (arr - m) / s
        o["target_breakdown"] = d["target_breakdown"]
        return o

    N = {n: norm(D[n], n == "train") for n in names}

    def T(d):
        return {k: torch.tensor(v, dtype=torch.float32) for k, v in d.items()}
    N = {n: T(N[n]) for n in names}

    bd = [D["train"]["kernel_config"].shape[1], D["train"]["workload"].shape[1],
          D["train"]["source_specs"].shape[1], D["train"]["target_specs"].shape[1],
          D["train"]["derived"].shape[1]]
    model = MultiBranchMLP(
        branch_dims=bd, shared_branch_indices=[[2, 3]] if bd[2] == bd[3] else None,
        branch_hidden=64, shared_hidden=128, n_shared_layers=2,
        regression_outputs=D["train"]["target_regression"].shape[1],
        breakdown_outputs=D["train"]["target_breakdown"].shape[1], dropout=0.1)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def fwd(b):
        return model(b["kernel_config"], b["workload"], b["source_specs"],
                     b["target_specs"], b["derived"])

    def _masked_mse(pred, tgt):
        mask = ~torch.isnan(tgt)
        diff = torch.where(mask, pred - tgt, torch.zeros_like(pred))
        n = mask.sum().clamp(min=1)
        return (diff * diff).sum() / n

    def _masked_kl_breakdown(pred, tgt):
        valid = ~torch.isnan(tgt).any(dim=-1)
        if not valid.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        p = pred[valid]
        t = tgt[valid]
        log_p = torch.log(p.clamp(min=1e-10))
        return -(t * log_p).sum(dim=-1).mean()

    def lf(p, b):
        return _masked_mse(p["regression"], b["target_regression"]) + \
            brkw * _masked_kl_breakdown(p["breakdown"], b["target_breakdown"])

    n_tr = len(tr)
    for ep in range(EPOCHS):
        model.train()
        idx = torch.randperm(n_tr)
        for i in range(0, n_tr, BATCH):
            bi = idx[i:i + BATCH]
            b = {k: v[bi] for k, v in N["train"].items()}
            opt.zero_grad()
            lf(fwd(b), b).backward()
            opt.step()

    # Run inference once per split, then slice by tgt_gpu for metrics.
    rmn, rsd = stats["target_regression"]
    out_rows = []
    for split in names:
        model.eval()
        with torch.no_grad():
            po = fwd(N[split])
        pr = po["regression"].numpy() * rsd + rmn
        tg = N[split]["target_regression"].numpy() * rsd + rmn
        et_p = recover_et_relative(pr[:, ET], src_ns[split])
        et_t = true_et[split]
        pb = po["breakdown"].numpy()
        tb = N[split]["target_breakdown"].numpy()
        tgt_gpus = np.array([s["tgt_gpu"] for s in split_pairs[split]])

        # Per-target slicing
        unique_tgts = sorted(set(tgt_gpus.tolist()))
        for tgt in unique_tgts + ["ALL"]:
            mask = np.ones(len(tgt_gpus), dtype=bool) if tgt == "ALL" \
                else (tgt_gpus == tgt)
            if not mask.any():
                continue
            n_samples = int(mask.sum())
            _emit_metrics(out_rows, src_filter_label, tgt, brkw, split,
                          et_p[mask], et_t[mask],
                          pr[mask], tg[mask], pb[mask], tb[mask], n_samples)
    return out_rows


def _emit_metrics(rows, src_filter, tgt, brkw, split,
                  et_p, et_t, pr, tg, pb, tb, n):
    def reg(name, p, t_):
        p = np.asarray(p, dtype=np.float64)
        t_ = np.asarray(t_, dtype=np.float64)
        valid = ~(np.isnan(p) | np.isnan(t_))
        if not valid.any():
            return ("nan",) * 7
        pv, tv = p[valid], t_[valid]
        mae = float(np.mean(np.abs(pv - tv)))
        mk = tv > 0
        mape = (float(np.mean(np.abs((tv[mk] - pv[mk]) / tv[mk]))) * 100
                if mk.sum() else 0.0)
        return (f"{r2(pv, tv):.4f}", f"{mae:.4f}", "", f"{mape:.2f}", "",
                f"{tv.mean():.4f}", f"{pv.min():.4f}", f"{pv.max():.4f}")

    def brk(p_raw, t_raw):
        valid = ~(np.isnan(p_raw) | np.isnan(t_raw))
        if not valid.any():
            return ("nan",) * 7
        p, t_ = p_raw[valid], t_raw[valid]
        mae_pp = float(np.mean(np.abs(p - t_))) * 100
        den = (np.abs(p) + np.abs(t_)) / 2.0
        mk = den > 1e-9
        sm = (float(np.mean(np.abs(p[mk] - t_[mk]) / den[mk])) * 100
              if mk.sum() else 0.0)
        return (f"{r2(p, t_):.4f}", "", f"{mae_pp:.4f}", "", f"{sm:.2f}",
                f"{t_.mean() * 100:.4f}", f"{p.min() * 100:.4f}",
                f"{p.max() * 100:.4f}")

    cells = reg("Execution Time", et_p, et_t)
    rows.append([src_filter, tgt, brkw, split, "Execution Time", *cells, n])
    for i, col in enumerate(OUTPUT_REGRESSION):
        if i == ET:
            continue
        cells = reg(col, pr[:, i], tg[:, i])
        rows.append([src_filter, tgt, brkw, split, col, *cells, n])
    for i, lab in enumerate(BRK):
        cells = brk(pb[:, i], tb[:, i])
        rows.append([src_filter, tgt, brkw, split, f"breakdown_{lab}",
                     *cells, n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True,
                    choices=["a100", "h100", "gb200", "all"],
                    help="Source GPU filter (or 'all' = combined baseline)")
    ap.add_argument("--brkw", type=float, required=True,
                    help="Breakdown-head loss weight (BRK_W)")
    ap.add_argument("--out", required=True,
                    help="Output CSV path")
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    args = ap.parse_args()

    np.seterr(over="ignore", invalid="ignore")
    sys.stdout.reconfigure(line_buffering=True)

    t0 = time.time()
    print(f"[{args.src}/brkw={args.brkw}] Loading wide-zip ...")
    full = {g: extract_gpu_data(zp, pt) for g, (zp, pt) in ALL_ZIPS.items()}
    samples = build_training_pairs(full)

    if args.src == "all":
        filt = samples
        label = "all"
    else:
        filt = [s for s in samples if s["src_gpu"].lower() == args.src]
        label = args.src
    print(f"[{label}/brkw={args.brkw}] {len(filt)} pairs after filter "
          f"(of {len(samples)} total).")
    if not filt:
        sys.exit(f"No pairs after src filter '{args.src}' — aborting.")

    rows = train_one(filt, args.brkw, args.seed, label)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src_filter", "tgt_gpu", "brkw", "split", "output",
                    "R2", "MAE", "MAE_pp", "MAPE", "sMAPE",
                    "truth_mean", "pred_min", "pred_max", "n"])
        w.writerows(rows)
    print(f"[{label}/brkw={args.brkw}] Done in {time.time() - t0:.1f}s "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
