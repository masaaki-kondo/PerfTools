"""Variance-reduction eval for the problematic breakdown cells.

Reuses the exact per-source pipeline (GB10-excluded copy) and adds:
  - single   : one model on a fixed group-by-kernel split (baseline)
  - ensemble5: average the test predictions of 5 models (seeds 42-46) on the
               SAME split -> cancels training-init noise (#2)
  - cv5      : 5-fold group-by-kernel CV, pooled out-of-fold predictions (#3)

No locked files modified. Pooled over targets (tgt = ALL). WITH mem_lat+sys_bw.
"""
import argparse
import os
import sys
import numpy as np
import torch

ROOFLINE = os.environ.get("ROOFLINE") == "1"
# v1.5: keep v1 target/output, but add v2's roofline quantities as INPUT features.
ROOFLINE_INPUTS = os.environ.get("ROOFLINE_INPUTS") == "1"
# v3: plain MLP, but normalize ExecTime target by t_roof (not source time):
#   predict r = ET / t_roof ; recover ET = r * t_roof. Same architecture as v1.
V3 = os.environ.get("V3") == "1"

sys.path.insert(0, ".")
import MLP_NN.test_per_source_model_nogb10 as M
from MLP_NN.test_per_source_model_nogb10 import (
    ALL_ZIPS, group_split, make_derived_raw, recover_et_relative,
    extract_gpu_data, build_training_pairs, stack_samples, OUTPUT_REGRESSION,
    ET, BRK, SRC_LOG_IDX, EPOCHS, LR, BATCH,
)
from MLP_NN import MultiBranchMLP

KEYS = ["kernel_config", "workload", "source_specs", "target_specs",
        "derived", "target_regression"]


def _roofline_ns(samples):
    """Full roofline time (ns) = max(bytes/BW, fp32/peak_fp32, fp64/peak_fp64),
    with REAL flops. Uses each sample's (possibly overridden) target specs."""
    b = np.array([s.get("roofline_bytes", np.nan) for s in samples])
    f32 = np.array([s.get("roofline_fp32_flops", np.nan) for s in samples])
    f64 = np.array([s.get("roofline_fp64_flops", np.nan) for s in samples])
    bw = np.array([s.get("target_dram_bw", np.nan) for s in samples])
    p32 = np.array([s.get("target_peak_fp32", np.nan) for s in samples])
    p64 = np.array([s.get("target_peak_fp64", np.nan) for s in samples])
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.fmax(np.fmax(b / bw, f32 / p32), f64 / p64)
    return t * 1e9     # ns


def _stack_target(samples):
    D = stack_samples(samples)
    src_ns = np.expm1(D["derived"][:, SRC_LOG_IDX]).clip(min=1.0)
    true_et = D["target_regression"][:, ET].copy()
    rg = D["target_regression"] = D["target_regression"].copy()
    if ROOFLINE:
        t_roof = _roofline_ns(samples)
        with np.errstate(divide="ignore", invalid="ignore"):
            rg[:, ET] = np.where(true_et > 0, t_roof / true_et, np.nan)  # eta in (0,1]
        scale = t_roof
    elif V3:                                  # r = ET / t_roof in LOG space (heavy-tailed)
        t_roof = _roofline_ns(samples)
        with np.errstate(divide="ignore", invalid="ignore"):
            rr = true_et / t_roof
            rg[:, ET] = np.where((t_roof > 0) & (rr > 0), np.log(rr), np.nan)
        scale = t_roof
    else:
        rg[:, ET] = true_et / src_ns
        scale = src_ns
    D["derived"] = make_derived_raw(D["derived"])
    if ROOFLINE:                                  # Fix 2: t_mem as an input
        extra = np.log1p(np.where(np.isfinite(t_roof) & (t_roof > 0), t_roof, 0.0))
        D["derived"] = np.concatenate([D["derived"], extra.reshape(-1, 1)], axis=1)
    if ROOFLINE_INPUTS:        # v1.5: roofline quantities as plain input features
        b = np.array([s.get("roofline_bytes", np.nan) for s in samples])
        f32 = np.array([s.get("roofline_fp32_flops", np.nan) for s in samples])
        f64 = np.array([s.get("roofline_fp64_flops", np.nan) for s in samples])
        bw = np.array([s.get("target_dram_bw", np.nan) for s in samples])
        p32 = np.array([s.get("target_peak_fp32", np.nan) for s in samples])
        p64 = np.array([s.get("target_peak_fp64", np.nan) for s in samples])
        with np.errstate(divide="ignore", invalid="ignore"):
            t_mem = b / bw
            t_comp = np.fmax(f32 / p32, f64 / p64)
            ai = (f32 + f64) / b
            bott = (np.log1p(np.clip(t_mem, 0, None))
                    - np.log1p(np.clip(t_comp, 0, None)))
        L = lambda x: np.log1p(np.clip(x, 0, None))
        feats = np.column_stack([L(b), L(f32), L(f64), L(t_mem), L(t_comp), L(ai), bott])
        feats = np.where(np.isfinite(feats), feats, np.nan)
        D["derived"] = np.concatenate([D["derived"], feats], axis=1)
    return D, scale, true_et


def _fit_stats(D):
    stats = {}
    for k in KEYS:
        a = D[k]
        m = np.nanmean(a, axis=0); s = np.nanstd(a, axis=0) + 1e-8
        m = np.where(np.isnan(m), 0.0, m); s = np.where(np.isnan(s), 1.0, s)
        stats[k] = (m, s)
    return stats


def _T(d):
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in d.items()}


def _norm(D, stats):
    o = {k: (D[k] - stats[k][0]) / stats[k][1] for k in KEYS}
    o["target_breakdown"] = D["target_breakdown"]
    return o


def _fwd(model, b):
    return model(b["kernel_config"], b["workload"], b["source_specs"],
                 b["target_specs"], b["derived"])


def train_model(train_samples, brkw, init_seed):
    torch.manual_seed(init_seed); np.random.seed(init_seed); torch.set_num_threads(4)
    D, _, _ = _stack_target(train_samples)
    stats = _fit_stats(D)
    N = _T(_norm(D, stats))
    bd = [D["kernel_config"].shape[1], D["workload"].shape[1],
          D["source_specs"].shape[1], D["target_specs"].shape[1],
          D["derived"].shape[1]]
    model = MultiBranchMLP(
        branch_dims=bd, shared_branch_indices=[[2, 3]] if bd[2] == bd[3] else None,
        branch_hidden=64, shared_hidden=128, n_shared_layers=2,
        regression_outputs=D["target_regression"].shape[1],
        breakdown_outputs=D["target_breakdown"].shape[1], dropout=0.1)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def mse(p, t):
        m = ~torch.isnan(t); d = torch.where(m, p - t, torch.zeros_like(p))
        return (d * d).sum() / m.sum().clamp(min=1)

    def kl(p, t):
        v = ~torch.isnan(t).any(dim=-1)
        if not v.any():
            return torch.tensor(0.0, requires_grad=True)
        return -(t[v] * torch.log(p[v].clamp(min=1e-10))).sum(dim=-1).mean()

    n = len(train_samples)
    for _ in range(EPOCHS):
        model.train()
        idx = torch.randperm(n)
        for i in range(0, n, BATCH):
            bi = idx[i:i + BATCH]
            b = {k: v[bi] for k, v in N.items()}
            opt.zero_grad()
            p = _fwd(model, b)
            (mse(p["regression"], b["target_regression"])
             + brkw * kl(p["breakdown"], b["target_breakdown"])).backward()
            opt.step()
    return model, stats


def predict(model, stats, samples):
    D, src_ns, true_et = _stack_target(samples)
    N = _T(_norm(D, stats))
    model.eval()
    with torch.no_grad():
        po = _fwd(model, N)
    rmn, rsd = stats["target_regression"]
    pr = po["regression"].numpy() * rsd + rmn
    if ROOFLINE:                          # exec_time = t_roofline / eta
        eta = np.clip(pr[:, ET], 1e-3, 5.0)
        et_p = src_ns / eta               # src_ns slot holds t_roofline (ns) here
    elif V3:                              # exec_time = exp(log_r) * t_roofline
        r = np.exp(np.clip(pr[:, ET], np.log(0.05), np.log(1e5)))
        et_p = r * src_ns                 # src_ns slot holds t_roofline (ns) here
    else:
        et_p = recover_et_relative(pr[:, ET], src_ns)
    pb = po["breakdown"].numpy()
    tg = D["target_regression"]              # ET col is ratio; others real
    tb = D["target_breakdown"]
    return dict(et_p=et_p, pr=pr, pb=pb, et_t=true_et, tg=tg, tb=tb)


def _r2(p, t):
    p = np.asarray(p, float); t = np.asarray(t, float)
    m = ~(np.isnan(p) | np.isnan(t))
    if not m.any():
        return float("nan")
    p, t = p[m], t[m]
    ss = np.sum((t - p) ** 2); st = np.sum((t - t.mean()) ** 2)
    return float(1 - ss / st) if st > 0 else 0.0


def _mape(p, t):
    mk = t > 0
    return float(np.mean(np.abs((t[mk] - p[mk]) / t[mk])) * 100) if mk.any() else float("nan")


def metrics(P):
    """Return {output: (R2, MAE, MAPE)} pooled over all samples.
    MAE in real units (pp for Mem%/Occ/breakdowns; ns for Execution Time)."""
    out = {}
    p, t = P["et_p"], P["et_t"]
    m = ~(np.isnan(p) | np.isnan(t)); pv, tv = p[m], t[m]
    out["Execution Time"] = (_r2(pv, tv),
                             float(np.mean(np.abs(pv - tv))) if len(pv) else float("nan"),
                             _mape(pv, tv))
    for name in ("Memory Throughput [%]", "Achieved Occupancy"):
        i = OUTPUT_REGRESSION.index(name)
        p, t = P["pr"][:, i], P["tg"][:, i]
        m = ~(np.isnan(p) | np.isnan(t)); pv, tv = p[m], t[m]
        out[name] = (_r2(pv, tv),
                     float(np.mean(np.abs(pv - tv))) if len(pv) else float("nan"),
                     _mape(pv, tv))
    for i, lab in enumerate(BRK):
        p, t = P["pb"][:, i], P["tb"][:, i]
        m = ~(np.isnan(p) | np.isnan(t)); pv, tv = p[m], t[m]
        out[f"breakdown_{lab}"] = (_r2(pv, tv),
                                   float(np.mean(np.abs(pv - tv)) * 100) if len(pv) else float("nan"),
                                   _mape(pv, tv))
    return out


def cat(ps):
    return dict(
        et_p=np.concatenate([p["et_p"] for p in ps]),
        et_t=np.concatenate([p["et_t"] for p in ps]),
        pr=np.concatenate([p["pr"] for p in ps]),
        tg=np.concatenate([p["tg"] for p in ps]),
        pb=np.concatenate([p["pb"] for p in ps]),
        tb=np.concatenate([p["tb"] for p in ps]),
    )


def avg_preds(preds):
    """Average predictions of several models (same eval order)."""
    base = preds[0]
    return dict(
        et_p=np.nanmean([p["et_p"] for p in preds], axis=0),
        pr=np.nanmean([p["pr"] for p in preds], axis=0),
        pb=np.nanmean([p["pb"] for p in preds], axis=0),
        et_t=base["et_t"], tg=base["tg"], tb=base["tb"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--brkw", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    full = {g: extract_gpu_data(zp, pt) for g, (zp, pt) in ALL_ZIPS.items()}
    samples = build_training_pairs(full)
    if args.src != "all":
        samples = [s for s in samples if s["src_gpu"].lower() == args.src]

    rows = []
    SEEDS = [42, 43, 44, 45, 46]

    # ---- fixed split for single + ensemble ----
    tr, va, te = group_split(list(samples), 42)
    models = [train_model(tr, args.brkw, s) for s in SEEDS]
    preds_te = [predict(m, st, te) for m, st in models]
    single = metrics(preds_te[0])                  # seed 42 only
    ens = metrics(avg_preds(preds_te))             # 5-model ensemble

    # ---- 5-fold group-by-kernel CV (pooled OOF) ----
    groups = sorted({(s["benchmark"], s["kernel_name"]) for s in samples})
    rng = np.random.RandomState(42); rng.shuffle(groups)
    folds = [set(groups[i::5]) for i in range(5)]
    oof = []
    for fi in range(5):
        te_g = folds[fi]
        tr_s = [s for s in samples if (s["benchmark"], s["kernel_name"]) not in te_g]
        te_s = [s for s in samples if (s["benchmark"], s["kernel_name"]) in te_g]
        if not te_s or not tr_s:
            continue
        m, st = train_model(tr_s, args.brkw, 42)
        oof.append(predict(m, st, te_s))
    cv = metrics(cat(oof))

    for method, mm, n in [("single", single, len(te)),
                          ("ensemble5", ens, len(te)),
                          ("cv5", cv, len(samples))]:
        for outk, (r2v, mae, mape) in mm.items():
            rows.append(f"{args.src},{args.brkw},{method},{outk},{r2v:.4f},{mae:.6g},{mape:.4f},{n}")

    with open(args.out, "w") as f:
        f.write("src,brkw,method,output,R2,mae,mape,n\n")
        f.write("\n".join(rows) + "\n")
    print(f"[{args.src}/brkw={args.brkw}] wrote {args.out}: single/ens/cv done")


if __name__ == "__main__":
    main()
