"""Self-contained inference core for v40 -- the RELEASED no-ET cross-GPU estimator.

v40 (formerly v6b) predicts a kernel's target-GPU ExecTime, Memory Throughput %,
Achieved Occupancy and stall breakdown WITHOUT using the source Execution Time as an
input (Kondo constraint). It reconstructs absolute performance from the other counters
via a residual-against-roofline target, source-roofline + log-rate features, and
calibrated-scaling distillation baked into training (so the NN itself extrapolates).

Artifact (MLP_NN/v40/v40_artifact/): model_seed{0..N}.pt + meta.pkl.
predict() runs the seed ensemble and recovers physical outputs. No ExecTime input,
no runtime formula -- a pure NN at inference.
"""
import os, sys, pickle
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "v1.5"))
import v15_core as vc                      # reuse MultiBranchMLP + feature lists + extractors
# GPU datasheet constants (inlined for a self-contained deliverable)
GPU_HW_FALLBACK = {"a100": {"dram_bw": 2.039e12}, "h100": {"dram_bw": 3.35e12},
                   "gb200": {"dram_bw": 8.0e12}, "gb10": {"dram_bw": 0.273e12}}
GPU_PEAK_SPECS = {"a100": {"peak_fp32": 19.5e12, "peak_fp64": 9.7e12},
                  "h100": {"peak_fp32": 67e12, "peak_fp64": 34e12},
                  "gb200": {"peak_fp32": 80e12, "peak_fp64": 40e12},
                  "gb10": {"peak_fp32": 31e12, "peak_fp64": 0.46e12}}

IB = ["kernel_config", "workload", "source_specs", "target_specs", "derived"]
KEEP = [0, 1, 2, 3, 4]; MEMBPS_COL = 6
ET, MEMPCT, OCC = 0, 1, 2
ART = os.path.join(os.path.dirname(__file__), "v40_artifact")


def load(artdir=ART):
    meta = pickle.load(open(os.path.join(artdir, "meta.pkl"), "rb"))
    models = []
    for sd in range(meta["seeds"]):
        m = vc.MultiBranchMLP(branch_dims=meta["branch_dims"], shared_branch_indices=meta["shared"],
                              branch_hidden=64, shared_hidden=128, n_shared_layers=2,
                              regression_outputs=meta["reg_out"], breakdown_outputs=meta["brk_out"], dropout=0.1)
        m.load_state_dict(torch.load(os.path.join(artdir, f"model_seed{sd}.pt"), map_location="cpu", weights_only=False))
        m.eval(); models.append(m)
    return models, meta


def _features(samples):
    """No-ET input features + the per-sample roofline-floor (ns), matching train_v40."""
    D = {k: np.stack([np.asarray(s[k], float) for s in samples], 0) for k in IB}
    b = np.array([s.get("roofline_bytes", np.nan) for s in samples])
    f32 = np.array([s.get("roofline_fp32_flops", np.nan) for s in samples])
    f64 = np.array([s.get("roofline_fp64_flops", np.nan) for s in samples])
    bw = np.array([s.get("target_dram_bw", np.nan) for s in samples])
    p32 = np.array([s.get("target_peak_fp32", np.nan) for s in samples])
    p64 = np.array([s.get("target_peak_fp64", np.nan) for s in samples])
    L = lambda x: np.log1p(np.clip(x, 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_mem = b / bw; t_comp = np.fmax(f32 / p32, f64 / p64); ai = (f32 + f64) / b
        bott = L(t_mem) - L(t_comp); troof_ns = np.fmax(t_mem, t_comp) * 1e9
    troof_ns = np.where(np.isfinite(troof_ns) & (troof_ns > 0), troof_ns, 1.0)
    D["derived"] = D["derived"][:, KEEP]
    feats = np.column_stack([L(b), L(f32), L(f64), L(t_mem), L(t_comp), L(ai), bott])
    sbw = np.array([GPU_HW_FALLBACK[s["src_gpu"].lower()]["dram_bw"] for s in samples])
    sp32 = np.array([GPU_PEAK_SPECS[s["src_gpu"].lower()]["peak_fp32"] for s in samples])
    sp64 = np.array([GPU_PEAK_SPECS[s["src_gpu"].lower()]["peak_fp64"] for s in samples])
    with np.errstate(divide="ignore", invalid="ignore"):
        tms = b / sbw; tcs = np.fmax(f32 / sp32, f64 / sp64); roof_s = np.fmax(tms, tcs)
    mem_bps = D["workload"][:, MEMBPS_COL]
    feats = np.column_stack([feats, L(tms), L(tcs), L(roof_s), L(mem_bps)])
    feats = np.where(np.isfinite(feats), feats, np.nan)
    D["derived"] = np.concatenate([D["derived"], feats], axis=1)
    return D, troof_ns


def predict(models, meta, samples):
    """Return dict(et_p[ns], mem[%], occ[%], brk[4]). No ExecTime input; pure NN."""
    D, troof = _features(samples)
    cm = meta["cm"]; stats = meta["stats"]
    for k in IB:
        D[k] = np.where(np.isnan(D[k]), cm[k], D[k])
    N = {k: torch.tensor((D[k] - stats[k][0]) / stats[k][1], dtype=torch.float32) for k in IB}
    rmn, rsd = stats["target_regression"]
    rlo, rhi = meta["clip"]
    lps, mems, occs, brks = [], [], [], []
    with torch.no_grad():
        for m in models:
            po = m(N["kernel_config"], N["workload"], N["source_specs"], N["target_specs"], N["derived"])
            pr = po["regression"].numpy() * rsd + rmn
            lps.append(pr[:, ET]); mems.append(pr[:, MEMPCT]); occs.append(pr[:, OCC]); brks.append(po["breakdown"].numpy())
    et = np.exp(np.clip(np.mean(lps, 0), rlo, rhi)) * troof          # recover absolute ExecTime
    mem = np.clip(np.mean(mems, 0), 0.0, 100.0)
    occ = np.clip(np.mean(occs, 0), 0.0, 100.0)
    brk = np.mean(brks, 0)
    return dict(et_p=et, mem=mem, occ=occ, brk=brk)
