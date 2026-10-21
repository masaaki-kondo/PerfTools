"""Self-contained inference core for v3 -- no-ET cross-GPU estimator + confidence.

v40 (formerly v6b) predicts a kernel's target-GPU ExecTime, Memory Throughput %,
Achieved Occupancy and stall breakdown WITHOUT using the source Execution Time as an
input (Kondo constraint). It reconstructs absolute performance from the other counters
via a residual-against-roofline target, source-roofline + log-rate features, and
calibrated-scaling distillation baked into training (so the NN itself extrapolates).

Artifact (PINN/v3.25/v325_artifact/): model_seed{0..N}.pt + meta.pkl.
predict() runs the seed ensemble and recovers physical outputs. No ExecTime input,
no runtime formula -- a pure NN at inference.
"""
import os, sys, pickle
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # own v15_core copy
import v15_core as vc
from MLP_NN.data_pipeline_v2 import tcomp_anchor, troof_floor, t_lat_of, t_wave_of, t_memcap_of                      # reuse MultiBranchMLP + feature lists + extractors
# GPU datasheet constants (inlined for a self-contained deliverable)
GPU_HW_FALLBACK = {"a100": {"dram_bw": 2.039e12}, "h100": {"dram_bw": 3.35e12},
                   "gb200": {"dram_bw": 8.0e12}, "gb10": {"dram_bw": 0.273e12}}
GPU_PEAK_SPECS = {"a100": {"peak_fp32": 19.5e12, "peak_fp64": 9.7e12},
                  "h100": {"peak_fp32": 67e12, "peak_fp64": 34e12},
                  "gb200": {"peak_fp32": 80e12, "peak_fp64": 40e12},
                  "gb10": {"peak_fp32": 31e12, "peak_fp64": 0.46e12}}
# 2026-10-09 FIX: the trainer builds the SOURCE-roofline features from data_pipeline_v2's tables (spec-sheet
# overridden: H100 3657 GB/s, GB200 6580 GB/s, plus H200/GH200/L40S/MIG), but this inlined copy said H100
# 3350 / GB200 8000 and had no H200/GH200/L40S at all -> train/inference skew on 4 derived features for
# H100- and GB200-source rows, and a KeyError for the other sources. Use the pipeline's values.
import MLP_NN.data_pipeline_v2 as _dp
GPU_HW_FALLBACK = {k: dict(v) for k, v in _dp.GPU_HW_FALLBACK.items()}
GPU_PEAK_SPECS = {k: dict(v) for k, v in _dp.GPU_PEAK_SPECS.items()}

IB = ["kernel_config", "workload", "source_specs", "target_specs", "derived"]
# ET_INPUT: put the SOURCE execution time back into the derived branch (log1p, index 7 of
# compute_derived_features, the column v1.5 used). The no-ET rule (Kondo, "3-point model")
# exists because with ET_src available the net learns ET_tgt = ET_src x (ratio it memorised
# per GPU pair) and has nothing to say about a GPU it never saw (v1.5 LOGO: negative R2 on
# every fold). Gated so the effect on interpolation vs extrapolation can be MEASURED.
KEEP = [0, 1, 2, 3, 4] + ([7] if os.environ.get("ET_INPUT", "0") == "1" else []); MEMBPS_COL = 6
ET, MEMPCT, OCC = 0, 1, 2
ART = os.path.join(os.path.dirname(__file__), "v325_artifact")


def load(artdir=ART):
    meta = pickle.load(open(os.path.join(artdir, "meta.pkl"), "rb")); meta["_artdir"] = artdir
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
        inst = np.array([s.get("roofline_inst", 0.0) for s in samples]); pinst = np.array([s.get("target_peak_inst", np.nan) for s in samples])
        t_mem = b / bw; t_comp = tcomp_anchor(f32, f64, inst, p32, p64, pinst); ai = (f32 + f64) / b
        bott = L(t_mem) - L(t_comp)
        tlat = np.array([t_lat_of(s) for s in samples]); twave = np.array([t_wave_of(s) for s in samples])
        troof_ns = troof_floor(t_mem, t_comp, inst, pinst, f32, f64, tlat, twave) * 1e9
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
    if _dp.BUSY_FEAT: feats = np.column_stack([feats, np.array([_dp.busy_feats(s) for s in samples], float)])   # BUSY_FEAT (2026-10-19)
    feats = np.where(np.isfinite(feats), feats, np.nan)
    D["derived"] = np.concatenate([D["derived"], feats], axis=1)
    D["_twave_ns"] = np.nan_to_num(twave, nan=0.0) * 1e9      # WAVE_CLAMP ingredient; not a branch (IB), ignored by the net
    D["_memcap_ns"] = np.array([t_memcap_of(x) for x in samples], float) * 1e9   # MEM_CAP ingredient (inf = no cap)
    D["_compcap_ns"] = np.array([_dp.t_compcap_of(x) for x in samples], float) * 1e9   # COMP_CAP ingredient (inf = no cap)
    return D, troof_ns


def predict(models, meta, samples):
    """Return dict(et_p[ns], mem[%], occ[%], brk[4]). No ExecTime input; pure NN."""
    D, troof = _features(samples)
    # SCALE_INVARIANT must match TRAINING exactly: zero the inputs that change under
    # whole-GPU scaling, so residual(f) == residual(1) and ET scales exactly as 1/f.
    _si = os.environ.get("SCALE_INVARIANT", "off")
    if _si != "off":
        if _si == "full":
            for _j in (6, 7, 9):
                if _j < D["target_specs"].shape[1]: D["target_specs"][:, _j] = 0.0
        for _j in (8, 9):
            if _j < D["derived"].shape[1]: D["derived"][:, _j] = 0.0
    cm = meta["cm"]; stats = meta["stats"]
    for k in IB:
        D[k] = np.where(np.isnan(D[k]), cm[k], D[k])
    N = {k: torch.tensor(vc.safe_standardize(D[k], stats[k][0], stats[k][1]), dtype=torch.float32) for k in IB}
    _bc = float(os.environ.get("BUSY_FEAT_CLIP", meta.get("busy_feat_clip", 0)))   # BUSY_FEAT: clip the 3 busy inputs to +-z of training (0 = off; default from meta)
    if meta.get("busy_feat") and _bc > 0:
        N["derived"][:, -3:] = N["derived"][:, -3:].clamp(-_bc, _bc)
    rmn, rsd = stats["target_regression"]
    rlo, rhi = meta["clip"]
    lps, mems, occs, brks = [], [], [], []
    with torch.no_grad():
        for m in models:
            po = m(N["kernel_config"], N["workload"], N["source_specs"], N["target_specs"], N["derived"])
            pr = po["regression"].numpy() * rsd + rmn
            lps.append(pr[:, ET]); mems.append(pr[:, MEMPCT]); occs.append(pr[:, OCC]); brks.append(po["breakdown"].numpy())
    et = np.exp(np.clip(np.mean(lps, 0), rlo, rhi)) * troof          # recover absolute ExecTime
    if _dp.MEM_CAP: et = np.minimum(et, D["_memcap_ns"])              # memory-efficiency cap (before the floors)
    if _dp.COMP_CAP: et = np.minimum(et, D["_compcap_ns"])            # compute-saturation cap (before the floors)
    if _dp.WAVE_CLAMP: et = np.fmax(et, D["_twave_ns"])                # one-wave floor, inference only
    mem = np.clip(np.mean(mems, 0), 0.0, 100.0)
    occ = np.clip(np.mean(occs, 0), 0.0, 100.0)
    brk = np.mean(brks, 0)
    return dict(et_p=et, mem=mem, occ=occ, brk=brk)


def predict_seeds(models, meta, samples, _no_transport=False):
    """Same forward pass as predict(), but ALSO returns every seed's own prediction so the
    ensemble spread can be used as a confidence signal.

    Returns the usual mean fields plus `*_seeds` arrays of shape (n_seeds, n_samples[, 4])
    and `troof` (the roofline floor, ns) for physics-consistency checks.
    Averaging matches predict() exactly: ExecTime in log space, the rest directly."""
    D, troof = _features(samples)
    # SCALE_INVARIANT must match TRAINING exactly: zero the inputs that change under
    # whole-GPU scaling, so residual(f) == residual(1) and ET scales exactly as 1/f.
    _si = os.environ.get("SCALE_INVARIANT", "off")
    if _si != "off":
        if _si == "full":
            for _j in (6, 7, 9):
                if _j < D["target_specs"].shape[1]: D["target_specs"][:, _j] = 0.0
        for _j in (8, 9):
            if _j < D["derived"].shape[1]: D["derived"][:, _j] = 0.0
    cm = meta["cm"]; stats = meta["stats"]
    for k in IB:
        D[k] = np.where(np.isnan(D[k]), cm[k], D[k])
    N = {k: torch.tensor(vc.safe_standardize(D[k], stats[k][0], stats[k][1]), dtype=torch.float32) for k in IB}
    _bc = float(os.environ.get("BUSY_FEAT_CLIP", meta.get("busy_feat_clip", 0)))   # BUSY_FEAT: clip the 3 busy inputs to +-z of training (0 = off; default from meta)
    if meta.get("busy_feat") and _bc > 0:
        N["derived"][:, -3:] = N["derived"][:, -3:].clamp(-_bc, _bc)
    rmn, rsd = stats["target_regression"]
    rlo, rhi = meta["clip"]
    lps, mems, occs, brks = [], [], [], []
    with torch.no_grad():
        for m in models:
            po = m(N["kernel_config"], N["workload"], N["source_specs"], N["target_specs"], N["derived"])
            pr = po["regression"].numpy() * rsd + rmn
            lps.append(pr[:, ET]); mems.append(pr[:, MEMPCT]); occs.append(pr[:, OCC]); brks.append(po["breakdown"].numpy())
    lps = np.asarray(lps); mems = np.asarray(mems); occs = np.asarray(occs); brks = np.asarray(brks)
    # TRUST_Z: outside the training range, pull the residual toward the training-typical
    # value (rmn[ET], the mean residual) -- see v15_core.trust_weight. No-op unless set.
    _w = vc.trust_weight(D["target_specs"], stats["target_specs"][0], stats["target_specs"][1])
    if _w is not None:
        # anchor residual: TRUST_MODE=mean -> training-mean residual (typical efficiency);
        # TRUST_MODE=src  -> the kernel's OWN residual on its source GPU, log(ET_src / t_roof_src):
        # "outside the trained envelope, assume the same roofline efficiency as on the source".
        # Uses ET_src -- declared, and only as the far-out fallback (w < 1).
        _anchor = np.full(len(samples), float(rmn[ET]))
        if os.environ.get("TRUST_MODE", "mean") in ("class", "cap"):
            import json as _json
            _tp = os.path.join(meta.get("_artdir", ART), "trust_anchor.json")
            _ta = _json.load(open(_tp)) if os.path.exists(_tp) else {}      # missing -> mean anchor (make_trust_anchor.py writes it)
            for _i, _s in enumerate(samples):
                _b = float(_s.get("roofline_bytes", 0) or 0); _f = float(_s.get("roofline_fp32_flops", 0) or 0) + float(_s.get("roofline_fp64_flops", 0) or 0)
                _mp = _s.get("src_mem_pct", np.nan); _mp = _mp if _mp == _mp else 50.0
                _tm = _b / float(_s["target_dram_bw"]); _tc = max(float(_s["roofline_fp32_flops"]) / float(_s["target_peak_fp32"]), float(_s["roofline_fp64_flops"]) / float(_s["target_peak_fp64"]), 1e-30)
                _cls = "COMM" if (_tm > _tc and _mp >= 50) else ("COMP" if (_tc > _tm and _f > 0) else "LATENCY")
                _anchor[_i] = float(np.clip(_ta.get(_cls, float(rmn[ET])), rlo, rhi))
        if os.environ.get("TRUST_MODE", "mean") == "src":
            from MLP_NN.data_pipeline_v2 import GPU_HW_FALLBACK as _HW, GPU_PEAK_SPECS as _PK
            for _i, _s in enumerate(samples):
                _g = str(_s.get("src_gpu", "")).lower(); _et = float(_s.get("src_dur_ns", np.nan) or np.nan)
                if _g in _HW and _et == _et and _et > 0:
                    _tm = _s["roofline_bytes"] / _HW[_g]["dram_bw"]
                    _tc = max(_s["roofline_fp32_flops"] / _PK[_g]["peak_fp32"], _s["roofline_fp64_flops"] / _PK[_g]["peak_fp64"], 1e-30)
                    _tr = max(_tm, _tc) * 1e9
                    if _tr > 0 and np.isfinite(_tr):
                        _anchor[_i] = float(np.clip(np.log(max(_et, 1.0) / max(_tr, 1.0)), rlo, rhi))
        if os.environ.get("TRUST_MODE", "mean") == "cap":
            # TRUST_MODE=cap (2026-10-07): no blending. Outside the trained envelope (w < 1) the residual
            # may not exceed the class-typical residual + TRUST_CAP_MARGIN nats (default 1.0 = the
            # kernel may be at most e^1 ~ 2.7x less efficient than its class median). Motivation: with
            # L40S held out, FP64 GEMM rows saturate at the training clip (pred ~1000x the roofline,
            # true ~1.2x) because the FP32/FP64 ratio (64) is ~20 sigma outside the trained range (~2).
            # Rows inside the envelope, and rows outside it with a plausible residual, are untouched.
            _cap = _anchor + float(os.environ.get("TRUST_CAP_MARGIN", "1.0"))
            # Applied PER SEED (seeds disagree wildly out of range: one row had seed residuals
            # -10 / +70 / -30); the lower clip at rlo stays on the ensemble mean as everywhere else, so
            # a row whose seeds mostly say 'at the roofline' lands on the roofline floor.
            # Trigger only on ARCHITECTURE axes (per-SM structure, FP32/FP64, flop/byte, L2/SM, latency,
            # host link; TRUST_CAP_AXES), never on capacity (axes 6/7/9: BW per SM, L2 per SM, tensor flop/s): a
            # whole-GPU x8-x32 projection is far out of range on capacity BY DESIGN and must not be capped.
            _ax = [int(a) for a in os.environ.get("TRUST_CAP_AXES", "0,1,2,3,4,5,8,10,11,12").split(",")]
            _ax = [a for a in _ax if a < D["target_specs"].shape[1]]
            _wc = vc.trust_weight(np.asarray(D["target_specs"])[:, _ax], np.asarray(stats["target_specs"][0])[_ax], np.asarray(stats["target_specs"][1])[_ax])
            lps = np.where((_wc < 1.0)[None, :], np.minimum(lps, _cap[None, :]), lps)
        else:
            lps = _w[None, :] * lps + (1.0 - _w[None, :]) * _anchor[None, :]
    if _dp.AXIS_TRANSPORT and _dp.REF_BUILDER is not None and not _no_transport:   # single-axis fix (2026-10-16)
        # Efficiency (residual over the roofline floor) is taken from the model AT THE NEAREST REAL GPU in balance
        # space and multiplied by the target's own roofline: continuous with the model where data exists, exact
        # roofline behaviour on every single axis far from it. Whole-GPU scaling keeps a real GPU's balance -> w=1.
        import re as _re
        _m = _re.search(r"_holdout_([a-z0-9_]+)", os.path.basename(str(meta.get("_artdir", ""))))
        _wt, _near = _dp.axis_weight(samples, exclude=[_m.group(1)] if _m else [])
        # rev 9 (2026-10-16): REFERENCE CAP. Every SCALED or hypothetical target (specs differ from its reference GPU by
        # >5%) is also capped at AT_REFCAP x the far answer, at any distance: inside the blend region the network alone
        # over-reacted to less compute (H200 compute x1/2 -> 1.59, x1/4 -> 1.37 after transport: non-monotone; measured
        # real pairs show the network 1.31x too slow on compute slow-downs). Unmodified real targets are untouched.
        _rc = float(os.environ.get("AT_REFCAP", "1.25"))
        _cand = [i for i in range(len(samples)) if _near[i] is not None and (_wt[i] < 1.0 or _rc > 0)]
        _ref0 = {i: _dp.REF_BUILDER(samples[i], _near[i]) for i in _cand}
        _held = {_m.group(1)} if (_m and os.environ.get("AT_HELDOUT_REF", "0") == "1" and os.environ.get("AT_HELDOUT_CAP", "0") != "1") else set()
        def _scaled(s, r):
            if str(s.get("tgt_gpu", "")).lower() in _held and str(r.get("tgt_gpu", "")).lower() != str(s.get("tgt_gpu", "")).lower():
                return False   # held-out real GPU redirected to a training GPU: distance blend only, no refcap / clamp
            ks = ("target_dram_bw", "target_peak_fp32", "target_peak_fp64", "target_n_sms")
            try:
                return any(abs(float(s[k]) / float(r[k]) - 1.0) > 0.05 for k in ks)   # 5%: GB200 rows carry 152 SMs vs the table 148 (2.7%)
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                return False
        _idx = [i for i in _cand if _ref0[i] is not None and (_wt[i] < 1.0 or _scaled(samples[i], _ref0[i]))]
        if _idx:
            _ref = [_ref0[i] for i in _idx]
            _ok = list(range(len(_idx)))
            if _ok:
                _R = predict_seeds(models, meta, [_ref[j] for j in _ok], _no_transport=True)
                _rr = np.asarray(_R["log_et_seeds"]); _ii = np.array([_idx[j] for j in _ok])
                if os.environ.get("AT_MEM", "1") == "1":                   # rev 3: memory term with Little's-law efficiency
                    _troof_ref = np.asarray(_R["troof"], float)
                    _et_ref = np.exp(np.clip(_rr, rlo, rhi)) * _troof_ref[None, :]          # per-seed reference ET [ns]
                    _good = np.zeros(len(_ok), bool)
                    for jj, j in enumerate(_ok):
                        gf = _dp.far_factors(samples[_idx[j]], _ref[j], _troof_ref[jj])
                        if gf is None:
                            continue
                        g1, g2, g3, g4, g4r, g5 = gf; e = _et_ref[:, jj]; _good[jj] = True
                        er = np.minimum(e * g4, np.maximum(e * g4r, g5)) if os.environ.get("AT_COMP_SAT", "1") == "1" else e * g4   # rev 7: compute saturation
                        etf = np.maximum(np.maximum(e * g1, np.minimum(g2, e * g3)), er)
                        _rr[:, jj] = np.log(np.maximum(etf, 1e-3) / max(troof[_idx[j]], 1e-12))
                _wa = _wt[_ii][None, :]
                lps[:, _ii] = _wa * lps[:, _ii] + (1.0 - _wa) * _rr
                if _rc > 0 and os.environ.get("AT_MEM", "1") == "1":       # rev 9: reference cap (far answer x AT_REFCAP)
                    lps[:, _ii[_good]] = np.minimum(lps[:, _ii[_good]], _rr[:, _good] + np.log(_rc))
                if os.environ.get("AT_CLAMP", "1") == "1" and os.environ.get("AT_MEM", "1") == "1" and _good.any():
                    # rev 10 (2026-10-16): DIRECTION CLAMP for scaled / hypothetical targets. The network may only move the
                    # answer between the reference GPU's own prediction and the far (physics) answer: less of a resource is
                    # never faster than the reference nor slower than physics, more is never slower nor faster than physics.
                    # Removes the blend-region wobbles (compute x1/2 1.27 > x1/4 1.13; zero-flop kernels +21% at x1/4
                    # compute; whole-GPU x1/8 latency 9.1 > 8). Unmodified real targets never reach this block.
                    _sc = _good & np.array([_scaled(samples[_idx[j]], _ref[j]) for j in _ok])   # deliberately scaled only
                    if _sc.any():
                        _gi = _ii[_sc]
                        _lref = np.log(np.maximum(_et_ref[:, _sc], 1e-3) / np.maximum(troof[_gi], 1e-12)[None, :])
                        _lo = np.minimum(_lref, _rr[:, _sc]); _hi = np.maximum(_lref, _rr[:, _sc])
                        lps[:, _gi] = np.clip(lps[:, _gi], _lo, _hi)
    et_seeds = np.exp(np.clip(lps, rlo, rhi)) * troof              # per-seed absolute ExecTime
    _etp = np.exp(np.clip(lps.mean(0), rlo, rhi)) * troof
    if _dp.MEM_CAP:                                                 # memory-efficiency cap, inference only (floors applied after)
        _mc = np.fmax(D["_memcap_ns"], troof); et_seeds = np.minimum(et_seeds, _mc[None, :]); _etp = np.minimum(_etp, _mc)   # never below the roofline (2026-10-16)
    if _dp.COMP_CAP:                                                # compute-saturation cap (2026-10-16), never below the roofline
        _cc = np.fmax(D["_compcap_ns"], troof); et_seeds = np.minimum(et_seeds, _cc[None, :]); _etp = np.minimum(_etp, _cc)
    if _dp.WAVE_CLAMP:                                              # one-wave floor, inference only
        et_seeds = np.fmax(et_seeds, D["_twave_ns"][None, :]); _etp = np.fmax(_etp, D["_twave_ns"])
    _mem = np.clip(mems.mean(0), 0.0, 100.0); _il = np.zeros(len(samples), bool)
    if os.environ.get("ISSUE_FLOOR", "1") == "1":                    # instruction-issue limit (2026-10-20; in the core since 2026-10-21)
        # a scheduler issues at most 1 warp instruction per cycle: time >= issued / (4 schedulers x SMs x clock) = issued / target
        # peak_inst. Hard physical floor on the ensemble answer (like the roofline/wave floors); same bytes over a longer time ->
        # Mem% scaled. Needs the source run's issued count (src_issued_inst); rows without it are untouched. ISSUE_FLOOR=0 = off.
        _iss = np.array([s.get("src_issued_inst", np.nan) for s in samples], float)
        _pin = np.array([s.get("target_peak_inst", np.nan) for s in samples], float)
        with np.errstate(divide="ignore", invalid="ignore"):
            _ti = _iss / _pin * 1e9; _il = _ti > _etp
        _mem[_il] *= _etp[_il] / _ti[_il]; _etp = np.where(_il, _ti, _etp)
    mem_seeds = np.clip(mems, 0.0, 100.0); occ_seeds = np.clip(occs, 0.0, 100.0)
    return dict(trust_w=(_w if _w is not None else np.ones(len(samples))),
                et_p=_etp,
                mem=_mem, occ=np.clip(occs.mean(0), 0.0, 100.0), issue_limited=_il,
                brk=brks.mean(0), et_seeds=et_seeds, mem_seeds=mem_seeds, occ_seeds=occ_seeds,
                brk_seeds=brks, log_et_seeds=lps, troof=troof)


# --------------------------------------------------------------------------- #
# v3 confidence estimation                                                   #
# --------------------------------------------------------------------------- #
# Two independent, ground-truth-FREE signals:
#   (a) ENSEMBLE VARIANCE  - how much the 3 seeds disagree (epistemic uncertainty)
#   (b) ROOFLINE CONSISTENCY - does the answer obey physics, and how far does it sit
#       from the roofline floor the model is anchored to
#
# CALIBRATION (2026-10-21) on HELD-OUT predictions of PINN v3.25: the 7 leave-one-GPU-out folds (H200, H100, A100, GH200,
# L40S, GB200, B300), i.e. GPUs the model never saw -- the situation of a user predicting a new GPU. 4,784 kernel rows
# for Execution Time (truth: nsys / ncu time); 3,877 for Mem% / occupancy / breakdown (truth: the target GPU's own ncu
# profile of the same run and kernel; B300 has none). Thresholds q60 / q80 of each signal over those rows.
# Rank correlation signal-vs-error: ExecTime +0.18, Mem% +0.46, Occupancy +0.50, breakdown +0.77.
#
# Bucket = top 20% of the signal -> "low", next 20% -> "medium", rest -> "high".
# `exp_err` is the MEASURED mean held-out error of that bucket, i.e. what to expect on a GPU the model never saw.
# Read exp_err, not only the label: where exp_err does not rise from high to low, the label does not rank the error.
CONF_CAL = {
    #  signal                       q60        q80    exp_err(high, medium, low)   unit
    "ExecTime": dict(sig="ens_cv_et", q60=0.06808, q80=0.11924, exp=(17.83, 24.46, 33.32), unit="%"),
    "Mem%": dict(sig="ens_sd_mem", q60=3.01021, q80=8.99044, exp=(6.19, 12.5, 35.9), unit="pp"),
    "Occupancy": dict(sig="ens_sd_occ", q60=2.59925, q80=17.6027, exp=(3.18, 6.4, 41.96), unit="pp"),
    "breakdown": dict(sig="ens_sd_brk", q60=1.16216, q80=6.49497, exp=(1.52, 4.81, 12.3), unit="pp"),
}


def confidence(R, samples):
    """Confidence signals + per-output labels for a predict_seeds() result. No truth needed.

    Returns per-sample arrays: the raw signals (ens_cv_et / ens_sd_mem / ens_sd_occ /
    ens_sd_brk / eta / eta_violation / log_dist_roof / mem_consistency) plus, per output,
    conf_<output> in {high, medium, low} and exp_err_<output> (see CONF_CAL).

    A physics violation (eta > 1: predicted faster than the roofline allows) forces 'low'
    on Execution Time no matter how much the seeds agree -- the failure mode ensemble
    variance alone cannot see.
    """
    et_p = np.asarray(R["et_p"], float)
    troof = np.asarray(R["troof"], float)
    et_s = np.asarray(R["et_seeds"], float)
    bw = np.array([s["target_dram_bw"] for s in samples], float)
    byts = np.array([s["roofline_bytes"] for s in samples], float)

    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = et_s.mean(0)
        out["ens_cv_et"] = et_s.std(0) / np.where(np.abs(mu) > 0, np.abs(mu), np.nan)
        out["ens_sd_mem"] = np.asarray(R["mem_seeds"], float).std(0)
        out["ens_sd_occ"] = np.asarray(R["occ_seeds"], float).std(0)
        out["ens_sd_brk"] = (np.asarray(R["brk_seeds"], float).std(0) * 100.0).mean(1)
        eta = troof / np.where(et_p > 0, et_p, np.nan)
        out["eta"] = eta
        out["eta_violation"] = (eta > 1.0).astype(int)
        out["log_dist_roof"] = np.abs(np.log(np.where(et_p > 0, et_p, np.nan)
                                             / np.where(troof > 0, troof, np.nan)))
        implied = byts / (et_p * 1e-9) / bw * 100.0
        out["mem_consistency"] = np.abs(np.asarray(R["mem"], float) - np.clip(implied, 0, 100))

    for name, c in CONF_CAL.items():
        v = np.asarray(out[c["sig"]], float)
        lab = np.where(~np.isfinite(v), "unknown",
                       np.where(v >= c["q80"], "low", np.where(v >= c["q60"], "medium", "high")))
        if name == "ExecTime":                      # physics veto
            lab = np.where(out["eta_violation"] == 1, "low", lab)
        exp = np.select([lab == "high", lab == "medium", lab == "low"], list(c["exp"]), default=np.nan)
        out[f"conf_{name}"] = lab
        out[f"exp_err_{name}"] = exp
    return out


BRK_NAMES = ["memory", "pipeline_contention", "sync", "scheduling_overhead"]
_ERR_SPEC = [("ExecTime", "Execution Time [ns]", "S-Execution Time [ns]", "pct", 1.0),
             ("Mem%", "Memory Throughput [%]", "S-Memory Throughput [%]", "abs", 1.0),
             ("Occupancy", "Achieved Occupancy", "S-Achieved Occupancy", "abs", 1.0)] + \
            [(f"brk_{b}", f"brk_{b}", f"S-brk_{b}", "abs", 100.0) for b in BRK_NAMES]


def attach_errors(out_rows, truth_path):
    """Score predictions against a row-matched ground-truth CSV (meta-/S- schema).

    Adds err_<output> to each row (ExecTime as APE %, the rest as absolute error in pp),
    prints a per-output breakdown split by self/cross, and returns the added column names.
    Row i of the truth file must correspond to row i of the input -- which is how the
    *_from_Yoshida-san_*.csv pair is generated (see MLP_NN/examples/make_full_dataset.py).
    """
    import csv as _csv
    with open(truth_path) as f:
        T = [r for r in _csv.DictReader(l for l in f if not l.startswith("#"))]
    if len(T) != len(out_rows):
        raise SystemExit(f"--truth row mismatch: {len(T)} truth vs {len(out_rows)} predictions")

    def _f(x):
        try:
            v = float(x); return v if np.isfinite(v) else np.nan
        except (TypeError, ValueError):
            return np.nan

    added, acc = [], {}
    for name, pcol, tcol, kind, scale in _ERR_SPEC:
        col = f"err_{name}"; added.append(col); vals = []
        for r, t in zip(out_rows, T):
            p, gt = _f(r.get(pcol)) * scale, _f(t.get(tcol)) * scale
            if kind == "pct":
                e = abs(p - gt) / abs(gt) * 100.0 if (np.isfinite(p) and np.isfinite(gt) and abs(gt) > 1e-9) else np.nan
            else:
                e = abs(p - gt) if (np.isfinite(p) and np.isfinite(gt)) else np.nan
            r[col] = e; vals.append(e)
        acc[name] = (np.asarray(vals, float), "%" if kind == "pct" else "pp")

    pt = np.array([t.get("meta-pair_type", "") for t in T])
    sys.stderr.write("\n=== v3 error breakdown vs %s ===\n" % os.path.basename(truth_path))
    sys.stderr.write(f"{'output':<26}{'ALL':>18}{'self':>18}{'cross':>18}\n")
    for name, (v, unit) in acc.items():
        cells = []
        for m in (np.ones(len(v), bool), pt == "self", pt == "cross"):
            w = v[m & np.isfinite(v)]
            cells.append(f"{w.mean():.2f}{unit} (n{len(w)})" if len(w) else "n/a")
        sys.stderr.write(f"{name:<26}" + "".join(f"{c:>18}" for c in cells) + "\n")
    sys.stderr.write("(ExecTime = mean absolute percentage error; others = mean absolute error in pp)\n")
    return added


CONF_LEGEND = [
    "",
    "=== confidence measures (no ground truth needed) ===",
    "Two independent signals per prediction:",
    "",
    "  A) ENSEMBLE VARIANCE -- the model's own uncertainty.",
    "     The 3 seeds each predict independently; their spread is the model's own",
    "     uncertainty. Wide spread = the model does not agree with itself.",
    "       ens_cv_et    relative spread of the 5 ExecTime predictions (std/mean)",
    "       ens_sd_mem   spread of Memory Throughput [%]      (pp)",
    "       ens_sd_occ   spread of Achieved Occupancy         (pp)",
    "       ens_sd_brk   mean spread of the 4 stall fractions (pp)",
    "",
    "  B) ROOFLINE CONSISTENCY -- the prediction checked against physics.",
    "     eta = t_roof / ExecTime must lie in (0, 1]: a kernel cannot run faster than",
    "     the roofline allows. eta > 1 is physically impossible and forces conf=low",
    "     however much the seeds agree -- the failure mode variance alone cannot see.",
    "       eta              predicted efficiency (must be <= 1)",
    "       eta_violation    1 = impossible prediction",
    "       log_dist_roof    distance of the answer from the roofline floor",
    "       mem_consistency  predicted Mem% vs bytes/(ExecTime*BW), an internal check",
    "",
    "Reported per output: conf_<output> = high | medium | low, and exp_err_<output> =",
    "the MEASURED mean error of that bucket (top 20% of the signal -> low, next 20% ->",
    "medium, rest -> high). Calibrated on HELD-OUT predictions (7 GPUs the model never",
    "saw; 4,784 kernels for time, 3,877 for the others). Mean error high / medium / low: ExecTime 17.8 / 24.5 / 33.3 %,",
    "Mem% 6.2 / 12.5 / 35.9 pp, occupancy 3.2 / 6.4 / 42.0 pp, breakdown 1.5 / 4.8 / 12.3 pp.",
    "Use exp_err: where it does not rise from high to low, the label does not rank the",
    "error. Ensemble variance measures seed disagreement, NOT correctness; exp_err is a",
    "typical (mean) error, not a guaranteed interval.",
]


def print_confidence_legend(stream=None):
    """Explain the two confidence signals in the run report."""
    (stream or sys.stderr).write("\n".join(CONF_LEGEND) + "\n")
