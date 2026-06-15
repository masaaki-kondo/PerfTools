"""v2.1 cross-GPU kernel performance estimator -- CLI (self-contained).

v2.1 = the HYBRID model: the v1.5 NN in the trained range + a calibrated analytical
anchor for OUT-OF-RANGE (future) GPUs. Most accurate, but it (a) uses Execution Time
as an input and (b) evaluates an analytical formula at runtime -- the two things Kondo
rejects for the shipped model (use v4.0 for that). Provided as the accuracy reference.

Same CLI and I/O as v1.5 / v4.0 (drop-in). Fully offline. Uses the v1.5 per-source
artifact (v15_artifact); the analytical slopes are baked in (no data needed at runtime).

TWO INPUT MODES:
  1. full input via CLI options.
  2. --csv FILE --row {N|all}.
"""
import os, sys, csv, re, json, pickle, argparse, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
V15 = os.path.join(HERE, "..", "v1.5")
sys.path.insert(0, V15); sys.path.insert(0, os.path.join(HERE, "..", ".."))
import numpy as np
import pandas as pd
import torch
import v15_core as core
# GPU datasheet constants (inlined for a self-contained deliverable)
GPU_HW_FALLBACK = {"a100": {"dram_bw": 2.039e12}, "h100": {"dram_bw": 3.35e12},
                   "gb200": {"dram_bw": 8.0e12}, "gb10": {"dram_bw": 0.273e12}}
GPU_PEAK_SPECS = {"a100": {"peak_fp32": 19.5e12, "peak_fp64": 9.7e12},
                  "h100": {"peak_fp32": 67e12, "peak_fp64": 34e12},
                  "gb200": {"peak_fp32": 80e12, "peak_fp64": 40e12},
                  "gb10": {"peak_fp32": 31e12, "peak_fp64": 0.46e12}}

SPEC = core.GPU_SPEC_FEATURES
SRC_PFX, TGT_PFX = "SRC ", "TGT "
OUT_COLS = ["Execution Time [ns]", "Memory Throughput [%]", "Achieved Occupancy",
            "brk_memory", "brk_pipeline_contention", "brk_sync",
            "brk_scheduling_overhead", "t_mem_ns", "t_comp_ns", "t_roof_ns", "efficiency_eta"]
FP_COUNTERS = [
    "Predicated-On FFMA Operations Per Cycle [inst]",
    "Predicated-On FADD Thread Instructions Executed Per Cycle [inst/cycle]",
    "Predicated-On FMUL Thread Instructions Executed Per Cycle [inst/cycle]",
    "Predicated-On DFMA Operations Per Cycle [inst]",
    "Predicated-On DADD Thread Instructions Executed Per Cycle [inst/cycle]",
    "Predicated-On DMUL Thread Instructions Executed Per Cycle [inst/cycle]",
    "Elapsed Cycles [cycle]"]
NCU_COLS = (core.KERNEL_CONFIG_FEATURES + core.WORKLOAD_PROFILE_FEATURES
            + core.STALL_FEATURES + ["Execution Time"] + FP_COUNTERS)
SPEC_COLS = ([f"{SRC_PFX}{c}" for c in SPEC] + [f"{TGT_PFX}{c}" for c in SPEC]
             + ["TGT peak_fp32", "TGT peak_fp64", "TGT dram_bw"])
INPUT_COLS = ["src_gpu", "tgt_gpu"] + NCU_COLS + SPEC_COLS
_flag = lambda c: "--" + re.sub(r"[^a-z0-9]+", "-", c.lower()).strip("-")
_dest = lambda c: re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_")
DEST2COL = {_dest(c): c for c in INPUT_COLS}

# calibrated (bandwidth-exp, peak-exp) per class, fit on real A100/H100/GB200 pairs.
# order within each class: (bandwidth, mem-latency, compute)
SLOPES = {
    "ET":  {"bandwidth": (0.289, 0.229), "mem-latency": (0.146, 0.187), "compute": (0.041, 0.150)},
    "MEM": {"bandwidth": (0.220, -0.060), "mem-latency": (0.654, -0.453), "compute": (0.281, -0.158)},
    "OCC": {"bandwidth": (0.002, 0.103), "mem-latency": (0.026, 0.030), "compute": (0.017, 0.000)},
    "BRK": {"bandwidth": (0.078, 0.007), "mem-latency": (0.005, -0.008), "compute": (0.337, -0.299)}}
GB_BW = GPU_HW_FALLBACK["gb200"]["dram_bw"]; GB_P = GPU_PEAK_SPECS["gb200"]["peak_fp32"]
sig = lambda x: 1.0 / (1.0 + np.exp(-x))
def cont(metric, bal, u0):
    d = SLOPES[metric]; gm = sig(bal / 0.5); gb = sig((u0 - 50) / 10.0)
    a = gm * (gb * d["bandwidth"][0] + (1 - gb) * d["mem-latency"][0]) + (1 - gm) * d["compute"][0]
    b = gm * (gb * d["bandwidth"][1] + (1 - gb) * d["mem-latency"][1]) + (1 - gm) * d["compute"][1]
    return a, b


def load_source_model(art, src):
    ck = torch.load(f"{art}/model_{src}.pt", map_location="cpu", weights_only=False)
    m = core.MultiBranchMLP(branch_dims=ck["branch_dims"], shared_branch_indices=ck["shared"],
                            branch_hidden=64, shared_hidden=128, n_shared_layers=2,
                            regression_outputs=ck["reg_out"], breakdown_outputs=ck["brk_out"], dropout=0.1)
    m.load_state_dict(ck["state_dict"]); m.eval()
    with open(f"{art}/stats_{src}.pkl", "rb") as f:
        stats = pickle.load(f)
    return m, stats, ck


def resolve_specs(row):
    if (SRC_PFX + SPEC[0]) not in row or row.get(SRC_PFX + SPEC[0]) != row.get(SRC_PFX + SPEC[0]):
        return None
    src_vec = np.array([float(row[SRC_PFX + c]) for c in SPEC])
    tgt_vec = np.array([float(row[TGT_PFX + c]) for c in SPEC])
    peaks = {"peak_fp32": float(row[TGT_PFX + "peak_fp32"]),
             "peak_fp64": float(row[TGT_PFX + "peak_fp64"]), "dram_bw": float(row[TGT_PFX + "dram_bw"])}
    return src_vec, tgt_vec, peaks


def build_sample(srow, src_gpu, tgt_gpu, src_vec, tgt_vec, peaks, reg_out, brk_out):
    s = pd.Series(srow)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = tgt_vec / np.where(np.abs(src_vec) > 1e-8, src_vec, 1e-8)

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
        "target_dram_bw": peaks["dram_bw"], "target_peak_fp32": peaks["peak_fp32"], "target_peak_fp64": peaks["peak_fp64"],
        # source measurements (for the analytical anchor out-of-range)
        "src_et": g0("Execution Time"), "src_mem": g0("Memory Throughput [%]"),
        "src_occ": g0("Achieved Occupancy"), "src_brk": float(core.compute_breakdown(s)[0])}


def predict_hybrid(model, stats, samples):
    """v1.5 NN in-range; analytical anchor (on the source measurement) out-of-range."""
    P = core.predict(model, stats, samples)
    et = P["et_p"].astype(float).copy(); mem = P["pr"][:, core.MEMPCT].astype(float).copy()
    occ = P["pr"][:, core.OCC].astype(float).copy(); brk = P["pb"].astype(float).copy()
    for i, s in enumerate(samples):
        tbw, tp = s["target_dram_bw"], s["target_peak_fp32"]
        if tbw <= GB_BW * 1.02 and tp <= GB_P * 1.02:
            continue                                    # in trained band -> keep the NN (v1.5)
        sbw = GPU_HW_FALLBACK[s["src_gpu"].lower()]["dram_bw"]; sp = GPU_PEAK_SPECS[s["src_gpu"].lower()]["peak_fp32"]
        fb, fp = tbw / sbw, tp / sp
        tcf = max(s["roofline_fp32_flops"] / tp, s["roofline_fp64_flops"] / s["target_peak_fp64"], 1e-30)
        tmem = s["roofline_bytes"] / tbw
        bal = np.log(max(tmem, 1e-30) / tcf); u0 = s["src_mem"] if s["src_mem"] > 0 else 50.0
        aE, bE = cont("ET", bal, u0); aM, bM = cont("MEM", bal, u0)
        aO, bO = cont("OCC", bal, u0); aS, bS = cont("BRK", bal, u0)
        if s["src_et"] > 0:
            et[i] = max(s["src_et"] * fb ** (-aE) * fp ** (-bE), max(tmem, tcf) * 1e9)
        mem[i] = float(np.clip(s["src_mem"] * fb ** (-aM) * fp ** (-bM), 0, 100))
        occ[i] = float(np.clip(s["src_occ"] * fb ** (-aO) * fp ** (-bO), 0, 100))
        if s["src_brk"] > 0:
            nm = float(np.clip(s["src_brk"] * fb ** (-aS) * fp ** (-bS), 1e-4, 0.95))
            oth = brk[i, 1:]; so = float(oth.sum())
            brk[i] = np.concatenate([[nm], oth * (1 - nm) / so]) if so > 0 else np.concatenate([[nm], np.full(3, (1 - nm) / 3)])
            brk[i] = brk[i] / brk[i].sum()
    return {"et_p": et, "mem": mem, "occ": occ, "brk": brk}


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
                    "t_roof_ns": float(t_roof[i]), "efficiency_eta": float(eta[i])})
    return out


PROVENANCE = [
    "# v2.1 cross-GPU kernel estimator (hybrid: v1.5 NN in-range + analytical anchor out-of-range)",
    "# INPUT SOURCES: kernel stats = NCU on the SOURCE GPU; GPU specs = spec sheet (SRC/TGT cols).",
    "# MODEL: v2.1 -- uses Execution Time as input AND an analytical formula out-of-range",
    "#   (accuracy reference; v4.0 is the pure-NN shipped model).",
    "# OUTPUTS: ExecTime/Mem%/Occupancy/brk_* = v2.1 predictions; t_*_ns + efficiency_eta = roofline.",
]


def _to_num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def main():
    ap = argparse.ArgumentParser(
        description="v2.1 cross-GPU kernel estimator (hybrid). Two input modes: "
                    "(1) full input via CLI options, or (2) --csv FILE [--row N].")
    ap.add_argument("--artifact", default=f"{V15}/v15_artifact")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log")
    ap.add_argument("--no-comments", action="store_true")
    m2 = ap.add_argument_group("Mode 2: CSV input")
    m2.add_argument("--csv", "--kernel-stats", dest="csv", help="input CSV (all columns)")
    m2.add_argument("--row", help="REQUIRED with --csv: a 1-based data row number, or 'all'")
    m1 = ap.add_argument_group("Mode 1: full input via CLI options (one prediction)")
    for c in INPUT_COLS:
        m1.add_argument(_flag(c), dest=_dest(c), metavar="V", help=c)
    args = ap.parse_args()

    meta = json.load(open(f"{args.artifact}/meta.json"))
    reg_out, brk_out, known = meta["reg_out"], meta["brk_out"], set(meta["sources"])
    cache = {}

    def model_for(src):
        s = src.lower()
        if s not in cache:
            cache[s] = load_source_model(args.artifact, s)
        return cache[s]

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
            ap.error(f"no per-source model for src '{sg}' (have: {sorted(known)})")

    out_rows = [None] * len(rows_in)
    by_src = {}
    for i, (_, sg, _) in enumerate(rows_in):
        by_src.setdefault(sg.lower(), []).append(i)
    for src, idxs in by_src.items():
        model, stats, _ = model_for(src)
        sub = []
        for i in idxs:
            row, sg, tg = rows_in[i]
            spec = resolve_specs(row)
            if spec is None:
                ap.error("input is missing the GPU spec columns (SRC ... / TGT ... / TGT peak_*).")
            sv, tv, pk = spec
            sub.append(build_sample(row, sg, tg, sv, tv, pk, reg_out, brk_out))
        for i, r in zip(idxs, to_rows(predict_hybrid(model, stats, sub), sub)):
            out_rows[i] = r

    cols = ["kernel_name", "src_gpu", "tgt_gpu"] + OUT_COLS
    with open(args.out, "w", newline="") as f:
        if not args.no_comments:
            f.write("\n".join(PROVENANCE) + "\n")
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in out_rows:
            w.writerow({c: (f"{r[c]:.6g}" if isinstance(r[c], float) else r[c]) for c in cols})

    log = [f"v2.1 cross-GPU estimator (hybrid)  run @ {datetime.datetime.now().isoformat(timespec='seconds')}",
           f"model    : v2.1  (v1.5 NN + analytical anchor out-of-range)", "specs    : read from the input",
           f"mode     : {'CSV' if args.csv else 'CLI'}" + (f", row {args.row}" if args.row else ""),
           f"kernels  : {len(rows_in)}",
           f"mean predicted ExecTime: {np.nanmean([r['Execution Time [ns]'] for r in out_rows]):.1f} ns",
           f"output   : {args.out}"]
    text = "\n".join(log); sys.stderr.write(text + "\n")
    if args.log:
        open(args.log, "w").write(text + "\n")


if __name__ == "__main__":
    main()
