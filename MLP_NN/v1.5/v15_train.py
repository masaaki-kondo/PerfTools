"""Train the v1.5 models and save a self-contained artifact for the CLI.

v1.5 = v1 (multi-branch MLP) + roofline quantities as extra INPUT features.
PER-SOURCE models: one model per source GPU (a100, h100, gb200), each predicting
its kernels on the other targets. BRK_W=1.0, GB10 excluded.

Writes (under MLP_NN/v1.5/):
  v15_artifact/model_<src>.pt    state_dict + branch dims, per source
  v15_artifact/stats_<src>.pkl   per-branch normalization (mean,std)
  v15_artifact/meta.json         feature schemas + output list
  specs.csv                      STATIC per-GPU spec file (separate input; 13
                                 spec features + abs peaks/BW for roofline)

Run from repo root on a compute node:  sbatch MLP_NN/v1.5/run_v15_train.sbatch
"""
import os, sys, json, pickle, csv
os.environ["ROOFLINE_INPUTS"] = "1"            # <-- v1.5: roofline terms as inputs
sys.path.insert(0, ".")
import numpy as np
import torch

import MLP_NN.eval_ensemble_cv as E
from MLP_NN.test_per_source_model_nogb10 import (
    ALL_ZIPS, extract_gpu_data, build_training_pairs)
from MLP_NN.data_pipeline_v2 import (
    GPU_SPEC_FEATURES, KERNEL_CONFIG_FEATURES, WORKLOAD_PROFILE_FEATURES,
    STALL_FEATURES, GPU_PEAK_SPECS, GPU_HW_FALLBACK, extract_features)

HERE = "MLP_NN/v1.5"
ART = f"{HERE}/v15_artifact"
os.makedirs(ART, exist_ok=True)
FP_COUNTERS = [
    "Predicated-On FFMA Operations Per Cycle [inst]",
    "Predicated-On FADD Thread Instructions Executed Per Cycle [inst/cycle]",
    "Predicated-On FMUL Thread Instructions Executed Per Cycle [inst/cycle]",
    "Predicated-On DFMA Operations Per Cycle [inst]",
    "Predicated-On DADD Thread Instructions Executed Per Cycle [inst/cycle]",
    "Predicated-On DMUL Thread Instructions Executed Per Cycle [inst/cycle]",
    "Elapsed Cycles [cycle]",
]


def main():
    full = {g: extract_gpu_data(zp, pt) for g, (zp, pt) in ALL_ZIPS.items()}
    samples = build_training_pairs(full)
    print(f"[train] {len(samples)} pairs, ROOFLINE_INPUTS={E.ROOFLINE_INPUTS}")

    # ---- STATIC per-GPU spec file (full precision: per-source source specs have
    #      ~zero variance, so rounding here explodes after (x-mean)/std) ----
    with open(f"{HERE}/specs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gpu"] + GPU_SPEC_FEATURES + ["peak_fp32", "peak_fp64", "dram_bw"])
        for g, benchdict in full.items():
            rows = [extract_features(r, GPU_SPEC_FEATURES)
                    for df in benchdict.values() for _, r in df.iterrows()]
            vec = np.nanmedian(np.array(rows, float), axis=0)
            lc = g.lower()
            w.writerow([lc] + [repr(float(x)) for x in vec]
                       + [repr(float(GPU_PEAK_SPECS[lc]["peak_fp32"])),
                          repr(float(GPU_PEAK_SPECS[lc]["peak_fp64"])),
                          repr(float(GPU_HW_FALLBACK[lc]["dram_bw"]))])
    print(f"[train] wrote {HERE}/specs.csv (static per-GPU specs)")

    # ---- branch dims (same for every source) ----
    D, _, _ = E._stack_target(samples)
    bd = [D["kernel_config"].shape[1], D["workload"].shape[1],
          D["source_specs"].shape[1], D["target_specs"].shape[1], D["derived"].shape[1]]
    reg_out, brk_out = D["target_regression"].shape[1], D["target_breakdown"].shape[1]
    shared = [[2, 3]] if bd[2] == bd[3] else None

    # ---- one model PER SOURCE GPU ----
    sources = sorted({s["src_gpu"].lower() for s in samples})
    for src in sources:
        sub = [s for s in samples if s["src_gpu"].lower() == src]
        model, stats = E.train_model(sub, brkw=1.0, init_seed=42)
        torch.save({"state_dict": model.state_dict(), "branch_dims": bd,
                    "shared": shared, "reg_out": reg_out, "brk_out": brk_out},
                   f"{ART}/model_{src}.pt")
        with open(f"{ART}/stats_{src}.pkl", "wb") as f:
            pickle.dump(stats, f)
        print(f"[train] source {src}: {len(sub)} pairs -> model_{src}.pt")

    meta = {
        "model": "v1.5 (MLP + roofline input features), PER-SOURCE, BRK_W=1.0",
        "sources": sources, "branch_dims": bd, "reg_out": reg_out, "brk_out": brk_out,
        "kernel_stats_columns": (KERNEL_CONFIG_FEATURES + WORKLOAD_PROFILE_FEATURES
                                 + STALL_FEATURES + ["Execution Time"] + FP_COUNTERS),
        "gpu_spec_features": GPU_SPEC_FEATURES,
        "outputs": ["Execution Time [ns]", "Memory Throughput [%]", "Achieved Occupancy",
                    "brk_memory", "brk_pipeline_contention", "brk_sync",
                    "brk_scheduling_overhead", "t_mem_ns", "t_comp_ns", "t_roof_ns",
                    "efficiency_eta"],
    }
    with open(f"{ART}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[train] saved per-source artifacts to {ART}/ (sources={sources}, bd={bd})")


if __name__ == "__main__":
    main()
