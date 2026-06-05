"""Write INPUT_SCHEMA.csv — the provenance (source) of every input column."""
import os, sys, csv, json
sys.path.insert(0, ".")
from MLP_NN.data_pipeline_v2 import (
    KERNEL_CONFIG_FEATURES, WORKLOAD_PROFILE_FEATURES, STALL_FEATURES)

HERE = "MLP_NN/v1.5"
meta = json.load(open(f"{HERE}/v15_artifact/meta.json"))
KSTAT = meta["kernel_stats_columns"]


def src_of(col):
    if col in KERNEL_CONFIG_FEATURES:   return ("NCU (source GPU)", "kernel launch config")
    if col in WORKLOAD_PROFILE_FEATURES: return ("NCU (source GPU)", "workload / throughput profile")
    if col in STALL_FEATURES:           return ("NCU (source GPU)", "warp-stall reason (per inst)")
    if "Per Cycle" in col or "Elapsed Cycles" in col:
        return ("NCU (source GPU)", "raw counter -> reconstructs FLOPs for roofline")
    if col == "Execution Time":         return ("NCU (source GPU)", "source-GPU kernel runtime (ns)")
    return ("NCU (source GPU)", "")


with open(f"{HERE}/INPUT_SCHEMA.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["column", "source", "role"])
    w.writerow(["src_gpu", "user (GPU name)", "row in specs.csv the kernel was profiled on"])
    w.writerow(["tgt_gpu", "user (GPU name)", "row in specs.csv to estimate for"])
    for c in KSTAT:
        s, role = src_of(c)
        w.writerow([c, s, role])
    w.writerow(["<GPU spec features>", "static specs.csv (by --src/--tgt name)",
                "NOT supplied per run; shipped static per-generation file (from the spec sheet)"])
print(f"wrote {HERE}/INPUT_SCHEMA.csv ({len(KSTAT) + 3} rows)")
