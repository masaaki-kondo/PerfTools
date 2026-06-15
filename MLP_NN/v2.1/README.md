# v2.1 -- cross-GPU CUDA kernel performance estimator (hybrid)

Predicts a CUDA kernel's metrics on a TARGET GPU from its NCU profile measured on a
SOURCE GPU. **v2.1 is the HYBRID / accuracy-reference model:** the v1.5 NN inside the
trained GPU range, plus a calibrated analytical anchor for OUT-OF-RANGE (future) GPUs.
It is the most accurate option, but it (a) uses the source Execution Time as an input
and (b) evaluates an analytical formula at runtime. For a pure neural net that does
neither, use **v4.0** (the shipped model).

- Depends only on: Python 3.11+, numpy, pandas, torch, and `../v1.5/v15_core.py`.
- Reuses the v1.5 per-source artifact (`../v1.5/v15_artifact`); the analytical slopes
  are baked in (no data needed at runtime).
- Supported GPUs: A100, H100, GB200 (HBM family) and future HBM GPUs.

## Quick start
```
python MLP_NN/v2.1/predict_v21.py \
       --csv MLP_NN/examples/example_kernels.csv --row all --out pred.csv
```

## What it predicts (per kernel)
| Output                       | Meaning                                  |
|------------------------------|------------------------------------------|
| Execution Time [ns]          | predicted kernel time on the target GPU  |
| Memory Throughput [%]        | predicted DRAM throughput                |
| Achieved Occupancy           | predicted occupancy                      |
| brk_memory / brk_pipeline_contention / brk_sync / brk_scheduling_overhead | stall breakdown (fractions, sum to 1) |
| t_mem_ns / t_comp_ns / t_roof_ns | roofline times (computed in-tool)    |
| efficiency_eta               | t_roof / predicted time                  |

Behavior: **in the trained range v2.1 == v1.5 exactly** (the NN). Beyond GB200 it
switches to the calibrated analytical anchor (extrapolation reference, ~21% median
ExecTime to a generation-ahead GPU). It also keeps per-kernel ordering best of all
variants and is the only one that responds to bandwidth and SM/peak independently.

## Two ways to run
**Mode 1 -- every input as a CLI flag (one prediction):**
```
python MLP_NN/v2.1/predict_v21.py --src-gpu a100 --tgt-gpu gb200 \
       --execution-time 12345 ... --out one.csv
```
**Mode 2 -- CSV (batch or single row):**
```
python MLP_NN/v2.1/predict_v21.py --csv FILE --row all   --out all.csv
python MLP_NN/v2.1/predict_v21.py --csv FILE --row 3     --out row3.csv
```

## Predict for a FUTURE GPU (different specs)
Like v4.0, v2.1 reads the TARGET GPU's specs from the input. To predict a GPU that does
not exist yet, supply its datasheet specs in the `TGT` columns and keep the kernel's NCU
counters from your real SOURCE profile:

| Column            | Meaning                          | Example (a "2x GB200")   |
|-------------------|----------------------------------|--------------------------|
| `src_gpu`         | the real GPU you profiled on     | a100                     |
| `TGT dram_bw`     | target memory bandwidth [byte/s] | 16e12                    |
| `TGT peak_fp32`   | target FP32 peak [flop/s]        | 160e12                   |
| `TGT peak_fp64`   | target FP64 peak [flop/s]        | 80e12                    |
| `TGT <spec>` (13) | target spec features             | from datasheet           |

```
python MLP_NN/v2.1/predict_v21.py --csv one_kernel.csv --row 1 \
       --tgt-dram-bw 16e12 --tgt-peak-fp32 160e12 --tgt-peak-fp64 80e12 --out future.csv
```
When the target bandwidth/peak exceeds GB200's, v2.1 applies the calibrated analytical
slope (anchored on your source measurement) instead of the raw NN -- that is the hybrid
out-of-range path.

## Building an input CSV
```
python MLP_NN/examples/prepare_data.py --src A100,H100,GB200 --n N --out my.csv
```
`example_kernels.csv` (all 147 kernels) is provided ready to use.

## Input columns (79 per row)
- metadata: `Kernel Name`, `src_gpu`, `tgt_gpu`
- NCU (source profile, ~47 cols): kernel config, workload, stalls, `Execution Time`,
  `Elapsed Cycles`, the Predicated-On FP-op counters
- specs (29 cols): `SRC <13>`, `TGT <13>`, `TGT peak_fp32`, `TGT peak_fp64`, `TGT dram_bw`

## Output
- `--out pred.csv` : `kernel_name, src_gpu, tgt_gpu` + the 11 metrics (6 sig figs;
  provenance comment header unless `--no-comments`).
- `--log run.log`  : plain-text run summary (also written to stderr).

## Files
```
predict_v21.py     CLI (same arguments and output as v1.5 predict_v15.py)
README.md          this file
```
Model weights: reuses `../v1.5/v15_artifact` (per-source models); analytical slopes are
hard-coded in `predict_v21.py`.
