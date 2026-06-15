# v4.0 -- cross-GPU CUDA kernel performance estimator (no-ET pure NN)

Predicts a CUDA kernel's metrics on a TARGET GPU from its NCU profile measured on a
SOURCE GPU. **v4.0 is the going-forward shipped model:** the source Execution Time is
NOT a predictive input, and inference is a plain neural-net forward pass -- there is no
analytical formula evaluated at runtime. Fully offline.

- Depends only on: Python 3.11+, numpy, pandas, torch, and `../v1.5/v15_core.py`.
- Supported GPUs: A100, H100, GB200 (the HBM datacenter family) and future HBM GPUs.
- Not for edge/LPDDR parts (e.g. GB10) -- a different regime.

## Quick start
```
python MLP_NN/v4.0/predict_v40.py \
       --csv MLP_NN/examples/example_inputs.csv --row all --out pred.csv
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

Accuracy (held-out, leave-GB200-out): **in-range ~5% median ExecTime**; **~17-23%
median to a generation-ahead GPU**; Memory ~13pp, Occupancy ~5pp, stall ~3pp out of
range. Trends (shape) are more reliable than one-off absolute magnitudes far out of range.

## Two ways to run
**Mode 1 -- every input as a CLI flag (one prediction):**
```
python MLP_NN/v4.0/predict_v40.py --src-gpu a100 --tgt-gpu gb200 \
       --execution-time 12345 --memory-throughput-byte-s 1.2e12 ... --out one.csv
```
**Mode 2 -- CSV (batch or single row):**
```
python MLP_NN/v4.0/predict_v40.py --csv FILE --row all   --out all.csv
python MLP_NN/v4.0/predict_v40.py --csv FILE --row 3     --out row3.csv
```

## Predict for a FUTURE GPU (different specs)
v4.0 reads the TARGET GPU's specs from the input, so to predict a GPU that does not
exist yet you simply supply its datasheet specs in the `TGT` columns and keep the
kernel's NCU counters from your real SOURCE profile:

| Column            | Meaning                          | Example (a "2x GB200")   |
|-------------------|----------------------------------|--------------------------|
| `src_gpu`         | the real GPU you profiled on     | a100                     |
| `TGT dram_bw`     | target memory bandwidth [byte/s] | 16e12                    |
| `TGT peak_fp32`   | target FP32 peak [flop/s]        | 160e12                   |
| `TGT peak_fp64`   | target FP64 peak [flop/s]        | 80e12                    |
| `TGT <spec>` (13) | target spec features (warps, shared-mem, L2/SM, DRAM latency, ...) | from datasheet |

Easiest path: copy one row of `example_inputs.csv` for your kernel, then overwrite the
`TGT ...` columns with the future GPU's numbers and run `--row 1`. Or pass them as
flags in Mode 1:
```
python MLP_NN/v4.0/predict_v40.py --csv one_kernel.csv --row 1 \
       --tgt-dram-bw 16e12 --tgt-peak-fp32 160e12 --tgt-peak-fp64 80e12 --out future.csv
```
Reliability is best up to ~one generation beyond GB200 and along the generation-scaling
direction (bandwidth and compute scaling together, as real GPU generations do).

## Building an input CSV
```
python MLP_NN/examples/prepare_data.py --src A100,H100,GB200 --n N --out my.csv
```
Needs a raw NCU export (not shipped); the spec sheet `data/gpu_microarch_specs.csv` is
shipped. `example_inputs.csv` (all 147 kernels) is provided ready to use.

## Input columns (79 per row)
- metadata: `Kernel Name`, `src_gpu`, `tgt_gpu`
- NCU (source profile, ~47 cols): kernel config, workload, stalls, `Execution Time`,
  `Elapsed Cycles`, the Predicated-On FP-op counters
- specs (29 cols): `SRC <13>`, `TGT <13>`, `TGT peak_fp32`, `TGT peak_fp64`, `TGT dram_bw`

`Execution Time` is read only to compute `bytes = MemThr[byte/s] x ExecTime`; it is
NOT a predictive feature for v4.0 (that is the point of v4.0).

## Output
- `--out pred.csv` : `kernel_name, src_gpu, tgt_gpu` + the 11 metrics above (6 sig figs;
  a provenance comment header unless `--no-comments`).
- `--log run.log`  : plain-text run summary (also written to stderr).

## Files
```
predict_v40.py     CLI (same arguments and output as v1.5 predict_v15.py)
v40_core.py        model + feature construction + inference (pure NN, no runtime formula)
v40_artifact/      model_seed0-4.pt (5-seed ensemble) + meta.pkl
README.md          this file
```
