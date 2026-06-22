# v4.1 -- cross-GPU CUDA kernel performance estimator (no-ET NN + single-axis trends)

Predicts a CUDA kernel's metrics on a TARGET GPU from its NCU profile measured on a
SOURCE GPU. **v4.1 = v4.0 (the no-ET pure NN) plus single-axis training**, so its
secondary metrics (Memory Throughput %, Occupancy, stall breakdown) respond when you
change memory throughput OR SM size **independently** -- not only when both scale
together. Like v4.0: Execution Time is NOT a predictive input, and inference is a plain
neural-net forward pass (no analytical formula at runtime). Fully offline.

- Depends only on: Python 3.11+, numpy, pandas, torch, and `../v1.5/v15_core.py`.
- Supported GPUs: A100, H100, GB200 (HBM family) and future HBM GPUs.
- Same CLI and output as v1.5 / v4.0 (drop-in).

## How v4.1 differs from v4.0
v4.0 was trained only along the GPU-generation direction (bandwidth and compute scaling
together, as real generations do), so its Mem%/Occ/stall heads stayed flat when you
varied a single axis. v4.1 adds synthetic training samples that vary **bandwidth only**
and **compute/SM only**, so those heads learn a per-axis response.

Honest scope of the change:
- The single-axis response is **partial**: clear for compute-bound kernels, weak for
  memory-bound ones (the bandwidth-vs-compute split is only loosely determined by 3
  collinear HBM GPUs; fully separating the axes needs decorrelated data, e.g. MIG).
- **Interpolation accuracy is unchanged from v4.0** (same in-range fit, within noise).
- Execution Time behaves the same as v4.0 (it is anchored to the roofline, which already
  separates bandwidth and compute by physics).

## Quick start
```
python MLP_NN/v4.1/predict_v41.py \
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

## Two ways to run
**Mode 1 -- every input as a CLI flag (one prediction):**
```
python MLP_NN/v4.1/predict_v41.py --src-gpu a100 --tgt-gpu gb200 \
       --execution-time 12345 --memory-throughput-byte-s 1.2e12 ... --out one.csv
```
**Mode 2 -- CSV (batch or single row):**
```
python MLP_NN/v4.1/predict_v41.py --csv FILE --row all   --out all.csv
python MLP_NN/v4.1/predict_v41.py --csv FILE --row 3     --out row3.csv
```

## Predict for a FUTURE GPU (different specs)
v4.1 reads the TARGET GPU's specs from the input, so to predict a GPU that does not
exist yet you supply its datasheet specs in the `TGT` columns and keep the kernel's NCU
counters from your real SOURCE profile:

| Column            | Meaning                          | Example (a "2x GB200")   |
|-------------------|----------------------------------|--------------------------|
| `src_gpu`         | the real GPU you profiled on     | a100                     |
| `TGT dram_bw`     | target memory bandwidth [byte/s] | 16e12                    |
| `TGT peak_fp32`   | target FP32 peak [flop/s]        | 160e12                   |
| `TGT peak_fp64`   | target FP64 peak [flop/s]        | 80e12                    |
| `TGT <spec>` (13) | target spec features             | from datasheet           |

Easiest path: copy one row of `example_inputs.csv` for your kernel, overwrite the
`TGT ...` columns, and run `--row 1`. Or pass them as flags:
```
python MLP_NN/v4.1/predict_v41.py --csv one_kernel.csv --row 1 \
       --tgt-dram-bw 16e12 --tgt-peak-fp32 160e12 --tgt-peak-fp64 80e12 --out future.csv
```
Because v4.1 has per-axis sensitivity, you can also probe a GPU that scales only ONE
knob (e.g. more bandwidth, same compute) -- but see the honest scope note above.

## Building an input CSV
```
python MLP_NN/examples/prepare_data.py --src A100,H100,GB200 --n N --out my.csv
```
`example_inputs.csv` (all 147 kernels) is provided ready to use.

## Input columns (79 per row)
- metadata: `Kernel Name`, `src_gpu`, `tgt_gpu`
- NCU (source profile, ~47 cols): kernel config, workload, stalls, `Execution Time`,
  `Elapsed Cycles`, the Predicated-On FP-op counters
- specs (29 cols): `SRC <13>`, `TGT <13>`, `TGT peak_fp32`, `TGT peak_fp64`, `TGT dram_bw`

`Execution Time` is read only to compute `bytes = MemThr[byte/s] x ExecTime`; it is NOT
a predictive feature.

## Output
- `--out pred.csv` : `kernel_name, src_gpu, tgt_gpu` + the 11 metrics above (6 sig figs;
  provenance comment header unless `--no-comments`).
- `--log run.log`  : plain-text run summary (also to stderr).

## Files
```
predict_v41.py     CLI (same arguments and output as v1.5 predict_v15.py)
v40_core.py        model + feature construction + inference (pure NN, shared with v4.0)
v41_artifact/      model_seed0-4.pt (5-seed ensemble) + meta.pkl
README.md          this file
```

## Profiling records (`--dump-records`)

`--dump-records PATH` writes a second CSV next to `--out`: one row per kernel in the
**io_signal** schema, for scoring predictions against a separate ground-truth file.

| group | columns |
|---|---|
| `meta-*` | `model, src_gpu, tgt_gpu, kernel, pair_type` (`self` if src==tgt else `cross`) |
| `I-*` | every INPUT this model consumes — `kcfg / wl / srcspec / tgtspec(ratio) / derived` |
| `O-*` | the model's PREDICTIONS (the 7 outputs) |
| `aux-roofline::` | in-tool `t_mem_ns, t_comp_ns, t_roof_ns, efficiency_eta` |

It writes **predictions only — no ground truth**. To score, compare row-for-row (same
row order) against a truth file of measured TARGET outputs (`S-*`); for `pair_type=self`
the input's own measured outputs ARE the target truth.

```
python MLP_NN/v4.1/predict_v41.py --csv inputs.csv --row all \
       --out preds.csv --dump-records records.csv
```

Because v4.1 is no-ET, the record has **no** `I-derived::src_raw_exec_time_ns`. Unlike
v1.5 / v2.1, v4.1 was trained on diagonal pairs and clips its outputs, so it predicts
**same-GPU (`pair_type=self`) reliably** as well as cross-GPU.
