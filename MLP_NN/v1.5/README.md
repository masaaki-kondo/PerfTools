# Cross-GPU CUDA Kernel Performance Estimator (v1.5)

Predict how a CUDA kernel will perform on a **target** GPU from its Nsight Compute
(NCU) profile measured on a **source** GPU. A pre-trained, per-source
neural-network model (v1.5 = multi-branch MLP + roofline-derived input features).

- **Supported GPUs:** A100, H100, GB200 (any pair, source → target)
- **Dependencies:** Python 3.11 + `numpy`, `pandas`, `torch` (CPU is fine)
- **Offline:** no GPU, no network, no database; runs from the repo root

## Quick start

```bash
python MLP_NN/v1.5/predict_v15.py \
    --csv MLP_NN/examples/example_input_mixed-src_20kernels.csv \
    --row all --out pred.csv --log run.log
```

## What it predicts (per kernel)

| output | meaning |
|---|---|
| `Execution Time [ns]` | predicted runtime on the target GPU |
| `Memory Throughput [%]` | achieved DRAM-bandwidth utilization |
| `Achieved Occupancy` | achieved occupancy (%) |
| `brk_memory`, `brk_pipeline_contention`, `brk_sync`, `brk_scheduling_overhead` | stall-time breakdown (4 fractions, sum to 1) |
| `t_mem_ns`, `t_comp_ns`, `t_roof_ns`, `efficiency_eta` | roofline quantities (`t_roof`=max(mem,compute); `eta`=`t_roof`/ExecTime) |

**Accuracy:** on 5-fold (group-by-kernel) cross-validation, Execution Time
reaches **R² ≈ 0.99, ~7% MAPE**. v1.5 is the **in-range** estimator (existing
GPUs); targets far beyond the training GPUs are extrapolation and less reliable.
`Memory Throughput [%]` is an unclamped regression output and can fall slightly
outside [0, 100] (read as ≈0 / ≈100).

## Two ways to run

**Mode 1 — full input via CLI options** (one prediction; every input is a flag):
```bash
python MLP_NN/v1.5/predict_v15.py --out pred.csv \
    --src-gpu A100 --tgt-gpu H100 \
    --block-size 256 --grid-size 4096 --execution-time 5000 ... \
    --src-dram-bw-per-sm-byte-s-sm 1.8e10 --tgt-peak-fp32 6.7e13 ...
```
Run `predict_v15.py --help` to list the flag for every input column.

**Mode 2 — CSV + `--row`** (one or many kernels). `--row` is **required**: a
1-based data row, or `all` for every row (the header is metadata):
```bash
# every kernel in the CSV
python MLP_NN/v1.5/predict_v15.py --csv input.csv --row all --out pred.csv --log run.log

# just one row
python MLP_NN/v1.5/predict_v15.py --csv input.csv --row 3 --out row3.csv
```
In both modes the estimator reads **only the inputs you give it** — every input
column below must be present (no hidden lookups).

### Building an input CSV

`MLP_NN/examples/prepare_data.py` builds an all-columns CSV from a raw NCU export
+ the spec sheet. It is **self-contained** (numpy/pandas + the spec sheet only —
no other project code), so it runs from this repo:

```bash
python MLP_NN/examples/prepare_data.py --raw data/<your_ncu_wide>.zip --n 20
```

It reads the NCU columns from the export and fills the `SRC `/`TGT ` spec columns
from the spec sheet (`data/gpu_microarch_specs.csv`). **Note:** the raw NCU export
(Yoshida-san's profiling data) is large and is **not included in `/data`** — place
the wide zip there first. The shipped `example_input_mixed-src_20kernels.csv` was
built this way.

## Input schema

NCU columns are measured on the **source** GPU; spec columns come from the **spec
sheet** (`data/gpu_microarch_specs.csv`). 79 columns total per row.

| column(s) | source | role |
|---|---|---|
| `src_gpu`, `tgt_gpu` | you (GPU names) | which GPU each row is from / for |
| **Kernel config** (7): `Block Size`, `Grid Size`, `Threads`, `Registers Per Thread`, `Static/Dynamic/Shared Memory Per Block` | NCU (source) | launch configuration |
| **Workload** (16): `Achieved Occupancy`, `Compute (SM) Throughput [%]`, `Memory Throughput [%]`, `L1/L2 Cache Throughput [%]`, `Memory Throughput [byte/s]`, `Eligible/Active Warps`, `Executed Ipc`, `Waves Per SM`, block limits … | NCU (source) | throughput / occupancy profile |
| **Stalls** (16): `Stall Long Scoreboard`, `Stall Wait`, `Stall Barrier`, `Stall MIO Throttle`, … | NCU (source) | warp-stall reasons (per inst) |
| **FP counters** (7): `Predicated-On {FFMA,FADD,FMUL,DFMA,DADD,DMUL} … Per Cycle`, `Elapsed Cycles [cycle]` | NCU (source) | reconstruct FLOPs for the roofline |
| `Execution Time` | NCU (source) | source-GPU runtime (ns) |
| `SRC <spec>` × 13 | spec sheet | **source** GPU hardware specs |
| `TGT <spec>` × 13 | spec sheet | **target** GPU hardware specs |
| `TGT peak_fp32`, `TGT peak_fp64`, `TGT dram_bw` | spec sheet | target roofline peaks / bandwidth |

The 13 GPU spec features (`<spec>`, used with `SRC `/`TGT ` prefixes):
`GPU Maximum Warps Per Scheduler`, `Theoretical Active Warps per SM`,
`Theoretical Active Warps Per Scheduler`, `Shared Memory Configuration Size`,
`Block Limit Warps`, `Peak FP32 / Peak FP64`, `DRAM BW per SM`, `L2 Size per SM`,
`Peak FP32 / DRAM BW`, `Peak FP16 Tensor`, `Peak Tensor / Peak FP32`,
`DRAM Latency [ns]`, `CPU-GPU BW`.

The roofline terms (`bytes`, FLOPs, `t_mem`, `t_comp`, `t_roof`) are **computed
in-tool** from the columns above — not supplied.

## Output

`--out pred.csv` — one row per kernel; a leading `#` comment block documents each
column's source (`--no-comments` for a plain CSV). `--log run.log` — a plain-text
run summary (also printed to stderr). Example:

```
kernel_name,src_gpu,tgt_gpu,Execution Time [ns],Memory Throughput [%],Achieved Occupancy,brk_memory,...,efficiency_eta
adam,A100,H100,203149,5.5,68.1,0.41,...,0.312
calculateForce,H100,GB200,1413460,39.4,66.2,0.55,...,0.247
```

## Files

```
MLP_NN/v1.5/
  predict_v15.py     CLI (reads everything from the inputs you give it)
  v15_core.py        model + feature construction (self-contained)
  v15_artifact/      pre-trained per-source weights (model_<gpu>.pt, stats, meta)
  README.md
MLP_NN/examples/
  prepare_data.py                          build an input CSV from raw NCU + the spec sheet
  example_input_mixed-src_20kernels.csv    20-row example (all columns, mixed sources)
data/
  gpu_microarch_specs.csv                  the spec sheet
  (raw NCU data is large and NOT shipped — place here to rebuild examples)
```

The runtime estimator (`predict_v15.py` + `v15_core.py` + `v15_artifact/`) is
self-contained. `prepare_data.py` is also self-contained (numpy/pandas + the spec
sheet), but needs **Yoshida-san's raw NCU export** — it is large and is **not
included in `/data`**; place the wide zip there to (re)build an input CSV.
