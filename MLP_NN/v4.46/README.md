# MLP-NN v4.46: cross-GPU kernel performance estimator (release 2026-10-16)

Predicts, for one CUDA kernel profiled on a **source** GPU, its execution time, memory throughput, occupancy and
stall-cycle breakdown on a **target** GPU: a real GPU or a hypothetical one described by its specs.

The weights in `v446_artifact/` come from the v4.45 training run (3 seeds). The v4.46 code adds physics-guarded
handling of GPU specs far from any measured GPU (see below).

## Run

```
python MLP_NN/v4.46/predict_v446.py --csv inputs.csv --row all --out predictions.csv
```

`--row N` predicts one data row. `--confidence` adds per-output confidence estimates. `--artifact DIR` selects other
weights. `--runtime-inputs FILE` supplies your own actual GPU values (see below). Requires numpy, pandas, torch.

## Inputs

One row per kernel, same columns as `MLP_NN/examples/example_inputs.csv`:

- `src_gpu`, `tgt_gpu`: GPU names (column names of the spec sheet, e.g. `A100`, `H100`, `GB200`, `H200`, `GH200`,
  `L40S`, `B300`), or any name for a hypothetical target.
- The source kernel's ncu metrics (launch configuration, instruction mix, memory traffic, stalls, its measured
  `Execution Time` and memory throughput). These are the kernel's actual measurements; they come from your profile.
- `SRC ...` / `TGT ...` GPU spec columns and `TGT peak_fp32`, `TGT peak_fp64`, `TGT dram_bw`. For a GPU named in the
  spec sheet, the model uses its own GPU table instead (rows built from any shipped spec sheet are recognised);
  for a hypothetical target, these columns define it.

## Spec sheet and runtime inputs

- `data/gpu_microarch_specs_v3.csv`: **specs only** (vendor datasheets, architectural limits, published
  microbenchmarks). It holds no measured values.
- `runtime_inputs_training.csv` (this folder): the **actual** values the model was trained with, applied on top of the
  spec sheet: operating SM clocks (some GPUs ran below boost), the bandwidth values used in training, and the model's
  DRAM-latency estimates. These are the defaults; they reproduce the validated predictions.
- `--runtime-inputs FILE`: your own actual values, same layout (`spec,unit,<GPU columns>`), applied on top of the
  defaults, cell by cell. Leave cells empty to keep the default.

## Outputs

Per row: `Execution Time [ns]` (per launch), `Memory Throughput [%]`, `Achieved Occupancy`, stall-cycle breakdown
`brk_memory`, `brk_pipeline_contention`, `brk_sync`, `brk_scheduling_overhead` (shares, sum to 1), the roofline floor
`t_mem_ns`, `t_comp_ns`, `t_roof_ns`, `efficiency_eta` (= t_roof / predicted time) and `trust_w`.

## Accuracy

Leave-one-GPU-out (the model never saw the target GPU), execution-time MAPE against nsys run time:
H200 10.1%, H100 15.5%, A100 21.5%, GH200 20.7%, L40S 46.1%, GB200 30.7%, B300 30.0%. In-sample: 13.1%.

## GPU specs far from measured GPUs

Within about 2x of a measured GPU (in bandwidth per SM, FP32 per SM, flop/byte, FP32:FP64 and SM count) the network
answers; beyond 4x a physics answer anchored on the network's prediction for the base GPU takes over (memory
efficiency from Little's law, compute saturation, wave-count limits), with a smooth blend in between. Less of a
resource never makes a kernel faster and more never makes it slower; no prediction falls below the roofline.
Only execution time has this guard; the other outputs are network outputs.

## Known limitations

- L40S is the hardest GPU (46% held out).
- Five GPU-related input features (GPU Maximum Warps Per Scheduler, Theoretical Active Warps Per Scheduler,
  Theoretical Active Warps per SM, Shared Memory Configuration Size, Block Limit Warps) come from the profile. The
  first two are architectural constants; the other three depend on the kernel's launch configuration and can be
  computed from the limits now listed in the spec sheet (max warps / blocks per SM, registers, shared-memory
  carveouts). For a target GPU without a profile, use the source kernel's values or compute them from those limits.
