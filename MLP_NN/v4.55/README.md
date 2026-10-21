# MLP-NN v4.55: cross-GPU kernel performance estimator (release 2026-10-21)

Predicts, for one CUDA kernel profiled with ncu on a **source** GPU, its execution time, memory throughput, occupancy,
stall-cycle breakdown and execution-cycle breakdown on a **target** GPU: a real GPU or a hypothetical one described by
its specs. The target GPU never needs a profile.

MLP-NN v4.55 is the v4.46 inference code with new weights (`v455_artifact/`, 5 seeds), trained on Yoshida-san's
HeCBench profiles including the 2026-10-17 sweep (A100, H100, H200, GH200 96 GB, GB200, L40S, H100 MIG 1g/2g/4g; B300
as an nsys-only target). The previous release (2026-10-16) is in `MLP_NN/obsolete/v4.46/`; it still runs and its
outputs are unchanged.

## Quick start

```
python MLP_NN/examples/prepare_data.py --ncu <ncu files or folders> --src H200 --tgt H100,GB200 --out inputs.csv
python MLP_NN/v4.55/predict_v455.py --csv inputs.csv --row all --out predictions.csv
```

`--row N` predicts one data row. `--confidence` adds per-output confidence estimates. `--artifact DIR` selects other
weights. `--runtime-inputs FILE` supplies your own actual GPU values (see below). Requires numpy, pandas, torch.
For a whole application run on one GPU, see `extrapolate_app.py` in the repository root.

## Inputs

One row per kernel (and target GPU), same columns as `MLP_NN/examples/example_inputs.csv`:

- `src_gpu`, `tgt_gpu`: GPU names (columns of the spec sheet), or any name for a hypothetical target.
  `src_gpu` must be a GPU with ncu profiles: A100, H100, H200, GH200, GB200, GB10, L40S, H100_MIG_1g/2g/4g.
  B300 and GH200_144GB are targets only (GH200_144GB through its spec columns).
- The source kernel's ncu metrics (launch configuration, instruction mix, memory traffic, cache hit rates, stalls).
  These are the kernel's actual measurements on the source GPU.
- `SRC ...` / `TGT ...` GPU spec columns, `TGT peak_*` columns, and `TGT n_sms` for a GPU the predictor does not know.

Build the CSV with `MLP_NN/examples/prepare_data.py`. It reads Yoshida-san's raw long-form ncu-full files or wide
CSVs; `--src`, `--tgt` choose the GPUs, `--target-specs FILE` describes a GPU that is not in the spec sheet,
`--runtime-inputs FILE` adds actual values, `--smem-config` sets the shared-memory configuration for a new SM design.

Five spec features come from ncu's architectural metrics. Two are hardware values: GPU Maximum Warps Per Scheduler and
Theoretical Active Warps Per Scheduler (= `max_warps_per_scheduler`). Three are representative values of a typical
kernel: Theoretical Active Warps per SM = `max_warps_per_sm`, Block Limit Warps = `max_warps_per_sm` / 8 (256-thread
blocks), Shared Memory Configuration Size = 32 KB (16 KB on L40S and GB10). These per-GPU medians are what every
reported accuracy uses; training uses each kernel's own values.

Interpolation (target inside the trained range of specs) is validated below; extrapolation (beyond it) has no
measured error.

## Spec sheet and runtime inputs

- `data/v4/gpu_microarch_specs.csv`: datasheet values only. The `model_input` column marks each row as `yes` (a model
  input), `reference` or `no (descriptive)`. The earlier sheet `data/gpu_microarch_specs.csv` (LightGBM, MLP v1.5) is unchanged; the 10-16 one is in `data/obsolete/`.
- `runtime_inputs_training.csv` (this folder): the **actual** values the model was trained with, applied on top of the
  spec sheet: operating SM clocks, DRAM latency, measured bandwidths. These are the defaults; they
  reproduce the validated predictions.
- `--runtime-inputs FILE`: your own actual values, same layout (`spec,unit,<GPU columns>`), applied on top of the
  defaults, cell by cell. Leave cells empty to keep the default.

## Outputs

Per row:

- `Execution Time [ns]` (per launch), `Memory Throughput [%]`, `Achieved Occupancy`.
- Stall-cycle breakdown `brk_memory`, `brk_pipeline_contention`, `brk_sync`, `brk_scheduling_overhead` (sum to 1).
- `exec_cycles` (= time x clock) and the execution-cycle breakdown `cyc_running`, `cyc_memory`,
  `cyc_pipeline_contention`, `cyc_sync`, `cyc_scheduling_overhead` (running + 4 stall shares = 1);
  `running_cycles` = issued warp instructions of the source run (exact for a target with the same SM architecture).
  These are NaN when the input has no `Issued Instructions [inst]`.
- `issue_limited`: 1 when the time was raised to the instruction-issue limit (see below).
- Roofline floor `t_mem_ns`, `t_comp_ns`, `t_roof_ns`, `efficiency_eta` (= t_roof / predicted time), and `trust_w`
  (1 = target inside the trained spec envelope, < 1 = outside, prediction blended toward the physics anchor).
- Optional: `PRED_SEED_TIMES=1` in the environment adds the per-seed times `et_seed0` ... `et_seed4`.

## Accuracy

Kernel-level mean absolute percentage error of execution time. Held out = leave-one-GPU-out: each GPU is never seen
in training. Inputs use the representative GPU features above and the instruction-issue limit is on.

| | In-sample | H200 | H100 | A100 | GH200 | L40S | GB200 | B300 | Held-out mean |
|---|---|---|---|---|---|---|---|---|---|
| MLP-NN v4.55 | 13.6 | 9.2 | 14.2 | 21.6 | 18.2 | 32.3 | 27.2 | 23.4 | 20.86 |

Application level (sum of kernel time x launches), held out: about 20%.

## What is new (vs the 2026-10-16 release)

- Newest training data, including the 2026-10-17 sweep; 5 seeds instead of 3.
- Clock-aware latency floor; the setting is stored in the weights' metadata.
- Busy-SM rule for hypothetical targets: per-SM bandwidth is counted over the SMs a kernel actually keeps busy. The
  network also receives it as 3 extra inputs, clipped to the training range.
- Instruction-issue limit inside the model: a kernel cannot finish faster than issued warp instructions /
  (4 x SMs x clock). Output column `issue_limited`.
- New outputs `exec_cycles`, `cyc_*` and `running_cycles`; optional per-seed times (`PRED_SEED_TIMES=1`).

## GPU specs far from measured GPUs

Within about 2x of a measured GPU (in bandwidth per SM, FP32 per SM, flop/byte, FP32:FP64 and SM count) the network
answers; beyond 4x a physics answer anchored on the network's prediction for the base GPU takes over (memory
efficiency from Little's law, compute saturation, wave-count limits), with a smooth blend in between. Less of a
resource never makes a kernel faster and more never makes it slower; no prediction falls below the roofline.
Only execution time has this guard; the other outputs are network outputs.

## Known limitations

- L40S is the hardest GPU (32% held out).
- The three representative spec features (warps per SM, block limit, shared-memory configuration) describe a typical
  kernel, not the specific one; kernels with unusual launch configurations carry more error on a new target.

## Files

| File | Content |
|---|---|
| `predict_v455.py` | predictor (command line) |
| `v455_core.py`, `v15_core.py` | inference code |
| `v455_artifact/` | weights, 5 seeds, and `meta.pkl` |
| `runtime_inputs_training.csv` | actual GPU values used in training |

Shared code: `MLP_NN/data_pipeline_v2.py`, `MLP_NN/mlp.py`.
