# MLP-NN v4.55: cross-GPU kernel performance estimator (release 2026-10-21)

Predicts, for one CUDA kernel profiled with ncu on a **source** GPU, its execution time, memory throughput, occupancy,
stall-cycle breakdown and execution-cycle breakdown on a **target** GPU: a real GPU or a hypothetical one described by
its specs. The target GPU never needs a profile.

MLP-NN v4.55 is the v4.46 inference code with new weights (`v4.55/`, 5 seeds), trained on Yoshida-san's
HeCBench profiles including the 2026-10-17 sweep (A100, H100, H200, GH200 96 GB, GB200, L40S, H100 MIG 1g/2g/4g; B300
as an nsys-only target). The previous release (2026-10-16) is in `MLP_NN/obsolete/v4.46/` (weights only); run it with
`python MLP_NN/mlp-nn.py --artifact MLP_NN/obsolete/v4.46 ...` (current inference code, so outputs differ slightly from
the 2026-10-16 release).

## Quick start

```
python examples/kernel_inputs/prepare_data.py --ncu <ncu files or folders> --src H200 --tgt H100,GB200 --out inputs.csv
python MLP_NN/mlp-nn.py --csv inputs.csv --row all --out predictions.csv
```

`--row N` predicts one data row. `--confidence` adds per-output confidence levels and expected errors (see below). `--artifact DIR` selects other
weights. `--runtime-inputs FILE` supplies your own actual GPU values (see below). Requires numpy, pandas, torch.
For a whole application run on one GPU, see `examples/predict_app.py`.

## Inputs

One row per kernel (and target GPU), same columns as `examples/kernel_inputs/kernel_inputs.csv`:

- `src_gpu`, `tgt_gpu`: GPU names (columns of the spec sheet), or any name for a hypothetical target.
  `src_gpu` must be a GPU with ncu profiles: A100, H100, H200, GH200, GB200, GB10, L40S, H100_MIG_1g/2g/4g.
  B300 and GH200_144GB are targets only (GH200_144GB through its spec columns).
- The source kernel's ncu metrics (launch configuration, instruction mix, memory traffic, cache hit rates, stalls).
  These are the kernel's actual measurements on the source GPU.
- `SRC ...` / `TGT ...` GPU spec columns, `TGT peak_*` columns, and `TGT n_sms` for a GPU the predictor does not know.

Build the CSV with `examples/kernel_inputs/prepare_data.py`. It reads Yoshida-san's raw long-form ncu-full files or wide
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
  input), `reference` or `no (descriptive)`. The earlier sheet `data/gpu_microarch_specs.csv` (LightGBM, MLP v1.5) is unchanged; the 2026-10-16 sheet is `data/obsolete/gpu_microarch_specs_v3.csv`.
- `v4.55/runtime_inputs_training.csv` (in the weights folder): the **actual** values the model was trained with, applied on top of the
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
  `running_cycles` and `cyc_*` are NaN when the input has no `Issued Instructions [inst]`; `exec_cycles` is always given.
- `issue_limited`: 1 when the time was raised to the instruction-issue limit (see below).
- Roofline floor `t_mem_ns`, `t_comp_ns`, `t_roof_ns`, `efficiency_eta` (= t_roof / predicted time), and `trust_w`
  (1 = target inside the trained spec envelope, < 1 = outside, prediction blended toward the physics anchor).
- Optional: `PRED_SEED_TIMES=1` in the environment adds the per-seed times `et_seed0` ... `et_seed4`.

## Confidence, errors and output files

**Output files.** Every result is a CSV: the predictor writes one row per input row to `--out FILE` (a few `#` header
lines first unless `--no-comments`); `prepare_data.py --out` writes the input CSV; `predict_app.py --out` appends
one row per call (application level).

**Confidence (no measurement needed).** `--confidence` adds, per output (ExecTime, Mem%, Occupancy, breakdown),
`exp_err_<output>` = the error to expect and `conf_<output>` = `high` / `medium` / `low` derived from it (ExecTime:
<= 16 / <= 25 / > 25 %; Mem%: <= 6 / <= 12 / > 12 pp; occupancy: <= 3 / <= 6 / > 6 pp; breakdown: <= 2 / <= 5 / > 5 pp).
`exp_err` combines two signals:

- `gpu_distance` / `gpu_familiarity`: how far the TARGET's specs lie from the GPUs the model was trained on (balance
  space: bandwidth per SM, FP32 per byte, FP32/FP64, size; about 0 for a training GPU described with its own specs, as prepare_data.py writes them).
  near < 0.3 <= mid < 1.0 <= far.
  Examples: a training GPU 0 (near); GH200 144 GB 0.02 (near); H200 x2 0.5 (mid); H200 x8 1.9 (far).
- the disagreement between the ensemble members (`ens_cv_et`, `ens_sd_*`); within the near and mid bins its top 20% is
  the low level, the next 20% medium, the rest high. An execution time faster than the roofline gets the low level of its bin.

Calibrated on held-out predictions: the 7 GPUs the model never saw, at their distance from the GPUs each fold trained on
(H100 0.03, GH200 0.02, H200 0.17 near; GB200 0.49, A100 0.61 mid; L40S, B300 2.29 far). 4,784 kernels for time; 3,877
with a measured target profile for the other outputs. Mean error, levels high / medium / low:

| Output | near | mid | far | unit |
|---|---|---|---|---|
| ExecTime | 10.8 / 15.6 / 21.1 | 20.1 / 22.7 / 40.1 | 27.7 | % (absolute percentage error) |
| Mem% | 3.7 / 5.2 / 7.1 | 7.5 / 11.3 / 13.1 | 33.6 | percentage points |
| Occupancy | 1.4 / 1.6 / 4.1 | 4.9 / 4.3 / 5.1 | 27.4 | percentage points |
| breakdown | 0.7 / 1.9 / 3.4 | 1.3 / 3.1 / 5.9 | 17.3 | percentage points (mean over the 4 shares) |

The distance carries most of the information: the mean error grows from near to mid to far for every output. Within
near and mid the ensemble spread separates the least reliable kernels (low level; mid low can exceed far), though not
always medium from high (e.g. occupancy in mid). The far bin is calibrated on L40S and B300 for time and on L40S alone
for the other outputs, so one value is given there. The label follows `exp_err`. Other raw signals: `eta`,
`eta_violation`, `log_dist_roof`, `mem_consistency` (roofline checks). `exp_err` is a typical (mean) error, not a
guaranteed interval; beyond the trained range (far) the true error can be larger than the held-out value.

**Spread of the ensemble.** `PRED_SEED_TIMES=1` adds each ensemble member's time (`et_seed<j>`).

**Errors against measurements.** If the kernels were also measured on the target, `--truth FILE` (row-matched CSV with
`S-Execution Time [ns]`, and optionally `S-Memory Throughput [%]`, `S-Achieved Occupancy`, `S-brk_*`) adds `err_ExecTime`, `err_Mem%`,
`err_Occupancy` and `err_brk_<bucket>` (time as absolute % error, the rest in percentage points) and prints an error summary:

    python MLP_NN/mlp-nn.py --csv inputs.csv --row all --truth measured.csv --confidence --out predictions.csv

**Applications.** `examples/predict_app.py` sums kernels x launches for one profiled run on one GPU and
reports: the actual error on the base GPU; with `--target-nsys` (run measured on the target) `status = known` and the
actual error there (`model_err_target_%`);
for a target inside the trained range (`status = interpolated`) an 80% range `likely_target_ms_lo` / `_hi` from the
held-out application errors; for a target beyond it (`status = extrapolated`) no calibrated range, only the seed spread
(`*_seed_min` / `*_seed_max`).

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
  The per-SM bandwidth term is weighted by the kernel's achieved DRAM fraction on the source: full at 50% or more,
  dropped for atomics- or compute-bound kernels, so only memory-bound kernels gain from it (`AT_BUSY_MEMW`, 0 = off).
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
| `mlp-nn.py` | entry point (command line) |
| `mlpnn_core.py`, `network.py`, `data_pipeline_v2.py` | inference code |
| `v4.55/` | weights (5 seeds, `meta.pkl`) and `runtime_inputs_training.csv` (actual GPU values used in training) |
| `obsolete/<version>/` | earlier versions |


