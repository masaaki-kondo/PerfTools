# PINN v3.26: physics-informed cross-GPU kernel performance estimator (release 2026-10-22)

Same task, inputs and outputs as MLP-NN v4.55 (`MLP_NN/README.md`): one CUDA kernel profiled with ncu on a
source GPU, predicted on a real or hypothetical target GPU. The difference is training: the network is also trained
against three physics laws.

PINN v3.26 is the v3.22 inference code with new weights (`v3.26/`, 5 seeds, all kept), trained on Yoshida-san's HeCBench
profiles including the 2026-10-17 sweep (A100, H100, H200, GH200 96 GB, GB200, L40S, H100 MIG 1g/2g/4g; B300 as an
nsys-only target). Earlier releases are in `PINN/obsolete/` (weights only): `v3.25/` (2026-10-21; the first 3 of the
5 seeds) and `v3.22/` (2026-10-16, published as `PINN/v1.0`). Run one with `python PINN/pinn.py --artifact
PINN/obsolete/v3.22 ...` (current inference code and confidence calibration, so outputs differ slightly from the original
releases).

## Quick start

```
python examples/kernel_inputs/prepare_data.py --ncu <ncu files or folders> --src H200 --tgt H100,GB200 --out inputs.csv
python PINN/pinn.py --csv inputs.csv --row all --out predictions.csv
```

Options as in MLP-NN v4.55 (`--row`, `--confidence`, `--artifact`, `--runtime-inputs`; `PRED_SEED_TIMES=1` adds
`et_seed0` ... `et_seed4`). The separate in-range model is not shipped; routing to it is off.

## Inputs, spec sheet and runtime inputs

As in MLP-NN v4.55: one row per kernel built with `examples/kernel_inputs/prepare_data.py`; specs from
`data/v4/gpu_microarch_specs.csv` (datasheet values only); the actual values the model was trained with in
`v3.26/runtime_inputs_training.csv` (in the weights folder); your own on top with `--runtime-inputs FILE`. `src_gpu` must be a GPU with
ncu profiles; B300 and GH200_144GB are targets only.

## Outputs

As in MLP-NN v4.55: execution time, memory throughput, occupancy, stall breakdown `brk_*`, execution-cycle breakdown
`exec_cycles` / `cyc_*` / `running_cycles`, `issue_limited`, roofline quantities and `trust_w`; plus `model_used`.

## The three physics laws

During training, kernels are placed on a grid of hypothetical GPUs (DRAM bandwidth and compute peak each scaled up to
256x). No labels are needed there; the loss checks the model's own predictions:

1. **Roofline floor**: never faster than the roofline (memory, compute or instruction-issue limit, latency and
   one-wave floors).
2. **Monotonicity**: more bandwidth or compute is never slower than at the base point.
3. **Scaling law**: time follows bandwidth^-a x compute^-b with a + b = 1 (whole-GPU scaling exact); the exponents
   depend on what limits the kernel (bandwidth-, memory-latency- or compute-bound).

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
| ExecTime | 12.6 / 15.8 / 21.5 | 19.0 / 28.3 / 43.6 | 30.6 | % (absolute percentage error) |
| Mem% | 4.7 / 5.8 / 8.4 | 8.0 / 12.0 / 12.4 | 36.1 | percentage points |
| Occupancy | 2.2 / 2.6 / 6.3 | 4.6 / 3.5 / 8.0 | 42.8 | percentage points |
| breakdown | 1.1 / 2.4 / 4.1 | 1.4 / 4.1 / 5.9 | 14.6 | percentage points (mean over the 4 shares) |

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

    python PINN/pinn.py --csv inputs.csv --row all --truth measured.csv --confidence --out predictions.csv

**Applications.** `examples/predict_app.py` sums kernels x launches for one profiled run on one GPU and
reports: the actual error on the base GPU; with `--target-nsys` (run measured on the target) `status = known` and the
actual error there (`model_err_target_%`);
for a target inside the trained range (`status = interpolated`) an 80% range `likely_target_ms_lo` / `_hi` from the
held-out application errors; for a target beyond it (`status = extrapolated`) no calibrated range, only the seed spread
(`*_seed_min` / `*_seed_max`).

## Accuracy

Kernel-level mean absolute percentage error of execution time. Held out = leave-one-GPU-out: each GPU is never seen
in training. Inputs use the representative GPU features and the instruction-issue limit is on.

| | In-sample | H200 | H100 | A100 | GH200 | L40S | GB200 | B300 | Held-out mean |
|---|---|---|---|---|---|---|---|---|---|
| PINN v3.26 | 14.5 | 12.2 | 14.6 | 22.3 | 18.1 | 36.2 | 28.8 | 25.3 | 22.50 |

Every fold uses all 5 seeds; no seed is selected on held-out results. Application level (sum of kernel time x launches),
held out: about 21%.

## What is new

vs v3.25 (2026-10-21): 5 seeds instead of 3 (the same 3 plus 2 new ones, same recipe); confidence calibration (expected
errors and spread quantiles) and the application range refitted to the 5-seed spread.

vs the 2026-10-16 release:

- Newest training data, including the 2026-10-17 sweep.
- Clock-aware latency floor; the setting is stored in the weights' metadata.
- Busy-SM rule for hypothetical targets: per-SM bandwidth is counted over the SMs a kernel actually keeps busy.
  The per-SM bandwidth term is weighted by the kernel's achieved DRAM fraction on the source: full at 50% or more,
  dropped for atomics- or compute-bound kernels, so only memory-bound kernels gain from it (`AT_BUSY_MEMW`, 0 = off).
- Instruction-issue limit inside the model: a kernel cannot finish faster than issued warp instructions /
  (4 x SMs x clock). Output column `issue_limited`.
- New outputs `exec_cycles`, `cyc_*` and `running_cycles`; optional per-seed times (`PRED_SEED_TIMES=1`).

## Known limitations

As in MLP-NN v4.55 (L40S is the hardest GPU; the representative spec features describe a typical kernel).

## Files

| File | Content |
|---|---|
| `pinn.py` | entry point (command line) |
| `pinn_core.py`, `network.py`, `data_pipeline_v2.py` | inference code |
| `v3.26/` | weights (5 seeds, `meta.pkl`, `trust_anchor.json`) and `runtime_inputs_training.csv` (actual GPU values used in training) |
| `obsolete/<version>/` | earlier versions |
