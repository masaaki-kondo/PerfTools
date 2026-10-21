# PINN v3.25: physics-informed cross-GPU kernel performance estimator (release 2026-10-21)

Same task, inputs and outputs as MLP-NN v4.55 (`MLP_NN/v4.55/README.md`): one CUDA kernel profiled with ncu on a
source GPU, predicted on a real or hypothetical target GPU. The difference is training: the network is also trained
against three physics laws.

PINN v3.25 is the v3.22 inference code with new weights (`v325_artifact/`, 3 seeds), trained on Yoshida-san's HeCBench
profiles including the 2026-10-17 sweep (A100, H100, H200, GH200 96 GB, GB200, L40S, H100 MIG 1g/2g/4g; B300 as an
nsys-only target). The previous release (2026-10-16, published as `PINN/v1.0`) is now `PINN/obsolete/v3.22/`, named
after its internal version; it still runs and its outputs are unchanged.

## Quick start

```
python MLP_NN/examples/prepare_data.py --ncu <ncu files or folders> --src H200 --tgt H100,GB200 --out inputs.csv
python PINN/v3.25/predict_v325.py --csv inputs.csv --row all --out predictions.csv
```

Options as in MLP-NN v4.55 (`--row`, `--confidence`, `--artifact`, `--runtime-inputs`; `PRED_SEED_TIMES=1` adds
`et_seed0` ... `et_seed2`). The separate in-range model is not shipped; routing to it is off.

## Inputs, spec sheet and runtime inputs

As in MLP-NN v4.55: one row per kernel built with `MLP_NN/examples/prepare_data.py`; specs from
`data/v4/gpu_microarch_specs.csv` (datasheet values only); the actual values the model was trained with in
`runtime_inputs_training.csv` (this folder); your own on top with `--runtime-inputs FILE`. `src_gpu` must be a GPU with
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
lines first unless `--no-comments`); `prepare_data.py --out` writes the input CSV; `extrapolate_app.py --out` appends
one row per call (application level).

**Confidence (no measurement needed).** `--confidence` adds, per output (ExecTime, Mem%, Occupancy, breakdown),
`conf_<output>` = `high` / `medium` / `low` and `exp_err_<output>` = the error to expect at that level, plus the raw
signals (`ens_cv_et`, `ens_sd_*` = disagreement between the ensemble members; `eta`, `eta_violation`, `log_dist_roof`,
`mem_consistency` = roofline checks; an execution time faster than the roofline is always `low`). The levels and errors are
calibrated on held-out predictions: the 7 GPUs the model never saw (4,784 kernels for time; 3,877 with a measured target
profile for the other outputs). Mean error per level:

| Output | high | medium | low | unit |
|---|---|---|---|---|
| ExecTime | 17.8 | 24.5 | 33.3 | % (absolute percentage error) |
| Mem% | 6.2 | 12.5 | 35.9 | percentage points |
| Occupancy | 3.2 | 6.4 | 42.0 | percentage points |
| breakdown | 1.5 | 4.8 | 12.3 | percentage points (mean over the 4 shares) |

For this model the levels rank the error for every output: `low` carries about 2x (time) to 13x (occupancy) the error
of `high`. `exp_err` is a typical (mean) error, not a guaranteed interval. The thresholds were set with the held-out models'
signals and are applied unchanged to the production weights.

**Spread of the ensemble.** `PRED_SEED_TIMES=1` adds each ensemble member's time (`et_seed<j>`).

**Errors against measurements.** If the kernels were also measured on the target, `--truth FILE` (row-matched CSV with
`S-Execution Time [ns]`, and optionally `S-Memory Throughput [%]`, `S-Achieved Occupancy`, `S-brk_*`) adds `err_ExecTime`, `err_Mem%`,
`err_Occupancy` and `err_brk_<bucket>` (time as absolute % error, the rest in percentage points) and prints an error summary:

    python PINN/v3.25/predict_v325.py --csv inputs.csv --row all --truth measured.csv --confidence --out predictions.csv

**Applications.** `extrapolate_app.py` (repository root) sums kernels x launches for one profiled run on one GPU and
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
| PINN v3.25 | 14.7 | 12.2 | 14.2 | 22.1 | 18.5 | 31.3 | 29.0 | 26.1 | 21.92 |

Application level (sum of kernel time x launches), held out: about 21%.
The L40S fold uses the best 3 of 5 seeds, chosen on that fold, so the L40S figure is optimistic.

## What is new (vs the 2026-10-16 release)

- Newest training data, including the 2026-10-17 sweep.
- Clock-aware latency floor; the setting is stored in the weights' metadata.
- Busy-SM rule for hypothetical targets: per-SM bandwidth is counted over the SMs a kernel actually keeps busy.
- Instruction-issue limit inside the model: a kernel cannot finish faster than issued warp instructions /
  (4 x SMs x clock). Output column `issue_limited`.
- New outputs `exec_cycles`, `cyc_*` and `running_cycles`; optional per-seed times (`PRED_SEED_TIMES=1`).

## Known limitations

As in MLP-NN v4.55 (L40S is the hardest GPU; the representative spec features describe a typical kernel).

## Files

| File | Content |
|---|---|
| `predict_v325.py` | predictor (command line) |
| `v325_core.py`, `v15_core.py` | inference code |
| `v325_artifact/` | weights, 3 seeds, `meta.pkl` and `trust_anchor.json` |
| `runtime_inputs_training.csv` | actual GPU values used in training |
