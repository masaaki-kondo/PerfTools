# PerfTools

GPU kernel performance estimators: given one CUDA kernel profiled (ncu) on a source GPU, predict its execution time,
memory throughput, occupancy and stall breakdown on another GPU, real or hypothetical. The target GPU needs no
profile, only its specs.

| Model | Folder | Status |
|---|---|---|
| MLP-NN v4.55 | `MLP_NN/v4.55/` | current (2026-10-21) |
| PINN v3.25 (physics-informed) | `PINN/v3.25/` | current (2026-10-21) |
| MLP-NN v4.46, PINN v3.22 | `MLP_NN/obsolete/v4.46/`, `PINN/obsolete/v3.22/` | previous release (2026-10-16; PINN was `PINN/v1.0`), still runnable |
| MLP v1.5, v2.1, v4.1 | `MLP_NN/obsolete/` | obsolete, kept for reference |
| LightGBM v1.0, v1.1 | `LightGBM_model/` | see its README |

Each model folder has a README with usage, inputs, outputs and accuracy.

## Which model to use

Both current models take the same inputs and give the same outputs. MLP-NN v4.55 is slightly more accurate on held-out
GPUs (mean kernel-level error 20.9% vs 21.9%; application level about 20% vs 21%). PINN v3.25 is trained against
physics laws (roofline floor, monotonicity, scaling law). Running both and comparing is a cheap check.

## Repository layout

| Path | Content |
|---|---|
| `MLP_NN/v4.55/`, `PINN/v3.25/` | current models (predictor, inference code, weights, training runtime inputs) |
| `MLP_NN/examples/prepare_data.py` | builds the model input CSV from ncu profiles on a source GPU and the spec sheet |
| `MLP_NN/examples/example_inputs.csv` | example input |
| `extrapolate_app.py` | predicts one profiled application run on one GPU from a spec sheet (see below) |
| `MLP_NN/data_pipeline_v2.py`, `MLP_NN/mlp.py` | shared code |
| `data/v4/gpu_microarch_specs.csv` | GPU spec sheet v4 (latest; datasheet values only; used by MLP-NN v4.55 / PINN v3.25) |
| `data/gpu_microarch_specs.csv` | earlier spec sheet, unchanged (used by LightGBM and MLP v1.5) |
| `data/obsolete/gpu_microarch_specs_v3.csv` | spec sheet of the 2026-10-16 models |
| `data/dummy_gpu_specs_ship5.csv` | hypothetical GPUs: H200 with 2x to 32x resources at fixed clock |
| `MLP_NN/obsolete/`, `PINN/obsolete/` | earlier models |
| `LightGBM_model/` | LightGBM models (v1.0, v1.1) |

## GPU specs

`data/v4/gpu_microarch_specs.csv` holds datasheet values only; its `model_input` column marks each row as `yes` (a model
input), `reference` or `no (descriptive)`. Actual values (operating clock, DRAM latency, measured
bandwidths) are runtime inputs: `runtime_inputs_training.csv` in each model folder holds the values used
in training, and `--runtime-inputs FILE` applies your own on top.

## Whole applications: extrapolate_app.py

Reads one application run (ncu profile and nsys kernel times on a base GPU), predicts every kernel on a target GPU
taken from a spec sheet, and sums kernel time x launches. Output: application time, speedup vs the base GPU, anchored
time (measured base time / predicted speedup), a status (`known`, `interpolated` or `extrapolated`), an 80% range for
interpolated targets, and the seed spread. `--model mlpnn` or `--model pinn`; usage in the script's header.
