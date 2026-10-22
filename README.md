# PerfTools

GPU kernel performance estimators: given one CUDA kernel profiled (ncu) on a source GPU, predict its execution time,
memory throughput, occupancy and stall breakdown on another GPU, real or hypothetical. The target GPU needs no
profile, only its specs.

| Model | Folder | Status |
|---|---|---|
| LightGBM v1.0, v1.1 | `LightGBM_model/` | see its README |
| MLP-NN v4.55 | `MLP_NN/` | current (2026-10-21) |
| PINN v3.26 (physics-informed) | `PINN/` | current (2026-10-22) |

Each model folder has a README with usage, inputs, outputs and accuracy.

## Which model to use

Both current models take the same inputs and give the same outputs. MLP-NN v4.55 is slightly more accurate on held-out
GPUs (mean kernel-level error 20.9% vs 22.5%; application level about 20% vs 21%). PINN v3.26 is trained against
physics laws (roofline floor, monotonicity, scaling law). Running both and comparing is a cheap check.

## Examples

Kernel prediction (one row per kernel and target GPU; the example holds kernels profiled on A100, H100 and GB200):

```
python MLP_NN/mlp-nn.py --csv examples/kernel_inputs/kernel_inputs.csv --row all --confidence --out kernel_pred.csv
```

Application prediction (one attention-cuda run profiled on H200, predicted on H100 and on a hypothetical H200 x 8):

```
python examples/predict_app.py --ncu examples/app_inputs/ncu/attention-cuda__894646bb.csv \
    --nsys examples/app_inputs/nsys/attention-cuda__894646bb.csv --base H200 \
    --specs data/v4/gpu_microarch_specs.csv --gpu H100 --model mlpnn --out app_pred.csv
python examples/predict_app.py --ncu examples/app_inputs/ncu/attention-cuda__894646bb.csv \
    --nsys examples/app_inputs/nsys/attention-cuda__894646bb.csv --base H200 \
    --specs data/dummy_gpu_specs_ship5.csv --gpu G3_8X --model mlpnn --out app_pred.csv
```

Your own kernels: build the input CSV from ncu profiles with `examples/kernel_inputs/prepare_data.py`. Details in `examples/README.md`.

## Repository layout

| Path | Content |
|---|---|
| `LightGBM_model/` | LightGBM models (v1.0, v1.1) |
| `MLP_NN/`, `PINN/` | current models: entry point (`MLP_NN/mlp-nn.py`, `PINN/pinn.py`), inference code, README |
| `MLP_NN/v4.55/`, `PINN/v3.26/` | weights of the current versions, with the runtime inputs used in training |
| `examples/` | `kernel_inputs/` (example kernel input, and `prepare_data.py` to build one from your ncu profiles), `app_inputs/` (example application run) and `predict_app.py` (one application run on one GPU); see `examples/README.md` |
| `data/` | GPU spec sheets (latest: `data/v4/gpu_microarch_specs.csv`), hypothetical GPUs and older data; see `data/README.md` |
| `MLP_NN/obsolete/`, `PINN/obsolete/` | earlier versions |

## GPU specs

`data/v4/gpu_microarch_specs.csv` holds datasheet values only; its `model_input` column marks each row as `yes` (a model
input), `reference` or `no (descriptive)`. Actual values (operating clock, DRAM latency, measured
bandwidths) are runtime inputs: `runtime_inputs_training.csv` in each weights folder holds the values used
in training, and `--runtime-inputs FILE` applies your own on top.

## Whole applications: examples/predict_app.py

Reads one application run (ncu profile and nsys kernel times on a base GPU), predicts every kernel on a target GPU
taken from a spec sheet, and sums kernel time x launches. Output: application time, speedup vs the base GPU, anchored
time (measured base time / predicted speedup), a status (`known`, `interpolated` or `extrapolated`), an 80% range for
interpolated targets, and the seed spread. `--model mlpnn` or `--model pinn`; usage in the script's header.
