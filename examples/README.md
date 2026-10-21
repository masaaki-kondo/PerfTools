# examples

Example inputs and the two helper scripts. Run every command from the repository root. Requires numpy, pandas, torch.

| File | Content |
|---|---|
| `kernel_inputs.csv` | example input for kernel prediction: 1,007 rows (147 kernels, Yoshida-san's HeCBench profiles), each a source -> target pair between two different GPUs among A100, H100 and GB200 |
| `app_inputs/ncu/attention-cuda__894646bb.csv` | example input for application prediction: the ncu profile (long form, `ncu --csv`) of one attention-cuda run on H200 |
| `app_inputs/nsys/attention-cuda__894646bb.csv` | the nsys kernel times of the same run (launch count and measured time of every kernel) |
| `prepare_data.py` | builds the kernel input CSV from your own ncu profiles |
| `predict_app.py` | predicts one application run on one GPU |

## Kernel prediction

One row per kernel and target GPU: the kernel's ncu metrics on the source GPU (`src_gpu`), plus the spec columns of the
source and the target GPU (`tgt_gpu`). The target GPU is never profiled; it is described by its specs only.

```
python MLP_NN/mlp-nn.py --csv examples/kernel_inputs.csv --row all --confidence --out kernel_pred.csv
python PINN/pinn.py   --csv examples/kernel_inputs.csv --row 3 --out kernel_pred_row3.csv
```

Output: per kernel, the predicted execution time, memory throughput, occupancy and stall breakdown on the target GPU;
`--confidence` adds confidence levels and expected errors. Details in `MLP_NN/README.md` and `PINN/README.md`.

### Your own kernels: prepare_data.py

Builds the same CSV from ncu profiles (files or folders; long-form `ncu --csv` exports, or wide CSVs with one row per
kernel) taken on a source GPU, for one or more target GPUs:

```
python examples/prepare_data.py --ncu examples/app_inputs/ncu/attention-cuda__894646bb.csv \
    --src H200 --tgt H100,GB200 --out inputs.csv
python MLP_NN/mlp-nn.py --csv inputs.csv --row all --out pred.csv
```

- Targets are columns of `data/v4/gpu_microarch_specs.csv`. For a GPU that is not in it, pass its specs with
  `--target-specs my_gpu.csv` (same layout; `data/dummy_gpu_specs_ship5.csv` is a worked example).
- `--runtime-inputs FILE` adds actual values (e.g. the target's DRAM latency or operating clock).
- `--smem-config BYTES` sets the target's shared-memory carveout (for a new SM design).

## Application prediction: predict_app.py

Reads one application run on a base GPU (the ncu profile and the nsys kernel times), predicts every kernel on the
target GPU and sums time x launches.

```
# a GPU from the spec sheet
python examples/predict_app.py --ncu examples/app_inputs/ncu/attention-cuda__894646bb.csv \
    --nsys examples/app_inputs/nsys/attention-cuda__894646bb.csv --base H200 \
    --specs data/v4/gpu_microarch_specs.csv --gpu H100 --model mlpnn --out app_pred.csv
# a hypothetical GPU (H200 x 8)
python examples/predict_app.py --ncu examples/app_inputs/ncu/attention-cuda__894646bb.csv \
    --nsys examples/app_inputs/nsys/attention-cuda__894646bb.csv --base H200 \
    --specs data/dummy_gpu_specs_ship5.csv --gpu G3_8X --model mlpnn --out app_pred.csv
```

| Option | Meaning |
|---|---|
| `--ncu`, `--nsys` | the run's ncu profile and nsys kernel-time file on the base GPU |
| `--base` | the GPU the run was profiled on |
| `--specs`, `--gpu` | spec sheet and the target column in it |
| `--model` | `mlpnn` (MLP-NN v4.55, default) or `pinn` (PINN v3.25) |
| `--target-nsys` | optional: nsys file of the same run measured on the target, to report the actual error |
| `--out` | CSV the result row is appended to (one row per call) |

Output (one line on stdout, one row in `--out`): predicted base and target times, speedup, anchored target time
(measured base time / speedup), the seed spread, and a status: `known` (measured on the target, actual error given),
`interpolated` (inside the trained range, 80% range given) or `extrapolated` (outside it, seed spread only). The
script's header describes every column.
