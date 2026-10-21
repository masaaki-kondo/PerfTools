# PINN v1.0: physics-informed cross-GPU kernel performance estimator (release 2026-10-16)

> **Obsolete (2026-10-21)**: superseded by PINN/v3.25. Released 2026-10-16 as PINN/v1.0 (= internal v3.22); kept runnable here.

Same task, inputs and outputs as MLP-NN v4.46 (`MLP_NN/obsolete/v4.46/README.md`): one CUDA kernel profiled on a source
GPU, predicted on a real or hypothetical target GPU. The difference is training: the network is also trained against
three physics laws.

PINN v1.0 is the first public PINN release: the internal v3.22 inference code with the weights of the internal v3.21
production training run (`v322_artifact/`, 3 seeds). The inference code includes the physics-guarded handling of
distant GPU specs shared with MLP-NN v4.46.

## Run

```
python PINN/obsolete/v3.22/predict_v322.py --csv inputs.csv --row all --out predictions.csv
```

Options as in MLP-NN v4.46 (`--row`, `--confidence`, `--artifact`, `--runtime-inputs`). The separate in-range model
is not shipped; routing to it is off.

## The three physics laws

During training, each of 399 kernels is placed on a 10 x 10 grid of hypothetical GPUs (DRAM bandwidth and compute
peak each scaled 1, 2, 3, 5, 8, 16, 32, 64, 128, 256x). No labels are needed there; the loss checks the model's own
predictions (weight 0.5 against the data loss, from epoch 20):

1. **Roofline floor**: never faster than the roofline (memory, compute or instruction-issue limit, latency and
   one-wave floors).
2. **Monotonicity**: more bandwidth or compute is never slower than at the base point.
3. **Scaling law**: time follows bandwidth^-a x compute^-b with a + b = 1 (whole-GPU scaling exact). The exponents,
   fitted on measured cross-GPU pairs, depend on what limits the kernel: bandwidth-bound a = 0.65, b = 0.35;
   memory-latency-bound 0.43 / 0.57; compute-bound 0.29 / 0.71.

## Spec sheet and runtime inputs

As in MLP-NN v4.46: `data/obsolete/gpu_microarch_specs_v3.csv` holds specs only; `runtime_inputs_training.csv` in this folder
holds the actual values the model was trained with; `--runtime-inputs FILE` applies your own on top.

## Accuracy

Leave-one-GPU-out, execution-time MAPE against nsys run time: H200 12.1%, H100 15.9%, A100 21.1%, GH200 18.4%,
L40S 47.9%, GB200 31.0%, B300 25.4%. In-sample: 14.4%.

## Known limitations

As in MLP-NN v4.46 (L40S; the five profile-derived GPU features).
