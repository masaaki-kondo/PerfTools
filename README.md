# PerfTools

GPU kernel performance estimators: given one CUDA kernel profiled (ncu) on a source GPU, predict its execution time,
memory throughput, occupancy and stall breakdown on another GPU, real or hypothetical.

| Model | Folder | Status |
|---|---|---|
| MLP-NN v4.46 | `MLP_NN/v4.46/` | current (2026-10-16) |
| PINN v1.0 (physics-informed) | `PINN/v1.0/` | current (2026-10-16) |
| MLP v1.5, v2.1, v4.1 | `MLP_NN/obsolete/` | obsolete, kept for reference |
| LightGBM v1.0, v1.1 | `LightGBM_model/` | see its README |

Each model folder has a README with usage, inputs, outputs and accuracy. Shared code: `MLP_NN/data_pipeline_v2.py`,
`MLP_NN/mlp.py`. GPU specs: `data/gpu_microarch_specs_v3.csv` (specs only; used by the current models);
`data/gpu_microarch_specs.csv` is the earlier sheet used by the obsolete models and `MLP_NN/examples/prepare_data.py`.
