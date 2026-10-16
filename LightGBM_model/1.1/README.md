# LightGBM AI Inference Model (v1.1) - User Guide

This directory contains the standalone, fully trained AI model package (v1.1) for predicting GPU performance across different architectures (**A100, H100, and GB200**). It takes raw Nsight Compute (NCU) profiling metrics from a source GPU (e.g. A100) and predicts the execution time and stall breakdowns on a target GPU (e.g. H100 or GB200).

---

## What's New in v1.1
This release addresses feedback regarding input feature alignment, target specification leakage, and launch configuration handling:
1. **Strictly Decoupled Feature Set (72 Clean Features)**:
   - Dropped 7 legacy target-ratio features (`I-tgtspec(ratio)::*`) that represented dynamic NCU counters (`GPU Maximum Warps Per Scheduler`, `Theoretical Active Warps per SM`, `Theoretical Active Warps Per Scheduler`, `Max Cluster Size`, `Shared Memory Configuration Size`, `Block Limit Warps`, `Block Limit Barriers`). These previously required target-side NCU profiling.
   - Target ratios in v1.1 are now 100% computed from static physical microarchitecture specifications (`gpu_microarch_specs.csv`), eliminating target profiling dependencies during forward inference.
2. **Fixed Metric Prefix Routing**:
   - `Theoretical Active Warps per SM [warp]` and `Block Limit Warps [block]` are now consistently populated across both workload (`I-wl::`) and source specification (`I-srcspec::`) namespaces.
3. **Architecture-Aware Baseline Defaults**:
   - Features not present in older architectures (e.g., `Max Cluster Size` for Ampere A100) are explicitly set to their architectural default (`0.0`), preventing missing-column warnings.
4. **Multi-Launch Configuration Support**:
   - Added `--group_by_config` flag in CLI inference. When enabled, kernels with different launch parameters (Grid Size, Block Size) are evaluated independently rather than averaged by kernel name.

---

## Repository Structure & GitHub Upload

When uploading this package to GitHub or deploying it, **only the core logic and models** need to be committed.

### Core Files (Must be uploaded to GitHub)
```text
gpu_perf_analysis/
├── data/
│   └── gpu_microarch_specs.csv    <-- REQUIRED: Exact path must be `../../data/gpu_microarch_specs.csv` relative to 1.1/
├── LightGBM_model/
│   └── 1.1/
│       ├── README.md                  <-- This user guide
│       ├── extract_features.py        <-- Script to parse NCU raw files and hardware specs
│   └── AI_model/
│       ├── run_inference.py       <-- Main CLI entry point for the frontend
│       └── model/                 <-- Directory containing pre-trained LightGBM weight files (.txt) and .pkl mappings
```

### Runtime Generated Files (Do NOT upload / Mark as gitignore)
```text
LightGBM_model/
└── 1.1/
    ├── __pycache__/                   <-- Automatically generated Python bytecode cache
    ├── config.yaml                    <-- Example: Automatically generated/updated configuration file
    ├── extracted_features.csv         <-- Example: Intermediate Feature Matrix generated during inference
    └── predictions/                   <-- Example: Directory where final prediction CSVs and logs are saved
```

---

## File Explanations

1. **`AI_model/run_inference.py`**: The main executable script. Handles argument parsing, feature extraction, loads pre-trained LightGBM v1.1 models, and writes predictions.
2. **`extract_features.py`**: Reads raw NCU files and `gpu_microarch_specs.csv`. Computes hardware ratios (Target vs Source) and derives the 72 `I-*` input features.
3. **`extracted_features.csv`**: Intermediate feature matrix generated during inference. Contains metadata (`meta-kernel`, `meta-src_gpu`, `meta-tgt_gpu`) and all 72 input features.
4. **`AI_model/model/*.txt`**: Fully trained LightGBM models for the 7 predicted targets.
5. **`config.yaml`**: Automatically logged configuration parameters for tracking runs.

---

## Environment Requirements

1. **Python Version**: Python 3.8 or higher.
2. **Required Packages**:
   ```bash
   pip install pandas numpy lightgbm pyyaml
   ```
   *(Note: LightGBM CPU inference requires no GPU or CUDA dependencies.)*

---

## How to Run Inference

Ensure you are running the command from inside the `LightGBM_model/1.1` directory:

```bash
python AI_model/run_inference.py \
    --src_gpu=A100 \
    --tgt_gpu=H100 \
    --ncu_csv=../../data/v3/raw_A100.csv \
    --out=predictions/prediction.csv \
    --log=predictions/report.txt
```

### Command Line Arguments:
- `--src_gpu`: Source architecture where NCU profile was captured. **(`A100`, `H100`, or `GB200`)**
- `--tgt_gpu`: Target architecture to predict performance for. **(`A100`, `H100`, or `GB200`)**
- `--ncu_csv`: Path to directory of NCU exported `.csv` files (or a single pre-extracted wide CSV).
- `--out`: Destination path for the predicted outputs CSV.
- `--log`: Destination path for the summary text report.
- `--group_by_config` *(Optional)*: If set, kernels with distinct grid/block dimensions are preserved as separate rows instead of averaged.
