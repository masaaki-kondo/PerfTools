# v1.5 cross-GPU kernel performance estimator

Estimates a CUDA kernel's metrics on a **target** GPU from its NCU profile
measured on a **source** GPU. Per-source MLP models with roofline input
features. Fully offline.

This README answers the BenchKit/CX integration questions directly.

## Language / framework
Python 3.11, single CLI script. Libraries: `numpy`, `pandas`, `torch` (CPU is fine).

## Runtime restrictions
None. Any machine with Python 3.11 + numpy/pandas/torch. No GPU needed at
inference. No network, no external database.

## Required inputs
1. **Kernel stats** — NCU profile of the kernel on the *source* GPU. Either:
   - batch CSV (`--kernel-stats kernels.csv`, one row per kernel), or
   - a single kernel via `--set "Col=Val"` flags.
   Columns required are listed in `INPUT_SCHEMA.csv` (all from NCU, source GPU).
2. **GPU names** — `--src` / `--tgt` (e.g. `A100`, `H100`, `GB200`).
3. **Static spec file** — `specs.csv` (ships with the tool; one row per GPU,
   derived from the spec sheet `data/gpu_microarch_specs.csv`). Kept **static
   per generation** — not collected at runtime.

## Outputs
- **`--out pred.csv`** — one row per kernel: predicted `Execution Time [ns]`,
  `Memory Throughput [%]`, `Achieved Occupancy`, 4 stall-breakdown fractions,
  plus roofline quantities (`t_mem_ns`, `t_comp_ns`, `t_roof_ns`,
  `efficiency_eta`). A `#`-comment header documents each column's source
  (suppress with `--no-comments` for a pure CSV).
- **`--log run.log`** — plain-text run summary (also echoed to stderr).

## Command line
```bash
# batch
python MLP_NN/v1.5/predict_v15.py \
    --kernel-stats MLP_NN/examples/kernel_stats_example.csv \
    --out pred.csv --log run.log

# one row of a CSV
python MLP_NN/v1.5/predict_v15.py \
    --kernel-stats MLP_NN/examples/kernel_stats_example.csv --row 3 --out row3.csv

# single kernel via flags
python MLP_NN/v1.5/predict_v15.py --src A100 --tgt GB200 --out one.csv \
    --set "Block Size=256" --set "Execution Time=5000" ...
```
`src_gpu` / `tgt_gpu` may instead be columns in the input CSV (per-row).

## Dependencies
Python 3.11 with `numpy`, `pandas`, `torch`. Nothing else.

## Network / DB
None. Fully offline.

## Working directory / layout
Run from the repo root. Flat layout:
```
data/                         spec sheet (gpu_microarch_specs.csv); raw NCU data
                              is NOT shipped — data/ is otherwise empty
MLP_NN/                       model code (imported by the CLI)
MLP_NN/v1.5/                  this tool
  predict_v15.py              the CLI
  v15_train.py                (re)train the models  [needs raw data in data/]
  prepare_input.py            build a kernel-stats CSV from raw NCU + spec sheet
  make_schema.py              (re)write INPUT_SCHEMA.csv
  specs.csv                   static per-GPU spec file
  INPUT_SCHEMA.csv            provenance of every input column
  v15_artifact/               pre-trained weights (model_<src>.pt, stats, meta)
MLP_NN/examples/              20-row example input + example outputs
```
Inference needs only `MLP_NN/` (code + `v1.5/v15_artifact` + `v1.5/specs.csv`)
and the examples. Retraining (`v15_train.py`) additionally needs the raw NCU
data placed in `data/`.

## Known GPUs
`A100`, `H100`, `GB200` (rows in `specs.csv`). Add a generation by adding a row.

## Notes
- v1.5 = the in-range estimator (existing GPUs). Accuracy is preliminary.
- Predictions for a target far beyond the training GPUs are extrapolation and
  less reliable (a separate roofline-bounded variant, v4, addresses that).
