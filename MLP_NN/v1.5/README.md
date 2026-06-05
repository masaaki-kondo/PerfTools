# v1.5 cross-GPU kernel performance estimator

Estimates a CUDA kernel's metrics on a **target** GPU from its NCU profile
measured on a **source** GPU. Pre-trained, fully offline, self-contained.

This is a **deliverable** — no training step. Answers the BenchKit/CX questions:

## Language / framework
Python 3.11, single CLI. Libraries: `numpy`, `pandas`, `torch` (CPU is fine).

## Runtime restrictions
None. Any machine with Python 3.11 + numpy/pandas/torch. No GPU, no network, no DB.

## Inputs
1. **Kernel stats** — NCU profile of the kernel on the *source* GPU: a batch CSV
   (`--kernel-stats`) or a single kernel via `--set "Col=Val"`.
2. **GPU specs** — from the spec sheet (`data/gpu_specs.csv`), selected by
   `--src` / `--tgt` GPU name. The shipped example CSV also carries the spec
   columns inline (`SRC <spec>` / `TGT <spec>`), so each row is self-describing.

`INPUT_SCHEMA.csv` lists every input column and its source (NCU vs spec sheet).

## Outputs
- **`--out pred.csv`** — per kernel: `Execution Time [ns]`, `Memory Throughput [%]`,
  `Achieved Occupancy`, 4 stall-breakdown fractions, and roofline quantities
  (`t_mem_ns`, `t_comp_ns`, `t_roof_ns`, `efficiency_eta`). A `#`-comment header
  documents column sources (`--no-comments` for a pure CSV).
- **`--log run.log`** — plain-text run summary (also to stderr).

## Command line
```bash
# batch
python MLP_NN/v1.5/predict_v15.py \
    --kernel-stats MLP_NN/examples/kernel_stats_example.csv --out pred.csv --log run.log

# one row of a CSV
python MLP_NN/v1.5/predict_v15.py \
    --kernel-stats MLP_NN/examples/kernel_stats_example.csv --row 3 --out row3.csv

# single kernel via flags (specs looked up by --src/--tgt)
python MLP_NN/v1.5/predict_v15.py --src A100 --tgt GB200 --out one.csv \
    --set "Block Size=256" --set "Execution Time=5000" ...
```
`src_gpu` / `tgt_gpu` may instead be columns in the input CSV (per-row).

## Dependencies / network / DB
Python 3.11 + `numpy`, `pandas`, `torch`. Nothing else. Fully offline.

## Working directory / layout
Run from the repo root. Self-contained:
```
data/
  gpu_microarch_specs.csv   spec sheet (raw)
  gpu_specs.csv             per-GPU specs used by the tool
  (raw NCU data is large and NOT shipped — place here to regenerate examples)
MLP_NN/v1.5/                THE DELIVERABLE
  predict_v15.py            CLI
  v15_core.py               model + feature construction (no other deps)
  v15_artifact/             pre-trained per-source weights (a100/h100/gb200)
  README.md  INPUT_SCHEMA.csv
MLP_NN/examples/
  prepare_example.py        rebuild the example CSV from raw NCU + spec sheet
  kernel_stats_example.csv  20-row example (all columns)
  pred_example.csv ...      example outputs
```
The deliverable depends on nothing else in the repo.

## Known GPUs
`A100`, `H100`, `GB200` (rows in `data/gpu_specs.csv`). Add a generation by adding a row.

## Notes
v1.5 is the in-range estimator (existing GPUs); accuracy is preliminary.
Targets far beyond the training GPUs are extrapolation and less reliable.
