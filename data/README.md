# data

GPU spec sheets and the data files used by the models. All spec sheets share one layout: one row per spec
(`spec`, `unit`), one column per GPU, then `confidence_notes` and `source` (`v4/` also has `model_input`).

| File | Content | Used by |
|---|---|---|
| `v4/gpu_microarch_specs.csv` | latest spec sheet: datasheet values of A100, H100, H200, GH200 (96 GB and 144 GB), GB200, B300, L40S, GB10 and the H100 MIG slices; its `model_input` column marks each row `yes` (model input), `reference` or `no (descriptive)` | MLP-NN v4.55, PINN v3.26, `examples/` |
| `dummy_gpu_specs_ship5.csv` | hypothetical GPUs G1_2X ... G5_32X: H200 with 2x to 32x the SMs, DRAM bandwidth, L2 and peaks at the same clock (bandwidth scaled from 4,534 GB/s, an older H200 value; the current sheet has 4,800); also a template for describing a new GPU | `--specs` / `--target-specs` of the examples |
| `gpu_microarch_specs.csv` | earlier spec sheet (A100, H100, GB200, GB10), kept unchanged | LightGBM v1.0 / v1.1, MLP v1.5 |
| `obsolete/gpu_microarch_specs_v3.csv` | spec sheet of the 2026-10-16 release | nothing current (kept for reference) |
| `obsolete/raw_inputs_20260522.csv` | 2,486 kernel rows (A100, H100, GB200 pairs) in the wide input format of the early models | MLP v1.5, v2.1, v4.1 (obsolete) |
| `obsolete/tagged_ground_truth_20260522.csv` | the measured outputs for the same rows, columns tagged by role | MLP v1.5, v2.1, v4.1 (obsolete) |

Actual values (operating clock, DRAM latency, measured bandwidths) are not spec-sheet values: they are runtime inputs.
The values used in training are in `runtime_inputs_training.csv` inside each weights folder (`MLP_NN/v4.55/`,
`PINN/v3.26/`); pass your own with `--runtime-inputs FILE`.
