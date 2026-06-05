#!/bin/bash
#SBATCH --job-name=v15deliv -p genoa -N1 -n1 --cpus-per-task=16 --time=00:40:00 --output=data/v15deliv_%j.out
set -euo pipefail; cd "$SLURM_SUBMIT_DIR"; source .venv/bin/activate
EX=MLP_NN/examples

echo "### 1. train per-source v1.5 models + write specs.csv ###"
python -u MLP_NN/v1.5/v15_train.py

echo; echo "### 2. INPUT_SCHEMA.csv (provenance) ###"
python -u MLP_NN/v1.5/make_schema.py

echo; echo "### 3. prepare 20-row example input from raw NCU + spec sheet ###"
python -u MLP_NN/v1.5/prepare_input.py --raw data/20260522_wide.zip --src A100 \
  --n 20 --out $EX/kernel_stats_example.csv

echo; echo "### 4. BATCH mode -> example output ###"
python -u MLP_NN/v1.5/predict_v15.py --kernel-stats $EX/kernel_stats_example.csv \
  --out $EX/pred_example.csv --log $EX/run_example.log

echo; echo "### 5. --row mode (single row of the CSV) ###"
python -u MLP_NN/v1.5/predict_v15.py --kernel-stats $EX/kernel_stats_example.csv \
  --row 3 --out $EX/pred_row3.csv

echo; echo "### 6. SINGLE mode (one kernel via --set) ###"
python -u MLP_NN/v1.5/predict_v15.py --src A100 --tgt H100 --out $EX/pred_single.csv \
  --set "Block Size=256" --set "Grid Size=4096" --set "Threads=1048576" \
  --set "Execution Time=5000" --set "Elapsed Cycles [cycle]=120000" \
  --set "Memory Throughput [byte/s]=8.0e11" --set "Memory Throughput [%]=60" \
  --set "Achieved Occupancy=0.5"

echo; echo "--- pred_example.csv (first 8 lines) ---"; head -8 $EX/pred_example.csv
echo; echo ALLDONE