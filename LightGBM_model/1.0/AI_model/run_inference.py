"""
run_inference.py
================
Main CLI entry point for the frontend team.
Loads pre-trained LightGBM models and runs inference on NCU data.

Usage (run from LightGBM_model/ directory):
    python AI_model/run_inference.py \\
        --src_gpu=A100 --tgt_gpu=H100 \\
        --ncu_csv=../data/v3/raw_A100.csv \\
        --out=predictions/prediction.csv \\
        --log=predictions/report.txt
"""
import os
import sys
import argparse
import pickle
import re
import yaml
import pandas as pd
import numpy as np

# Add LightGBM_model root to path so we can import extract_features
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import extract_features


def clean_col(name):
    return re.sub('[^A-Za-z0-9_]+', '_', name)


OUTPUT_COLS = [
    'O-Execution Time',
    'O-Memory Throughput [%]',
    'O-Achieved Occupancy',
    'O-breakdown_memory',
    'O-breakdown_pipeline_contention',
    'O-breakdown_sync',
    'O-breakdown_scheduling_overhead'
]


def main():
    parser = argparse.ArgumentParser(description="Run LightGBM Inference for GPU Performance Prediction.")
    parser.add_argument('--src_gpu', type=str, required=True, help="Source GPU (e.g. A100, H100, GB200)")
    parser.add_argument('--tgt_gpu', type=str, required=True, help="Target GPU (e.g. A100, H100, GB200)")
    parser.add_argument('--ncu_csv', type=str, required=True, help="Path to NCU raw CSV file or directory")
    parser.add_argument('--out', type=str, required=True, help="Path to output prediction CSV")
    parser.add_argument('--log', type=str, required=True, help="Path to output log/report TXT")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, 'AI_model', 'model')
    specs_csv = os.path.join(base_dir, '..', '..', 'data', 'gpu_microarch_specs.csv')
    features_csv = os.path.join(base_dir, 'extracted_features.csv')

    # Update config.yaml
    config_path = os.path.join(base_dir, 'config.yaml')
    config = {
        'src_gpu': args.src_gpu,
        'tgt_gpu': args.tgt_gpu,
        'ncu_csv_path': os.path.abspath(args.ncu_csv),
        'output_csv': args.out,
        'output_log': args.log,
        'gpu_specs_csv': specs_csv,
        'model_dir': model_dir,
    }
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)

    # ---------------------------------------------------------------
    # Step 1: Extract Features
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[Step 1] Extracting Features from NCU Data")
    print("=" * 60)
    try:
        df_features = extract_features.build_inference_features(
            ncu_path=args.ncu_csv,
            specs_csv=specs_csv,
            src_gpu=args.src_gpu,
            tgt_gpu=args.tgt_gpu,
            output_csv=features_csv
        )
    except Exception as e:
        print(f"ERROR during feature extraction: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    if len(df_features) == 0:
        print("ERROR: No kernels were extracted. Check the NCU CSV path.")
        sys.exit(1)

    # ---------------------------------------------------------------
    # Step 2: Load Saved Models and Feature Map
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[Step 2] Loading Pre-Trained Models")
    print("=" * 60)

    # Load the feature names and clean map that were saved during training
    feature_names_path = os.path.join(model_dir, 'feature_names.pkl')
    clean_map_path = os.path.join(model_dir, 'clean_map.pkl')

    with open(feature_names_path, 'rb') as f:
        expected_features = pickle.load(f)  # List of cleaned feature names
    with open(clean_map_path, 'rb') as f:
        clean_map = pickle.load(f)  # Dict: original col -> cleaned col

    print(f"  Model expects {len(expected_features)} features.")

    # Build X matrix: align extracted features to match model's expected columns
    # The clean_map maps original I-* column names -> cleaned names
    reverse_clean_map = {v: k for k, v in clean_map.items()}

    # Create a DataFrame with all expected features, filled with NaN for missing ones
    X = pd.DataFrame(index=df_features.index)
    missing_features = []
    for clean_name in expected_features:
        original_name = reverse_clean_map.get(clean_name)
        if original_name and original_name in df_features.columns:
            X[clean_name] = df_features[original_name].values
        else:
            X[clean_name] = 0.0  # Fill missing features with 0
            missing_features.append(clean_name)

    if missing_features:
        print(f"  WARNING: {len(missing_features)} features not found in extracted data (filled with 0):")
        for mf in missing_features[:10]:
            print(f"    - {mf}")
        if len(missing_features) > 10:
            print(f"    ... and {len(missing_features) - 10} more")

    # ---------------------------------------------------------------
    # Step 3: Run Inference
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[Step 3] Running Inference")
    print("=" * 60)

    model_files = {
        'O-Execution Time': 'lgbm_Execution_Time.txt',
        'O-Memory Throughput [%]': 'lgbm_Memory_Throughput_pct.txt',
        'O-Achieved Occupancy': 'lgbm_Achieved_Occupancy.txt',
        'O-breakdown_memory': 'lgbm_breakdown_memory.txt',
        'O-breakdown_pipeline_contention': 'lgbm_breakdown_pipeline_contention.txt',
        'O-breakdown_sync': 'lgbm_breakdown_sync.txt',
        'O-breakdown_scheduling_overhead': 'lgbm_breakdown_scheduling_overhead.txt',
    }

    import lightgbm as lgb

    for o_col in OUTPUT_COLS:
        df_features[o_col] = np.nan

    for o_col, model_filename in model_files.items():
        model_path = os.path.join(model_dir, model_filename)
        if not os.path.exists(model_path):
            print(f"  WARNING: Model file not found: {model_filename}. Skipping {o_col}.")
            continue

        model = lgb.Booster(model_file=model_path)
        preds = model.predict(X)

        # Post-process Execution Time (model predicts ratio, multiply by src exec time)
        if 'Execution Time' in o_col:
            src_exec_clean_name = clean_map.get('I-derived::src_raw_exec_time_ns', 'I_derived_src_raw_exec_time_ns')
            if src_exec_clean_name in X.columns:
                preds = preds * X[src_exec_clean_name].values
            else:
                # Fallback: try original name
                if 'I-derived::src_raw_exec_time_ns' in df_features.columns:
                    preds = preds * df_features['I-derived::src_raw_exec_time_ns'].values

        # Clip
        if 'breakdown' in o_col:
            preds = np.clip(preds, 0.0, 1.0)
        elif 'Throughput' in o_col or 'Occupancy' in o_col:
            preds = np.clip(preds, 0.0, 100.0)

        df_features[o_col] = preds
        print(f"  {o_col:45s} mean={np.nanmean(preds):.4f}")

    # ---------------------------------------------------------------
    # Step 4: Save Outputs
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[Step 4] Saving Outputs")
    print("=" * 60)

    # Output CSV: metadata + O-* columns only
    out_cols = ['meta-kernel', 'meta-src_gpu', 'meta-tgt_gpu'] + OUTPUT_COLS
    df_out = df_features[[c for c in out_cols if c in df_features.columns]]

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)
    df_out.to_csv(args.out, index=False)
    print(f"  Predictions saved to: {args.out}")

    # Log report
    log_lines = [
        "=" * 60,
        "      GPU PERFORMANCE PREDICTIONS",
        "=" * 60,
        f"Source GPU: {args.src_gpu.upper()}",
        f"Target GPU: {args.tgt_gpu.upper()}",
        f"NCU Data:   {os.path.abspath(args.ncu_csv)}",
        f"Number of Kernels: {len(df_out)}",
        "",
        "--- Mean Predictions Across All Kernels ---",
    ]
    for o_col in OUTPUT_COLS:
        if o_col in df_out.columns:
            mean_val = df_out[o_col].mean()
            if 'Execution Time' in o_col:
                log_lines.append(f"  {o_col:45s} : {mean_val:,.0f} ns")
            elif 'breakdown' in o_col:
                log_lines.append(f"  {o_col:45s} : {mean_val:.4f} ({mean_val*100:.2f}%)")
            else:
                log_lines.append(f"  {o_col:45s} : {mean_val:.2f}%")

    log_lines.append("")
    log_lines.append("--- Per-Kernel Predictions (Top 10) ---")
    for _, row in df_out.head(10).iterrows():
        kname = row.get('meta-kernel', '?')
        if len(kname) > 60:
            kname = kname[:57] + '...'
        log_lines.append(f"\n  Kernel: {kname}")
        for o_col in OUTPUT_COLS:
            if o_col in row:
                val = row[o_col]
                if 'Execution Time' in o_col:
                    log_lines.append(f"    {o_col:43s} : {val:,.0f} ns")
                elif 'breakdown' in o_col:
                    log_lines.append(f"    {o_col:43s} : {val:.4f}")
                else:
                    log_lines.append(f"    {o_col:43s} : {val:.2f}%")

    log_lines.append("\n" + "=" * 60)

    log_text = "\n".join(log_lines)

    os.makedirs(os.path.dirname(args.log) if os.path.dirname(args.log) else '.', exist_ok=True)
    with open(args.log, 'w') as f:
        f.write(log_text)

    print(f"  Report saved to: {args.log}")
    print("\nDone!")


if __name__ == "__main__":
    main()
