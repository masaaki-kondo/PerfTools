"""
extract_features.py  (LightGBM_model)
======================================
Adapted from extract_and_build_cv5.py — the battle-tested NCU extraction pipeline.
This version is stripped down for INFERENCE ONLY:
  - Reads NCU raw files for ONE source GPU only
  - Constructs the full I-* feature matrix (I-kcfg, I-wl, I-srcspec, I-tgtspec(ratio), I-derived)
  - Does NOT produce S-* or O-* columns
  - Output: extracted_features.csv
"""
import os
import glob
import gc
import pandas as pd
import numpy as np

# ==============================================================================
# load_hardware_specs  (EXACT COPY from extract_and_build_cv5.py)
# ==============================================================================
def load_hardware_specs(specs_csv):
    """Load and transpose the GPU microarchitecture specs."""
    print(f"Loading hardware specifications from {specs_csv}")
    specs_df = pd.read_csv(specs_csv)
    specs_df = specs_df.dropna(subset=['spec'])
    specs_df = specs_df.set_index('spec')

    gpus_to_keep = ['A100', 'H100', 'GB200']
    existing_gpus = [c for c in gpus_to_keep if c in specs_df.columns]

    gpu_specs = {}
    for gpu in existing_gpus:
        gpu_dict = {}
        for spec_name in specs_df.index:
            raw_val = specs_df.loc[spec_name, gpu]
            try:
                gpu_dict[spec_name] = float(raw_val)
            except (ValueError, TypeError):
                gpu_dict[spec_name] = raw_val

        # Compute derived features (EXACT same logic as extract_and_build_cv5.py)
        try:
            if 'peak_fp16_tensor' in gpu_dict:
                gpu_dict['Peak FP16 Tensor [flop/s]'] = gpu_dict['peak_fp16_tensor'] * 1e12
            if 'peak_fp32' in gpu_dict and 'peak_fp64' in gpu_dict and gpu_dict['peak_fp64'] != 0:
                gpu_dict['Peak FP32 / Peak FP64'] = gpu_dict['peak_fp32'] / gpu_dict['peak_fp64']
            if 'peak_fp16_tensor' in gpu_dict and 'peak_fp32' in gpu_dict and gpu_dict['peak_fp32'] != 0:
                gpu_dict['Peak Tensor / Peak FP32'] = gpu_dict['peak_fp16_tensor'] / gpu_dict['peak_fp32']
            if 'peak_fp32' in gpu_dict and 'dram_bandwidth' in gpu_dict and gpu_dict['dram_bandwidth'] != 0:
                gpu_dict['Peak FP32 / DRAM BW [flop/byte]'] = (gpu_dict['peak_fp32'] * 1e12) / (gpu_dict['dram_bandwidth'] * 1e9)
            if 'dram_bandwidth' in gpu_dict and 'num_sms' in gpu_dict and gpu_dict['num_sms'] != 0:
                gpu_dict['DRAM BW per SM [byte/s/SM]'] = (gpu_dict['dram_bandwidth'] * 1e9) / gpu_dict['num_sms']
            if 'l2_cache_total' in gpu_dict and 'num_sms' in gpu_dict and gpu_dict['num_sms'] != 0:
                gpu_dict['L2 Size per SM [byte/SM]'] = (gpu_dict['l2_cache_total'] * 1e6) / gpu_dict['num_sms']
            if 'cpu_gpu_bandwidth' in gpu_dict:
                gpu_dict['CPU-GPU BW [byte/s]'] = gpu_dict['cpu_gpu_bandwidth'] * 1e9
            if 'dram_latency_model_recommended' in gpu_dict:
                gpu_dict['DRAM Latency [ns]'] = gpu_dict['dram_latency_model_recommended']
        except Exception:
            pass

        gpu_specs[gpu] = gpu_dict

    return gpu_specs


# ==============================================================================
# extract_ncu_data  (EXACT COPY from extract_and_build_cv5.py)
# ==============================================================================
def extract_ncu_data(gpu_name, ncu_path):
    """
    Parse NCU CSV files for a single GPU and return a DataFrame of aggregated metrics.
    Supports TWO formats:
      1. NCU long format: columns include 'Kernel Name', 'Metric Name', 'Metric Value'
      2. Pre-extracted wide format: 'Kernel Name' as first column, metrics as column headers
    """
    print(f"\nExtracting NCU data for {gpu_name} from {ncu_path}...")

    # Support both single file and directory
    if os.path.isfile(ncu_path):
        valid_files = [ncu_path]
    else:
        all_files = glob.glob(os.path.join(ncu_path, "*.csv"))
        valid_files = [f for f in all_files if "memory_workload_analysis" not in f and "source_counters" not in f]

    print(f"  Found {len(valid_files)} primary NCU CSV files.")

    # --- Auto-detect format by peeking at the first file ---
    if valid_files:
        peek_df = pd.read_csv(valid_files[0], nrows=2, dtype=str)
        is_wide_format = ('Kernel Name' in peek_df.columns and 'Metric Name' not in peek_df.columns)
    else:
        is_wide_format = False

    if is_wide_format:
        # ===== WIDE FORMAT: Already extracted (e.g. raw_A100.csv) =====
        print(f"  Detected WIDE format (pre-extracted). Reading directly...")
        all_dfs = []
        for f in valid_files:
            try:
                df = pd.read_csv(f)
                # Clean comma-separated numbers in all numeric columns
                for col in df.columns:
                    if col == 'Kernel Name':
                        continue
                    if df[col].dtype == object:
                        df[col] = df[col].str.replace(',', '', regex=False)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                all_dfs.append(df)
            except Exception as e:
                print(f"    [-] Skipped {os.path.basename(f)}: {e}")

        if not all_dfs:
            print(f"  [!] No data extracted for {gpu_name}.")
            return None

        master_df = pd.concat(all_dfs, ignore_index=True)
        master_df = master_df.groupby('Kernel Name', as_index=False).mean(numeric_only=True)
        # Re-add Kernel Name if lost
        if 'Kernel Name' not in master_df.columns:
            knames = pd.concat(all_dfs, ignore_index=True)['Kernel Name']
            master_df['Kernel Name'] = knames.groupby(knames).first().values

    else:
        # ===== LONG FORMAT: Original NCU CSVs =====
        print(f"  Detected LONG format (original NCU). Parsing...")
        all_dfs = []

        for idx, f in enumerate(valid_files):
            try:
                chunk_iter = pd.read_csv(f, chunksize=10000, dtype=str)
                sum_df = None
                count_df = None

                for chunk in chunk_iter:
                    req_cols = ['Kernel Name', 'Metric Name', 'Metric Unit', 'Metric Value']
                    if not all(c in chunk.columns for c in req_cols):
                        continue

                    chunk['Metric Value'] = chunk['Metric Value'].replace(['', 'N/A', 'nan', 'NaN'], np.nan)

                    # Strip commas before converting (CRITICAL FIX from earlier debugging)
                    if hasattr(chunk['Metric Value'], 'str'):
                        chunk['Metric Value'] = chunk['Metric Value'].str.replace(',', '', regex=False)

                    chunk['Metric Value'] = pd.to_numeric(chunk['Metric Value'], errors='coerce')

                    chunk['Metric Unit'] = chunk['Metric Unit'].fillna('')
                    chunk['Metric_Header'] = chunk['Metric Name'] + ' [' + chunk['Metric Unit'] + ']'
                    chunk['Metric_Header'] = chunk['Metric_Header'].str.replace(' []', '', regex=False)

                    agg_sum = chunk.groupby(['Kernel Name', 'Metric_Header'], as_index=False)['Metric Value'].sum(min_count=1)
                    agg_count = chunk.groupby(['Kernel Name', 'Metric_Header'], as_index=False)['Metric Value'].count()

                    if sum_df is None:
                        sum_df = agg_sum
                        count_df = agg_count
                    else:
                        sum_df = pd.concat([sum_df, agg_sum]).groupby(['Kernel Name', 'Metric_Header'], as_index=False)['Metric Value'].sum(min_count=1)
                        count_df = pd.concat([count_df, agg_count]).groupby(['Kernel Name', 'Metric_Header'], as_index=False)['Metric Value'].sum(min_count=1)

                    del chunk

                if sum_df is None:
                    continue

                mean_df = sum_df.copy()
                mean_df['Metric Value'] = sum_df['Metric Value'] / count_df['Metric Value']

                wide_df = mean_df.pivot_table(index='Kernel Name', columns='Metric_Header', values='Metric Value').reset_index()
                all_dfs.append(wide_df)

                del mean_df, sum_df, count_df, wide_df
                gc.collect()

                if (idx + 1) % 20 == 0:
                    print(f"    Processed {idx + 1}/{len(valid_files)} files...")

            except Exception as e:
                print(f"    [-] Skipped {os.path.basename(f)}: {e}")

        if not all_dfs:
            print(f"  [!] No data extracted for {gpu_name}.")
            return None

        master_df = pd.concat(all_dfs, ignore_index=True)
        master_df = master_df.groupby('Kernel Name', as_index=False).mean(numeric_only=True)
        if 'Kernel Name' not in master_df.columns:
            master_df = pd.concat(all_dfs, ignore_index=True)
            master_df = master_df.groupby('Kernel Name', as_index=False).first()

    master_df['Source_GPU'] = gpu_name

    # Calculate Breakdown Stalls (EXACT same logic as extract_and_build_cv5.py)
    def safe_get(df, cols):
        existing = [c for c in cols if c in df.columns]
        if not existing:
            return pd.Series(0.0, index=df.index)
        return df[existing].fillna(0).sum(axis=1)

    base_stall_names = [
        "Stall Barrier", "Stall Branch Resolving", "Stall Dispatch Stall", "Stall Drain",
        "Stall LG Throttle", "Stall Long Scoreboard", "Stall MIO Throttle", "Stall Math Pipe Throttle",
        "Stall Membar", "Stall Misc", "Stall No Instruction", "Stall Not Selected",
        "Stall Short Scoreboard", "Stall Sleeping", "Stall Tex Throttle", "Stall Wait"
    ]
    exact_stall_cols = [f"{s} [inst]" for s in base_stall_names]
    total_stall = safe_get(master_df, exact_stall_cols)

    comp_stalls = [f"{s} [inst]" for s in [
        "Stall Math Pipe Throttle", "Stall Wait", "Stall No Instruction",
        "Stall Branch Resolving", "Stall Dispatch Stall"
    ]]
    master_df['breakdown_pipeline_contention'] = safe_get(master_df, comp_stalls) / total_stall

    mem_stalls = [f"{s} [inst]" for s in [
        "Stall Long Scoreboard", "Stall Short Scoreboard", "Stall LG Throttle",
        "Stall MIO Throttle", "Stall Tex Throttle", "Stall Drain"
    ]]
    master_df['breakdown_memory'] = safe_get(master_df, mem_stalls) / total_stall

    sync_stalls = [f"{s} [inst]" for s in ["Stall Barrier", "Stall Membar"]]
    master_df['breakdown_sync'] = safe_get(master_df, sync_stalls) / total_stall

    exec_stalls = [f"{s} [inst]" for s in ["Stall Not Selected", "Stall Sleeping", "Stall Misc"]]
    master_df['breakdown_scheduling_overhead'] = safe_get(master_df, exec_stalls) / total_stall

    master_df.loc[total_stall == 0, ['breakdown_pipeline_contention', 'breakdown_memory',
                                      'breakdown_sync', 'breakdown_scheduling_overhead']] = np.nan

    print(f"  Extracted {len(master_df)} unique kernels.")
    return master_df


# ==============================================================================
# build_inference_features  (Adapted from build_unified_dataset)
# ==============================================================================
def build_inference_features(ncu_path, specs_csv, src_gpu, tgt_gpu, output_csv):
    """
    Build the extracted_features.csv for inference.
    Only processes ONE src_gpu's NCU data, paired with ONE tgt_gpu.
    No S-* or O-* columns are generated.
    """
    gpu_specs = load_hardware_specs(specs_csv)
    if gpu_specs is None:
        raise RuntimeError("Failed to load hardware specs.")

    # Standardize names
    gpu_name_map = {'a100': 'A100', 'h100': 'H100', 'gb200': 'GB200'}
    src_gpu = gpu_name_map.get(src_gpu.lower(), src_gpu)
    tgt_gpu = gpu_name_map.get(tgt_gpu.lower(), tgt_gpu)

    if src_gpu not in gpu_specs:
        raise ValueError(f"Source GPU '{src_gpu}' specs not found. Available: {list(gpu_specs.keys())}")
    if tgt_gpu not in gpu_specs:
        raise ValueError(f"Target GPU '{tgt_gpu}' specs not found. Available: {list(gpu_specs.keys())}")

    # Extract NCU data for source GPU
    src_df = extract_ncu_data(src_gpu, ncu_path)
    if src_df is None or len(src_df) == 0:
        raise RuntimeError(f"No NCU data extracted for {src_gpu}.")

    src_df['Kernel Name'] = src_df['Kernel Name'].str.replace(r'^\d+\s+', '', regex=True)
    src_df = src_df.reset_index(drop=True)
    print(f"Extracted {len(src_df)} kernels for {src_gpu}.")

    # Build unified format (SAME LOGIC as extract_and_build_cv5.py lines 238-321)
    unified = pd.DataFrame(index=range(len(src_df)))

    # meta-*
    unified['meta-kernel'] = src_df['Kernel Name'].values
    unified['meta-src_gpu'] = src_gpu
    unified['meta-tgt_gpu'] = tgt_gpu

    # I-srcspec::*
    src_spec_data = gpu_specs[src_gpu]
    for spec_key, spec_val in src_spec_data.items():
        # Ensure scalar value (some spec values can be Series after pandas operations)
        if isinstance(spec_val, (pd.Series, np.ndarray)):
            spec_val = spec_val.iloc[0] if hasattr(spec_val, 'iloc') else spec_val[0]
        if isinstance(spec_val, (int, float, np.integer, np.floating)):
            unified[f'I-srcspec::{spec_key}'] = float(spec_val)

    # I-tgtspec(ratio)::*
    tgt_spec_data = gpu_specs[tgt_gpu]
    for spec_key, tgt_val in tgt_spec_data.items():
        if isinstance(tgt_val, (pd.Series, np.ndarray)):
            tgt_val = tgt_val.iloc[0] if hasattr(tgt_val, 'iloc') else tgt_val[0]
        if isinstance(tgt_val, (int, float, np.integer, np.floating)):
            unified[f'I-tgtspec::{spec_key}'] = float(tgt_val)
        src_val = src_spec_data.get(spec_key)
        if isinstance(src_val, (pd.Series, np.ndarray)):
            src_val = src_val.iloc[0] if hasattr(src_val, 'iloc') else src_val[0]
        try:
            if src_val and isinstance(src_val, (int, float, np.integer, np.floating)) and isinstance(tgt_val, (int, float, np.integer, np.floating)) and float(src_val) != 0:
                unified[f'I-tgtspec(ratio)::{spec_key}'] = float(tgt_val) / float(src_val)
        except Exception:
            pass

    # I-kcfg::*
    kcfg_map = {
        'Grid Size': 'I-kcfg::Grid Size',
        'Block Size': 'I-kcfg::Block Size',
        'Registers Per Thread [register/thread]': 'I-kcfg::Registers Per Thread [register/thread]',
        'Shared Memory Configuration Size [byte]': 'I-kcfg::Shared Memory Configuration Size [byte]'
    }
    for raw_col, new_col in kcfg_map.items():
        if raw_col in src_df.columns:
            unified[new_col] = src_df[raw_col].values

    # I-wl::*
    exclude_prefixes = ['meta-', 'S-', 'I-kcfg', 'I-srcspec', 'I-tgtspec', 'Source_GPU', 'Kernel Name']
    kcfg_features = {
        'Block Size', 'Grid Size', 'Registers Per Thread [register/thread]',
        'Static Shared Memory Per Block [byte/block]', 'Dynamic Shared Memory Per Block [byte/block]',
        'Shared Memory Per Block [byte/block]', 'Threads'
    }
    srcspec_features = {
        'GPU Maximum Warps Per Scheduler [warp]',
        'Theoretical Active Warps Per Scheduler [warp]',
        'Theoretical Active Warps per SM [warp]',
        'Block Limit Warps [block]',
        'Block Limit Barriers [block]',
        'Max Cluster Size [block]',
        'Shared Memory Configuration Size [byte]'
    }

    for c in src_df.columns:
        if any(c.startswith(pre) for pre in exclude_prefixes):
            continue
        if c in kcfg_features:
            unified[f'I-kcfg::{c}'] = src_df[c].values
        elif c in srcspec_features:
            unified[f'I-srcspec::{c}'] = src_df[c].values
        else:
            if c == 'Duration [ns]':
                unified['I-wl::Execution Time [ns]'] = src_df[c].values
            elif c in ['breakdown_pipeline_contention', 'breakdown_memory', 'breakdown_sync', 'breakdown_scheduling_overhead']:
                # These are the source breakdowns used as input features
                # Map to the I-wl::src_brk_* naming
                unified[f'I-wl::src_brk_{c.replace("breakdown_", "")}'] = src_df[c].values
            else:
                unified[f'I-wl::{c}'] = src_df[c].values

    # Compute I-kcfg::Threads if not present
    if 'I-kcfg::Threads' not in unified.columns:
        bs = unified.get('I-kcfg::Block Size', pd.Series(256, index=unified.index))
        gs = unified.get('I-kcfg::Grid Size', pd.Series(1, index=unified.index))
        unified['I-kcfg::Threads'] = bs * gs

    # I-derived::*
    num_sms = src_spec_data.get('num_sms', 108)
    unified['I-derived::grid_to_sm_ratio'] = unified.get('I-kcfg::Grid Size', 1) / num_sms
    unified['I-derived::threads_to_sm_ratio'] = unified.get('I-kcfg::Threads', 1) / num_sms
    unified['I-derived::log_grid'] = np.log1p(unified.get('I-kcfg::Grid Size', 1).astype(float))
    unified['I-derived::fills_gpu'] = np.clip(unified['I-derived::grid_to_sm_ratio'], 0, 32)
    unified['I-derived::block_size_anchor'] = unified.get('I-kcfg::Block Size', 256)

    # src_raw_exec_time_ns
    if 'Duration [ns]' in src_df.columns:
        unified['I-derived::src_raw_exec_time_ns'] = src_df['Duration [ns]'].values
    elif 'I-wl::Execution Time [ns]' in unified.columns:
        unified['I-derived::src_raw_exec_time_ns'] = unified['I-wl::Execution Time [ns]']
    else:
        unified['I-derived::src_raw_exec_time_ns'] = 0.0

    # Save
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
    unified.to_csv(output_csv, index=False)
    print(f"\nExtraction complete! {len(unified)} kernels saved to {output_csv}")
    return unified
