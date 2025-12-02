import numpy as np
import pandas as pd
import os
import glob
import joblib
import time
import re
import warnings
import hashlib
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from nilearn.glm.first_level import compute_regressor
import argparse
from config_utils import load_config, standardize, make_dir
from tqdm import tqdm
import gc

# <<< MODIFICATION: Imports for Atlas Masking & Stats >>>
from nilearn import datasets
from nilearn.maskers import NiftiMasker
from scipy.stats import rankdata
# <<< END MODIFICATION >>>

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Fit LLM activations to fMRI data.")
parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
parser.add_argument('--output_suffix', type=str, default=None, help='Suffix for the output directory.')
parser.add_argument('--layers', type=int, nargs='+', required=False, help='List of layer indices to process for LLMs.')
parser.add_argument('--inputs', type=str, nargs='+', required=False, help='One or more activation keys to combine as features for LLMs.')
parser.add_argument('--pivot_input', type=str, required=False, help='[llm-mode2 ONLY] One of the inputs to use as a pivot for feature selection size.')
parser.add_argument('--run',
                    type=str,
                    nargs='+',
                    default=['llm-mode1', 'glove', 'random'],
                    choices=['llm-mode1', 'llm-mode2', 'glove', 'random'],
                    help='Specify which parts to run: llm-mode1 (original), llm-mode2 (pivot/2-stage), glove, random.')
parser.add_argument('--subject',
                    type=str,
                    nargs='+',
                    default=['average'],
                    help="Subject(s) to process. Options: 'average' (default), 'all', or a list of subject indices (e.g., '1' or '1 2 3').")
# <<< MODIFICATION: Parcel & Significance Arguments >>>
parser.add_argument('--parcel', type=str, default='all', 
                    help='Name of the Harvard-Oxford atlas parcel to use for masking (e.g., "Temporal Fusiform Cortex, posterior division"). Defaults to "all".')
parser.add_argument('--significance_testing', action='store_true', 
                    help='If set, perform Bootstrap and FDR correction (Benjamini-Hochberg) to generate p-values and significance masks.')
parser.add_argument('--n_bootstraps', type=int, default=1000, 
                    help='Number of bootstrap iterations for significance testing. Defaults to 1000.')
# <<< END MODIFICATION >>>

args = parser.parse_args()

# --- Validation ---
run_llm_mode1 = 'llm-mode1' in args.run
run_llm_mode2 = 'llm-mode2' in args.run
run_llm_any = run_llm_mode1 or run_llm_mode2

if run_llm_any and (args.layers is None or args.inputs is None):
    parser.error("--layers and --inputs are required when running 'llm-mode1' or 'llm-mode2'.")

if run_llm_mode2 and args.pivot_input is None:
    parser.error("--pivot_input is required when running 'llm-mode2'.")

# --- Configuration & Paths ---
print(f"Loading configuration from {args.config}...")
config = load_config(args.config)
enabled_models = [model for model in config['models'] if model.get('enabled', False)]
if run_llm_any and not enabled_models:
    print("Warning: LLM mode was specified to run, but no LLMs are enabled in the 'models' section of the configuration file.")

# --- DYNAMIC CONFIGURATION START ---
dataset_name = config['dataset']['name']
lang = config['experiment']['language'].lower()
assert lang in ['en', 'fr', 'cn'], "Language must be 'en', 'fr', or 'cn'"
print(f"Configuring for dataset: {dataset_name}, language: {lang}")

output_suffix = args.output_suffix
if output_suffix is None:
    output_suffix = config['general'].get('output_baseline', '') 
if output_suffix != "":
    output_suffix = '_' + output_suffix

activation_folder = config['paths']['llms_activations']
base_output_folder = f"{config['paths']['llms_brain_correlations']}{output_suffix}"
t_r = config['general']['t_r']
trim_timepoints = config['preprocessing']['trim_timepoints']
alphas = np.logspace(config['preprocessing']['alpha_exp_min'], config['preprocessing']['alpha_exp_max'], config['preprocessing']['alpha_num_points'])
hrf_model = 'glover'

# Define dataset-specific paths and templates
if dataset_name == 'lpp':
    output_dir_name = f'lpp_{lang}_per_subject'
    average_dir_name = f'lpp_{lang}_average_subject'
    subject_str_template = f'sub-{lang.upper()}{{subject_id:03d}}' 
elif dataset_name == 'narratives':
    output_dir_name = 'narratives_per_subject'
    average_dir_name = f'narratives_{lang}_average_subject'
    subject_str_template = f'sub-{{subject_id:03d}}' 
else:
    raise ValueError(f"Dataset '{dataset_name}' not recognized in fit_baseline.py.")
# --- DYNAMIC CONFIGURATION END ---


def compute_regressor_from_activations(activations, onsets, offsets, frame_times):
    durations = offsets - onsets
    nn_signals = []
    for feature_amplitudes in tqdm(activations.T, desc="  Convolving features with HRF", leave=False):
        exp_condition = np.array((onsets, durations, feature_amplitudes))
        signal, _ = compute_regressor(exp_condition, hrf_model, frame_times)
        nn_signals.append(signal[:,0])
    return np.array(nn_signals).T

# <<< MODIFICATION: Significance Testing Logic >>>
def perform_bootstrap_fdr(y_true_all, y_pred_all, n_boot=1000, alpha=0.05):
    """
    Performs non-parametric bootstrap resampling to estimate p-values 
    and applies Benjamini-Hochberg FDR correction.
    
    y_true_all: (n_total_timepoints, n_voxels)
    y_pred_all: (n_total_timepoints, n_voxels)
    """
    n_samples, n_voxels = y_true_all.shape
    
    # Pre-calculate z-scores for fast correlation (r = mean(z_x * z_y))
    # Add small epsilon to std to avoid division by zero
    y_true_z = (y_true_all - np.mean(y_true_all, axis=0)) / (np.std(y_true_all, axis=0) + 1e-12)
    y_pred_z = (y_pred_all - np.mean(y_pred_all, axis=0)) / (np.std(y_pred_all, axis=0) + 1e-12)
    
    print(f"    [Stats] Running {n_boot} bootstraps on {n_voxels} voxels...")
    
    # Counter for how many times the bootstrap correlation is <= 0
    neg_count = np.zeros(n_voxels)
    
    rng = np.random.default_rng(42) # Fixed seed for reproducibility
    
    # Vectorized bootstrap loop
    for _ in range(n_boot):
        # Resample indices with replacement
        idx = rng.integers(0, n_samples, n_samples)
        
        # Select resampled data
        yt_b = y_true_z[idx]
        yp_b = y_pred_z[idx]
        
        # Compute correlation efficiently
        # Since vectors are z-scored, corr is just the mean of the element-wise product
        corr_b = np.mean(yt_b * yp_b, axis=0)
        
        # Accumulate null cases (r <= 0)
        neg_count += (corr_b <= 0)
        
    # Calculate p-values (fraction of nulls)
    # Adding 1 to num/denom is a standard bias correction for Monte Carlo p-values
    p_values = (neg_count + 1) / (n_boot + 1)
    
    # FDR Correction (Benjamini-Hochberg)
    # 1. Rank p-values
    ranked_p_values = rankdata(p_values)
    
    # 2. Calculate BH threshold for each rank
    fdr_threshold = (ranked_p_values / n_voxels) * alpha
    
    # 3. Find the largest p-value that is below its specific threshold
    is_below = p_values <= fdr_threshold
    
    reject = np.zeros(n_voxels, dtype=bool)
    if np.any(is_below):
        # Find the max rank that satisfies the condition
        max_rank = np.max(ranked_p_values[is_below])
        # Reject everything with a rank better (smaller) than max_rank
        reject = ranked_p_values <= max_rank
        
    return p_values, reject
# <<< END MODIFICATION >>>

# --- Load MASTER Run Labels and Timing ---
run_labels_file = os.path.join(activation_folder, f'run_labels_{dataset_name}_{lang}.gz')
print(f"Loading MASTER run labels from {run_labels_file}...")
try:
    master_run_labels = joblib.load(run_labels_file)
except FileNotFoundError:
    print(f"ERROR: Could not find {run_labels_file}.")
    print("Please run the baseline (e.g., GloVe) extraction script first to generate this file.")
    raise
    
n_master_runs = len(master_run_labels)
print(f"Found {n_master_runs} master runs/stories: {master_run_labels}")

# Load MASTER onsets/offsets
onsets_offsets_file = os.path.join(activation_folder, f'onsets_offsets_{dataset_name}_{lang}.gz')
print(f"Loading MASTER timing from {onsets_offsets_file}...")
try:
    master_runs_onsets_offsets = joblib.load(onsets_offsets_file)
except FileNotFoundError:
    print(f"ERROR: Could not find {onsets_offsets_file}.")
    raise

if len(master_runs_onsets_offsets) != n_master_runs:
    print(f"ERROR: Mismatch! Found {n_master_runs} master run labels but {len(master_runs_onsets_offsets)} timing entries.")
    raise ValueError("Run label and timing file mismatch.")

master_story_map = {}
for i, story_name in enumerate(master_run_labels):
    master_story_map[story_name] = (master_runs_onsets_offsets[i][0], master_runs_onsets_offsets[i][1])

# --- Target Subjects ---
subjects_to_process = []
if args.subject == ['average']:
    subjects_to_process = ['average']
elif args.subject == ['all']:
    subjects_to_process = config['general'].get('subjects', []) 
    if not subjects_to_process:
        print("ERROR: '--subject all' specified but no 'subjects' list found in config['general']. Exiting.")
        exit()
else:
    try:
        subjects_to_process = [int(s) for s in args.subject] 
    except ValueError:
        parser.error("Invalid subject index. Must be 'average', 'all', or a list of numbers (e.g., '1' or '1 57 102').")

print(f"Targets specified: {args.subject}. Will process: {subjects_to_process}")

# --- MAIN SUBJECT LOOP ---
for subject_id in subjects_to_process:
    
    subject_fmri_runs = []
    subject_run_labels = [] 
    subject_run_keys = []   
    subject_runs_onsets = []
    subject_runs_offsets = []
    
    try:
        # --- Load Data ---
        if dataset_name == 'lpp' or subject_id == 'average':
            print(f"\n{'='*80}\nProcessing {subject_id} (STATIC RUNS)\n{'='*80}")
            
            if subject_id == 'average':
                fmri_data_path = os.path.join(config['paths']['home_folder'], f'{average_dir_name}{output_suffix}')
                file_prefix = 'average_subject'
                subject_output_folder = base_output_folder
            else: 
                subject_str = subject_str_template.format(subject_id=subject_id)
                fmri_base_folder = os.path.join(config['paths']['home_folder'], f'{output_dir_name}{output_suffix}')
                fmri_data_path = os.path.join(fmri_base_folder, subject_str)
                file_prefix = subject_str
                subject_output_folder = os.path.join(base_output_folder, subject_str)

            make_dir(subject_output_folder)
            print(f"Loading pre-processed fMRI data from {fmri_data_path}...")

            subject_run_labels = master_run_labels
            subject_run_keys = master_run_labels
            subject_runs_onsets = [master_story_map[story][0] for story in master_run_labels]
            subject_runs_offsets = [master_story_map[story][1] for story in master_run_labels]
            
            for run_idx, run_label in enumerate(subject_run_labels):
                if dataset_name == 'lpp':
                    fname = f'{file_prefix}_run-{run_idx}.gz'
                else: 
                    fname = f'{file_prefix}_{run_label}.gz'
                
                fpath = os.path.join(fmri_data_path, fname)
                subject_fmri_runs.append(joblib.load(fpath))

        elif dataset_name == 'narratives' and subject_id != 'average':
            subject_str = subject_str_template.format(subject_id=subject_id)
            print(f"\n{'='*80}\nProcessing {subject_str} (DYNAMIC RUNS)\n{'='*80}")
            
            fmri_base_folder = os.path.join(config['paths']['home_folder'], f'{output_dir_name}{output_suffix}')
            fmri_data_path = os.path.join(fmri_base_folder, subject_str)
            subject_output_folder = os.path.join(base_output_folder, subject_str)
            make_dir(subject_output_folder)

            print(f"Scanning for fMRI runs in {fmri_data_path}...")
            found_files = sorted(glob.glob(os.path.join(fmri_data_path, f'{subject_str}_task-*.gz')))
            if not found_files:
                raise FileNotFoundError(f"No fMRI files found for {subject_str} with pattern '{subject_str}_task-*.gz'")

            print(f"Found {len(found_files)} fMRI files. Mapping to stories and correcting timings...")
            for fpath in found_files:
                basename = os.path.basename(fpath)
                run_key = basename.replace(f'{subject_str}_task-', '').replace('.gz', '')
                story_name = run_key.split('_run-')[0]
                
                if story_name in master_story_map:
                    raw_onsets, raw_offsets = master_story_map[story_name]
                    raw_onsets = np.copy(raw_onsets)
                    raw_offsets = np.copy(raw_offsets)

                    if raw_onsets.size == 0:
                        continue

                    event_fname = f'sub-{subject_id:03d}_task-{run_key}_events.tsv'
                    event_fpath = os.path.join(config['paths']['home_folder'], 'narratives', f'sub-{subject_id:03d}', 'func', event_fname)
                                            
                    fmri_story_onset_time = 0.0 
                    try:
                        events_df = pd.read_csv(event_fpath, sep='\t')
                        if not events_df.empty:
                            fmri_story_onset_time = events_df.iloc[0]['onset']
                    except Exception:
                        pass

                    stimulus_t0 = 0.0
                    word_onset_gaps = np.diff(raw_onsets) 
                        
                    if word_onset_gaps.size == 0: 
                        stimulus_t0 = raw_onsets[0]
                    else:
                        first_normal_gap_idx = -1
                        for i, gap in enumerate(word_onset_gaps):
                            if gap < 5.0: 
                                first_normal_gap_idx = i
                                break
                        
                        if first_normal_gap_idx == 0:
                            stimulus_t0 = raw_onsets[0]
                        elif first_normal_gap_idx > 0:
                            stimulus_t0 = raw_onsets[first_normal_gap_idx]
                        else:
                            stimulus_t0 = raw_onsets[0]

                    shift_amount = fmri_story_onset_time 
                    final_onsets = raw_onsets + shift_amount
                    final_offsets = raw_offsets + shift_amount
                    
                    subject_fmri_runs.append(joblib.load(fpath))
                    subject_runs_onsets.append(final_onsets)
                    subject_runs_offsets.append(final_offsets)
                    subject_run_labels.append(story_name) 
                    subject_run_keys.append(run_key)

    except FileNotFoundError as e:
        print(f"❌ ERROR: Could not find fMRI data for {subject_id}. {e}")
        continue
    except Exception as e:
        print(f"❌ An unexpected error occurred loading data for {subject_id}: {e}")
        continue
    
    if dataset_name == 'narratives' and subject_id != 'average':
        if not subject_fmri_runs:
            print("No runs loaded after timing correction. Skipping.")
            continue 

        print(f"Found {len(subject_fmri_runs)} valid runs. Selecting longest run for 10-fold CV.")
        original_scans = [fmri_run.shape[0] for fmri_run in subject_fmri_runs]
        longest_run_idx = np.argmax(original_scans)
        
        subject_fmri_runs = [subject_fmri_runs[longest_run_idx]]
        subject_run_labels = [subject_run_labels[longest_run_idx]]
        subject_run_keys = [subject_run_keys[longest_run_idx]]
        subject_runs_onsets = [subject_runs_onsets[longest_run_idx]]
        subject_runs_offsets = [subject_runs_offsets[longest_run_idx]]
    
    n_runs_subject = len(subject_fmri_runs)
    if n_runs_subject == 0:
        print(f"❌ ERROR: No valid fMRI runs were loaded for {subject_id}. Skipping subject.")
        continue
    elif n_runs_subject < 2:
        if dataset_name == 'narratives' and subject_id != 'average' and n_runs_subject == 1:
            print("Proceeding with 1 run for 10-fold CV.")
        else:
            print(f"❌ ERROR: Only {n_runs_subject} run found. Need at least 2 for cross-validation. Skipping subject.")
            continue
        
    n_scans_runs = [fmri_run.shape[0] for fmri_run in subject_fmri_runs]
    
    # --- Trim, Standardize, AND Clean NaNs ---
    print(f"Applying trimming (first/last {trim_timepoints} TRs) and final standardization...")
    for run in range(n_runs_subject):
        trimmed_data = subject_fmri_runs[run][trim_timepoints:-trim_timepoints]
        standardized_data = standardize(trimmed_data)
        np.nan_to_num(standardized_data, copy=False, nan=0.0)
        subject_fmri_runs[run] = standardized_data
        
    # <<< MODIFICATION: APPLY PARCEL MASKING >>>
    if args.parcel != 'all':
        try:
            parcel_name_str = args.parcel
            print(f"\n{'!'*20}\nAPPLYING MASK: '{parcel_name_str}'\n{'!'*20}")

            # 1. Set up NiftiMasker
            mask_dir = config['paths']['roi_masks']
            mask_file = os.path.join(mask_dir, f'mask_{dataset_name}_{lang}.nii.gz')
            
            if not os.path.exists(mask_file):
                mask_file_fallback = os.path.join(mask_dir, f'mask_{dataset_name}.nii.gz')
                if os.path.exists(mask_file_fallback):
                    mask_file = mask_file_fallback
                else:
                    raise FileNotFoundError(f"Brain mask file not found in {mask_dir}.")
            
            print(f"  Using brain mask: {mask_file}")
            nifti_masker = NiftiMasker(mask_img=mask_file)
            nifti_masker.fit()

            # 2. Fetch the Harvard-Oxford atlas
            atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-1mm')
            atlas_labels = atlas.labels
            
            # 3. Find index
            try:
                parcel_index = atlas_labels.index(parcel_name_str)
            except ValueError:
                available_parcels = "\n - ".join(atlas_labels[1:]) 
                raise ValueError(f"Parcel '{parcel_name_str}' not found in Harvard-Oxford atlas.")
            
            print(f"  Found '{parcel_name_str}' at index {parcel_index}.")

            # 4. Create mask
            atlas_data_1d = nifti_masker.transform(atlas.maps).flatten().astype(int)
            mask = (atlas_data_1d == parcel_index)
            
            if not np.any(mask):
                raise ValueError(f"Parcel '{parcel_name_str}' resulted in empty mask.")

            # 5. Apply to ALL runs for this subject
            for run in range(n_runs_subject):
                original_shape = subject_fmri_runs[run].shape
                subject_fmri_runs[run] = subject_fmri_runs[run][:, mask]
                print(f"    Run {run} masked: {original_shape} -> {subject_fmri_runs[run].shape}")

        except Exception as e:
            print(f"❌ FATAL ERROR applying parcel mask: {e}")
            exit()
    else:
        print("No parcel masking applied (using whole brain).")
    
    n_voxels = subject_fmri_runs[0].shape[1]
    print(f"Final voxel count for analysis: {n_voxels}")
    # <<< END MODIFICATION >>>

    # --- LLM Processing ---
    if run_llm_any:
        print(f"\n{'='*80}\nProcessing LLM Models for {subject_id}\n{'='*80}")
        
        def get_part_num(filepath):
            match = re.search(r'_part-(\d+)_activations\.gz$', filepath)
            return int(match.group(1)) if match else -1
            
        for model_info in enabled_models:
            model_name = model_info['name']
            model_name_sanitized = model_name.replace("/", "_")
            print(f"\n{'-'*80}\nProcessing model: {model_name}\n{'-'*80}")
            
            for idx_layer in tqdm(args.layers, desc=f"Processing layers for {model_name}"):
                print('='*62)
                print(f'Processing layer {idx_layer}', flush=True)
                try:
                    llm_activations_map = {} 
                    print("  Pre-loading all master story activations for this layer...")
                    
                    input_dims, input_boundaries, current_dim_end = {}, {}, 0
                    
                    keys_to_load = list(args.inputs)
                    if run_llm_mode2 and args.pivot_input and args.pivot_input not in keys_to_load:
                        keys_to_load.append(args.pivot_input)

                    for story_name in tqdm(master_run_labels, desc=f"  Loading story chunks", leave=False):
                        chunk_files_pattern = os.path.join(
                            activation_folder,
                            f'{model_name_sanitized}_{dataset_name}_{lang}_run-{story_name}_part-*_activations.gz'
                        )
                        chunk_files = glob.glob(chunk_files_pattern)
                        
                        if not chunk_files:
                            llm_activations_map[story_name] = np.array([])
                            continue

                        chunk_files.sort(key=get_part_num)
                        word_activations_for_this_story = {key: [] for key in keys_to_load}
                        
                        for chunk_file in chunk_files:
                            chunk_data = joblib.load(chunk_file) 
                            for key in keys_to_load:
                                if key in chunk_data and idx_layer < len(chunk_data[key]):
                                    word_activations_for_this_story[key].extend(chunk_data[key][idx_layer])

                        if not word_activations_for_this_story[args.inputs[0]]:
                            llm_activations_map[story_name] = np.array([])
                            continue

                        num_words = len(word_activations_for_this_story[args.inputs[0]])
                        concatenated_run_data = []
                        for word_idx in range(num_words):
                            word_vecs_to_concat = [word_activations_for_this_story[key][word_idx] for key in args.inputs]
                            concatenated_word_vec = np.concatenate(word_vecs_to_concat)
                            concatenated_run_data.append(concatenated_word_vec)
                        
                        llm_activations_map[story_name] = np.array(concatenated_run_data)
                        
                        if not input_dims:
                            current_dim_end = 0
                            for idx, key in enumerate(args.inputs): 
                                word_vec = word_activations_for_this_story[key][0]
                                dim = word_vec.shape[0]
                                input_dims[key] = dim
                                input_boundaries[key] = (current_dim_end, current_dim_end + dim)
                                current_dim_end += dim
                            
                            if run_llm_mode2 and args.pivot_input and args.pivot_input not in input_dims:
                                pivot_vec = word_activations_for_this_story[args.pivot_input][0]
                                input_dims[args.pivot_input] = pivot_vec.shape[0]
                    
                    print("  Mapping pre-loaded activations to subject's runs...")
                    activations_for_one_layer = []
                    for story_name in subject_run_labels:
                        if story_name not in llm_activations_map:
                            raise KeyError(f"Missing LLM data for story: {story_name}")
                        activations_for_one_layer.append(llm_activations_map[story_name])
                    
                    regressors_runs = []
                    print("  Computing regressors...")
                    for run in range(n_runs_subject):
                        activations_words_neurons = activations_for_one_layer[run]
                        
                        if activations_words_neurons.size == 0:
                            n_features = 0
                            for arr in activations_for_one_layer:
                                if arr.size > 0: n_features = arr.shape[1]; break
                            
                            n_scans_trimmed = subject_fmri_runs[run].shape[0]
                            regressors_runs.append(np.zeros((n_scans_trimmed, n_features)))
                            continue

                        frame_times = np.arange(n_scans_runs[run]) * t_r + .5 * t_r
                        regressor_run = compute_regressor_from_activations(
                            activations_words_neurons, 
                            subject_runs_onsets[run], 
                            subject_runs_offsets[run], 
                            frame_times
                        )
                        regressor_run = regressor_run[trim_timepoints:-trim_timepoints]
                        regressor_run = standardize(regressor_run) 
                        np.nan_to_num(regressor_run, copy=False, nan=0.0)
                        regressors_runs.append(regressor_run)

                    # =====================================================
                    # LLM MODE 1 (ORIGINAL LOGIC)
                    # =====================================================
                    if run_llm_mode1:
                        print(f"  [MODE 1] Running standard Ridge Regression...")
                        corr_runs_with_nans = [] 
                        
                        # <<< MODIFICATION: Stats Init >>>
                        if args.significance_testing:
                            stored_y_test = []
                            stored_y_pred = []
                        # <<< END MODIFICATION >>>
                        
                        if dataset_name == 'lpp' or subject_id == 'average':
                            print(f"  Performing {n_runs_subject}-fold cross-validation...")
                            for run_test in range(n_runs_subject):
                                tic = time.time()
                                runs_train = np.setdiff1d(np.arange(n_runs_subject), run_test)
                                
                                x_train_list = [regressors_runs[r] for r in runs_train]
                                y_train_list = [subject_fmri_runs[r] for r in runs_train]
                                x_test, y_test = regressors_runs[run_test], subject_fmri_runs[run_test]

                                x_train = np.vstack(x_train_list)
                                y_train = np.vstack(y_train_list)

                                run_val = runs_train[0]
                                runs_train_val = np.setdiff1d(runs_train, run_val)
                                x_train_val_list = [regressors_runs[r] for r in runs_train_val]
                                y_train_val_list = [subject_fmri_runs[r] for r in runs_train_val]
                                
                                if not runs_train_val.size:
                                    x_train_val, y_train_val = x_test, y_test
                                else:
                                    x_train_val = np.vstack(x_train_val_list)
                                    y_train_val = np.vstack(y_train_val_list)
                                    
                                x_val, y_val = regressors_runs[run_val], subject_fmri_runs[run_val]
                                
                                corr_val_list = []
                                for alpha in alphas:
                                    model = Ridge(alpha=alpha, fit_intercept=False)
                                    model.fit(x_train_val, y_train_val)
                                    y_pred = model.predict(x_val)
                                    corrs_with_nan = np.array([np.corrcoef(y_val[:,i], y_pred[:,i])[0,1] for i in range(n_voxels)])
                                    corr_val_list.append(corrs_with_nan)
                                
                                with warnings.catch_warnings():
                                    warnings.filterwarnings('ignore', r'Mean of empty slice')
                                    best_alpha = alphas[np.nanargmax([np.nanmean(c) for c in corr_val_list])]

                                model = Ridge(alpha=best_alpha, fit_intercept=False)
                                model.fit(x_train, y_train)
                                y_pred = model.predict(x_test)
                                
                                # <<< MODIFICATION: Store Predictions >>>
                                if args.significance_testing:
                                    stored_y_test.append(y_test)
                                    stored_y_pred.append(y_pred)
                                # <<< END MODIFICATION >>>
                                
                                corrs_with_nan = np.array([np.corrcoef(y_test[:,i], y_pred[:,i])[0,1] for i in range(n_voxels)])
                                corr_runs_with_nans.append(corrs_with_nan)
                                toc = time.time()
                                print(f'      ... Fold {run_test+1} mean = {np.nanmean(corrs_with_nan):.03f}, time = {toc-tic:.03f}s')

                        else:
                            N_FOLDS = 10
                            print(f"  Performing {N_FOLDS}-fold cross-validation on single run...")
                            X_full = regressors_runs[0]
                            Y_full = subject_fmri_runs[0]
                            kf = KFold(n_splits=N_FOLDS, shuffle=False)
                            
                            fold_num = 1
                            for train_index, test_index in kf.split(X_full):
                                tic = time.time()
                                x_train, x_test = X_full[train_index], X_full[test_index]
                                y_train, y_test = Y_full[train_index], Y_full[test_index]

                                val_size = len(train_index) // N_FOLDS 
                                if val_size < 1: val_size = 1
                                x_val, y_val = x_train[:val_size], y_train[:val_size]
                                x_train_val, y_train_val = x_train[val_size:], y_train[val_size:]
                                if x_train_val.shape[0] == 0:
                                    x_train_val, y_train_val = x_train, y_train
                                    x_val, y_val = x_test, y_test
                                
                                corr_val_list = []
                                for alpha in alphas:
                                    model = Ridge(alpha=alpha, fit_intercept=False)
                                    model.fit(x_train_val, y_train_val)
                                    y_pred = model.predict(x_val)
                                    corrs_with_nan = np.array([np.corrcoef(y_val[:,i], y_pred[:,i])[0,1] for i in range(n_voxels)])
                                    corr_val_list.append(corrs_with_nan)
                                
                                with warnings.catch_warnings():
                                    warnings.filterwarnings('ignore', r'Mean of empty slice')
                                    best_alpha = alphas[np.nanargmax([np.nanmean(c) for c in corr_val_list])]
                                
                                model = Ridge(alpha=best_alpha, fit_intercept=False)
                                model.fit(x_train, y_train)
                                y_pred = model.predict(x_test)
                                
                                # <<< MODIFICATION: Store Predictions >>>
                                if args.significance_testing:
                                    stored_y_test.append(y_test)
                                    stored_y_pred.append(y_pred)
                                # <<< END MODIFICATION >>>
                                
                                corrs_with_nan = np.array([np.corrcoef(y_test[:,i], y_pred[:,i])[0,1] for i in range(n_voxels)])
                                corr_runs_with_nans.append(corrs_with_nan)
                                toc = time.time()
                                print(f'      ... Fold {fold_num} mean = {np.nanmean(corrs_with_nan):.03f}, time = {toc-tic:.03f}s')
                                fold_num += 1

                        corr_runs_for_saving = np.nan_to_num(corr_runs_with_nans, nan=0.0)
                        layers_str = f"layer-{idx_layer}"
                        inputs_str = f"inputs-{'_'.join(args.inputs)}" 
                        save_suffix = f"{layers_str}_{inputs_str}"
                        if args.parcel != 'all':
                            sanitized_parcel = re.sub(r'[^\w\-_.]', '_', args.parcel)
                            save_suffix += f"_parcel-{sanitized_parcel}"

                        filename = os.path.join(subject_output_folder, f'{model_name_sanitized}_{dataset_name}_{lang}_{save_suffix}_corr.gz')
                        with open(filename, 'wb') as f:
                            joblib.dump(np.mean(corr_runs_for_saving, axis=0), f, compress=4)
                            
                        # <<< MODIFICATION: Save Stats >>>
                        if args.significance_testing:
                            try:
                                Y_total = np.vstack(stored_y_test)
                                P_total = np.vstack(stored_y_pred)
                                pvals, mask = perform_bootstrap_fdr(Y_total, P_total, n_boot=args.n_bootstraps)
                                
                                with open(filename.replace('_corr.gz', '_pval.gz'), 'wb') as f:
                                    joblib.dump(pvals, f, compress=4)
                                with open(filename.replace('_corr.gz', '_sig_mask.gz'), 'wb') as f:
                                    joblib.dump(mask, f, compress=4)
                                print(f"    [Stats] Saved significance outputs.")
                            except Exception as e:
                                print(f"    [Stats Error] {e}")
                        # <<< END MODIFICATION >>>
                    
                    # =====================================================
                    # LLM MODE 2 (PIVOT / 2-STAGE REGRESSION)
                    # =====================================================
                    if run_llm_mode2:
                        print(f"  [MODE 2] Running Two-Stage Pivot Regression (Pivot: {args.pivot_input})")
                        if not input_dims: continue
                        num_features_to_select = input_dims[args.pivot_input]
                        
                        corr_runs_s1, corr_runs_s2, all_fold_betas_s1 = [], [], []
                        
                        # <<< MODIFICATION: Stats Init >>>
                        if args.significance_testing:
                            stored_y_test = []
                            stored_y_pred_s2 = []
                        # <<< END MODIFICATION >>>

                        if dataset_name == 'lpp' or subject_id == 'average':
                            print(f"  Performing {n_runs_subject}-fold cross-validation...")
                            for run_test in range(n_runs_subject):
                                tic = time.time()
                                runs_train = np.setdiff1d(np.arange(n_runs_subject), run_test)
                                x_train_list = [regressors_runs[r] for r in runs_train]
                                y_train_list = [subject_fmri_runs[r] for r in runs_train]
                                x_test, y_test = regressors_runs[run_test], subject_fmri_runs[run_test]
                                x_train = np.vstack(x_train_list)
                                y_train = np.vstack(y_train_list)
                                
                                run_val = runs_train[0]
                                runs_train_val = np.setdiff1d(runs_train, run_val)
                                x_train_val_list = [regressors_runs[r] for r in runs_train_val]
                                y_train_val_list = [subject_fmri_runs[r] for r in runs_train_val]
                                if not runs_train_val.size:
                                    x_train_val, y_train_val = x_test, y_test
                                else:
                                    x_train_val = np.vstack(x_train_val_list)
                                    y_train_val = np.vstack(y_train_val_list)
                                x_val, y_val = regressors_runs[run_val], subject_fmri_runs[run_val]
                                
                                corr_val_list = []
                                for alpha in alphas:
                                    model = Ridge(alpha=alpha, fit_intercept=False)
                                    model.fit(x_train_val, y_train_val)
                                    y_pred = model.predict(x_val)
                                    corrs_with_nan = np.array([np.corrcoef(y_val[:,i], y_pred[:,i])[0,1] for i in range(n_voxels)])
                                    corr_val_list.append(corrs_with_nan)
                                with warnings.catch_warnings():
                                    warnings.filterwarnings('ignore', r'Mean of empty slice')
                                    best_alpha = alphas[np.nanargmax([np.nanmean(c) for c in corr_val_list])]

                                initial_model = Ridge(alpha=best_alpha, fit_intercept=False)
                                initial_model.fit(x_train, y_train)
                                beta_weights_stage1 = initial_model.coef_
                                y_pred_stage1 = initial_model.predict(x_test)
                                correlations_stage1 = np.array([np.corrcoef(y_test[:,i], y_pred_stage1[:,i])[0,1] for i in range(n_voxels)])
                                
                                feature_importance = np.mean(np.abs(beta_weights_stage1), axis=0)
                                ranked_indices = np.argsort(feature_importance)[::-1]
                                selected_indices = ranked_indices[:num_features_to_select]
                                x_test_selected = x_test[:, selected_indices]
                                
                                final_model = Ridge(alpha=best_alpha, fit_intercept=False)
                                final_model.fit(x_train[:, selected_indices], y_train)
                                y_pred_stage2 = final_model.predict(x_test_selected)
                                correlations_stage2 = np.array([np.corrcoef(y_test[:,i], y_pred_stage2[:,i])[0,1] for i in range(n_voxels)])

                                # <<< MODIFICATION: Store >>>
                                if args.significance_testing:
                                    stored_y_test.append(y_test)
                                    stored_y_pred_s2.append(y_pred_stage2)
                                # <<< END MODIFICATION >>>

                                corr_runs_s1.append(correlations_stage1)
                                corr_runs_s2.append(correlations_stage2)
                                all_fold_betas_s1.append(beta_weights_stage1)
                                toc = time.time()
                                print(f'      ... Fold {run_test+1} S1={np.nanmean(correlations_stage1):.3f}, S2={np.nanmean(correlations_stage2):.3f}, t={toc-tic:.2f}s')

                        else:
                            N_FOLDS = 10
                            print(f"  Performing {N_FOLDS}-fold cross-validation on single run...")
                            X_full = regressors_runs[0]
                            Y_full = subject_fmri_runs[0]
                            kf = KFold(n_splits=N_FOLDS, shuffle=False) 
                            
                            fold_num = 1
                            for train_index, test_index in kf.split(X_full):
                                tic = time.time()
                                x_train, x_test = X_full[train_index], X_full[test_index]
                                y_train, y_test = Y_full[train_index], Y_full[test_index]
                                
                                val_size = len(train_index) // N_FOLDS 
                                if val_size < 1: val_size = 1
                                x_val, y_val = x_train[:val_size], y_train[:val_size]
                                x_train_val, y_train_val = x_train[val_size:], y_train[val_size:]
                                if x_train_val.shape[0] == 0:
                                    x_train_val, y_train_val = x_train, y_train
                                    x_val, y_val = x_test, y_test 

                                corr_val_list = []
                                for alpha in alphas:
                                    model = Ridge(alpha=alpha, fit_intercept=False)
                                    model.fit(x_train_val, y_train_val)
                                    y_pred = model.predict(x_val)
                                    corrs_with_nan = np.array([np.corrcoef(y_val[:,i], y_pred[:,i])[0,1] for i in range(n_voxels)])
                                    corr_val_list.append(corrs_with_nan)
                                with warnings.catch_warnings():
                                    warnings.filterwarnings('ignore', r'Mean of empty slice')
                                    best_alpha = alphas[np.nanargmax([np.nanmean(c) for c in corr_val_list])]
                                
                                initial_model = Ridge(alpha=best_alpha, fit_intercept=False)
                                initial_model.fit(x_train, y_train)
                                beta_weights_stage1 = initial_model.coef_
                                
                                feature_importance = np.mean(np.abs(beta_weights_stage1), axis=0)
                                ranked_indices = np.argsort(feature_importance)[::-1]
                                selected_indices = ranked_indices[:num_features_to_select]
                                x_test_selected = x_test[:, selected_indices]
                                
                                final_model = Ridge(alpha=best_alpha, fit_intercept=False)
                                final_model.fit(x_train[:, selected_indices], y_train)
                                y_pred_stage2 = final_model.predict(x_test_selected)
                                correlations_stage2 = np.array([np.corrcoef(y_test[:,i], y_pred_stage2[:,i])[0,1] for i in range(n_voxels)])

                                # <<< MODIFICATION: Store >>>
                                if args.significance_testing:
                                    stored_y_test.append(y_test)
                                    stored_y_pred_s2.append(y_pred_stage2)
                                # <<< END MODIFICATION >>>

                                corr_runs_s1.append(correlations_stage1)
                                corr_runs_s2.append(correlations_stage2)
                                all_fold_betas_s1.append(beta_weights_stage1)
                                toc = time.time()
                                print(f'      ... Fold {fold_num} S1={np.nanmean(correlations_stage1):.3f}, S2={np.nanmean(correlations_stage2):.3f}, t={toc-tic:.2f}s')
                                fold_num += 1
                        
                        layers_str = f"layer-{idx_layer}"
                        inputs_combined_str = '_'.join(args.inputs)
                        inputs_hash = hashlib.md5(inputs_combined_str.encode()).hexdigest()[:8]
                        inputs_str = f"inputs-{inputs_hash}"
                        pivot_str = f"pivot-{args.pivot_input}"
                        
                        sanitized_parcel_name = re.sub(r'[^\w\-_.]', '_', args.parcel)
                        parcel_str = f"parcel-{sanitized_parcel_name}"
                        save_suffix = f"{layers_str}_{inputs_str}_{pivot_str}_{parcel_str}"
                        
                        mean_corr_runs_s1 = np.nanmean(corr_runs_s1, axis=0)
                        mean_corr_runs_s2 = np.nanmean(corr_runs_s2, axis=0)
                        corr_diff_map = mean_corr_runs_s2 - mean_corr_runs_s1
                        
                        filename_corr_s1 = os.path.join(subject_output_folder, f'{model_name_sanitized}_{dataset_name}_{lang}_{save_suffix}_corr_stage1.gz')
                        with open(filename_corr_s1, 'wb') as f:
                            joblib.dump(mean_corr_runs_s1, f, compress=4)

                        filename_corr_s2 = os.path.join(subject_output_folder, f'{model_name_sanitized}_{dataset_name}_{lang}_{save_suffix}_corr_stage2.gz')
                        with open(filename_corr_s2, 'wb') as f:
                            joblib.dump(mean_corr_runs_s2, f, compress=4)

                        filename_corr_diff = os.path.join(subject_output_folder, f'{model_name_sanitized}_{dataset_name}_{lang}_{save_suffix}_corr_diff.gz')
                        with open(filename_corr_diff, 'wb') as f:
                            joblib.dump(corr_diff_map, f, compress=4)
                        
                        mean_beta_weights = np.mean(np.stack(all_fold_betas_s1, axis=0), axis=0)
                        final_betas_to_save = {
                            key: mean_beta_weights[:, start:end]
                            for key, (start, end) in input_boundaries.items()
                        }
                        filename_details = os.path.join(subject_output_folder, f'{model_name_sanitized}_{dataset_name}_{lang}_{save_suffix}_selection_details.gz')
                        print(f"  Saving selection details to {filename_details}")
                        with open(filename_details, 'wb') as f:
                            joblib.dump(final_betas_to_save, f, compress=4)

                        # <<< MODIFICATION: Save Stats Stage 2 >>>
                        if args.significance_testing:
                            try:
                                Y_total = np.vstack(stored_y_test)
                                P_total = np.vstack(stored_y_pred_s2)
                                pvals, mask = perform_bootstrap_fdr(Y_total, P_total, n_boot=args.n_bootstraps)
                                
                                base_fn = os.path.join(subject_output_folder, f'{model_name_sanitized}_{dataset_name}_{lang}_{save_suffix}')
                                with open(f'{base_fn}_stage2_pval.gz', 'wb') as f:
                                    joblib.dump(pvals, f, compress=4)
                                with open(f'{base_fn}_stage2_sig_mask.gz', 'wb') as f:
                                    joblib.dump(mask, f, compress=4)
                                print(f"    [Stats] Saved Stage 2 significance outputs.")
                            except Exception as e:
                                print(f"    [Stats Error] {e}")
                        # <<< END MODIFICATION >>>

                    del llm_activations_map, activations_for_one_layer, regressors_runs
                    gc.collect()

                except Exception as e:
                    print(f"❌ Error processing layer {idx_layer} for model {model_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            print("\nAll specified layers processed!")
    else:
        print(f"\nSkipping LLM processing for {subject_id} as per --run argument.")

    # --- Baseline Processing ---
    print(f"\n{'='*80}\nProcessing Baseline Models for {subject_id}\n{'='*80}")
    baseline_models = []
    if 'glove' in args.run and config.get('baselines', {}).get('glove', {}).get('enabled', False):
        if lang == 'en':
            baseline_models.append({'name': 'glove', 'file': f'glove_{dataset_name}_{lang}.gz'})
    if 'random' in args.run and config.get('baselines', {}).get('random_embeddings', {}).get('enabled', False):
        for rc in config['baselines']['random_embeddings'].get('configs', []):
            nm = f"random_{rc.get('type','embedding')}_{rc.get('n_dims',300)}d_seed{rc.get('seed',1)}_{dataset_name}_{lang}"
            baseline_models.append({'name': nm, 'file': f'{nm}.gz'})

    for model_info in tqdm(baseline_models, desc="Processing baseline models"):
        model_name = model_info['name']
        try:
            act_file = os.path.join(activation_folder, model_info['file'])
            if not os.path.exists(act_file): continue
            
            baseline_activations_map = {} 
            baseline_data_raw = joblib.load(act_file) 
            for run_idx, story_name in enumerate(master_run_labels):
                baseline_activations_map[story_name] = baseline_data_raw[run_idx]
            
            activations_data = [baseline_activations_map[story] for story in subject_run_labels]
            n_layers = len(activations_data[0])
            
            for idx_layer in range(n_layers):
                regressors_runs = []
                for run in range(n_runs_subject): 
                    act_run = np.array(activations_data[run][idx_layer]) 
                    if act_run.size == 0:
                        n_scans_trimmed = subject_fmri_runs[run].shape[0]
                        regressors_runs.append(np.zeros((n_scans_trimmed, 0)))
                        continue
                    frame_times = np.arange(n_scans_runs[run])*t_r + .5*t_r
                    reg_run = compute_regressor_from_activations(
                        act_run, subject_runs_onsets[run], subject_runs_offsets[run], frame_times
                    )
                    reg_run = reg_run[trim_timepoints:-trim_timepoints]
                    reg_run = standardize(reg_run)
                    np.nan_to_num(reg_run, copy=False, nan=0.0)
                    regressors_runs.append(reg_run)

                corr_runs_with_nans = [] 
                
                # <<< MODIFICATION: Stats Init >>>
                if args.significance_testing:
                    stored_y_test = []
                    stored_y_pred = []
                # <<< END MODIFICATION >>>

                if dataset_name == 'lpp' or subject_id == 'average':
                    for run_test in range(n_runs_subject):
                        runs_train = np.setdiff1d(np.arange(n_runs_subject), run_test)
                        x_train_list = [regressors_runs[r] for r in runs_train]
                        y_train_list = [subject_fmri_runs[r] for r in runs_train]
                        x_test, y_test = regressors_runs[run_test], subject_fmri_runs[run_test]
                        
                        x_train = np.vstack(x_train_list)
                        y_train = np.vstack(y_train_list)
                        run_val = runs_train[0]
                        runs_train_val = np.setdiff1d(runs_train, run_val)
                        x_train_val_list = [regressors_runs[r] for r in runs_train_val]
                        y_train_val_list = [subject_fmri_runs[r] for r in runs_train_val]
                        if not runs_train_val.size:
                            x_train_val, y_train_val = x_test, y_test
                        else:
                            x_train_val = np.vstack(x_train_val_list)
                            y_train_val = np.vstack(y_train_val_list)
                        x_val, y_val = regressors_runs[run_val], subject_fmri_runs[run_val]
                        
                        corr_val_list = []
                        for alpha in alphas:
                            model = Ridge(alpha=alpha, fit_intercept=False)
                            model.fit(x_train_val, y_train_val)
                            y_pred = model.predict(x_val)
                            corrs_with_nan = np.array([np.corrcoef(y_val[:,i], y_pred[:,i])[0,1] for i in range(n_voxels)])
                            corr_val_list.append(corrs_with_nan)
                        
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', r'Mean of empty slice')
                            best_alpha = alphas[np.nanargmax([np.nanmean(c) for c in corr_val_list])]
                        
                        model = Ridge(alpha=best_alpha, fit_intercept=False)
                        model.fit(x_train, y_train)
                        y_pred = model.predict(x_test)
                        
                        # <<< MODIFICATION: Store >>>
                        if args.significance_testing:
                            stored_y_test.append(y_test)
                            stored_y_pred.append(y_pred)
                        # <<< END MODIFICATION >>>

                        corr_runs_with_nans.append(np.array([np.corrcoef(y_test[:,i], y_pred[:,i])[0,1] for i in range(n_voxels)]))
                else:
                    X_full = regressors_runs[0]
                    Y_full = subject_fmri_runs[0]
                    kf = KFold(n_splits=10, shuffle=False)
                    for train_index, test_index in kf.split(X_full):
                        x_train, x_test = X_full[train_index], X_full[test_index]
                        y_train, y_test = Y_full[train_index], Y_full[test_index]
                        val_size = len(train_index) // 10 
                        if val_size < 1: val_size = 1
                        x_val, y_val = x_train[:val_size], y_train[:val_size]
                        x_train_val, y_train_val = x_train[val_size:], y_train[val_size:]
                        if x_train_val.shape[0] == 0:
                            x_train_val, y_train_val = x_train, y_train
                            x_val, y_val = x_test, y_test
                        
                        corr_val_list = []
                        for alpha in alphas:
                            model = Ridge(alpha=alpha, fit_intercept=False)
                            model.fit(x_train_val, y_train_val)
                            y_pred = model.predict(x_val)
                            corrs_with_nan = np.array([np.corrcoef(y_val[:,i], y_pred[:,i])[0,1] for i in range(n_voxels)])
                            corr_val_list.append(corrs_with_nan)
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', r'Mean of empty slice')
                            best_alpha = alphas[np.nanargmax([np.nanmean(c) for c in corr_val_list])]
                        
                        model = Ridge(alpha=best_alpha, fit_intercept=False)
                        model.fit(x_train, y_train)
                        y_pred = model.predict(x_test)
                        
                        # <<< MODIFICATION: Store >>>
                        if args.significance_testing:
                            stored_y_test.append(y_test)
                            stored_y_pred.append(y_pred)
                        # <<< END MODIFICATION >>>

                        corr_runs_with_nans.append(np.array([np.corrcoef(y_test[:,i], y_pred[:,i])[0,1] for i in range(n_voxels)]))

                corr_runs_for_saving = np.nan_to_num(corr_runs_with_nans, nan=0.0)
                
                save_suffix = ""
                if args.parcel != 'all':
                    sanitized_parcel = re.sub(r'[^\w\-_.]', '_', args.parcel)
                    save_suffix = f"_parcel-{sanitized_parcel}"
                
                filename = os.path.join(subject_output_folder, f'{model_name}_layer-{idx_layer}{save_suffix}_corr.gz')
                
                with open(filename, 'wb') as f:
                    joblib.dump(np.mean(corr_runs_for_saving, axis=0), f, compress=4)
                    
                # <<< MODIFICATION: Save Stats >>>
                if args.significance_testing:
                    try:
                        Y_total = np.vstack(stored_y_test)
                        P_total = np.vstack(stored_y_pred)
                        pvals, mask = perform_bootstrap_fdr(Y_total, P_total, n_boot=args.n_bootstraps)
                        
                        with open(filename.replace('_corr.gz', '_pval.gz'), 'wb') as f:
                            joblib.dump(pvals, f, compress=4)
                        with open(filename.replace('_corr.gz', '_sig_mask.gz'), 'wb') as f:
                            joblib.dump(mask, f, compress=4)
                    except Exception as e:
                        print(f"    [Stats Error] {e}")
                # <<< END MODIFICATION >>>
            
            del baseline_activations_map, activations_data, regressors_runs
            gc.collect()
        except Exception as e:
            print(f"❌ Error processing baseline {model_name}: {e}")
            continue

    print(f"--- Finished processing {subject_id}. Cleaning up memory. ---")
    del subject_fmri_runs
    gc.collect()

print("\nScript finished.")
