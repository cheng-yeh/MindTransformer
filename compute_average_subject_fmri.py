import numpy as np
import os
import glob
import joblib
import argparse
from tqdm.auto import tqdm
from nilearn.input_data import NiftiMasker
import nibabel as nib
import time
from sklearn.linear_model import Ridge

# Import the configuration utilities
from config_utils import load_config, make_dir, standardize

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml',
                    help='Path to configuration file')
args = parser.parse_args()

# Load configuration
config = load_config(args.config)

# --- Dynamic Path Configuration ---

# Get dataset info
dataset_name = config['dataset']['name']
lang = config['experiment']['language'].lower() # Still needed for LPP

# Get common paths from config
home_folder = config['paths']['home_folder']
fmri_data_avg_subject = config['paths']['fmri_data_avg_subject_input']
output_masks_dir = config['paths']['roi_masks']
isc_output_dir = config['paths']['isc_output']
make_dir(fmri_data_avg_subject)
make_dir(isc_output_dir)

# Set up dataset-specific paths and filenames
if dataset_name == 'lpp':
    assert lang in ['en', 'fr', 'cn'], 'LPP language must be en, fr, or cn.'
    resampled_folder_name = f'lpp_{lang}_resampled'
    subject_glob_pattern = config['dataset']['subject_glob'].replace('LANG', lang.upper())
    mask_filename = f'mask_lpp_{lang}.nii.gz'
    isc_filename = f'isc_{config["preprocessing"]["isc_trials"]}trials_{lang}.gz'
    
elif dataset_name == 'narratives':
    resampled_folder_name = 'narratives_resampled'
    subject_glob_pattern = config['dataset']['subject_glob'] # sub-[0-9][0-9][0-9]
    mask_filename = 'mask_narratives.nii.gz'
    isc_filename = f'isc_{config["preprocessing"]["isc_trials"]}trials_narratives.gz'
    
else:
    raise ValueError(f"Dataset '{dataset_name}' not recognized.")

# Construct dynamic paths
fmri_data_resampled = os.path.join(home_folder, resampled_folder_name)
mask_path = os.path.join(output_masks_dir, mask_filename)
isc_output_path = os.path.join(isc_output_dir, isc_filename)

print(f"--- Running compute_average_subject_fmri.py for dataset: {dataset_name} ---")

# --- Get Experimental Parameters ---
n_runs = config['general']['n_runs']
t_r = config['general']['t_r']
random_seed = config['experiment']['seed']
n_trials = config['preprocessing'].get('isc_trials', 10)
high_pass_filter = config['preprocessing'].get('high_pass_filter', 1/128)
alphas = np.logspace(
    config['preprocessing'].get('alpha_exp_min', 2),
    config['preprocessing'].get('alpha_exp_max', 7),
    config['preprocessing'].get('alpha_num_points', 16)
)
trim_timepoints = config['preprocessing'].get('trim_timepoints', 10)

# --- Get Subject List ---
subject_list = np.sort(glob.glob(os.path.join(fmri_data_resampled, subject_glob_pattern)))
if not subject_list.any():
    raise FileNotFoundError(f"No resampled subject folders found at: {os.path.join(fmri_data_resampled, subject_glob_pattern)}")
n_subjects = len(subject_list)
print(f"Found {n_subjects} subjects.")

# --- Get Voxel Count (Bug Fix) ---
# Load the mask once to get the number of voxels
try:
    mask_img = nib.load(mask_path)
    n_voxels = np.sum(mask_img.get_fdata() > 0).astype(int)
    print(f"Loaded mask '{mask_filename}' with {n_voxels} voxels.")
except FileNotFoundError:
    raise FileNotFoundError(f"Mask file not found at {mask_path}. Did you run compute_mask.py first?")

# --- Process fMRI Data ---
fmri_subs_runs = []
for sub_id in tqdm(subject_list, desc="Loading, masking, and standardizing subjects"):
    fmri_imgs_sub = sorted(glob.glob(os.path.join(sub_id, '*.nii.gz')))
    
    # Check if subject has the expected number of runs
    if len(fmri_imgs_sub) != n_runs:
        print(f"Warning: Subject {os.path.basename(sub_id)} has {len(fmri_imgs_sub)} runs, but config expects {n_runs}. Skipping subject.")
        continue

    fmri_runs = []  # n_runs x n_timesteps x n_voxels
    for fmri_img in fmri_imgs_sub:
        # A new masker is created for each run, which is correct
        # as detrending and standardizing should be run-specific.
        nifti_masker = NiftiMasker(
            mask_img=mask_path,
            detrend=True,
            standardize=True,
            high_pass=high_pass_filter,
            t_r=t_r
        )
        fmri_runs.append(nifti_masker.fit_transform(fmri_img))
    fmri_subs_runs.append(fmri_runs)

# Update n_subjects in case any were skipped
n_subjects = len(fmri_subs_runs)
if n_subjects == 0:
    raise ValueError("No subjects were successfully processed. Check run counts and file paths.")
print(f"Successfully processed {n_subjects} subjects.")

# --- Save Average Subject Data ---
print(f"Saving average subject data to: {fmri_data_avg_subject}")
for run in range(n_runs):
    # Average across all processed subjects for this run
    fmri_mean_sub = np.mean([fmri_sub_runs[run] for fmri_sub_runs in fmri_subs_runs], axis=0)
    # Standardize the averaged run
    fmri_mean_sub = standardize(fmri_mean_sub, axis=0)
    
    filename = os.path.join(fmri_data_avg_subject, f'average_subject_run-{run}.gz')
    with open(filename, 'wb') as f:
        joblib.dump(fmri_mean_sub, f, compress=4)

# --- Compute Reliable Voxels (ISC) ---
print("Starting ISC calculation...")
np.random.seed(random_seed)

corr_split = []
for i_trial in range(n_trials):
    print('='*80)
    print(f'Trial {i_trial+1}/{n_trials}')
    
    idx_random = np.arange(n_subjects)
    np.random.shuffle(idx_random)
    
    idx_group_1 = idx_random[:n_subjects//2]
    idx_group_2 = idx_random[n_subjects//2:]
    
    # Using trim_timepoints to trim start and end timepoints
    regressors_runs = [
        np.mean([fmri_subs_runs[idx_sub][run][trim_timepoints:-trim_timepoints] for idx_sub in idx_group_1], axis=0)
        for run in range(n_runs)
    ]
    fmri_runs = [
        np.mean([fmri_subs_runs[idx_sub][run][trim_timepoints:-trim_timepoints] for idx_sub in idx_group_2], axis=0)
        for run in range(n_runs)
    ]

    corr_runs = []
    for run_test in range(n_runs):
        tic = time.time()
        
        runs_train = np.setdiff1d(np.arange(n_runs), run_test)
        x_train = np.vstack([regressors_runs[run_train] for run_train in runs_train])
        x_test = regressors_runs[run_test]
        y_train = np.vstack([fmri_runs[run_train] for run_train in runs_train])
        y_test = fmri_runs[run_test]
        
        # Start nested CV
        # Handle cases with few runs (e.g., n_runs=3 for Narratives)
        if len(runs_train) > 1:
            run_val = runs_train[0]
            runs_train_val = np.setdiff1d(runs_train, run_val)
            x_train_val = np.vstack([regressors_runs[run_train_val] for run_train_val in runs_train_val])
            x_val = regressors_runs[run_val]
            y_train_val = np.vstack([fmri_runs[run_train] for run_train in runs_train_val]) # Bug in original, should be runs_train_val
            y_val = fmri_runs[run_val]

            corr_val = []
            for alpha in alphas:
                model = Ridge(alpha=alpha, fit_intercept=False)
                model.fit(x_train_val, y_train_val)
                y_pred = model.predict(x_val)
                corr_tmp = [np.corrcoef(y_val[:,i], y_pred[:,i])[0,1] for i in range(n_voxels)]
                corr_val.append(corr_tmp)

            idx_best_alpha = np.argmax(np.mean(corr_val, axis=1))
            alpha = alphas[idx_best_alpha]
        
        # If n_runs is 2 or less, can't do nested CV, just pick first alpha
        # (This handles n_runs=3 -> len(runs_train)=2 -> len(runs_train_val)=1)
        elif len(runs_train) <= 1: 
            print(f"Run {run_test}: Not enough training runs for nested CV. Using default alpha.")
            alpha = alphas[0]
        # End nested CV
        
        model = Ridge(alpha=alpha, fit_intercept=False)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        corr_tmp = [np.corrcoef(y_test[:,i], y_pred[:,i])[0,1] for i in range(n_voxels)]

        corr_runs.append(corr_tmp)
        
        toc = time.time()
        
        print(f'Run {run_test}\tmean = {np.mean(corr_tmp):.3f}\tmax = {np.max(corr_tmp):.3f}\ttime elapsed = {toc-tic:.3f}s')

    corr_split.append(np.mean(corr_runs, axis=0))

# --- Save ISC results ---
print(f"Saving ISC results to: {isc_output_path}")
with open(isc_output_path, 'wb') as f:
    joblib.dump(np.array(corr_split), f, compress=4)

print(f"compute_average_subject_fmri.py finished for {dataset_name}.")
