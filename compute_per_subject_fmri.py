import numpy as np
import os
import glob
import joblib
import argparse
from tqdm.auto import tqdm
from nilearn.input_data import NiftiMasker
import nibabel as nib
import time

# Import the configuration utilities
from config_utils import load_config, make_dir, standardize

# --- Script Configuration ---
parser = argparse.ArgumentParser(description="Pre-process fMRI data on a per-subject basis.")
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
output_masks_dir = config['paths']['roi_masks']

# Set up dataset-specific paths and filenames
if dataset_name == 'lpp':
    assert lang in ['en', 'fr', 'cn'], 'LPP language must be en, fr, or cn.'
    resampled_folder_name = f'lpp_{lang}_resampled'
    output_dir_name = f'lpp_{lang}_per_subject'
    subject_glob_pattern = config['dataset']['subject_glob'].replace('LANG', lang.upper())
    mask_filename = f'mask_lpp_{lang}.nii.gz'
    
elif dataset_name == 'narratives':
    resampled_folder_name = 'narratives_resampled'
    output_dir_name = 'narratives_per_subject' # Following the same naming convention
    subject_glob_pattern = config['dataset']['subject_glob'] # sub-[0-9][0-9][0-9]
    mask_filename = 'mask_narratives.nii.gz'
    
else:
    raise ValueError(f"Dataset '{dataset_name}' not recognized.")

# Construct dynamic paths
fmri_data_resampled = os.path.join(home_folder, resampled_folder_name)
fmri_data_per_subject = os.path.join(home_folder, output_dir_name)
mask_path = os.path.join(output_masks_dir, mask_filename)

print(f"--- Running compute_per_subject_fmri.py for dataset: {dataset_name} ---")
print(f"Setting output directory to: {fmri_data_per_subject}")
make_dir(fmri_data_per_subject)

# --- Experimental Parameters ---
# n_runs = config['general']['n_runs'] # We don't use this strict check anymore
t_r = config['general']['t_r']
high_pass_filter = config['preprocessing'].get('high_pass_filter', 1/128)

# Get subject list
subject_list = np.sort(glob.glob(os.path.join(fmri_data_resampled, subject_glob_pattern)))

# --- Main Processing Loop ---
print(f"Found {len(subject_list)} subjects. Starting per-subject preprocessing...")

for sub_id_path in tqdm(subject_list, desc="Processing subjects"):
    sub_id_basename = os.path.basename(sub_id_path) # e.g., "sub-EN057" or "sub-001"
    
    # --- Create a save directory for this subject ---
    subject_output_dir = os.path.join(fmri_data_per_subject, sub_id_basename)
    make_dir(subject_output_dir)

    fmri_imgs_sub = sorted(glob.glob(os.path.join(sub_id_path, '*.nii.gz')))
    
    # We no longer check n_runs, as decided.
    
    # Process and save each run for this subject
    for run_index, fmri_img_path in enumerate(fmri_imgs_sub):
        
        # 1. Initialize NiftiMasker
        #    (Detrends and applies first standardization)
        nifti_masker = NiftiMasker(
            mask_img=mask_path, 
            detrend=True, 
            standardize=True, # First standardization
            high_pass=high_pass_filter, 
            t_r=t_r
        )
        
        # 2. Apply masker
        #    fmri_run_data is (n_timesteps, n_voxels)
        try:
            fmri_run_data = nifti_masker.fit_transform(fmri_img_path)
        except Exception as e:
            print(f"  ERROR: Failed to process {fmri_img_path} for subject {sub_id_basename}. Error: {e}")
            break # Skip to the next subject if a run fails

        # 2.1. Replace any NaNs created by the masker with 0.0
        np.nan_to_num(fmri_run_data, copy=False, nan=0.0)
    
        # 3. Apply the *same* post-masker standardization as the averaging script
        #    (This is the second standardization)
        fmri_run_data_standardized = standardize(fmri_run_data, axis=0)

        # 3.1. Clean up NaNs from the second standardization
        np.nan_to_num(fmri_run_data_standardized, copy=False, nan=0.0)
        
        # --- START MODIFICATION ---
        
        if dataset_name == 'lpp':
            # Original LPP logic: save as run-0, run-1, etc.
            # This is fine since LPP runs are identical across subjects
            output_filename = f'{sub_id_basename}_run-{run_index}.gz'
            
        elif dataset_name == 'narratives':
            # Narratives logic: Preserve the BIDS entities
            # 1. Get the original filename (e.g., "sub-001_task-pieman_run-1.nii.gz")
            original_basename = os.path.basename(fmri_img_path)
            
            # 2. Remove the .nii.gz extension
            output_key = original_basename.rsplit('.nii.gz', 1)[0]
            
            # 3. Create the new filename
            output_filename = f'{output_key}.gz'
            
        # --- END MODIFICATION ---

        # 4. Save the processed file
        filename = os.path.join(subject_output_dir, output_filename)
        
        with open(filename, 'wb') as f:
            joblib.dump(fmri_run_data_standardized, f, compress=4)

print("\n--- Per-subject preprocessing complete. ---")
print(f"All processed subject data saved in {fmri_data_per_subject}")
