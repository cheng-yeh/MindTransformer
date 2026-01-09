import numpy as np
import glob
import os
import argparse
from tqdm.auto import tqdm
from nilearn.image import resample_img
from pathlib import Path

# Import the configuration utilities
# Make sure you have this file available in your path
from config_utils import load_config, make_dir

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml',
                    help='Path to configuration file (e.g., config_lpp.yaml or config_narratives.yaml)')
args = parser.parse_args()

# Load configuration
config = load_config(args.config)

# --- Dynamic Dataset Configuration ---

# Get dataset-specific configuration
try:
    dataset_config = config['dataset']
    dataset_name = dataset_config['name']
    subject_glob_pattern = dataset_config['subject_glob']
    fmri_glob_pattern = dataset_config['fmri_glob_pattern']
except KeyError:
    raise KeyError("Config file is missing the required 'dataset' section with 'name', 'subject_glob', and 'fmri_glob_pattern'.")

# Get fMRI resampling parameters (from preprocessing block)
target_affine = np.array(config['preprocessing']['target_affine'])
target_shape = tuple(config['preprocessing']['target_shape'])

# Get common paths
home_folder = config['paths']['home_folder']
fmri_data_path = config['paths']['fmri_data']

# Set up dataset-specific paths and filenames
if dataset_name == 'lpp':
    lang = config['experiment']['language'].lower()
    assert lang in ['en', 'fr', 'cn'], 'LPP language must be en, fr, or cn.'
    resampled_folder_name = f'lpp_{lang}_resampled'
    subject_glob_pattern = subject_glob_pattern.replace('LANG', lang.upper())
    
elif dataset_name == 'narratives':
    resampled_folder_name = f'narratives_resampled'
    # subject_glob_pattern is already correct (e.g., "sub-[0-9][0-9][0-9]")
    
else:
    raise ValueError(f"Dataset '{dataset_name}' not recognized.")

# Set output path dynamically
output_path = os.path.join(home_folder, resampled_folder_name)
make_dir(output_path)

# Get subject list using the new dynamic glob pattern
subject_list = np.sort(glob.glob(os.path.join(fmri_data_path, subject_glob_pattern)))

if len(subject_list) == 0:
    print(f"Warning: No subjects found at {os.path.join(fmri_data_path, subject_glob_pattern)}")

print(f"Found {len(subject_list)} subjects for dataset '{dataset_name}'. Resampling...")

# --- Main Resampling Loop ---

for sub_id in tqdm(subject_list, desc="Processing Subjects"):
    sub_name = os.path.basename(sub_id)
    make_dir(os.path.join(output_path, sub_name))
    
    # Find fMRI runs using the new dynamic glob pattern
    search_path = os.path.join(sub_id, fmri_glob_pattern)
    fmri_imgs_sub = sorted(glob.glob(search_path))
    
    if not fmri_imgs_sub:
        print(f"Warning: No fMRI files found for {sub_name} using pattern {search_path}")
        continue
    
    # Loop over all found runs
    for run_index, fmri_img_path in enumerate(tqdm(fmri_imgs_sub, desc=f"Runs for {sub_name}", leave=False)):
        
        # Resample the image to the target 4x4x4mm grid
        img_resampled = resample_img(fmri_img_path,
                                     target_affine=target_affine,
                                     target_shape=target_shape,
                                     force_resample=True,
                                     copy_header=True)
        
        # --- START MODIFICATION ---
        
        if dataset_name == 'lpp':
            # Original LPP logic: Standardize to 'run1', 'run2', etc.
            # (The 'run' variable comes from enumerate)
            output_filename = f'{sub_name}_run{run_index+1}.nii.gz'
            
        elif dataset_name == 'narratives':
            # Narratives logic: Preserve the BIDS entities
            # 1. Get the original filename (e.g., "sub-001_task-pieman_run-1_space-...")
            original_basename = os.path.basename(fmri_img_path)
            
            # 2. Extract the key BIDS identifiers
            #    (e.g., "sub-001_task-pieman_run-1")
            #    This split is robust because your glob pattern ensures
            #    '_space-MNI152NLin2009cAsym' is in the name.
            bids_key = original_basename.split('_space-')[0]
            
            # 3. Create the new, clean filename
            output_filename = f'{bids_key}.nii.gz'
            
        # --- END MODIFICATION ---

        # Save with the new, correct filename
        img_resampled.to_filename(os.path.join(output_path,
                                               sub_name,
                                               output_filename))

print(f"\nResampling complete. Output saved to: {output_path}")
