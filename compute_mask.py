import numpy as np
import glob
import os
import argparse
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn.masking import compute_multi_epi_mask, intersect_masks
from nilearn.image import swap_img_hemispheres
from tqdm.auto import tqdm # Added for progress bar

# Import the configuration utilities
from config_utils import load_config, make_dir

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml',
                    help='Path to configuration file')
args = parser.parse_args()

# Load configuration
config = load_config(args.config)

# --- Dynamic Path Configuration ---

# Get dataset name
dataset_name = config['dataset']['name']

# Get common paths
home_folder = config['paths']['home_folder']
output_masks_dir = config['paths']['roi_masks']
figures_folder = config['paths']['figures_folder']
make_dir(output_masks_dir)
make_dir(figures_folder)

# Get language (still needed for LPP)
lang = config['experiment']['language'].lower()

# Set up dataset-specific paths and filenames
if dataset_name == 'lpp':
    resampled_folder_name = f'lpp_{lang}_resampled'
    subject_glob_pattern = config['dataset']['subject_glob'].replace('LANG', lang.upper())
    mask_filename = f'mask_lpp_{lang}.nii.gz'
    plot_title = f'Symmetrized fMRI mask - {lang.upper()}'
    plot_filename = f'mask_lpp_{lang}.png'
    
elif dataset_name == 'narratives':
    resampled_folder_name = f'narratives_resampled'
    subject_glob_pattern = config['dataset']['subject_glob'] # sub-[0-9][0-9][0-9]
    mask_filename = 'mask_narratives.nii.gz'
    plot_title = 'Symmetrized fMRI mask - Narratives'
    plot_filename = 'mask_narratives.png'
    
else:
    raise ValueError(f"Dataset '{dataset_name}' not recognized in compute_mask.py")

# --- Find Resampled Files (Unified Logic) ---

# Input path for resampled data
fmri_data_resampled = os.path.join(home_folder, resampled_folder_name)

# Get subject list from the resampled fMRI data
subject_list = np.sort(glob.glob(os.path.join(fmri_data_resampled, subject_glob_pattern)))

if len(subject_list) == 0:
    raise FileNotFoundError(f"No resampled subject folders found at: {os.path.join(fmri_data_resampled, subject_glob_pattern)}")

print(f"Found {len(subject_list)} subjects in {fmri_data_resampled}. Collecting run files...")

fmri_imgs_subs = []
for sub_id in tqdm(subject_list, desc="Scanning subject folders"):
    sub_id_basename = os.path.basename(sub_id)
    # Find all resampled runs for this subject
    run_files = sorted(glob.glob(os.path.join(sub_id, '*.nii.gz')))
    if run_files:
        fmri_imgs_subs.append(run_files)

if not fmri_imgs_subs:
    raise FileNotFoundError(f"No '*.nii.gz' run files found in any subject folder under {fmri_data_resampled}")

print(f"Found {len(np.concatenate(fmri_imgs_subs))} total run files. Computing global mask...")

# --- Compute and Save Mask (Unified Logic) ---

# Get mask threshold from config
mask_threshold = config.get('preprocessing', {}).get('mask_threshold', 0.5)

# Compute the mask
# This function takes all resampled BOLD runs and finds a common mask
mask = compute_multi_epi_mask(np.concatenate(fmri_imgs_subs), threshold=mask_threshold)

# Symmetrize the mask (with configurable intersection threshold)
intersection_threshold = config.get('preprocessing', {}).get('intersection_threshold', 1)
print("Symmetrizing mask...")
mask_sym = intersect_masks([mask, swap_img_hemispheres(mask)], threshold=intersection_threshold)

# Save the mask
mask_output_path = os.path.join(output_masks_dir, mask_filename)
print(f"Saving final symmetrized mask to: {mask_output_path}")
nib.save(mask_sym, mask_output_path)

# --- Optional Visualization ---
if config.get('experiment', {}).get('plot_masks', False):
    print("Generating mask plot...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    plotting.plot_roi(mask_sym, title=plot_title)
    
    plot_save_path = os.path.join(figures_folder, plot_filename)
    plt.savefig(plot_save_path, dpi=300)
    print(f"Mask plot saved to: {plot_save_path}")
    
    if config.get('experiment', {}).get('show_plots', False):
        plt.show()

print(f"compute_mask.py finished for {dataset_name}.")
