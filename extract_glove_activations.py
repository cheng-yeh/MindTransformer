import numpy as np
import pandas as pd
import os
import glob
import joblib
import argparse
from tqdm.auto import tqdm
from config_utils import load_config, make_dir
# --- NEW: Imports for Interpolation ---
from itertools import groupby
from operator import itemgetter

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml',
                    help='Path to the configuration file')
args = parser.parse_args()

# Load configuration
config = load_config(args.config)

# --- Dynamic Path Configuration ---
dataset_name = config['dataset']['name']
output_folder = config['paths']['llms_activations']
annotation_folder = config['paths']['annotation_folder']
home_folder = config['paths']['home_folder']
make_dir(output_folder)

# Set model name
model_name = 'glove'
lang = 'en' # GloVe embeddings are English-only

# --- Load GloVe Embeddings ---
glove_path = config['paths']['glove_path']
print(f"Loading GloVe embeddings from: {glove_path}")

def load_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

try:
    glove_embeddings = load_embeddings(glove_path)
    n_dims = glove_embeddings['the'].shape[0]
    print(f"Successfully loaded GloVe embeddings with dimension: {n_dims}")
except Exception as e:
    raise RuntimeError(f"Failed to load GloVe embeddings from {glove_path}: {str(e)}")

# --- Initialize Output Lists ---
runs_words_activations = []
onsets_offsets_runs = []
run_labels = [] 
unknown_words = []

print(f"--- Processing dataset: {dataset_name} ---")

# ==================================================================
# --- LPP Dataset Logic ---
# ==================================================================
if dataset_name == 'lpp':
    config_lang = config['experiment']['language'].lower()
    if config_lang != 'en':
        print(f"Warning: GloVe is English-only, but config lang is '{config_lang}'. Processing 'en'.")
    
    n_runs = config['general']['n_runs']
    
    # Load LPP word onsets
    filename = os.path.join(annotation_folder, 'EN', 'lppEN_word_information.csv')
    try:
        df_word_onsets = pd.read_csv(filename)
        print(f"Successfully loaded LPP word onsets from: {filename}")
    except Exception as e:
        raise RuntimeError(f"Failed to load word onsets from {filename}: {str(e)}")

    # Data cleaning for LPP
    df_word_onsets = df_word_onsets.drop([3919, 6775, 6781])

    for run in range(n_runs):
        run_label = f'run-{run+1}'
        run_labels.append(run_label)
        
        df_word_onsets_run = df_word_onsets[df_word_onsets.section == (run+1)]
        word_list_tmp = df_word_onsets_run.word.to_numpy()
        onsets_tmp = df_word_onsets_run.onset.to_numpy()
        offsets_tmp = df_word_onsets_run.offset.to_numpy()
        
        word_list = []
        onsets = []
        offsets = []
        
        # LPP Filtering Logic (Matches LLM script)
        for idx_word, (word, onset, offset) in enumerate(zip(word_list_tmp, onsets_tmp, offsets_tmp)):
            if isinstance(word, str) and word != ' ':
                word_list.append(word)
                onsets.append(onset)
                offsets.append(offset)
                
        onsets_offsets_runs.append((np.array(onsets), np.array(offsets)))
        
        # Process each run to create activations
        words_activations = []
        for word in word_list:
            # Apply LPP-specific preprocessing
            word = word.lower().replace("'", "").replace(';','')
            if word == 'na\ive':
                word = 'naive'
            if word == 'redfaced':
                word = 'red-faced'
            
            if word in glove_embeddings:
                words_activations.append(glove_embeddings[word])
            elif word == 'three two five':
                words_activations.append((glove_embeddings['three']+glove_embeddings['two']+glove_embeddings['five'])/3)
            elif word == 'three two six':
                words_activations.append((glove_embeddings['three']+glove_embeddings['two']+glove_embeddings['six'])/3)
            elif word == 'three two seven':
                words_activations.append((glove_embeddings['three']+glove_embeddings['two']+glove_embeddings['seven'])/3)
            elif word == 'three two eight':
                words_activations.append((glove_embeddings['three']+glove_embeddings['two']+glove_embeddings['eight'])/3)
            elif word == 'three two nine':
                words_activations.append((glove_embeddings['three']+glove_embeddings['two']+glove_embeddings['nine'])/3)
            elif word == 'three three zero':
                words_activations.append((glove_embeddings['three']*2.0+glove_embeddings['zero'])/3)
            else:
                # print(f'Unknown word in run {run_label}: {word}')
                unknown_words.append(word)
                words_activations.append(np.zeros(n_dims))
        
        runs_words_activations.append([words_activations]) 

# ==================================================================
# --- Narratives Dataset Logic ---
# ==================================================================
elif dataset_name == 'narratives':
    story_dirs = sorted(glob.glob(os.path.join(annotation_folder, '*/')))
    if not story_dirs:
        raise FileNotFoundError(f"No story directories found in {annotation_folder}")
    
    print(f"Found {len(story_dirs)} stories in {annotation_folder}. Processing each...")

    for story_dir in tqdm(story_dirs, desc="Processing Stories"):
        story_name = os.path.basename(os.path.normpath(story_dir))
        run_labels.append(story_name)
        
        align_file = os.path.join(story_dir, 'align.csv')
        
        try:
            # --- MODIFICATION: Robust Parsing & Interpolation (Matches extract_llm) ---
            # 1. Force dtype=str
            df_story = pd.read_csv(align_file, header=None, names=['word_orig', 'word', 'onset', 'offset'], dtype=str)
            
            # 2. Coerce to numeric
            df_story['onset'] = pd.to_numeric(df_story['onset'], errors='coerce')
            df_story['offset'] = pd.to_numeric(df_story['offset'], errors='coerce')

            # 3. Interpolate NaNs (Instead of Dropping)
            nan_mask = df_story['onset'].isna() | df_story['offset'].isna()
            if nan_mask.any():
                indices = df_story.index[nan_mask]
                for k, g in groupby(enumerate(indices), lambda ix: ix[0] - ix[1]):
                    group_indices = list(map(itemgetter(1), g))
                    start_idx = group_indices[0]
                    end_idx = group_indices[-1]

                    if start_idx == 0: t_start = 0.0
                    else: t_start = df_story.at[start_idx - 1, 'offset']

                    if end_idx == len(df_story) - 1:
                        t_end = t_start + (len(group_indices) * 1.0) # Heuristic tail
                    else:
                        t_end = df_story.at[end_idx + 1, 'onset']

                    duration = t_end - t_start
                    if duration < 0: duration = 0
                    step = duration / len(group_indices)

                    for i, idx in enumerate(group_indices):
                        df_story.at[idx, 'onset'] = t_start + (i * step)
                        df_story.at[idx, 'offset'] = t_start + ((i + 1) * step)
            
            # --- END MODIFICATION ---

        except Exception as e:
            print(f"Warning: Could not read {align_file}. Skipping story. Error: {e}")
            continue

        word_list = df_story['word'].astype(str).values
        onsets = df_story['onset'].values
        offsets = df_story['offset'].values
        
        onsets_offsets_runs.append((np.array(onsets), np.array(offsets)))
        
        # Process each story to create activations
        words_activations = []
        for word in word_list:
            word = word.lower().replace("'", "")
            
            if word in glove_embeddings:
                words_activations.append(glove_embeddings[word])
            else:
                unknown_words.append(word)
                words_activations.append(np.zeros(n_dims))
                
        runs_words_activations.append([words_activations]) 

# --- Summary and Save ---
if unknown_words:
    print(f"Found {len(unknown_words)} total unknown words ({len(set(unknown_words))} unique).")
else:
    print("All words were found in the GloVe embeddings.")

# Save activations
activation_file = os.path.join(output_folder, f'{model_name}_{dataset_name}_{lang}.gz')
print(f"Saving activations to: {activation_file}")
with open(activation_file, 'wb') as f:
    joblib.dump(runs_words_activations, f, compress=4)

# Save onsets/offsets
onsets_offsets_file = os.path.join(output_folder, f'onsets_offsets_{dataset_name}_{lang}.gz')
print(f"Saving onsets/offsets to: {onsets_offsets_file}")
with open(onsets_offsets_file, 'wb') as f:
    joblib.dump(onsets_offsets_runs, f, compress=4)

# Save Run Labels
run_labels_file = os.path.join(output_folder, f'run_labels_{dataset_name}_{lang}.gz')
print(f"Saving run labels to: {run_labels_file}")
with open(run_labels_file, 'wb') as f:
    joblib.dump(run_labels, f, compress=4)

print("GloVe processing completed successfully!")

