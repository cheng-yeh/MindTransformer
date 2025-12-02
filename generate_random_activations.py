import numpy as np
import pandas as pd
import os
import glob
import joblib
import argparse
from tqdm.auto import tqdm
from config_utils import load_config, make_dir

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml',
                    help='Path to the configuration file')
parser.add_argument('--type', type=str, default=None,
                    help='Override random type (embedding or vector)')
parser.add_argument('--n_dims', type=int, default=None,
                    help='Override number of dimensions')
parser.add_argument('--seed', type=int, default=None,
                    help='Override random seed')
args = parser.parse_args()

# Load configuration
config = load_config(args.config)

# --- Dynamic Path Configuration ---
dataset_name = config['dataset']['name']
output_folder = config['paths']['llms_activations']
annotation_folder = config['paths']['annotation_folder']
make_dir(output_folder)

# Set a language variable for saving files
if dataset_name == 'lpp':
    save_lang = config['experiment']['language'].lower()
elif dataset_name == 'narratives':
    save_lang = 'en' # Narratives is English-only

print(f"--- Processing dataset: {dataset_name} ---")

# --- Load Annotations (Unified) ---
word_list_runs = []
onsets_offsets_runs = []
run_labels = [] # To store run IDs or story names

# ==================================================================
# --- LPP Dataset Logic ---
# ==================================================================
if dataset_name == 'lpp':
    lang = config['experiment']['language'].lower()
    assert lang in ['en', 'fr', 'cn'], 'This language is not available. Please choose between en, fr or cn.'

    # Load LPP word onsets
    filename = os.path.join(annotation_folder, lang.upper(), f'lpp{lang.upper()}_word_information.csv')
    try:
        df_word_onsets = pd.read_csv(filename)
        print(f"Successfully loaded LPP word onsets from: {filename}")
    except Exception as e:
        raise RuntimeError(f"Failed to load word onsets from {filename}: {str(e)}")

    # Language-specific data cleaning
    if lang == 'en':
        df_word_onsets = df_word_onsets.drop([3919,6775,6781])
    elif lang == 'fr':
        df_word_onsets.loc[3332, 'word'] = 'de'
        df_word_onsets.loc[3379, 'word'] = 'trois'
        # ... (all other 'fr' cleaning lines) ...
        df_word_onsets = df_word_onsets.drop([338, 1204, 3333])
    elif lang == 'cn':
        pass

    n_runs_config = config['general']['n_runs']
    print(f"Loading {n_runs_config} runs for LPP...")

    for run in range(n_runs_config):
        run_labels.append(f'run-{run+1}')
        df_word_onsets_run = df_word_onsets[df_word_onsets.section==(run+1)]
        word_list_tmp = df_word_onsets_run.word.to_numpy()
        onsets_tmp = df_word_onsets_run.onset.to_numpy()
        offsets_tmp = df_word_onsets_run.offset.to_numpy()
        
        word_list = []
        onsets = []
        offsets = []
        
        for idx_word, (word, onset, offset) in enumerate(zip(word_list_tmp, onsets_tmp, offsets_tmp)):
            if isinstance(word, str) and word != ' ':
                word_list.append(word)
                onsets.append(onset)
                offsets.append(offset)
                
        onsets_offsets_runs.append((np.array(onsets), np.array(offsets)))
        word_list_runs.append(word_list)

# ==================================================================
# --- Narratives Dataset Logic ---
# ==================================================================
elif dataset_name == 'narratives':
    story_dirs = sorted(glob.glob(os.path.join(annotation_folder, '*/')))
    if not story_dirs:
        raise FileNotFoundError(f"No story directories found in {annotation_folder}")
    
    print(f"Found {len(story_dirs)} stories in {annotation_folder}. Loading each...")

    for story_dir in tqdm(story_dirs, desc="Loading Stories"):
        story_name = os.path.basename(os.path.normpath(story_dir))
        run_labels.append(story_name)
        
        align_file = os.path.join(story_dir, 'align.csv')
        
        try:
            # Load headerless CSV: 0=orig, 1=lower, 2=onset, 3=offset
            df_story = pd.read_csv(align_file, header=None, names=['word_orig', 'word', 'onset', 'offset'])
        except Exception as e:
            print(f"Warning: Could not read {align_file}. Skipping story. Error: {e}")
            continue

        word_list = df_story['word'].astype(str).values
        onsets = df_story['onset'].values
        offsets = df_story['offset'].values
        
        onsets_offsets_runs.append((np.array(onsets), np.array(offsets)))
        word_list_runs.append(word_list)

# --- End Data Loading ---

# Get the total number of runs/stories that were loaded
n_runs = len(word_list_runs)
if n_runs == 0:
    raise ValueError("No runs or stories were successfully loaded.")
print(f"Successfully loaded {n_runs} runs/stories.")

# --- Save Onsets/Offsets ---
# This file is identical to the one saved by the GloVe script,
# ensuring it's available regardless of which baseline is run.
onsets_offsets_file = os.path.join(output_folder, f'onsets_offsets_{dataset_name}_{save_lang}.gz')
if not os.path.exists(onsets_offsets_file):
    print(f"Saving onsets/offsets to: {onsets_offsets_file}")
    with open(onsets_offsets_file, 'wb') as f:
        joblib.dump(onsets_offsets_runs, f, compress=5)
else:
    print(f"Onsets/offsets file already exists: {onsets_offsets_file}")

# --- Save Run Labels ---
# (Also saved by GloVe script, but good to have here for consistency)
run_labels_file = os.path.join(output_folder, f'run_labels_{dataset_name}_{save_lang}.gz')
if not os.path.exists(run_labels_file):
    print(f"Saving run labels to: {run_labels_file}")
    with open(run_labels_file, 'wb') as f:
        joblib.dump(run_labels, f, compress=5)

# --- Configure Random Baselines ---
if args.type is not None or args.n_dims is not None or args.seed is not None:
    # Create a single config with the provided overrides
    configs_to_process = [{
        'type': args.type if args.type is not None else 'embedding',
        'n_dims': args.n_dims if args.n_dims is not None else 300,
        'seed': args.seed if args.seed is not None else 1
    }]
else:
    # Check if the baselines->random_embeddings structure exists
    if 'baselines' in config and 'random_embeddings' in config['baselines'] and config['baselines']['random_embeddings'].get('enabled', False):
        configs_to_process = config['baselines']['random_embeddings'].get('configs', [])
    else:
        # Fallback for old/missing config
        print("Warning: 'baselines.random_embeddings' not found or not enabled in config. Using default.")
        configs_to_process = [{ 'type': 'embedding', 'n_dims': 300, 'seed': 1 }]

if not configs_to_process:
     print("No random embedding configs found to process.")

# --- Process each configuration ---
for config_idx, random_config in enumerate(configs_to_process):
    random_type = random_config.get('type', 'embedding')
    n_dims = random_config.get('n_dims', 300)
    seed = random_config.get('seed', 1)
    
    assert random_type in ['embedding', 'vector'], f'Random type must be "embedding" or "vector", got {random_type}'
    
    # Construct model name with dataset info
    model_name = f'random_{random_type}_{n_dims}d_seed{seed}_{dataset_name}_{save_lang}'
    
    print(f"\nProcessing config {config_idx+1}/{len(configs_to_process)}")
    print(f"Generating random {random_type} with {n_dims} dimensions (seed {seed})")
    
    np.random.seed(seed)
    
    runs_words_activations = []
    
    if random_type == 'vector':
        print(f"Generating random vectors for each word")
        for run in range(n_runs): # n_runs is now the length of word_list_runs
            words_activations = [np.random.randn(n_dims) for _ in range(len(word_list_runs[run]))]
            runs_words_activations.append([words_activations])
            
    elif random_type == 'embedding':
        print(f"Generating random embeddings for each unique word")
        # First create random embeddings for all unique words
        word_embeddings = {}        
        for run in range(n_runs):
            for word in word_list_runs[run]:
                if word not in word_embeddings:
                    word_embeddings[word] = np.random.randn(n_dims)
        
        # Then use these embeddings for each run
        for run in range(n_runs):
            words_activations = [word_embeddings[word] for word in word_list_runs[run]]
            runs_words_activations.append([words_activations])

    # Save the generated activations (n_runs x 1 x n_words x n_neurons)
    activation_file = os.path.join(output_folder, f'{model_name}.gz')
    print(f"Saving activations to: {activation_file}")
    with open(activation_file, 'wb') as f:
        joblib.dump(runs_words_activations, f, compress=5)

print("\nRandom embeddings generation completed successfully!")
