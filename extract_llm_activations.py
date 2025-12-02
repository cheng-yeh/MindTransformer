import numpy as np
import pandas as pd
import os
import zipfile
import joblib
import argparse
import gc
import glob
from tqdm.auto import tqdm
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from config_utils import load_config, make_dir
from itertools import groupby
from operator import itemgetter

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- Global config-dependent variables ---
dataset_name = ''
lang = ''
punct = []
chinese_punct = []

def clear_cuda_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def simplify_word(word):
    word = str(word).lower().replace(' ', '')
    for p in punct: word = word.replace(p, '')
    for p in chinese_punct: word = word.replace(p, '')
    return word

def do_word_match(word_in_list, word_in_text):
    if word_in_list.lower() in word_in_text.lower():
        if word_in_list == simplify_word(word_in_text): return 'exact'
        return 'fuzzy_substring'
    
    word_in_text_simplified = simplify_word(word_in_text)
    if word_in_list == word_in_text_simplified: return 'exact'

    if dataset_name == 'lpp':
        if any([(word_in_list == 's' and word_in_text_simplified.endswith('s')),(word_in_list == 'nt' and word_in_text_simplified == 't'),(word_in_list == 'i' and word_in_text_simplified == 'ive'),(word_in_list == 'did' and word_in_text_simplified == 'didn'),(word_in_list == 'does' and word_in_text_simplified == "doesn"),(word_in_list == 'do' and word_in_text_simplified == "don"),(word_in_list == 'is' and word_in_text_simplified == "isn")]): return 'forgave_contraction'
        if any([(word_in_list == 'one' and word_in_text_simplified == '1'),(word_in_list == 'threetwofive' and word_in_text_simplified == "3"),(word_in_list == 'threetwosix' and word_in_text_simplified == "3"),(word_in_list == 'threetwoseven' and word_in_text_simplified == "3"),(word_in_list == 'threetwoeight' and word_in_text_simplified == "3"),(word_in_list == 'threetwonine' and word_in_text_simplified == "3"),(word_in_list == 'threethreezero' and word_in_text_simplified == "3")]): return 'forgave_number'
        if any([(word_in_list == 'na\ive' and word_in_text_simplified == 'naive'),(word_in_list == 'repondit' and word_in_text_simplified == 'répondit'),(word_in_list == 'oeuvre' and word_in_text_simplified == 'œuvre'),(word_in_list == 'oeil' and word_in_text_simplified == 'œil'),(word_in_list == 'a' and word_in_text_simplified == 'à'),(word_in_list == 'coeur' and word_in_text_simplified == 'cœur')]): return 'forgave_special_char'
    
    if len(word_in_text_simplified) > 0 and word_in_list.startswith(word_in_text_simplified): return 'fuzzy_startswith'
    if len(word_in_list) > (1 - (lang == 'cn')) and word_in_list in word_in_text_simplified: return 'fuzzy_substring'
    if len(word_in_text_simplified) > (1 - (lang == 'cn')) and word_in_text_simplified in word_in_list: return 'fuzzy_substring'
    return False

# --- EXTRACTOR ---
class ActivationExtractor:
    def __init__(self, model, model_type, activation_keys, target_layer=-1):
        self.model = model
        self.model_type_str = model_type.lower()
        self.original_config = model.config
        self.config = model.config
        self.activation_keys = activation_keys
        self.target_layer = target_layer
        self.hooks = []
        self.activations = defaultdict(dict)
        self.rope_cos_sin = None
       
        self.is_gemma3_multimodal = self._is_gemma3_multimodal_case()
        if self.is_gemma3_multimodal:
            self.config = self.original_config.text_config
            self.text_model_component = self.model.model.language_model
            self.layers_path_within_component = "layers"
       
        self.arch_maps = {
            'llama_family': {
                'layers_path': 'model.layers',
                'first_norm': 'input_layernorm', 
                'q_proj': 'self_attn.q_proj', 'k_proj': 'self_attn.k_proj', 
                'v_proj': 'self_attn.v_proj', 'o_proj': 'self_attn.o_proj', 
                'second_norm': 'post_attention_layernorm', 'ffn_down_proj': 'mlp.down_proj',
                'gate_proj': 'mlp.gate_proj', 'up_proj': 'mlp.up_proj',
            },
            'gpt_oss_family': {
                'layers_path': 'model.layers', 'first_norm': 'input_layernorm',
                'q_proj': 'self_attn.q_proj', 'k_proj': 'self_attn.k_proj',
                'v_proj': 'self_attn.v_proj', 'o_proj': 'self_attn.o_proj',
                'second_norm': 'post_attention_layernorm', 'ffn_down_proj': 'mlp',
            }
        }
        self.model_family = self._get_model_family()

    def _is_gemma3_multimodal_case(self):
        return "gemma-3" in self.original_config.name_or_path and "forconditionalgeneration" in self.model.__class__.__name__.lower()

    def get_num_layers(self):
        if self.is_gemma3_multimodal: return self.config.num_hidden_layers
        layers_path_key = self.arch_maps[self.model_family]['layers_path']
        return len(self._get_module(self.model, layers_path_key))

    def _get_model_family(self):
        if 'gpt-oss' in self.model_type_str: return 'gpt_oss_family'
        if any(t in self.model_type_str for t in ['llama', 'gemma', 'mistral', 'qwen']): return 'llama_family'
        raise ValueError(f"Unsupported model type: {self.model_type_str}")

    def _get_module(self, parent_module, path_key):
        module = parent_module
        for part in path_key.split('.'): module = getattr(module, part)
        return module

    def _save_activation(self, key, layer_idx, tensor):
        if key.endswith('_cos_sin'):
            self.activations[key][layer_idx] = (tensor[0].detach().to(torch.float32).cpu(), tensor[1].detach().to(torch.float32).cpu())
        else:
            if isinstance(tensor, tuple): tensor = tensor[0]
            self.activations[key][layer_idx] = tensor.detach().to(torch.float32).cpu()

    def _create_hooks(self, layer_idx):
        layer = self._get_layer(layer_idx)
        arch_map = self.arch_maps[self.model_family]

        if 'first_norm' in arch_map:
            def hook_first_norm(module, module_input, module_output):
                self._save_activation('input_hidden_state', layer_idx, module_input[0])
                self._save_activation('pre_attn_norm', layer_idx, module_output)
            self.hooks.append(self._get_module(layer, arch_map['first_norm']).register_forward_hook(hook_first_norm))

        if 'q_proj' in arch_map:
            def create_qkv_hook(key_char):
                def hook(module, module_input, module_output):
                    self._save_activation(f'per_head_{key_char}', layer_idx, module_output)
                    B, S, D = module_output.shape
                    num_heads = self.config.num_attention_heads if key_char == 'q' else getattr(self.config, 'num_key_value_heads', self.config.num_attention_heads)
                    head_dim = D // num_heads
                    per_head = module_output.view(B, S, num_heads, head_dim).transpose(1, 2)
                    self._save_activation(f'head_averaged_{key_char}', layer_idx, per_head.mean(dim=1))
                return hook
            self.hooks.append(self._get_module(layer, arch_map['q_proj']).register_forward_hook(create_qkv_hook('q')))
            self.hooks.append(self._get_module(layer, arch_map['k_proj']).register_forward_hook(create_qkv_hook('k')))
            self.hooks.append(self._get_module(layer, arch_map['v_proj']).register_forward_hook(create_qkv_hook('v')))

        if 'o_proj' in arch_map:
            def hook_o_proj(module, module_input, module_output):
                self._save_activation('per_head_context_vector', layer_idx, module_input[0])
                B, S, D = module_input[0].shape
                num_heads = self.config.num_attention_heads
                head_dim = D // num_heads
                per_head_context = module_input[0].view(B, S, num_heads, head_dim).transpose(1, 2)
                self._save_activation('head_averaged_context_vector', layer_idx, per_head_context.mean(dim=1))
                self._save_activation('attn_output', layer_idx, module_output)
            self.hooks.append(self._get_module(layer, arch_map['o_proj']).register_forward_hook(hook_o_proj))

        if 'second_norm' in arch_map:
            def hook_second_norm(module, module_input, module_output):
                self._save_activation('post_attn_hidden_state', layer_idx, module_input[0])
                self._save_activation('pre_ffn_norm', layer_idx, module_output)
            self.hooks.append(self._get_module(layer, arch_map['second_norm']).register_forward_hook(hook_second_norm))

        if 'gate_proj' in arch_map:
             def hook_gate_proj(module, module_input, module_output): self._save_activation('ffn_gate', layer_idx, module_output)
             self.hooks.append(self._get_module(layer, arch_map['gate_proj']).register_forward_hook(hook_gate_proj))
             
        if 'up_proj' in arch_map:
             def hook_up_proj(module, module_input, module_output): self._save_activation('ffn_up', layer_idx, module_output)
             self.hooks.append(self._get_module(layer, arch_map['up_proj']).register_forward_hook(hook_up_proj))

        if 'ffn_down_proj' in arch_map:
            def hook_ffn_down_proj(module, module_input, module_output):
                ffn_output_tensor = module_output[0] if isinstance(module_output, tuple) else module_output
                self._save_activation('ffn_activated_state', layer_idx, module_input[0])
                self._save_activation('ffn_output', layer_idx, ffn_output_tensor)
                post_attn_state = self.activations['post_attn_hidden_state'][layer_idx]
                final_output = post_attn_state.to(ffn_output_tensor.device) + ffn_output_tensor
                self._save_activation('final_block_output', layer_idx, final_output)
            self.hooks.append(self._get_module(layer, arch_map['ffn_down_proj']).register_forward_hook(hook_ffn_down_proj))

    def _get_layer(self, layer_idx):
        if self.is_gemma3_multimodal: return getattr(self.text_model_component, self.layers_path_within_component)[layer_idx]
        return self._get_module(self.model, self.arch_maps[self.model_family]['layers_path'])[layer_idx]

    def start(self):
        if self.is_gemma3_multimodal: rope_module = getattr(self.text_model_component, 'rotary_emb', None)
        else: rope_module = getattr(self.model.model, 'rotary_emb', None)
        if rope_module:
            def rope_hook(module, module_input, module_output):
                self.rope_cos_sin = (module_output[0].detach().cpu(), module_output[1].detach().cpu())
            self.hooks.append(rope_module.register_forward_hook(rope_hook))
        layers_to_hook = range(self.get_num_layers()) if self.target_layer == -1 else [self.target_layer]
        for layer_idx in layers_to_hook: self._create_hooks(layer_idx)

    def stop(self):
        for hook in self.hooks: hook.remove()
        self.hooks = []
    
    def clear(self):
        self.activations.clear()
        self.rope_cos_sin = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--layer', type=int, default=-1, help='Specify a single layer to extract (0-indexed). Default is -1 to extract all layers.')
    args = parser.parse_args()

    config = load_config(args.config)
    enabled_models = [m for m in config['models'] if m.get('enabled', True)]
    if not enabled_models: raise ValueError("No enabled models found in configuration")

    global dataset_name, lang, punct, chinese_punct
    dataset_name = config['dataset']['name']

    if dataset_name == 'lpp': lang = config['experiment']['language'].lower()
    elif dataset_name == 'narratives': lang = 'en'
    assert lang in ['en', 'fr', 'cn']

    punct = config['text_processing']['punctuation']['standard']
    chinese_punct = config['text_processing']['punctuation']['chinese']

    print(f"--- Starting LLM Extraction for dataset: {dataset_name} (lang: {lang}) ---")

    # --- CONFIG LOADING ---
    requested_keys = config['experiment'].get('activation_mode', [])
    activation_keys = sorted(list(set(requested_keys)))
    print(f"INFO: Will extract {len(activation_keys)} activation types: {activation_keys}")

    api_keys = {'hidden_state', 'per_head_k_cache', 'per_head_v_cache', 'head_averaged_k_cache', 'head_averaged_v_cache'}
    requested_api_keys = [k for k in activation_keys if k in api_keys]
    extract_from_api = len(requested_api_keys) > 0
    if extract_from_api and args.layer == -1:
        raise ValueError("To extract hidden_state or kv_cache, you must specify a single layer using the --layer argument.")

    if args.layer != -1: print(f"INFO: Targeting SINGLE layer extraction: Layer {args.layer}")

    seed = config['experiment'].get('model_seed', 0)
    text_proc_config = config.get('text_processing', {})
    proc_config = text_proc_config.get('processing', {})
    batch_size = proc_config.get('batch_size', 1)
    max_tokens = proc_config.get('max_tokens', 4096)
    CHUNK_SIZE = max_tokens

    print(f"INFO: Using max_tokens = {max_tokens}, batch_size = {batch_size}, CHUNK_SIZE = {CHUNK_SIZE}")

    if seed > 0: torch.manual_seed(seed)

    home_folder = config['paths']['home_folder']
    output_folder = config['paths']['llms_activations']
    annotation_folder = config['paths']['annotation_folder']
    make_dir(output_folder)

    word_match_attempts = config['text_processing']['word_match_attempts']
    trim_words_start = config['text_processing'].get('trim_words_start', 0)
    trim_words_end = config['text_processing'].get('trim_words_end', 0)

    run_labels_file = os.path.join(output_folder, f'run_labels_{dataset_name}_{lang}.gz')
    try:
        with open(run_labels_file, 'rb') as f: run_labels = joblib.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {run_labels_file}")
        raise
    
    # --- Verification Load ---
    master_onsets_file = os.path.join(output_folder, f'onsets_offsets_{dataset_name}_{lang}.gz')
    master_onsets_offsets = []
    try:
        with open(master_onsets_file, 'rb') as f:
            master_onsets_offsets = joblib.load(f)
        print(f"DEBUG: Loaded master onsets file with {len(master_onsets_offsets)} runs for verification.")
    except FileNotFoundError:
        print("DEBUG: Master onsets file not found. Verification steps will be skipped.")

    n_runs = len(run_labels)
    print(f"Found {n_runs} runs/stories to process.")

    for model_config in tqdm(enabled_models, desc="Processing models"):
        model_name, model_type = model_config['name'], model_config['type']
        print(f"\n\n{'='*50}\nProcessing model: {model_name} (Type: {model_type})\n{'='*50}")
        model_match_counts = {}

        try:
            from_pretrained_kwargs = {
                "output_attentions": False, "low_cpu_mem_usage": True,
                "device_map": "auto", "torch_dtype": "auto", "output_hidden_states": extract_from_api,
            }

            print(f"Loading model {model_name} with settings: {from_pretrained_kwargs}")
            token = config['auth'].get('huggingface_token')
            model = AutoModelForCausalLM.from_pretrained(model_name, token=token, **from_pretrained_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

            temp_extractor = ActivationExtractor(model, model_type, [], -1)
            n_layers = temp_extractor.get_num_layers()
            del temp_extractor
            clear_cuda_cache()

            extractor = ActivationExtractor(model, model_type, activation_keys, args.layer)

            for run in tqdm(range(n_runs), desc=f"Processing runs for {model_name}", leave=False):
                run_label = run_labels[run]

                # --- Load Data ---
                word_list = []
                fulltext_run = ""
                if dataset_name == 'lpp':
                    run_index_0 = int(run_label.split('-')[1]) - 1
                    text_zip_location = os.path.join(home_folder, config['text_processing']['text_zip'])
                    text_archive_password = config['text_processing']['text_archive_password'].encode('utf-8')
                    text_filename = os.path.join(f'lpp_{lang}_text', f'text_{lang}_run{run_index_0+1}.txt')
                    with zipfile.ZipFile(text_zip_location, 'r') as text_archive:
                        fulltext_run = text_archive.read(text_filename, pwd=text_archive_password).decode('utf8').replace('\n', ' ')
                    lpp_csv_file = os.path.join(annotation_folder, lang.upper(), f'lpp{lang.upper()}_word_information.csv')
                    df_word_onsets = pd.read_csv(lpp_csv_file)
                    if lang == 'en': df_word_onsets = df_word_onsets.drop([3919, 6775, 6781])
                    df_run = df_word_onsets[df_word_onsets.section == (run_index_0 + 1)]
                    valid_indices = [isinstance(word, str) and word != ' ' for word in df_run.word]
                    word_list_full = df_run.word[valid_indices].tolist()
                    print(f"DEBUG: [LPP] Run {run_label} - Valid Words: {len(word_list_full)}")
                    word_list = word_list_full
                    del df_word_onsets, df_run

                elif dataset_name == 'narratives':
                    story_dir = os.path.join(annotation_folder, run_label)
                    try:
                        with open(os.path.join(story_dir, 'transcript.txt'), 'r', encoding='utf-8') as f: fulltext_run = f.read().replace('\n', ' ')
                    except FileNotFoundError: continue
                    try:
                        df_story = pd.read_csv(os.path.join(story_dir, 'align.csv'), header=None, names=['word_orig', 'word_lower', 'onset', 'offset'], dtype=str)
                        raw_count = len(df_story)
                        df_story['onset'] = pd.to_numeric(df_story['onset'], errors='coerce')
                        df_story['offset'] = pd.to_numeric(df_story['offset'], errors='coerce')
                        
                        # Interpolate NaNs (Don't drop!)
                        nan_mask = df_story['onset'].isna() | df_story['offset'].isna()
                        if nan_mask.any():
                            indices = df_story.index[nan_mask]
                            for k, g in groupby(enumerate(indices), lambda ix: ix[0] - ix[1]):
                                group_indices = list(map(itemgetter(1), g))
                                start_idx = group_indices[0]
                                end_idx = group_indices[-1]
                                if start_idx == 0: t_start = 0.0
                                else: t_start = df_story.at[start_idx - 1, 'offset']
                                if end_idx == len(df_story) - 1: t_end = t_start + (len(group_indices) * 1.0)
                                else: t_end = df_story.at[end_idx + 1, 'onset']
                                duration = t_end - t_start
                                if duration < 0: duration = 0
                                step = duration / len(group_indices)
                                for i, idx in enumerate(group_indices):
                                    df_story.at[idx, 'onset'] = t_start + (i * step)
                                    df_story.at[idx, 'offset'] = t_start + ((i + 1) * step)
                        
                        clean_count = len(df_story)
                        print(f"DEBUG: [Narratives] Story {run_label} - Raw CSV: {raw_count}, Final (Interpolated): {clean_count}")
                        word_list_full = df_story['word_orig'].astype(str).values
                        word_list = word_list_full
                        del df_story
                    except Exception: continue

                # --- Verification ---
                if master_onsets_offsets and run < len(master_onsets_offsets):
                    gt_count = len(master_onsets_offsets[run][0])
                    current_count = len(word_list)
                    print(f"DEBUG: Validation Check -> Extracting {current_count} words. Master file expects {gt_count} words.")
                    if gt_count != current_count:
                        print(f"⚠️ WARNING: MISMATCH DETECTED! {current_count} vs {gt_count}. (Did you update GloVe to also keep NaNs?)")

                # --- Tokenize ---
                inputs = tokenizer(fulltext_run, return_tensors='pt', return_offsets_mapping=True, truncation=True, padding=True, max_length=max_tokens, return_overflowing_tokens=True, stride=64)

                # --- Map Words to Batches ---
                idx_word_to_idx_token = []
                idx_batch, idx_token = 0, 0
                for idx_word, word in enumerate(word_list):
                    word_simplified = simplify_word(word)
                    if idx_token >= inputs['offset_mapping'].shape[1]: idx_batch, idx_token = idx_batch + 1, 64
                    # --- REPLACEMENT CODE START ---
                    n, match_type = 0, False
                    
                    # Store skipped tokens for debugging if it fails
                    skipped_tokens_debug = [] 
                    
                    while n < word_match_attempts and not match_type:
                        i_start_char, i_stop_char = inputs['offset_mapping'][idx_batch, idx_token].numpy()
                        #print(i_start_char, i_stop_char)
                        token_text = fulltext_run[i_start_char:i_stop_char]
                        #print(token_text)

                        # =======================================================
                        # [FIX] SKIP GHOST/PADDING TOKENS
                        # If offset is (0,0) or (X,X), it is a BOS, EOS, or PAD token.
                        # We must skip it without counting it as a failed attempt.
                        # =======================================================
                        if i_start_char == i_stop_char:
                            idx_token += 1
                            # If we hit the end of this batch row, move to next batch
                            if idx_token >= inputs['offset_mapping'].shape[1]:
                                idx_batch, idx_token = idx_batch + 1, 64
                            continue
                        # =======================================================

                        # Debug: Match checking
                        match_type = do_word_match(word_simplified, token_text.lower())
                        
                        if not match_type:
                            skipped_tokens_debug.append(f"'{token_text}'")
                            idx_token += 1
                            if idx_token >= inputs['offset_mapping'].shape[1]: 
                                idx_batch, idx_token = idx_batch + 1, 64
                        n += 1

                    if not match_type:
                        # ENHANCED ERROR MESSAGE
                        print(f"\nCRITICAL FAILURE at Word: '{word}' (Simplified: '{word_simplified}')")
                        print(f"Skipped the following {n} tokens trying to find it: {skipped_tokens_debug}")
                        print(f"Next 5 tokens in queue: {[fulltext_run[inputs['offset_mapping'][idx_batch, idx_token+i].numpy()[0]:inputs['offset_mapping'][idx_batch, idx_token+i].numpy()[1]] for i in range(5) if idx_token+i < inputs['offset_mapping'].shape[1]]}")
                        raise Exception(f'Token error: {word}')
                    # --- REPLACEMENT CODE END ---
                    model_match_counts[match_type] = model_match_counts.get(match_type, 0) + 1
                    idx_word_to_idx_token.append((idx_batch, idx_token))
                    if idx_word + 1 < len(word_list):
                         if not do_word_match(simplify_word(word_list[idx_word + 1]), fulltext_run[inputs['offset_mapping'][idx_batch, idx_token].numpy()[0]:inputs['offset_mapping'][idx_batch, idx_token].numpy()[1]].lower()): idx_token += 1

                words_in_batch = defaultdict(list)
                word_end_batch = {}
                for idx_word in range(len(word_list)):
                    b_start, t_start = idx_word_to_idx_token[idx_word]
                    if idx_word < len(word_list) - 1: b_end_tok, t_end_tok = idx_word_to_idx_token[idx_word+1]
                    else: b_end_tok, t_end_tok = b_start, inputs['input_ids'][b_start].ne(tokenizer.pad_token_id).sum().item()
                    token_coords = []
                    if b_start == b_end_tok:
                        coords = [(b_start, t) for t in range(t_start, t_end_tok)]
                        token_coords.extend(coords)
                        words_in_batch[b_start].append((idx_word, coords))
                    else:
                        c1 = [(b_start, t) for t in range(t_start, max_tokens)]
                        c2 = [(b_end_tok, t) for t in range(64, t_end_tok)]
                        token_coords.extend(c1 + c2)
                        words_in_batch[b_start].append((idx_word, c1))
                        words_in_batch[b_end_tok].append((idx_word, c2))
                    word_end_batch[idx_word] = b_end_tok

                # --- Streaming Loop ---
                num_layers_to_process = n_layers if args.layer == -1 else 1
                word_accumulators = {key: [defaultdict(list) for _ in range(num_layers_to_process)] for key in activation_keys}
                
                # --- NEW: Cache for Previous Padding ---
                last_seen_shapes = {key: {} for key in activation_keys} # for dimensions
                last_valid_vectors = {key: {} for key in activation_keys} # for Forward Fill

                chunk_triggers = []
                for chunk_start in range(0, len(word_list), CHUNK_SIZE):
                    chunk_end = min(chunk_start + CHUNK_SIZE, len(word_list))
                    last_word_idx = chunk_end - 1
                    trigger_batch = word_end_batch[last_word_idx]
                    chunk_triggers.append((trigger_batch, chunk_end))

                current_chunk_idx = 0
                num_batches = inputs['input_ids'].shape[0]

                for k in tqdm(range(0, num_batches, batch_size), desc=f"Extracting {run_label}", leave=False):
                    k_end = min(k + batch_size, num_batches)
                    
                    # 1. Inference
                    input_ids = inputs['input_ids'][k:k_end].to(model.device)
                    attention_mask = inputs['attention_mask'][k:k_end].to(model.device)
                    position_ids = attention_mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 1)
                    extractor.clear()
                    extractor.start()
                    with torch.no_grad(): outputs = model(input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=extract_from_api)
                    extractor.stop()
                    
                    # 2. RoPE
                    if extractor.rope_cos_sin:
                        cos, sin = extractor.rope_cos_sin
                        layers_to_process = list(extractor.activations.get('per_head_q', {}).keys())
                        for layer_idx in layers_to_process:
                             if 'per_head_q' in extractor.activations and layer_idx in extractor.activations['per_head_q']:
                                q_raw = extractor.activations['per_head_q'][layer_idx]
                                cos_dev, sin_dev = cos.to(q_raw.device), sin.to(q_raw.device)
                                B, S, D = q_raw.shape
                                num_heads = extractor.config.num_attention_heads
                                head_dim = D // num_heads
                                q_reshaped = q_raw.view(B, S, num_heads, head_dim).transpose(1, 2)
                                q_rot, q_pass = q_reshaped[..., :cos_dev.shape[-1]], q_reshaped[..., cos_dev.shape[-1]:]
                                q_rot_emb, _ = apply_rotary_pos_emb(q_rot, q_rot, cos_dev, sin_dev, position_ids)
                                extractor.activations['per_head_q_rope'][layer_idx] = torch.cat((q_rot_emb, q_pass), dim=-1).transpose(1, 2).contiguous().view(B, S, -1)
                             if 'per_head_k' in extractor.activations and layer_idx in extractor.activations['per_head_k']:
                                k_raw = extractor.activations['per_head_k'][layer_idx]
                                num_kv_heads = getattr(extractor.config, 'num_key_value_heads', num_heads)
                                k_reshaped = k_raw.view(B, S, num_kv_heads, head_dim).transpose(1, 2)
                                k_rot, k_pass = k_reshaped[..., :cos_dev.shape[-1]], k_reshaped[..., cos_dev.shape[-1]:]
                                _, k_rot_emb = apply_rotary_pos_emb(k_rot, k_rot, cos_dev, sin_dev, position_ids)
                                extractor.activations['per_head_k_rope'][layer_idx] = torch.cat((k_rot_emb, k_pass), dim=-1).transpose(1, 2).contiguous().view(B, S, -1)

                    # 3. Extract
                    for internal_b in range(k, k_end):
                        for idx_word, coords in words_in_batch[internal_b]:
                            local_indices = [t for (b, t) in coords if b == internal_b]
                            if not local_indices: continue
                            local_b = internal_b - k
                            for key in activation_keys:
                                if key in api_keys: continue
                                for i in range(num_layers_to_process):
                                    layer_idx = args.layer if args.layer != -1 else i
                                    if layer_idx in extractor.activations.get(key, {}):
                                        tensor = extractor.activations[key][layer_idx]
                                        data = tensor[local_b, local_indices, :].detach().cpu()
                                        word_accumulators[key][i][idx_word].append(data)
                                        # Cache shape
                                        if i not in last_seen_shapes[key]: 
                                            last_seen_shapes[key][i] = data.shape[-1]

                    # 4. Cleanup
                    extractor.clear()
                    del input_ids, attention_mask, outputs, position_ids
                    clear_cuda_cache()

                    # 5. Save Chunk
                    if current_chunk_idx < len(chunk_triggers):
                        trigger_batch, chunk_end_word_idx = chunk_triggers[current_chunk_idx]
                        if k >= trigger_batch:
                            start_word = 0 if current_chunk_idx == 0 else chunk_triggers[current_chunk_idx-1][1]
                            end_word = chunk_end_word_idx
                            part_num = current_chunk_idx + 1
                            print(f"  Saving chunk {part_num} (Words {start_word}-{end_word})...")
                            chunk_save_data = {key: [[] for _ in range(num_layers_to_process)] for key in activation_keys}
                            for w_idx in range(start_word, end_word):
                                for key in activation_keys:
                                    if key in api_keys: continue
                                    for i in range(num_layers_to_process):
                                        partials = word_accumulators[key][i].pop(w_idx, [])
                                        if partials:
                                            all_tokens = torch.cat(partials, dim=0)
                                            act_mean = all_tokens.mean(dim=0).float().numpy()
                                            chunk_save_data[key][i].append(act_mean)
                                            # Update previous pad cache
                                            last_valid_vectors[key][i] = act_mean 
                                        else:
                                            # --- FIX: Previous Padding (Forward Fill) ---
                                            if i in last_valid_vectors[key]:
                                                chunk_save_data[key][i].append(last_valid_vectors[key][i])
                                            else:
                                                # Fallback: Zero Vector if first word
                                                dim = last_seen_shapes[key].get(i, None)
                                                if dim is not None:
                                                    zero_vec = np.zeros(dim, dtype=np.float32)
                                                    chunk_save_data[key][i].append(zero_vec)
                                                    last_valid_vectors[key][i] = zero_vec
                                                else:
                                                    print(f"    WARNING: Word {w_idx} skipped (No tokens, no previous vector).")

                            sanitized_model_name = model_name.replace("/", "_")
                            base_filename = f'{sanitized_model_name}_{dataset_name}_{lang}_run-{run_label}_part-{part_num}_activations.gz'
                            if seed > 0: base_filename = f'{sanitized_model_name}_untrained_seed{seed}_{dataset_name}_{lang}_run-{run_label}_part-{part_num}_activations.gz'
                            filename = os.path.join(output_folder, base_filename)
                            make_dir(os.path.dirname(filename))
                            
                            processed = [key for key in chunk_save_data if any(len(x) > 0 for x in chunk_save_data[key])]
                            missing = [key for key in chunk_save_data if not any(len(x) > 0 for x in chunk_save_data[key])]
                            print(f"  [Report] Extracted: {processed}")
                            if missing: print(f"  [Report] Missing: {missing}")

                            with open(filename, 'wb') as f: joblib.dump(chunk_save_data, f, compress=4)
                            del chunk_save_data
                            gc.collect()
                            current_chunk_idx += 1

                # --- 6. Post-Loop Flush (CRITICAL FIX) ---
                # Ensure any remaining full or partial chunks are saved
                while current_chunk_idx < len(chunk_triggers):
                    start_word = 0 if current_chunk_idx == 0 else chunk_triggers[current_chunk_idx-1][1]
                    end_word = chunk_triggers[current_chunk_idx][1]
                    part_num = current_chunk_idx + 1

                    print(f"  Saving REMAINING chunk {part_num} (Words {start_word}-{end_word})...")
                    chunk_save_data = {key: [[] for _ in range(num_layers_to_process)] for key in activation_keys}

                    for w_idx in range(start_word, end_word):
                        for key in activation_keys:
                            if key in api_keys: continue
                            for i in range(num_layers_to_process):
                                partials = word_accumulators[key][i].pop(w_idx, [])
                                if partials:
                                    all_tokens = torch.cat(partials, dim=0)
                                    act_mean = all_tokens.mean(dim=0).float().numpy()
                                    chunk_save_data[key][i].append(act_mean)
                                    last_valid_vectors[key][i] = act_mean
                                else:
                                    if i in last_valid_vectors[key]:
                                        chunk_save_data[key][i].append(last_valid_vectors[key][i])
                                    else:
                                        dim = last_seen_shapes[key].get(i, None)
                                        if dim is not None:
                                            zero_vec = np.zeros(dim, dtype=np.float32)
                                            chunk_save_data[key][i].append(zero_vec)
                                            last_valid_vectors[key][i] = zero_vec
                                        else:
                                            print(f"    WARNING: Word {w_idx} skipped (No tokens, no previous vector).")

                    sanitized_model_name = model_name.replace("/", "_")
                    base_filename = f'{sanitized_model_name}_{dataset_name}_{lang}_run-{run_label}_part-{part_num}_activations.gz'
                    if seed > 0: base_filename = f'{sanitized_model_name}_untrained_seed{seed}_{dataset_name}_{lang}_run-{run_label}_part-{part_num}_activations.gz'
                    filename = os.path.join(output_folder, base_filename)
                    make_dir(os.path.dirname(filename))

                    with open(filename, 'wb') as f: joblib.dump(chunk_save_data, f, compress=4)
                    del chunk_save_data
                    gc.collect()
                    current_chunk_idx += 1
                # -----------------------------------------

                clear_cuda_cache()

            total_matches = sum(model_match_counts.values())
            if total_matches > 0:
                print(f"\nTotal Aligned Words: {total_matches}")
                for k in sorted(model_match_counts.keys()): print(f"  - {k}: {model_match_counts[k]} ({model_match_counts[k]/total_matches:.2%})")
            print("All runs saved.")

        except Exception as e:
            import traceback
            print(f"❌ Error processing model {model_name}: {str(e)}")
            traceback.print_exc()
        finally:
            if 'model' in locals(): del model
            if 'tokenizer' in locals(): del tokenizer
            clear_cuda_cache()

if __name__ == '__main__':
    main()
    print("\nProcessing completed!")
