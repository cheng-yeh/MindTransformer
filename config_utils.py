import yaml
import os
from pathlib import Path
import numpy as np

import re # Make sure to import re at the top of your config_utils.py

def resolve_variables(root_config, current_level_to_process=None):
    """
    Recursively resolves ${variable} placeholders in strings within a configuration dictionary.
    Modifies the configuration dictionary in place.

    Args:
        root_config: The top-level configuration dictionary, used for all lookups.
        current_level_to_process: The current dictionary or list being processed.
                                  Defaults to root_config on the first call.
    """
    if current_level_to_process is None:
        current_level_to_process = root_config

    if isinstance(current_level_to_process, dict):
        # Iterate over a copy of items if keys could be added/removed,
        # but for modifying values, direct iteration is usually fine.
        # Using list(current_level_to_process.items()) for safety if structure could change.
        for key, value in list(current_level_to_process.items()):
            if isinstance(value, dict):
                resolve_variables(root_config, value) # Process sub-dictionary
            elif isinstance(value, list):
                resolve_variables(root_config, value) # Process list (will be handled by the next elif block)
            elif isinstance(value, str) and "${" in value:
                original_string_for_key = value
                current_key_value = original_string_for_key
                
                placeholders = re.findall(r'\$\{(.+?)\}', original_string_for_key) # Non-greedy match

                for var_path in placeholders:
                    var_parts = var_path.split(".")
                    var_value_to_sub = root_config # Lookups always from the root
                    try:
                        for var_part in var_parts:
                            var_value_to_sub = var_value_to_sub[var_part]
                        
                        var_value_to_sub_str = str(var_value_to_sub)
                        current_key_value = current_key_value.replace("${" + var_path + "}", var_value_to_sub_str)
                    except (KeyError, TypeError):
                        # print(f"Warning: Variable '{var_path}' for key '{key}' not found or path error. Placeholder remains.")
                        pass # Placeholder remains if lookup fails
                
                if current_key_value != original_string_for_key:
                    current_level_to_process[key] = current_key_value

    elif isinstance(current_level_to_process, list):
        for i, item in enumerate(current_level_to_process):
            if isinstance(item, dict):
                resolve_variables(root_config, item) # Process dictionary items in list
            elif isinstance(item, list):
                resolve_variables(root_config, item) # Process list items in list (nested lists)
            elif isinstance(item, str) and "${" in item:
                original_item_string = item
                current_item_string = original_item_string

                placeholders = re.findall(r'\$\{(.+?)\}', original_item_string)

                for var_path in placeholders:
                    var_parts = var_path.split(".")
                    var_value_to_sub = root_config # Lookups always from the root
                    try:
                        for var_part in var_parts:
                            var_value_to_sub = var_value_to_sub[var_part]
                        var_value_to_sub_str = str(var_value_to_sub)
                        current_item_string = current_item_string.replace("${" + var_path + "}", var_value_to_sub_str)
                    except (KeyError, TypeError):
                        # print(f"Warning: Variable '{var_path}' for list item '{original_item_string}' not found or path error. Placeholder remains.")
                        pass # Placeholder remains

                if current_item_string != original_item_string:
                    current_level_to_process[i] = current_item_string # Modify list in place
    
    return root_config # Return the root_config, which has been modified in-place

def standardize(v, axis=0):
    """
    Standardize a numpy array by subtracting the mean and dividing by the standard deviation.
    
    Args:
        v (numpy.ndarray): The array to standardize
        axis (int, optional): The axis along which to compute the mean and std. Defaults to 0.
        
    Returns:
        numpy.ndarray: The standardized array
    """
    return (v - np.mean(v, axis=axis, keepdims=True)) / np.std(v, axis=axis, keepdims=True)

def make_dir(directory):
    """Create directory if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def is_file_path(path):
    """
    Check if a path is likely a file path.
    
    Args:
        path (str): The path to check
        
    Returns:
        bool: True if the path is likely a file path, False otherwise
    """
    # Check if the path has a file extension
    basename = os.path.basename(path)
    return '.' in basename and not basename.endswith('.')

def load_config(config_path):
    """
    Load and process the YAML configuration file.
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        dict: Processed configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Resolve variable substitutions
    config = resolve_variables(config)
    
    # Create directories if they don't exist
    for path_key, path_value in config['paths'].items():
        # Skip file paths, only create directories
        if is_file_path(path_value):
            # For file paths, create the parent directory if it exists
            parent_dir = os.path.dirname(path_value)
            if parent_dir:
                make_dir(parent_dir)
        else:
            # This is a directory path
            make_dir(path_value)
    
    return config
