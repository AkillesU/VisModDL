import os
import re
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.nn.modules import activation
from torchvision import transforms
from PIL import Image
from torchinfo import summary
import yaml
from tqdm import tqdm
import math
import torch.nn as nn
import pickle
from joblib import Parallel, delayed
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from itertools import combinations, product
import random
import hashlib
from collections import defaultdict
import torchvision                       
from torchvision.datasets.utils import download_url
import os, tarfile, hashlib
import copy
import statsmodels.formula.api as smf
from pathlib import Path
from typing import Sequence, Mapping, Tuple, List

def get_layer_from_path(model, layer_path):
    current = model
    for step in layer_path.split('.'):
        # If step is an integer, treat current as a sequence (e.g., nn.Sequential)
        if step.isdigit():
            current = current[int(step)]
        else:
            # First try getattr if current is a module
            if hasattr(current, step):
                current = getattr(current, step)
            # If current is a module and step is in its submodules, access via _modules
            elif isinstance(current, torch.nn.Module) and step in current._modules:
                current = current._modules[step]
            # If current is a dictionary, just index into it
            elif isinstance(current, dict) and step in current:
                current = current[step]
            else:
                raise AttributeError(f"Cannot find '{step}' in {current}. Make sure the path is correct.")
    return current


def get_all_weight_layers(model, base_path="", include_bias=False):
    """
    Recursively find all submodules under the given model that have a 'weight' parameter.
    Returns a list of layer paths (dot-separated) that lead to layers with weights.

    Parameters:
        model (nn.Module): The model or module to search.
        base_path (str): The starting path. If empty, we assume 'model' is the root.

    Returns:
        List[str]: A list of dot-separated paths to each module with weights.
    """
    weight_layers = []
    if include_bias:
        if hasattr(model, 'weight') or hasattr(model, 'bias'):
            # 'model' itself is a leaf layer with weights
            weight_layers.append(base_path)
    else:
        # Check if current module has a 'weight' parameter
        if hasattr(model, 'weight'):
            # 'model' itself is a leaf layer with weights
            weight_layers.append(base_path)

    # If not, or in addition, iterate through submodules
    for name, submodule in model._modules.items():
        if submodule is None:
            continue
        # If base_path is empty, just use the name. Otherwise, append it with a dot.
        if base_path == "":
            new_path = name
        else:
            new_path = base_path + "." + "_modules" + "." + name

        weight_layers.extend(get_all_weight_layers(submodule, new_path))

    return weight_layers


def get_all_conv_layers(model, base_path="", include_bias=False):
    """
    Recursively find all submodules under `model` that are nn.Conv2d,
    returning a list of layer paths (dot-separated) where each layer has weights.

    Parameters:
        model (nn.Module): The model or module to search.
        base_path (str): The starting path. If empty, we assume 'model' is the root.

    Returns:
        List[str]: A list of dot-separated paths to each Conv2d module.
    """
    conv_layers = []

    # If the current module itself is a Conv2d, record its path
    if isinstance(model, nn.Conv2d):
        # Make sure it has a weight parameter (it should)
        if include_bias:
            if hasattr(model, 'weight') or hasattr(model, 'bias'):
                conv_layers.append(base_path)
        else:
            if hasattr(model, 'weight'):
                conv_layers.append(base_path)
            # Once we've identified this as a Conv2d, we typically don't recurse further
            # because a single nn.Conv2d shouldn't have any of its own submodules.
        return conv_layers

    # Otherwise if base path is not Conv layer itself, recurse into children
    for name, submodule in model._modules.items():
        if submodule is None:
            continue
        if base_path == "":
            new_path = name
        else:
            new_path = base_path + "." + "_modules" + "." + name

        # Rerun function to check if child is conv layer...
        conv_layers.extend(get_all_conv_layers(submodule, new_path))

    return conv_layers


def load_model(model_info: dict, pretrained=True, layer_name='IT', layer_path="", model_time_steps=5):
    """
    Load a specified pretrained model and register a forward hook to capture activations.

    Parameters:
        model_class: The class (constructor) for the model (e.g., cornet_s).
        pretrained (bool): If True, load pretrained weights.
        layer_name (str): The layer name at which to hook and capture activations.
                          Possible values depend on the model architecture.

    Returns:
        model: The loaded and hooked model
        activations: A dictionary to store captured activations
    """
    # Define model load parameters
    model_source = model_info["source"]
    model_repo = model_info["repo"]
    model_name = model_info["name"]
    model_weights = model_info["weights"]
    if "time_steps" in model_info:
        model_time_steps = model_info["time_steps"]

    # Hook function to capture layer outputs
    def hook_fn(module, input, output):
        activations[layer_name] = output.cpu().detach().numpy()


    if model_source == "cornet":
        if model_name == "cornet_z":
            from cornet import cornet_z
            model = cornet_z(pretrained=pretrained, map_location=(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")))

        elif model_name == "cornet_s":
            from cornet import cornet_s
            model = cornet_s(pretrained=pretrained, map_location=(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")))

        elif model_name == "cornet_rt":
            from cornet import cornet_rt
            model = cornet_rt(pretrained=pretrained, map_location=(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")), times=model_time_steps)

        else:
            raise ValueError(f"CORnet model {model_name} not found. Check config file.")

    elif model_source == "pytorch_hub":
        if model_weights == "":
            model = torch.hub.load(model_repo, model_name, map_location=(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")))
        else:
            model = torch.hub.load(model_repo, model_name, weights=model_weights, map_location=(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")))
    else:
        raise ValueError(f"Check model source: {model_source}")
    
    # Print model summary
    #print(model)

    model.eval()
    activations = {} # Init activations dictionary for hook registration

    # early-exit guard 
    if layer_path in (None, ""):
        # caller does not need hooks – just return the model as-is
        return model, activations


    # We'll store all hooks in the same 'activations' dictionary, keyed by their path.
    if isinstance(layer_path, list):
        # If layer_path is already a list of paths, do multiple hooks
        def make_hook_fn(name):
            def hook_fn(module, input, output):
                activations[name] = output.cpu().detach().numpy()
            return hook_fn
        
        for lp in layer_path:
            target_layer = get_layer_from_path(model, lp)
            target_layer.register_forward_hook(make_hook_fn(lp))
    else:
        # Old single-layer logic
        def hook_fn(module, input, output):
            activations[layer_name] = output.cpu().detach().numpy()
        target_layer = get_layer_from_path(model, layer_path)
        target_layer.register_forward_hook(hook_fn)

    return model, activations


def preprocess_image(image_path):
    """
    Preprocess a single image for model input.

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        input_tensor (torch.Tensor): Preprocessed image tensor.
    """
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])
    img = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
    return input_tensor


def extract_activations(model, activations, image_dir, layer_name='IT'):
    """
    Extract activations from all images in a directory using the given model and hook.

    Parameters:
        model: A hooked model that updates 'activations' dict on forward pass
        activations (dict): Dictionary to store activations
        image_dir (str): Directory containing images
        layer_name (str): The layer name at which activations were hooked

    Returns:
        activations_df (pd.DataFrame): DataFrame of flattened activations indexed by image name
    """
    all_activations = []
    image_names = []

    for image_file in os.listdir(image_dir):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_dir, image_file)
            input_tensor = preprocess_image(img_path)

            with torch.no_grad():
                model(input_tensor)

            all_activations.append(activations[layer_name].flatten())
            image_names.append(image_file)

    activations_df = pd.DataFrame(all_activations, index=image_names)
    return activations_df


def apply_noise(model, noise_level, noise_dict, layer_paths=None, apply_to_all_layers=False, only_conv=True, include_bias=False):
    """
    Add Gaussian noise with std = noise_level to the specified layers' weights.
    """
    with torch.no_grad():
        if apply_to_all_layers:
            for name, param in model.named_parameters():
                if 'weight' in name and param.requires_grad:
                    noise = torch.randn_like(param) * noise_level
                    param += noise
        else:
            for path in layer_paths:
                target_layer = get_layer_from_path(model, path)
                if only_conv:
                    weight_layer_paths = get_all_conv_layers(target_layer, path, include_bias)
                else:
                    weight_layer_paths = get_all_weight_layers(target_layer, path, include_bias)

                for w_path in weight_layer_paths:
                    w_layer = get_layer_from_path(model, w_path)
                    if hasattr(w_layer, 'weight') and w_layer.weight is not None:
                        noise = torch.randn_like(w_layer.weight) * noise_dict[f"{w_path}.weight"] * noise_level
                        w_layer.weight += noise
                        # include bias
                        if include_bias and hasattr(w_layer, 'bias') and w_layer.bias is not None:
                            noise = torch.randn_like(w_layer.bias) * noise_dict[f"{w_path}.bias"] * noise_level
                            w_layer.bias += noise
                    else:
                        raise AttributeError(f"layer {w_path} does not have weights")

def apply_masking(
    model,
    fraction_to_mask,
    layer_paths=None,
    apply_to_all_layers=False,
    masking_level="connections",
    only_conv=True,
    include_bias=False,
):
    """
    Mask either individual connections or whole units (filters / neurons).

    • masking_level == "connections": randomly zero a fraction of individual
      weights. Biases are treated independently.

    • masking_level == "units": randomly zero a fraction of whole filters /
      neurons (dim‑0 rows).  **If include_bias is True the matching bias
      elements are zeroed with the *same* unit indices.**

    Note
    ----
    The helper functions `get_layer_from_path`, `get_all_conv_layers`,
    `get_all_weight_layers`, and `create_mask` are assumed to exist
    unchanged elsewhere in the codebase.
    """
    param_masks = {}

    # Build the mask tensors
    with torch.no_grad():
        if apply_to_all_layers:
            for name, param in model.named_parameters():
                if "weight" in name and param.requires_grad:
                    param_masks[name] = create_mask(
                        param, fraction_to_mask, masking_level=masking_level
                    )
        else:
            for path in layer_paths:
                target_layer = get_layer_from_path(model, path)
                if only_conv:
                    weight_layer_paths = get_all_conv_layers(
                        target_layer, path, include_bias
                    )
                else:
                    weight_layer_paths = get_all_weight_layers(
                        target_layer, path, include_bias
                    )

                for w_path in weight_layer_paths:
                    w_layer = get_layer_from_path(model, w_path)

                    # unit‑level masking with bias: draw indices once
                    if masking_level == "units" and include_bias:
                        num_units = w_layer.weight.shape[0]
                        k = int(fraction_to_mask * num_units)
                        device = w_layer.weight.device
                        if k == 0:
                            unit_idx = torch.empty(
                                0, dtype=torch.long, device=device
                            )
                        else:
                            unit_idx = torch.randperm(
                                num_units, device=device
                            )[:k]

                        # weight mask (same shape as weight tensor)
                        weight_mask = torch.ones_like(w_layer.weight)
                        if k > 0:
                            weight_mask[unit_idx, ...] = 0
                        param_masks[w_path] = weight_mask

                        # matching bias mask (1‑D)
                        if (
                            hasattr(w_layer, "bias")
                            and w_layer.bias is not None
                        ):
                            bias_mask = torch.ones_like(w_layer.bias)
                            if k > 0:
                                bias_mask[unit_idx] = 0
                            param_masks[f"{w_path}_bias"] = bias_mask

                    else:
                        if (
                            hasattr(w_layer, "weight")
                            and w_layer.weight is not None
                        ):
                            weight_mask = create_mask(
                                w_layer.weight,
                                fraction_to_mask,
                                masking_level=masking_level,
                            )
                            param_masks[w_path] = weight_mask

                            if (
                                include_bias
                                and hasattr(w_layer, "bias")
                                and w_layer.bias is not None
                            ):
                                bias_mask = create_mask(
                                    w_layer.bias,
                                    fraction_to_mask,
                                    masking_level=masking_level,
                                )
                                param_masks[f"{w_path}_bias"] = bias_mask
                        else:
                            raise AttributeError(
                                f"layer {w_path} does not have weights"
                            )

    # ------------------------------------------------------------------ #
    # 2. Register forward‑pre hooks that multiply in the masks           #
    # ------------------------------------------------------------------ #
    if not apply_to_all_layers:
        for path in layer_paths:
            if only_conv:
                weight_layer_paths = get_all_conv_layers(
                    get_layer_from_path(model, path), path
                )
            else:
                weight_layer_paths = get_all_weight_layers(
                    get_layer_from_path(model, path), path
                )

            if include_bias:
                for w_path in weight_layer_paths:
                    layer = get_layer_from_path(model, w_path)
                    w_mask = param_masks[w_path]
                    b_mask = (
                        param_masks.get(f"{w_path}_bias")
                        if layer.bias is not None
                        else None
                    )

                    def _hook(module, _input, w_mask=w_mask, b_mask=b_mask):
                        if hasattr(module, "weight"):
                            module.weight.data.mul_(w_mask)
                        if (
                            b_mask is not None
                            and hasattr(module, "bias")
                            and module.bias is not None
                        ):
                            module.bias.data.mul_(b_mask)
                        return None

                    layer.register_forward_pre_hook(_hook)
            else:
                for w_path in weight_layer_paths:
                    layer = get_layer_from_path(model, w_path)
                    w_mask = param_masks[w_path]

                    def _hook(module, _input, w_mask=w_mask):
                        if hasattr(module, "weight"):
                            module.weight.data.mul_(w_mask)
                        return None

                    layer.register_forward_pre_hook(_hook)

    else:
        param_to_mask = {
            param: mask
            for name, param in model.named_parameters()
            if name in param_masks
            for mask in [param_masks[name]]
        }

        def _global_hook(module, _input):
            for _n, p in module.named_parameters(recurse=False):
                if p in param_to_mask:
                    p.data.mul_(param_to_mask[p])
            return None

        for m in model.modules():
            if len(list(m.parameters(recurse=False))) > 0:
                m.register_forward_pre_hook(_global_hook)


def create_mask(param, fraction, masking_level='connections'):
    """
    Create a mask for the given parameter (weight tensor).
    If masking_level == 'connections', randomly zero out a fraction of all weight entries.
    If masking_level == 'units', randomly zero out a fraction of entire units (rows in linear layers or filters in conv layers).
    """
    shape = param.shape
    device = param.device
    if fraction <= 0:
        return torch.ones_like(param)

    # Flatten the weights to pick indices easily
    if masking_level == 'connections':
        param_data = param.view(-1)
        n = param_data.numel()
        k = int(fraction * n)
        if k == 0:
            return torch.ones_like(param)
        indices = torch.randperm(n, device=device)[:k]
        mask_flat = torch.ones(n, device=device)
        mask_flat[indices] = 0
        return mask_flat.view(shape)

    elif masking_level == 'units':
        # For a Linear layer: weight shape is [out_features, in_features]
        # Each out_feature is considered a unit
        # For a Conv layer: weight shape might be [out_channels, in_channels, ...]
        # We'll consider out_channels as units.
        
        # Identify dimension representing units: 
        # For Linear: dim 0 is out_features.
        # For Conv2d: dim 0 is out_channels.
        # We'll assume standard layers where first dimension corresponds to units.
        units_dim = 0
        num_units = param.shape[units_dim]
        k = int(fraction * num_units)
        if k == 0:
            return torch.ones_like(param)
        unit_indices = torch.randperm(num_units, device=device)[:k]
        # Create a mask of ones, then zero entire units
        mask = torch.ones_like(param)
        # zero out rows (units)
        mask[unit_indices, ...] = 0
        return mask

    else:
        raise ValueError(f"masking_level {masking_level} not recognized.")


def sort_activations_by_numeric_index(activations_df):
    """
    Extract a string prefix and numeric index from image names, sort the DataFrame
    alphabetically by the prefix and numerically by the numeric index. Keep the
    'numeric_index' column, and optionally create a 'sorted_numeric_index' column.

    Example indices: "face1", "face2", "face11", "object1", "object2"

    Parameters:
        activations_df (pd.DataFrame): DataFrame with image names as index.

    Returns:
        pd.DataFrame: DataFrame sorted and with the 'numeric_index' column retained.
    """
    # Ensure the index is string
    activations_df.index = activations_df.index.astype(str)
    
    # Extract the string part (all non-digits at the start).
    # This will be used for alphabetical sorting of the prefix.
    activations_df['string_part'] = activations_df.index.str.extract(r'^([^\d]+)', expand=False).fillna('')
    
    # Extract the numeric part (digits) -> 'numeric_index'.
    # Fallback to '0' if no digits found (i.e., purely alphabetic name).
    activations_df['numeric_index'] = (
        activations_df.index
            .str.extract(r'(\d+)', expand=False)
            .fillna('0')
            .astype(int)
    )
    
    # Sort first by string_part (alphabetical), then by numeric_index (numerical)
    activations_df_sorted = activations_df.sort_values(['string_part', 'numeric_index'])
    
    # (Optional) create a sorted numeric index to reflect the new order:
    activations_df_sorted['numeric_index'] = range(1, len(activations_df_sorted) + 1)

    # Drop 'string_part' if you no longer need it
    activations_df_sorted.drop(columns=['string_part'], inplace=True)

    return activations_df_sorted


def compute_correlations(activations_df_sorted):
    """
    Compute correlation matrix between activation vectors.

    Parameters:
        activations_df_sorted (pd.DataFrame): DataFrame with sorted activations, including numeric_index column.

    Returns:
        correlation_matrix (np.ndarray): Correlation matrix of activations.
        sorted_image_names (list): List of image names in sorted order.
    """
    sorted_image_names = activations_df_sorted.index.tolist()
    correlation_matrix = np.corrcoef(activations_df_sorted.drop(columns='numeric_index').values)
    return correlation_matrix, sorted_image_names


def plot_correlation_heatmap(correlation_matrix, sorted_image_names, layer_name='IT', vmax=0.4, model_name="untitled_model"):
    """
    Plot a heatmap of the correlation matrix.

    Parameters:
        correlation_matrix (np.ndarray): The correlation matrix.
        sorted_image_names (list): List of image names corresponding to matrix indices.
        layer_name (str): Name of the layer used in the title.
        vmax (float): Upper bound for colormap.
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=False, cmap="viridis",
                xticklabels=sorted_image_names, yticklabels=sorted_image_names, vmax=vmax, vmin=0)
    plt.title(f"Correlation of Activations Between Images (Layer: {layer_name})")
    plt.xlabel("Images")
    plt.ylabel("Images")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    # Save figure
    save_path = f"data/haupt_stim_activ/{model_name}/{layer_name}.png"
    # Create the directories if they don't already exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400)
    plt.show()


def assign_categories(sorted_image_names):
    """
    Assign category labels to images based on the non-numeric part of each filename.

    Example:
        "face1.jpg"   -> category "face"
        "object12.png" -> category "object"

    Parameters:
        sorted_image_names (list): Sorted list of image filenames.

    Returns:
        np.ndarray: Array of category labels (strings).
    """
    categories = []
    for image_name in sorted_image_names:
        # Split off the extension, e.g. "animal1.jpg" -> "animal1"
        base_name = image_name.rsplit('.', 1)[0]
        # Remove all digits from the base name using a regex
        category_name = re.sub(r'\d+', '', base_name)
        categories.append(category_name)
    
    return np.array(categories)


# POSSIBLY DEPRECATED FUNCTION. SEE calc_within_between() BELOW
def bootstrap_correlations(correlation_matrix, categories_array, n_bootstrap=10000):
    """
    Perform bootstrap analysis comparing within-category and between-category correlations.
    
    Parameters:
        correlation_matrix (np.ndarray): Correlation matrix of image activations.
        categories_array (np.ndarray): Array of category labels for each image (strings).
        n_bootstrap (int): Number of bootstrap iterations.
    
    Returns:
        results (dict): Dictionary containing analysis results for each category (string).
    """
    results = {}
    unique_categories = np.unique(categories_array)  # e.g. ["animal", "face", "object"]

    for category_name in unique_categories:
        # Indices of images in current category
        category_indices = np.where(categories_array == category_name)[0]
        other_indices = np.where(categories_array != category_name)[0]

        # Within-category correlations
        submatrix_within = correlation_matrix[np.ix_(category_indices, category_indices)]
        n_within = len(category_indices)
        # Exclude diagonal from within-category to avoid self-correlation
        mask_within = np.ones((n_within, n_within), dtype=bool)
        np.fill_diagonal(mask_within, False)
        within_correlations = submatrix_within[mask_within]

        # Between-category correlations
        submatrix_between = correlation_matrix[np.ix_(category_indices, other_indices)]
        between_correlations = submatrix_between.flatten()

        # Observed difference
        avg_within = np.mean(within_correlations)
        avg_between = np.mean(between_correlations)
        observed_difference = avg_within - avg_between

        # Bootstrap
        bootstrap_differences = []
        for i in range(n_bootstrap):
            resampled_within = np.random.choice(within_correlations, size=len(within_correlations), replace=True)
            resampled_between = np.random.choice(between_correlations, size=len(between_correlations), replace=True)

            resampled_avg_within = np.mean(resampled_within)
            resampled_avg_between = np.mean(resampled_between)
            bootstrap_differences.append(resampled_avg_within - resampled_avg_between)

        bootstrap_differences = np.array(bootstrap_differences)

        # p-value and confidence intervals
        p_value = np.mean(bootstrap_differences <= 0)
        ci_lower = np.percentile(bootstrap_differences, 2.5)
        ci_upper = np.percentile(bootstrap_differences, 97.5)

        # Store the results under the string-based category key
        results[category_name] = {
            'avg_within': avg_within,
            'avg_between': avg_between,
            'observed_difference': observed_difference,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }

    return results


def calc_within_between(correlation_matrix, categories_array):
    """
    Compute within-category and between-category correlations for each category,
    and also compute the grand averages across all categories.
    
    Parameters:
        correlation_matrix (np.ndarray): Correlation matrix of image activations.
        categories_array (np.ndarray): Array of category labels for each image (strings).
    
    Returns:
        results (dict): Dictionary containing analysis results for each category,
                        plus an "all_categories" subdictionary with overall stats.
    """
    results = {}
    unique_categories = np.unique(categories_array)
    
    # To store all within- and between-category values across ALL categories
    all_within_correlations = []
    all_between_correlations = []

    for category_name in unique_categories:
        # Indices for current category vs. other categories
        category_indices = np.where(categories_array == category_name)[0]
        other_indices = np.where(categories_array != category_name)[0]

        # Within-category correlations (excluding diagonal)
        submatrix_within = correlation_matrix[np.ix_(category_indices, category_indices)]
        n_within = len(category_indices)
        mask_within = np.ones((n_within, n_within), dtype=bool)
        np.fill_diagonal(mask_within, False)
        within_correlations = submatrix_within[mask_within]

        # Between-category correlations
        submatrix_between = correlation_matrix[np.ix_(category_indices, other_indices)]
        between_correlations = submatrix_between.flatten()

        # Mean values
        avg_within = np.mean(within_correlations)
        avg_between = np.mean(between_correlations)
        observed_difference = avg_within - avg_between

        # Save category-specific stats
        results[category_name] = {
            'avg_within': avg_within,
            'avg_between': avg_between,
            'observed_difference': observed_difference
        }

        # Collect for overall stats
        all_within_correlations.extend(within_correlations)
        all_between_correlations.extend(between_correlations)

    # Compute grand-average stats across all categories
    avg_within_all = np.mean(all_within_correlations) if len(all_within_correlations) > 0 else float('nan')
    avg_between_all = np.mean(all_between_correlations) if len(all_between_correlations) > 0 else float('nan')
    observed_diff_all = avg_within_all - avg_between_all

    # Store them in a subdictionary
    results['total'] = {
        'avg_within': avg_within_all,
        'avg_between': avg_between_all,
        'observed_difference': observed_diff_all
    }

    return results


def convert_np_to_native(value):
    if isinstance(value, np.generic):
        # Convert numpy scalar to Python scalar
        return value.item()
    elif isinstance(value, np.ndarray):
        # Convert numpy array to Python list
        return value.tolist()
    elif isinstance(value, dict):
        # Recursively convert dictionary KEYS and VALUES
        new_dict = {}
        for k, v in value.items():
            # Convert key if it's a NumPy type
            if isinstance(k, np.generic):
                k = k.item()
            # Ensure the key is a Python string (or something YAML can handle)
            if not isinstance(k, str):
                k = str(k)

            # Recursively convert the value
            new_dict[k] = convert_np_to_native(v)
        return new_dict
    elif isinstance(value, list):
        # Recursively convert list elements
        return [convert_np_to_native(v) for v in value]
    else:
        # Return the value as is if it's already a native type
        return value


def print_within_between(results, layer_name, model_name, output_path=None):
    """
    Print the bootstrap results in a readable format.
    """

    # 1) Convert results (including dictionary keys) to native Python types
    native_results = convert_np_to_native(results)

    # 2) Now iterate over the keys in native_results (not results)
    for category_name in native_results.keys():
        avg_within = native_results[category_name]['avg_within']
        avg_between = native_results[category_name]['avg_between']
        observed_difference = native_results[category_name]['observed_difference']
        p_value = native_results[category_name]['p_value']
        ci_lower = native_results[category_name]['ci_lower']
        ci_upper = native_results[category_name]['ci_upper']

        print(f"Category '{category_name}':")
        print(f"  Average within-category correlation: {avg_within:.4f}")
        print(f"  Average between-category correlation: {avg_between:.4f}")
        print(f"  Observed difference (within - between): {observed_difference:.4f}")
        print(f"  95% Confidence interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  P-value (one-tailed test): {p_value:.4f}\n")

    # 3) Save to YAML
    if output_path is None:
        save_path = f"data/haupt_stim_activ/{model_name}/{layer_name}_within-between.yaml"
    else:
        save_path = output_path

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 4) Dump the *native* version of results to YAML
    with open(save_path, "w") as f:
        yaml.safe_dump(native_results, f, sort_keys=False)


def generate_params_list(params=[0.1,10,0.1]):
    # Unpack the parameters
    start, length, step = params
    
    # Generate the values
    return [start + i * step for i in range(length)]


def get_params_sd(model: nn.Module) -> dict:
    """
    Utilizes `get_all_weight_layers` to find all modules that have weight/bias.
    Then computes std dev for each weight/bias (if they exist) and returns a dict:
        {
          'some.layer.path.weight': std_val,
          'some.layer.path.bias':  std_val,
          ...
        }
    """
    sd_dict = {}
    
    # 1. Gather all submodules that might have weight or bias
    layer_paths = get_all_weight_layers(model, base_path="", include_bias=True)
    
    # 2. Iterate over each layer path, retrieve submodule, compute std dev
    for path in layer_paths:
        submodule = get_layer_from_path(model, path)
        
        # If this submodule has a 'weight' param, compute std dev
        if hasattr(submodule, "weight") and submodule.weight is not None:
            # if submodule.weight.requires_grad:
            std_w = submodule.weight.std().item()
            sd_dict[f"{path}.weight"] = std_w
        
        # If this submodule has a 'bias' param, compute std dev
        if hasattr(submodule, "bias") and submodule.bias is not None:
            # if submodule.bias.requires_grad:
            std_b = submodule.bias.std().item()
            sd_dict[f"{path}.bias"] = std_b
    
    return sd_dict

       
def run_damage(
    model_info,
    pretrained,
    fraction_to_mask_params,
    noise_levels_params,
    layer_paths_to_damage,
    apply_to_all_layers,
    manipulation_method,          # "connections" or "noise"
    mc_permutations,
    layer_name,
    activation_layers_to_save,
    image_dir,
    only_conv,
    include_bias,
    masking_level="connections",  #  ← NEW (keeps old behaviour by default)
    run_suffix=""
):
    """
    A merged run_damage that:
      - Keeps  time_steps/run_suffix logic + per-image forward pass.
      - Incorporates earliest-damaged-block logic so only hook layers that come at or after that block.
      - Still uses an 'activations' dictionary, updated each time run a forward pass on a single image.
      - Allows hooking multiple layers (activation_layers_to_save).
    """

    # -------------------------------------------------------------------------
    # SECTION 2: (Unchanged) Time steps & run_suffix from original code
    # -------------------------------------------------------------------------
    if "time_steps" in model_info:
        time_steps = str(model_info['time_steps'])
    elif model_info['name'] == "cornet_rt":
        time_steps = "5"
    else:
        time_steps = ""

    # Keep original run_suffix logic
    run_suffix = (("_c" if only_conv else "_all") + ("+b" if include_bias else "")) + run_suffix

    # -------------------------------------------------------------------------
    # A) Determine the list of damage levels (same as before + optional temp load for noise_dict)
    # -------------------------------------------------------------------------
    if manipulation_method == "connections":
        damage_levels_list = generate_params_list(fraction_to_mask_params)
        noise_dict = None
    elif manipulation_method == "noise":
        damage_levels_list = generate_params_list(noise_levels_params)
        # For noise, we do the original approach of loading a model once
        model, _ = load_model(model_info, pretrained=pretrained, layer_name="temp", layer_path="module._modules.V1._modules.output")
        noise_dict = get_params_sd(model)
        
    else:
        raise ValueError("manipulation_method must be 'connections' or 'noise'.")

    # -------------------------------------------------------------------------
    # B) Identify earliest damaged block (new logic), filter activation layers
    # -------------------------------------------------------------------------
    top_level_blocks_in_order = ["V1", "V2", "V4", "IT"]
    block_order_map = {b: i for i, b in enumerate(top_level_blocks_in_order)}

    damage_indices = []
    for dmg_path in layer_paths_to_damage:
        top_block_dmg = dmg_path.split(".")[0]
        if top_block_dmg in block_order_map:
            damage_indices.append(block_order_map[top_block_dmg])

    if damage_indices:
        earliest_damage_idx = min(damage_indices)
    else:
        earliest_damage_idx = 0
    
    if isinstance(activation_layers_to_save, list):
        model, _ = load_model(model_info, pretrained=pretrained, layer_path=activation_layers_to_save[0])
    else:
        model, _ = load_model(model_info, pretrained=pretrained, layer_path=activation_layers_to_save)
    
        # Now filter activation_layers_to_save to keep only those whose top-level block >= earliest_damage_idx
    final_layers_to_hook = get_final_layers_to_hook(model,activation_layers_to_save,layer_paths_to_damage)
    
    print("Activations to be saved for ",layer_paths_to_damage, ": ", final_layers_to_hook)
    # If final_layers_to_hook is empty, we won't really collect multi-layer activations,
    # but we keep 'layer_name' for the old directory naming logic.

    total_iterations = len(damage_levels_list) * mc_permutations

    # ------------------------------------------------------------------
    # C)  Folder tag: 'noise', 'connections', or 'units'
    # ------------------------------------------------------------------
    if manipulation_method == "noise":
        dir_tag = "noise"
    elif manipulation_method == "connections":
        dir_tag = "units" if masking_level == "units" else "connections"
    else:
        dir_tag = manipulation_method          # fallback, should not occur

    # -------------------------------------------------------------------------
    # SECTION 4: (Kept) The original style: one forward pass per image.
    #            We'll do it slightly adapted so we can handle multi-layer hooking
    # -------------------------------------------------------------------------
    with tqdm(total=total_iterations, desc="Running alteration") as pbar:
        for damage_level in damage_levels_list:
            for permutation_index in range(mc_permutations):
                # 1) Load fresh model & attach multi-hook
                #    We'll store outputs in the original 'activations' dict but for multiple layers
                model, activations = load_model(
                    model_info=model_info,
                    pretrained=pretrained,
                    layer_name=layer_name,       # used for naming, but see note below
                    layer_path=final_layers_to_hook  # multi-layer hooking
                )
                model.eval()

                # 2) Apply the chosen damage
                if manipulation_method == "connections":
                    apply_masking(
                        model,
                        fraction_to_mask=damage_level,
                        layer_paths=layer_paths_to_damage,
                        apply_to_all_layers=apply_to_all_layers,
                        masking_level=masking_level,  
                        only_conv=only_conv,
                        include_bias=include_bias
                    )
                else:  # "noise"
                    apply_noise(
                        model,
                        noise_level=damage_level,
                        noise_dict=noise_dict,
                        layer_paths=layer_paths_to_damage,
                        apply_to_all_layers=apply_to_all_layers,
                        only_conv=only_conv,
                        include_bias=include_bias
                    )

                # 3) Now we do the old "extract_activations" approach (one image at a time)
                #    but we need multi-layer outputs. We'll do a custom loop similar to original code.

                # We'll store each image's flattened activation in a dictionary-of-lists, e.g.:
                #   per_layer_data = {layer_path: []}
                per_layer_data = {lp: [] for lp in final_layers_to_hook}

                image_files = [
                    f for f in os.listdir(image_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]
                image_files.sort()

                for image_file in image_files:
                    img_path = os.path.join(image_dir, image_file)
                    input_tensor = preprocess_image(img_path)

                    # run forward
                    with torch.no_grad():
                        model(input_tensor)

                    # 'activations[lp]' is now the last image's output for that layer
                    # We'll flatten & store in per_layer_data
                    for lp in final_layers_to_hook:
                        out_flat = activations[lp].flatten()  # assume hooking updated this
                        per_layer_data[lp].append(out_flat.cpu().numpy() if torch.is_tensor(out_flat) else out_flat)

                # 4) For each layer, build a DataFrame, compute correlation, selectivity, and save
                #    We'll keep the original directory structure: same as old, but insert 'lp' after RDM/selectivity etc.
                for lp in final_layers_to_hook:
                    # build a 2D array [N_images, features]
                    arr_2d = np.stack(per_layer_data[lp], axis=0)  # shape [N, F]
                    activations_df = pd.DataFrame(arr_2d, index=image_files)

                    # sort indices
                    activations_df_sorted = sort_activations_by_numeric_index(activations_df)

                    lp_name = lp.split(".")[2]
                    # save activations
                    activation_dir = (
                        f"data/haupt_stim_activ/damaged/{model_info['name']}{time_steps}{run_suffix}/"
                        f"{dir_tag}/{layer_name}/activations/{lp_name}/damaged_{round(damage_level,3)}"
                    )
                    os.makedirs(activation_dir, exist_ok=True)
                    activation_dir_path = os.path.join(activation_dir, f"{permutation_index}.pkl")

                    reduced_df = activations_df_sorted.astype(np.float16)
                    reduced_df.to_pickle(activation_dir_path)

                    # compute correlation matrix
                    correlation_matrix, sorted_image_names = compute_correlations(activations_df_sorted)

                    # save correlation matrix
                    corrmat_dir = (
                        f"data/haupt_stim_activ/damaged/{model_info['name']}{time_steps}{run_suffix}/"
                        f"{dir_tag}/{layer_name}/RDM/{lp_name}/damaged_{round(damage_level,3)}"
                    )
                    os.makedirs(corrmat_dir, exist_ok=True)
                    corrmat_path = os.path.join(corrmat_dir, f"{permutation_index}.pkl")
                    with open(corrmat_path, "wb") as f:
                        pickle.dump(correlation_matrix.tolist(), f)

                    # compute within-between
                    categories_array = assign_categories(sorted_image_names)
                    results = calc_within_between(correlation_matrix, categories_array)
                    results = convert_np_to_native(results)

                    # save selectivity
                    selectivity_dir = (
                        f"data/haupt_stim_activ/damaged/{model_info['name']}{time_steps}{run_suffix}/"
                        f"{dir_tag}/{layer_name}/selectivity/{lp_name}/damaged_{round(damage_level,3)}"
                    )
                    os.makedirs(selectivity_dir, exist_ok=True)
                    selectivity_path = os.path.join(selectivity_dir, f"{permutation_index}.pkl")
                    with open(selectivity_path, "wb") as f:
                        pickle.dump(results, f)

                # update progress
                pbar.update(1)

    print("All damage permutations completed!")



def get_sorted_filenames(image_dir):
    """
    List and sort image filenames in a directory by alphabetical prefix and numeric suffix.
    """
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    all_files = os.listdir(image_dir)

    # Extract valid filenames and their base names
    base_names = [os.path.splitext(f)[0] for f in all_files 
                    if os.path.splitext(f)[1].lower() in valid_exts]

    # Sort using a helper function
    return sorted(base_names, key=lambda name: extract_string_numeric_parts(name))


def extract_string_numeric_parts(name):
    """Extract the string prefix and numeric index (integer or float) from a name."""
    match = re.match(r"^([^\d]+)([\d\.]+)$", name)
    if match:
        prefix, num_str = match.groups()
        try:
            # Convert numeric part to float if possible
            num = float(num_str) if "." in num_str else int(num_str)
        except ValueError:
            num = 0
        return prefix, num
    return name, 0


def get_color_for_triple(layer, activation_layer, category):
    """
    Generate a stable RGB color from the hash of the triple.
    This will be consistent across runs for the same triple.
    """
    triple_str = f"{layer}_{activation_layer}_{category}"
    h = hashlib.md5(triple_str.encode("utf-8")).hexdigest()
    # Use the first 6 hex characters (3 bytes) for R, G, B
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return (r, g, b)


def safe_load_pickle(file_path):
    """Safely load a pickle file, checking if it's non-empty and catching any EOFError."""
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        try:
            with open(file_path, "rb") as f:
                return pickle.load(f)
        except EOFError:
            print(f"Warning: Encountered EOFError while loading {file_path}")
            return None
    else:
        print(f"Warning: File {file_path} is empty or does not exist.")
        return None


def categ_corr_lineplot(
    damage_layers,
    activations_layers,
    damage_type,
    main_dir="data/haupt_stim_activ/damaged/cornet_rt/",
    categories=("overall",),              # default for imagenet
    metric="observed_difference",          # or "top1"/"top5" for imagenet
    subdir_regex=r"damaged_([\d\.]+)$",
    plot_dir="plots/",
    data_type="selectivity",               # "selectivity" | "svm_15" | "imagenet"
    scatter=False,
    verbose=0,
    ylim=None,
    percentage=False,
    selectivity_fraction: float|None = None,
    selection_mode: str = "percentage",
    selectivity_file: str|None   = "unit_selectivity/all_layers_units_mannwhitneyu.pkl"):
    """
    Aggregate replicate files into mean±std curves.

    data_type == "selectivity"  -> original within-between correlation
    data_type.startswith("svm") -> original SVM accuracy
    data_type == "imagenet"     -> this NEW branch, needs .pkl of form
            { "overall": {"top1": .., "top5": ..},
                "classes": { cls_idx: {"top1": .., "top5": ..}, ... } }

    categories
    ----------
    selectivity / svm : list[str]   ("animal","face",...) total
    imagenet          : iterable of
       "overall"                → use content["overall"][metric]
       int 0-999 (or str digit) → use per-class entry
    """

    # ------------ 1. choose data sub-folder --------------------
    if data_type in ("selectivity",) or data_type.startswith("svm"):
        data_subfolder = data_type
    elif data_type == "imagenet":
        data_subfolder = "imagenet"
        if metric not in ("top1", "top5"):
            raise ValueError("metric must be 'top1' or 'top5' when data_type='imagenet'")
    else:
        raise ValueError(f"unknown data_type '{data_type}'")

    # ------------ 2. helper for cache filename ----------------
    def agg_fname(frac):
        if data_type == "imagenet":
            return f"avg_imagenet_{metric}_{frac}.pkl"
        else:
            return f"avg_{data_type}_{frac}.pkl"

    # ------------ 3. containers -------------------------------
    data       = {}   # (layer, act_key, cat) -> {frac:(mean,std,n)}
    raw_points = {}   # same keys -> {frac:[replicas]}

    # ------------ 4. crawl the directory tree -----------------
    for layer in damage_layers:
        for act in activations_layers:
            act_key = act
            if data_type == "selectivity" and selectivity_fraction is not None:
                if "total" in categories:
                    categories_rdm = ("face", "object", "animal", "place") # FIX THIS 

                # 1. Build RDM directory path
                rdm_dir = Path(main_dir) / damage_type / layer / f"RDM_{selectivity_fraction:.2f}_{selection_mode}" / act
                if not rdm_dir.exists():
                    # 2. Generate RDMs if missing
                    print("MISSING")
                    activ_root = os.path.join(main_dir, damage_type)
                    generate_category_selective_RDMs(
                        activations_root=activ_root,
                        layer_name=act,
                        top_frac=selectivity_fraction,
                        categories=categories_rdm,
                        selection_mode=selection_mode,
                        damage_layer=layer,
                        activation_layer=act,
                        selectivity_file=selectivity_file
                    )

                # 3. Prepare output directory for averages
                avg_dir = Path(main_dir) / damage_type / layer / f"avg_selectivity_top{selectivity_fraction:.2f}_{selection_mode}" / act
                avg_dir.mkdir(parents=True, exist_ok=True)

                # 4. For each category
                for cat in categories_rdm:
                    print(layer, act_key, cat)
                    data[(layer,act_key,cat)] = {}
                    raw_points[(layer,act_key,cat)] = {}
                    cat_dir = rdm_dir / f"{cat}_selective"
                    if not cat_dir.exists(): # Skip if dir doesn't exist
                        continue
                    for dmg in sorted(cat_dir.iterdir()):
                        if not dmg.is_dir():
                            continue
                        dmg_level = dmg.name.split("_")[-1]
                        avg_file = avg_dir / f"avg_selectivity_{cat}_{dmg_level}.pkl"
                        if avg_file.exists():
                            # Load if already computed
                            with open(avg_file, "rb") as f:
                                stats = pickle.load(f)
                        else:
                            # Compute within-between selectivity for all RDMs in this damage level
                            selectivities = []
                            for rdm_pkl in sorted(dmg.glob("*.pkl")):
                                with open(rdm_pkl, "rb") as f:
                                    content = pickle.load(f)
                                    R = content['RDM']
                                    image_names = assign_categories(content['image_names']) # Creates array from image names
                                # Compute within-between for this category
                                # (Assume you have a helper function for this)
                                sel_dict = calc_within_between(R, image_names)
                                sel = sel_dict[cat]["observed_difference"]
                                selectivities.append(sel)
                            # Aggregate
                            mean_sel = float(np.mean(selectivities))
                            std_sel = float(np.std(selectivities,ddof=1))
                            stats = {"mean": mean_sel, "std": std_sel, "n": len(selectivities)}
                            with open(avg_file, "wb") as f:
                                pickle.dump(stats, f)
                        # Store for plotting
                        data[(layer, act, cat)][float(dmg_level)] = (stats["mean"], stats["std"], stats["n"])
                if "total" in categories:
                    data[(layer, act_key, "total")] = {}
                    raw_points[(layer, act_key, "total")] = {}
                    # For each damage level, aggregate across all categories
                    # First, collect all damage levels present in any category
                    all_dmg_levels = set()
                    for cat in categories_rdm:
                        cat_dir = rdm_dir / f"{cat}_selective"
                        if not cat_dir.exists():
                            continue
                        for dmg in sorted(cat_dir.iterdir()):
                            if not dmg.is_dir():
                                continue
                            dmg_level = dmg.name.split("_")[-1]
                            all_dmg_levels.add(dmg_level)
                    for dmg_level in sorted(all_dmg_levels, key=float):
                        all_selectivities = []
                        for cat in categories_rdm:
                            cat_dir = rdm_dir / f"{cat}_selective"
                            dmg = cat_dir / f"damaged_{dmg_level}"
                            if not dmg.is_dir():
                                continue
                            for rdm_pkl in sorted(dmg.glob("*.pkl")):
                                with open(rdm_pkl, "rb") as f:
                                    content = pickle.load(f)
                                    R = content['RDM']
                                    image_names = assign_categories(content['image_names'])
                                sel_dict = calc_within_between(R, image_names)
                                # Use the category-specific value
                                sel = sel_dict[cat]["observed_difference"]
                                all_selectivities.append(sel)
                        if all_selectivities:
                            mean_sel = float(np.mean(all_selectivities))
                            std_sel = float(np.std(all_selectivities, ddof=1))
                            stats = {"mean": mean_sel, "std": std_sel, "n": len(all_selectivities)}
                            data[(layer, act_key, "total")][float(dmg_level)] = (mean_sel, std_sel, len(all_selectivities))
                            raw_points[(layer, act_key, "total")][float(dmg_level)] = all_selectivities


            else:
                # pick path & “act_key” (imagenet has no per-activation dir)
                if data_type == "imagenet":
                    layer_path = os.path.join(main_dir, damage_type, layer, "imagenet")
                    out_base   = os.path.join(main_dir, damage_type, layer, "avg_imagenet")
                    act_key    = "imagenet"
                else:
                    layer_path = os.path.join(main_dir, damage_type, layer, data_subfolder, act)
                    out_base   = os.path.join(main_dir, damage_type, layer, f"avg_{data_type}", act)
                    act_key    = act

                if not os.path.isdir(layer_path):
                    continue
                os.makedirs(out_base, exist_ok=True)

                # init dict slots
                for cat in categories:
                    data[(layer, act_key, cat)] = {}
                    raw_points[(layer, act_key, cat)] = {}
                print(data)

                # scan damaged_* subdirs
                for subdir in os.listdir(layer_path):
                    subdir_path = os.path.join(layer_path, subdir)
                    if not os.path.isdir(subdir_path):
                        continue
                    m = re.search(subdir_regex, subdir)
                    if not m:
                        continue
                    frac = round(float(m.group(1)), 3)
                    cache = os.path.join(out_base, agg_fname(frac))

                    # ---------- build cache if missing ----------
                    if not os.path.exists(cache):
                        agg = {}                      # cat -> list[values]

                        for fn in os.listdir(subdir_path):
                            if not fn.lower().endswith(".pkl"):
                                continue
                            content = safe_load_pickle(os.path.join(subdir_path, fn))
                            if content is None:
                                continue

                            # ---- IMAGE NET MODE ----
                            if data_type == "imagenet":
                                for cat in categories:
                                    if str(cat).lower() == "overall":
                                        val = content["overall"][metric]
                                    elif str(cat).isdigit():
                                        cls = int(cat)
                                        val = content["classes"].get(cls, {}).get(metric, np.nan)
                                    else:
                                        continue
                                    agg.setdefault(cat, []).append(float(val))

                            # ---- SELECTIVITY MODE ----
                            elif data_type == "selectivity":
                                if not isinstance(content, dict):
                                    continue
                                for cat_name, met_dict in content.items():
                                    if (cat_name in categories and
                                        metric in met_dict):
                                        agg.setdefault(cat_name, []).append(float(met_dict[metric]))

                            # ---- SVM MODE ----
                            else:   # data_type starts with "svm"
                                for cat_name in categories:
                                    if cat_name in content:
                                        agg.setdefault(cat_name, []).append(float(content[cat_name]["score"]))

                        # save aggregated stats
                        packed = {
                            c: {
                                metric: {
                                    "mean": float(np.mean(v)),
                                    "std":  float(np.std(v)),
                                    "n":    len(v),
                                    "vals": v
                                }
                            } for c, v in agg.items()
                        }
                        with open(cache, "wb") as f:
                            pickle.dump(packed, f)

                    # ---------- read cache and populate data ----------
                    agg_content = safe_load_pickle(cache) or {}
                    for cat in categories:
                        cat_key = str(cat).lower() if data_type.startswith("svm") else cat
                        if (cat_key in agg_content and
                            metric in agg_content[cat_key] and
                            "n" in agg_content[cat_key][metric]):
                            mean = agg_content[cat_key][metric]["mean"]
                            std  = agg_content[cat_key][metric]["std"]
                            n    = agg_content[cat_key][metric]["n"]
                            vals = agg_content[cat_key][metric]["vals"]

                            data[(layer, act_key, cat)][frac] = (mean, std, n)
                            raw_points[(layer, act_key, cat)][frac] = vals

    # ------------ 5. optional percentage scaling --------------
    if percentage:
        for key in data:
            frac_dict = data[key]
            raw_dict  = raw_points[key]
            if not frac_dict:
                continue
            base_frac = sorted(frac_dict.keys())[0]
            base_vals = np.array(raw_dict[base_frac], dtype=float)
            for frac in frac_dict:
                cur_vals = np.array(raw_dict[frac], dtype=float)
                min_len  = min(len(base_vals), len(cur_vals))
                ratio    = 100.0 * cur_vals[:min_len] / np.where(base_vals[:min_len]==0, np.nan, base_vals[:min_len])
                frac_dict[frac] = (float(np.nanmean(ratio)),
                                   float(np.nanstd(ratio)),
                                   len(ratio))
                raw_dict[frac]  = ratio.tolist()

    # ------------ 6. PLOT --------------------------------------
    plt.figure(figsize=(8, 6))
    for (layer, act_key, cat), frac_dict in data.items():
        if not frac_dict or cat not in categories:
            continue
        xs = sorted(frac_dict.keys())
        ys = [frac_dict[x][0] for x in xs]
        err= [frac_dict[x][1] for x in xs]
        lbl = f"{layer}-{act_key}-{cat}"
        color = get_color_for_triple(layer, act_key, str(cat))
        plt.errorbar(xs, ys, yerr=err, fmt='-o', capsize=4, label=lbl, color=color)
        if scatter:
            for x in xs:
                jitter = np.random.normal(0, 0.005, size=len(raw_points[(layer, act_key, cat)][x]))
                plt.scatter(x + jitter, raw_points[(layer, act_key, cat)][x],
                            color=color, alpha=0.5, s=10)

    plt.xlabel("Damage parameter")
    if data_type == "imagenet":
        ylabel = f"ImageNet {metric.upper()} accuracy"
    elif data_type == "selectivity":
        ylabel = "Differentiation (within-between)"
    else:
        ylabel = "SVM accuracy"
    if percentage:
        ylabel += " (scaled %)"
    plt.ylabel(ylabel)
    plt.title(f"{data_type} vs damage — {damage_type}")
    if ylim: plt.ylim(ylim)
    plt.legend()
    plt.tight_layout()

    os.makedirs(plot_dir, exist_ok=True)
    name_parts = [main_dir.strip("/").split("/")[-1],
                  data_type, damage_type, metric]
    name_parts.extend(damage_layers)
    name_parts.extend(activations_layers)
    if selectivity_fraction is not None:
        name_parts.append(f"top{selectivity_fraction:.2f}-{selection_mode}")
    plot_path = os.path.join(plot_dir, "_".join(name_parts) + ".png")
    plt.savefig(plot_path, dpi=500)
    if verbose: plt.show()
    else: plt.close()


def plot_avg_corr_mat(
    layers,
    damage_type,
    image_dir="stimuli/",
    output_dir="average_RDMs",
    subdir_regex=r"damaged_([\d\.]+)$",
    damage_levels=None, # List
    main_dir="data/haupt_stim_activ/damaged/cornet_rt/",
    vmax=1.0,
    plot_dir="plots/",
    verbose=0
):
    """
    1. Loop over subdirectories matching the damage type and layers.
    2. For each match, compute the average correlation matrix if not already saved.
    3. Collect and plot the matrices as subplots, allowing multiple layers and damage types.
    """
    # Prepare data structure
    fraction_to_matrix = {}

    # Build axis labels once
    sorted_image_names = get_sorted_filenames(image_dir)
    n_images=len(sorted_image_names)

    # Loop over layers and damage types
    for layer in layers:
        layer_output_dir = os.path.join(main_dir, damage_type, layer, output_dir)
        os.makedirs(layer_output_dir, exist_ok=True)

        layer_path = os.path.join(main_dir, damage_type, layer, "RDM")

        if not os.path.isdir(layer_path):
            continue

        for subdir_name in os.listdir(layer_path):
            subdir_path = os.path.join(layer_path, subdir_name)
            if not os.path.isdir(subdir_path):
                continue

            # Check for suffix or regex match
            if damage_levels:
                if not any(subdir_name.endswith(f"damaged_{suffix}") for suffix in damage_levels):
                    continue
            else:
                match = re.search(subdir_regex, subdir_name)
                if not match:
                    continue

            fraction = round(float(match.group(1)) if not damage_levels else extract_string_numeric_parts(subdir_name)[1],3)
            
            # Prepare output file name
            out_fname = f"avg_RDM_{fraction}.pkl"
            out_path = os.path.join(layer_output_dir, out_fname)
            # Check for precomputed matrix
            if os.path.exists(out_path):
                print(f"Found existing average RDM for fraction={fraction}, layer={layer}; loading it.")
                with open(out_path, "rb") as f:
                    avg_mat = np.array(pickle.load(f), dtype=np.float32)
            else:
                all_mats = []
                print(f"No precomputed average RDM for fraction={fraction}, layer={layer}; computing now.")
                for fname in os.listdir(subdir_path):
                    if fname.lower().endswith(".pkl"):
                        pkl_path = os.path.join(subdir_path, fname)
                        with open(pkl_path, "rb") as f:
                            matrix_list = pickle.load(f)
                        mat = np.array(matrix_list, dtype=np.float32)
                        all_mats.append(mat)

                avg_mat = np.mean(all_mats, axis=0) if all_mats else None
                if avg_mat is not None:
                    with open(out_path, "wb") as f:
                        pickle.dump(avg_mat.tolist(), f)

            if avg_mat is not None:
                fraction_to_matrix[(fraction, layer)] = avg_mat

    # Plot results
    sorted_fractions = sorted(fraction_to_matrix.keys())
    n_subplots = len(sorted_fractions)
    n_cols = int(math.ceil(n_subplots ** 0.5))
    n_rows = int(math.ceil(n_subplots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), squeeze=False)
    axes = axes.ravel()
    

    for i, (fraction, layer) in enumerate(sorted_fractions):
        avg_mat = fraction_to_matrix[(fraction, layer)]
        ax = axes[i]
        im = ax.imshow(avg_mat, cmap="viridis", vmin=0, vmax=vmax)
        ax.set_xticks(range(n_images))
        ax.set_yticks(range(n_images))
        ax.set_xticklabels(sorted_image_names, rotation=90, fontsize=4)
        ax.set_yticklabels(sorted_image_names, fontsize=4)
        ax.set_title(f"Fraction={fraction}, Layer={layer}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.suptitle(damage_type)
    plt.tight_layout()
        # Saving plot
    os.makedirs(plot_dir, exist_ok=True)
    model_name = main_dir.split("/")[-2] # Assuming that there is a slash after the model name ("/cornet/")
    plot_name = f"{model_name}_RDM_{damage_type}"

    for layer in layers:
        plot_name = plot_name + f"_{layer}"

    n_damages = len(damage_levels)
    
    plot_name = plot_name + f"_{n_damages}-levels"

    save_path = os.path.join(plot_dir, plot_name)

    if verbose==1:
        save_plot = input(f"Save plot under {save_path}? Y/N: ")

        if save_plot.capitalize() == "Y":
            plt.savefig(save_path, dpi=500)
        plt.show()
    elif verbose == 0:
        plt.savefig(save_path, dpi=500)
    else:
        ValueError(f"{verbose} is not a valid value. Use 0 or 1.")



def plot_correlation_heatmap(correlation_matrix, sorted_image_names, layer_name='IT', vmax=0.4, model_name="untitled_model"):
    """
    Plot a heatmap of the correlation matrix.

    Parameters:
        correlation_matrix (np.ndarray): The correlation matrix.
        sorted_image_names (list): List of image names corresponding to matrix indices.
        layer_name (str): Name of the layer used in the title.
        vmax (float): Upper bound for colormap.
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=False, cmap="viridis",
                xticklabels=sorted_image_names, yticklabels=sorted_image_names, vmax=vmax, vmin=0)
    plt.title(f"Correlation of Activations Between Images (Layer: {layer_name})")
    plt.xlabel("Images")
    plt.ylabel("Images")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    # Save figure
    save_path = f"data/haupt_stim_activ/{model_name}/{layer_name}.png"
    # Create the directories if they don't already exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=400)
    plt.show()


def plot_categ_differences(
    damage_layers,
    activations_layers,
    damage_type,
    main_dir="data/haupt_stim_activ/damaged/cornet_rt/",
    image_dir="stimuli/",
    mode='dirs',
    file_prefix='damaged_',
    damage_levels=["0"],
    comparison=False,
    plot_dir="plots/",
    verbose=0,
    scatter=False,
    ylim=None,
    data_type="selectivity",   # "selectivity" or e.g. "svm_15"
    percentage=False
):
    """
    Plot either:
      - Selectivity: bars = (within-cat) - (between-cat) for correlation-based data
      - SVM: bars = average classification accuracy (cat vs. other-cat)

    If comparison=False, each (damage_layer, activation_layer, damage_level) combo
    is its own row of subplots (columns = categories). If comparison=True, fewer
    subplots with grouped bars.

    When percentages=True, the first damage level (damage_levels[0]) is treated
    as the baseline. We compute ratio = (current / baseline) * 100 for each raw
    replicate. That includes re-scaling the baseline itself, so the baseline
    bar is at 100%.  

    data_type:
       "selectivity" -> correlation-based data from ".../RDM/<act_layer>"
       "svm_..." -> classification accuracy from ".../svm_15/<act_layer>".
    """

    # ---------------- HELPER FUNCTIONS ----------------
    def unify_image_name(cat_str):
        """ Convert 'scene' -> 'place' for selectivity categories. """
        cat_str = cat_str.lower()
        if cat_str == "scene":
            return "place"
        return cat_str


    def get_sorted_filenames(folder):
        """Return sorted list of filenames, ignoring subdirectories."""
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Image directory '{folder}' does not exist.")
        files = [
            f for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))
        ]
        return sorted(files)

    def extract_string_numeric_parts(fname):
        """
        e.g. "face01.jpg" -> ("face", "01"), "animal12.3.png" -> ("animal", "12.3").
        Used for identifying the category from the filename prefix.
        """
        match = re.match(r"([^\d]+)([\d\.]*)", os.path.splitext(fname)[0])
        if match:
            return (match.group(1), match.group(2))
        else:
            return (fname, "")

    # ----------- SELECTIVITY FUNCTIONS -------------
    def load_correlation_matrices(item_path, n_files):
        def _validate_matrix(mat, path_str):
            if len(mat) != n_files:
                raise ValueError(
                    f"Matrix in '{path_str}' has {len(mat)} rows, but expected {n_files}."
                )
            for row in mat:
                if len(row) != n_files:
                    raise ValueError(f"Non-square row in '{path_str}'.")
        mats = []
        if os.path.isfile(item_path):
            with open(item_path, 'rb') as f:
                mat = pickle.load(f)
            _validate_matrix(mat, item_path)
            mats.append(mat)
        elif os.path.isdir(item_path):
            for fname in sorted(os.listdir(item_path)):
                if fname.endswith(".pkl"):
                    path = os.path.join(item_path, fname)
                    with open(path, 'rb') as f:
                        mat = pickle.load(f)
                    _validate_matrix(mat, path)
                    mats.append(mat)
        else:
            raise FileNotFoundError(f"'{item_path}' is neither file nor directory.")
        return mats

    def compute_differences_selectivity(matrices, sorted_filenames, categories_list):
        """
        For each focal category, compute (within-cat) - (between-cat) for each other cat.
        Return diffs_dict[cat] = (other_cats, mean_vals, std_vals, raw_arrays).
        """
        accum = {
            cat: {oc: [] for oc in categories_list if oc != cat}
            for cat in categories_list
        }

        for mat in matrices:
            for cat in categories_list:
                within = []
                between = {oc: [] for oc in categories_list if oc != cat}
                for r_i, row in enumerate(mat):
                    cat_r = unify_image_name(extract_string_numeric_parts(sorted_filenames[r_i])[0])
                    for c_i, val in enumerate(row):
                        if r_i == c_i:
                            continue  # skip diagonal
                        cat_c = unify_image_name(extract_string_numeric_parts(sorted_filenames[c_i])[0])
                        if cat_r == cat and cat_c == cat:
                            within.append(val)
                        elif cat_r == cat and cat_c != cat:
                            between[cat_c].append(val)
                        elif cat_c == cat and cat_r != cat:
                            between[cat_r].append(val)

                w_mean = np.mean(within) if within else 0.0
                for oc in between:
                    b_mean = np.mean(between[oc]) if len(between[oc]) > 0 else 0.0
                    diff = w_mean - b_mean
                    accum[cat][oc].append(diff)

        # Convert accum -> final diffs_dict
        diffs_dict = {}
        for cat in accum:
            other_cats = sorted(accum[cat].keys())
            mean_vals, std_vals, raw_arrays = [], [], []
            for oc in other_cats:
                arr = np.array(accum[cat][oc])
                mean_vals.append(arr.mean())
                std_vals.append(arr.std() * 1.96)  # 95% CI
                raw_arrays.append(arr)
            diffs_dict[cat] = (other_cats, mean_vals, std_vals, raw_arrays)
        return diffs_dict

    # ----------- SVM FUNCTIONS -------------
    def load_svm_dataframes(item_path):
        """Load all .pkl files as DataFrames from a dir or a single file."""
        dfs = []
        if os.path.isfile(item_path):
            df = pd.read_pickle(item_path)
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"File '{item_path}' did not contain a DataFrame.")
            dfs.append(df)
        elif os.path.isdir(item_path):
            for fname in sorted(os.listdir(item_path)):
                if fname.endswith(".pkl"):
                    path = os.path.join(item_path, fname)
                    df = pd.read_pickle(path)
                    if not isinstance(df, pd.DataFrame):
                        raise ValueError(f"File '{path}' did not contain a DataFrame.")
                    dfs.append(df)
        else:
            raise FileNotFoundError(f"'{item_path}' is neither file nor directory.")
        return dfs

    def compute_svm_pairwise_accuracies(dataframes, categories_list):
        """
        For each focal cat 'cat' and other cat 'oc':
          - find columns containing BOTH cat and oc (e.g. 'object_vs_face')
          - average them => that bar = classification accuracy for cat vs oc.
        Return diffs_dict[cat] = (other_cats, mean_vals, std_vals, raw_arrays).
        """
        accum = {
            cat: {oc: [] for oc in categories_list if oc != cat}
            for cat in categories_list
        }

        for df in dataframes:
            lower_cols = [c.lower() for c in df.columns]
            for cat in categories_list:
                cat_l = cat 
                for oc in categories_list:
                    if oc == cat:
                        continue
                    oc_l = oc
                    # gather columns that have BOTH cat_l and oc_l
                    pair_cols_idx = [
                        i for i, col in enumerate(lower_cols)
                        if (cat_l in col) and (oc_l in col)
                    ]
                    if pair_cols_idx:
                        vals = df.iloc[:, pair_cols_idx].values.flatten()
                        mean_acc = np.mean(vals)
                    else:
                        mean_acc = 0.0
                    accum[cat][oc].append(mean_acc)

        # Convert accum -> final
        diffs_dict = {}
        for cat in accum:
            other_cats = sorted(accum[cat].keys())
            mean_vals, std_vals, raw_arrays = [], [], []
            for oc in other_cats:
                arr = np.array(accum[cat][oc])
                mean_vals.append(arr.mean())
                std_vals.append(arr.std() * 1.96)  # 95% CI
                raw_arrays.append(arr)
            diffs_dict[cat] = (other_cats, mean_vals, std_vals, raw_arrays)
        return diffs_dict

    # ---------------- Utility to scale to baseline ----------------
    def scale_to_baseline(diffs_dict_current, diffs_dict_baseline):
        """
        Given two diffs_dict structures:
          - diffs_dict_current: { cat: (oc_list, mean_vals, std_vals, raw_arrays) }
          - diffs_dict_baseline: same structure, used as baseline.

        Produce a new scaled version of diffs_dict_current where each raw replicate
        is (current / baseline) * 100. The baseline's own replicate also becomes 100%
        (since baseline / baseline = 1).

        If baseline replicate is 0, we get NaN for that ratio.
        """
        scaled_dict = {}
        for cat, (oc_list, mean_vals, std_vals, raw_arrays) in diffs_dict_current.items():
            # If cat not in baseline, keep original (or set to NaN).
            if cat not in diffs_dict_baseline:
                scaled_dict[cat] = (oc_list, mean_vals, std_vals, raw_arrays)
                continue

            base_oc_list, base_mean_vals, base_std_vals, base_raw_arrays = diffs_dict_baseline[cat]

            # We'll assume oc_list and base_oc_list are in the same sorted order
            # If they differ, you could do a more robust 'index matching'.
            scaled_oc_list = oc_list
            new_mean_vals = []
            new_std_vals = []
            new_raw_arrays = []

            for i, oc in enumerate(oc_list):
                # find the corresponding baseline index for oc
                if i >= len(base_oc_list) or oc != base_oc_list[i]:
                    # fallback: search by name
                    try:
                        base_i = base_oc_list.index(oc)
                    except ValueError:
                        new_mean_vals.append(np.nan)
                        new_std_vals.append(np.nan)
                        new_raw_arrays.append([])
                        continue
                else:
                    base_i = i

                arr_base = np.array(base_raw_arrays[base_i])
                arr_curr = np.array(raw_arrays[i])

                # If arrays differ in length for some reason, match the smaller
                min_len = min(len(arr_base), len(arr_curr))
                arr_base = arr_base[:min_len]
                arr_curr = arr_curr[:min_len]

                ratio_arr = []
                for vb, vc in zip(arr_base, arr_curr):
                    if vb != 0:
                        ratio_arr.append((vc / vb) * 100.0)
                    else:
                        ratio_arr.append(np.nan)

                ratio_arr = np.array(ratio_arr)
                new_mean_vals.append(np.nanmean(ratio_arr))
                new_std_vals.append(np.nanstd(ratio_arr) * 1.96)
                new_raw_arrays.append(ratio_arr.tolist())

            scaled_dict[cat] = (scaled_oc_list, new_mean_vals, new_std_vals, new_raw_arrays)
        return scaled_dict

    # ---------- 1) LOAD & GROUP IMAGES BY CATEGORY -----------
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory '{image_dir}' does not exist.")

    sorted_filenames = get_sorted_filenames(image_dir)
    n_files = len(sorted_filenames)

    # We'll define a custom category order with "place" (not "scene"):
    custom_category_order = ["face", "place", "object", "animal"]

    # Build categories_map: {cat_name: [filenames]}
    categories_map = {}
    for fname in sorted_filenames:
        cat_raw = extract_string_numeric_parts(fname)[0]
        cat = unify_image_name(cat_raw)  # unify "scene" -> "place"
        categories_map.setdefault(cat, []).append(fname)

    # Keep only categories that are actually found
    categories_list = [c for c in custom_category_order if c in categories_map]
    n_categories = len(categories_list)

    # ---------- 2) GATHER RESULTS (matrices or dataframes) -----------
    def get_base_path(dmg_layer, act_layer):
        if data_type.startswith("svm"):
            # e.g. .../<dmg_layer>/svm_15/<act_layer>/
            return os.path.join(main_dir, damage_type, dmg_layer, data_type, act_layer)
        else:
            # selectivity => correlation RDM
            return os.path.join(main_dir, damage_type, dmg_layer, "RDM", act_layer)

    all_results = []  # list of (dmg_layer, act_layer, suffix, item_path, diffs_dict)

    if not damage_layers:
        raise ValueError("No damage_layers specified.")
    if not activations_layers:
        raise ValueError("No activations_layers specified.")

    for dmg_layer in damage_layers:
        for act_layer in activations_layers:
            base_path = get_base_path(dmg_layer, act_layer)
            if not os.path.isdir(base_path):
                raise FileNotFoundError(
                    f"Directory '{base_path}' does not exist.\n"
                    f"Check your damage_type='{damage_type}', damage_layer='{dmg_layer}', or activation_layer='{act_layer}'."
                )

            for suffix in damage_levels:
                # find items (dirs or files) that match <file_prefix>*<suffix>
                items = []
                if mode == 'files':
                    for f in os.listdir(base_path):
                        fullpath = os.path.join(base_path, f)
                        if os.path.isfile(fullpath):
                            if f.startswith(file_prefix) and f.endswith(suffix):
                                items.append(fullpath)
                elif mode == 'dirs':
                    for d in os.listdir(base_path):
                        full_dpath = os.path.join(base_path, d)
                        if os.path.isdir(full_dpath):
                            if d.startswith(file_prefix) and d.endswith(suffix):
                                items.append(full_dpath)
                else:
                    raise ValueError("mode must be 'files' or 'dirs'.")

                if not items:
                    raise FileNotFoundError(
                        f"No matching items found in '{base_path}' for dmg_layer='{dmg_layer}', "
                        f"act_layer='{act_layer}', suffix='{suffix}', prefix='{file_prefix}', mode='{mode}'."
                    )

                # load each item and compute diffs
                for item_path in items:
                    if data_type.startswith("svm"):
                        dfs = load_svm_dataframes(item_path)
                        diffs_dict = compute_svm_pairwise_accuracies(dfs, categories_list)
                    else:
                        mats = load_correlation_matrices(item_path, n_files)
                        diffs_dict = compute_differences_selectivity(mats, sorted_filenames, categories_list)

                    all_results.append((dmg_layer, act_layer, suffix, item_path, diffs_dict))

    # ---------- 3) (OPTIONAL) CONVERT TO PERCENTAGES -----------
    if percentage:
        # Group entries by (damage_layer, act_layer) so we can scale 
        # each suffix to the baseline suffix = damage_levels[0] in that group.
        baseline_sfx = damage_levels[0]
        group_map = defaultdict(list)
        # We'll keep track of the index in all_results so we can overwrite.
        for idx, (dmg_layer, act_layer, sfx, ipath, diffs) in enumerate(all_results):
            group_map[(dmg_layer, act_layer)].append((sfx, ipath, diffs, idx))

        updated_all_results = list(all_results)  # make a mutable copy
        for group_key, items in group_map.items():
            # Find the baseline item for this group (the one with suffix == baseline_sfx)
            baseline_item = None
            for (sfx, ipath, diffs_d, idx_in_all) in items:
                if sfx == baseline_sfx:
                    baseline_item = (sfx, ipath, diffs_d, idx_in_all)
                    break

            if baseline_item is None:
                # no baseline in this group, skip
                continue
            baseline_diffs = baseline_item[2]

            # Scale *every* suffix (including the baseline) 
            # so that baseline ÷ baseline = 1 => 100%.
            for (sfx, ipath, current_diffs, idx_in_all) in items:
                scaled_diffs = scale_to_baseline(current_diffs, baseline_diffs)
                # Overwrite in updated_all_results
                old_tuple = list(updated_all_results[idx_in_all])
                old_tuple[4] = scaled_diffs
                updated_all_results[idx_in_all] = tuple(old_tuple)

        all_results = updated_all_results

    # ---------- 4) PLOTTING -----------
    if not comparison:
        # One row per (dmg_layer, act_layer, suffix, item_path)
        # columns = categories
        num_rows = len(all_results)
        fig, axes = plt.subplots(
            num_rows, n_categories,
            figsize=(3*n_categories, 3*num_rows),
            sharey=True
        )
        axes = np.array(axes, ndmin=2)  # ensure 2D

        for i, (dmg_layer, act_layer, suffix, item_path, diffs_dict) in enumerate(all_results):
            for j, cat in enumerate(categories_list):
                ax = axes[i, j]
                if cat not in diffs_dict:
                    ax.set_visible(False)
                    continue

                other_cats, mean_vals, std_vals, raw_diffs = diffs_dict[cat]
                x_pos = np.arange(len(other_cats))

                # bars
                ax.bar(
                    x_pos, mean_vals,
                    yerr=std_vals,
                    capsize=4
                )

                # scatter if desired
                if scatter:
                    for idx_oc, arr in enumerate(raw_diffs):
                        if len(arr) == 0:
                            continue
                        jitter = np.random.normal(0, 0.05, size=len(arr))
                        ax.scatter(
                            x_pos[idx_oc] + jitter,
                            arr,
                            alpha=0.4, s=20, zorder=0
                        )

                ax.set_xticks(x_pos)
                ax.set_xticklabels(other_cats, rotation=45, ha='right')
                if ylim is not None:
                    ax.set_ylim(ylim)

                if data_type.startswith("svm"):
                    ylabel = "Classification Accuracy"
                else:
                    ylabel = "Within - Between"
                if percentage:
                    ylabel += " (scaled %)"

                ax.set_ylabel(ylabel)
                ax.set_title(f"{dmg_layer}, {act_layer}, dmg={suffix}\n{cat}")

        plt.tight_layout()
        os.makedirs(plot_dir, exist_ok=True)
        model_name = main_dir.strip("/").split("/")[-1]
        plot_name = f"{model_name}_{data_type}_categ-diff_{damage_type}"
        for layer in damage_layers:
            plot_name += f"_{layer}"
        for al in activations_layers:
            plot_name += f"_{al}"
        plot_name += f"_{len(damage_levels)}-levels"
        if scatter:
            plot_name += "_with-points"
        if percentage:
            plot_name += "_percent"
        plot_path = os.path.join(plot_dir, plot_name)

        if verbose == 1:
            save_plot = input(f"Save plot under {plot_path}? Y/N: ")
            if save_plot.capitalize() == "Y":
                plt.savefig(plot_path, dpi=500)
            plt.show()
        elif verbose == 0:
            plt.savefig(plot_path, dpi=500)
            plt.show()
        else:
            raise ValueError(f"{verbose} is not valid. Use 0 or 1.")

    else:
        # comparison=True => group all combos in fewer subplots
        # (1 subplot per category, grouped bars for each combo).
        fig, axes = plt.subplots(1, n_categories, figsize=(4*n_categories, 4), sharey=True)
        if n_categories == 1:
            axes = [axes]

        # group results by category
        results_by_cat = {cat: {} for cat in categories_list}
        combos = []
        for (dmg_layer, act_layer, suffix, item_path, diffs_dict) in all_results:
            combo_key = (dmg_layer, act_layer, suffix, item_path)
            if combo_key not in combos:
                combos.append(combo_key)
            for cat in categories_list:
                results_by_cat[cat][combo_key] = diffs_dict.get(cat, ([], [], [], []))

        bar_width = 0.8 / len(combos)
        for idx_cat, cat in enumerate(categories_list):
            ax = axes[idx_cat]
            if data_type.startswith("svm"):
                ylabel = "Classification Accuracy"
            else:
                ylabel = "Within - Between"
            if percentage:
                ylabel += " (scaled %)"

            ax.set_ylabel(ylabel)
            ax.set_title(f"{cat}")

            for i, combo_key in enumerate(combos):
                (oc_list, mean_vals, std_vals, raw_diffs) = results_by_cat[cat][combo_key]
                if not oc_list:
                    continue

                # reorder oc_list to match custom_category_order minus cat
                index_map = {c: k for k, c in enumerate(oc_list)}
                new_oc_list = [c for c in custom_category_order if c != cat and c in index_map]

                new_mean_vals = [mean_vals[index_map[c]] for c in new_oc_list]
                new_std_vals = [std_vals[index_map[c]] for c in new_oc_list]
                new_raw_diffs = [raw_diffs[index_map[c]] for c in new_oc_list]

                x_pos = np.arange(len(new_oc_list))
                offset = i * bar_width
                color_id = f"C{i}"

                ax.bar(
                    x_pos + offset,
                    new_mean_vals,
                    yerr=new_std_vals,
                    width=bar_width,
                    label=f"{combo_key[0]}, {combo_key[1]}, dmg={combo_key[2]}",
                    capsize=4,
                    facecolor="none",
                    edgecolor=color_id,
                    linewidth=2
                )

                if scatter:
                    for idx_oc, arr in enumerate(new_raw_diffs):
                        if len(arr) == 0:
                            continue
                        jitter = np.random.normal(0, bar_width/6, size=len(arr))
                        ax.scatter(
                            (x_pos[idx_oc] + offset) + jitter,
                            arr,
                            alpha=0.4, s=20, zorder=0,
                            color=color_id
                        )

            ax.set_xticks(x_pos + bar_width*(len(combos)/2 - 0.5))
            ax.set_xticklabels(new_oc_list, rotation=45, ha='right')
            if ylim is not None:
                ax.set_ylim(ylim)

        ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()

        os.makedirs(plot_dir, exist_ok=True)
        model_name = main_dir.strip("/").split("/")[-1]
        plot_name = f"{model_name}_{data_type}_categ-diff_{damage_type}"
        for layer in damage_layers:
            plot_name += f"_{layer}"
        for al in activations_layers:
            plot_name += f"_{al}"
        plot_name += f"_{len(damage_levels)}-levels_comparison"
        if scatter:
            plot_name += "_with-points"
        if percentage:
            plot_name += "_percent"
        plot_path = os.path.join(plot_dir, plot_name)

        if verbose == 1:
            save_plot = input(f"Save plot under {plot_path}? Y/N: ")
            if save_plot.capitalize() == "Y":
                plt.savefig(plot_path, dpi=500)
            plt.show()
        elif verbose == 0:
            plt.savefig(plot_path, dpi=500)
            plt.show()
        else:
            raise ValueError(f"{verbose} is not valid. Use 0 or 1.")


def aggregate_permutations(
    main_dir,
    output_dir,
    pkl_ext=".pkl"
):
    """
    For each subdirectory in `main_dir`, this script:
      1) Looks for all files ending with `pkl_ext` (default: ".pkl").
      2) Reads each file as a dict of categories -> metrics -> value
      3) Aggregates (mean, std) across all files in that subdir
      4) Saves a single Pickle named after subdir to output_dir with the aggregate stats.

    Example subdirectory structure:
      main_dir/
        damaged_0.01/
          file1.pkl
          file2.pkl
          ...
        damaged_0.02/
          ...
    """

    # Loop over everything in main_dir
    for subdir_name in os.listdir(main_dir):
        
        subdir_path = os.path.join(main_dir, subdir_name)
        if not os.path.isdir(subdir_path):
            continue  # skip if it's not a directory

        # We'll store: aggregated_data[category][metric] = list of float values
        aggregated_data = {}

        # Look for .pkl files in the subdir
        pkl_files = [
            f for f in os.listdir(subdir_path)
            if f.lower().endswith(pkl_ext)
        ]
        if not pkl_files:
            # no pkl files found in this subdir
            continue

        for yf in pkl_files:
            pkl_path = os.path.join(subdir_path, yf)
            with open(pkl_path, "rb") as f:
                content = pickle.load(f)  # a dict like {"animal": {"avg_between": ...}}

            # Walk through the categories/metrics in this file and store values
            for category_name, metrics_dict in content.items():
                if category_name not in aggregated_data:
                    aggregated_data[category_name] = {}
                # metrics_dict = {"avg_between": val, "avg_within": val, "observed_difference": val, ...}
                for metric_name, value in metrics_dict.items():
                    if metric_name not in aggregated_data[category_name]:
                        aggregated_data[category_name][metric_name] = []
                    aggregated_data[category_name][metric_name].append(value)

        # Compute mean & std for each category/metric
        
        results_dict = {}
        for category_name, metric_dict in aggregated_data.items():
            results_dict[category_name] = {}
            for metric_name, values_list in metric_dict.items():
                arr = np.array(values_list, dtype=float)
                mean_val = float(np.mean(arr))
                std_val  = float(np.std(arr))
                results_dict[category_name][metric_name] = {
                    "mean": mean_val,
                    "std":  std_val
                }
            
        # 5) Save the aggregated stats to a single Pickle in the same subdirectory

        # Define output_filename as numeric part from 
        _, damage_value = extract_string_numeric_parts(subdir_name)
        output_filename = f"aggr_stats_{damage_value}.pkl"
        # Create output directory and save file
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, "wb") as f:
            pickle.dump(results_dict, f)

        print(f"Aggregated stats saved -> {output_path}")


def pair_corr_scatter_subplots(
    layers,
    damage_levels,
    damage_type,
    image1,              # e.g. "animal1" (exclude ".jpg")
    image2,              # e.g. "animal2" (exclude ".jpg")
    n_permutations=5,
    main_dir="data/haupt_stim_activ/damaged/cornet_rt/",
    plot_dir="plots/",
    use_log_scale=False,  # if True, x/y axes are log-scaled and we do log-log fits
    verbose=0,  # 0 or 1
    lower_limit=0
):
    """
    Creates a grid of subplots with rows = layers, cols = damage levels.
    In each subplot, we overlay up to n_permutations scatter plots for the two
    images, plus (optionally) a log-log fit if use_log_scale=True.

    1) For each (layer, damage_level):
       - Navigate to {main_dir}/{damage_type}/{layer}/activations/damaged_{damage_level}/
       - Load up to n_permutations .pkl files.
       - For each pkl file, get the row for image1.jpg, image2.jpg, scatter them.
       - Optionally do a log-space fit if use_log_scale=True (i.e., fit y = a * x^m).
    2) Return one figure with len(layers)*len(damage_levels) subplots in a grid.
    3) Save the figure (or prompt) according to verbose.
    """

    # Make sure the plot directory exists
    os.makedirs(plot_dir, exist_ok=True)

    # Create the figure grid
    nrows = len(layers)
    ncols = len(damage_levels)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 4 * nrows),
        squeeze=False  # ensures axes is a 2D array
    )

    # If you want to preserve the order of damage_levels as given, do so:
    # Otherwise, you could sort them if they're numeric. We'll keep them as is.
    # same for layers

    # Iterate over each row=layer, col=damage_level
    for row_idx, layer in enumerate(layers):
        layer_activ_path = os.path.join(main_dir, damage_type, layer, "activations")
        if not os.path.isdir(layer_activ_path):
            print(f"[WARN] {layer_activ_path} does not exist; skipping entire row {layer}")
            # Optionally turn off that entire row of subplots
            for col_idx in range(ncols):
                ax = axes[row_idx, col_idx]
                ax.set_title(f"{layer} not found")
                ax.axis("off")
            continue

        for col_idx, dmg_level in enumerate(damage_levels):
            ax = axes[row_idx, col_idx]
            """ax.set_xlim(left=lower_limit)
            ax.set_ylim(bottom=lower_limit)"""
            subdir_name = f"damaged_{dmg_level}"
            subdir_path = os.path.join(layer_activ_path, subdir_name)

            if not os.path.isdir(subdir_path):
                # Subdir doesn't exist; skip
                ax.set_title(f"Missing {layer}, level={dmg_level}")
                ax.axis("off")
                continue

            # Gather .pkl files
            pkl_files = [f for f in os.listdir(subdir_path) if f.lower().endswith(".pkl")]
            pkl_files.sort()

            # Colors for permutations
            cmap = plt.get_cmap("tab10")
            color_cycle = [cmap(k % 10) for k in range(n_permutations)]

            # We'll plot each of the first n_permutations
            for j, pkl_file in enumerate(pkl_files[:n_permutations]):
                pkl_path = os.path.join(subdir_path, pkl_file)
                df = safe_load_pickle(pkl_path)

                rowname1 = f"{image1}.jpg"
                rowname2 = f"{image2}.jpg"

                if rowname1 not in df.index or rowname2 not in df.index:
                    print(f"[WARN] {rowname1} or {rowname2} not in {pkl_path}, skipping this file.")
                    continue

                # Filter numeric columns
                numeric_cols = [c for c in df.columns if str(c).isdigit()]
                act1 = df.loc[rowname1, numeric_cols].to_numpy(dtype=float)
                act2 = df.loc[rowname2, numeric_cols].to_numpy(dtype=float)

                # Scatter
                color = color_cycle[j]
                corr = np.corrcoef(act1, act2)[0, 1]
                label_str = f"{pkl_file}: r={corr:.2f}"
       

                # Create a scatter plot with a label
                ax.scatter(
                    act1, act2,
                    alpha=0.5,
                    color=color,
                    s=10,
                    label=label_str)  # This is crucial for legend


                # Optionally fit a line in log space if use_log_scale
                if use_log_scale:
                    # Exclude any non-positive points
                    valid_mask = (act1 > 0) & (act2 > 0)
                    if np.count_nonzero(valid_mask) > 1:
                        log_x = np.log(act1[valid_mask])
                        log_y = np.log(act2[valid_mask])
                        slope, intercept = np.polyfit(log_x, log_y, 1)
                        # slope ~ exponent, intercept ~ ln(constant)

                        # Create a range in log domain
                        log_x_fit = np.linspace(log_x.min(), log_x.max(), 50)
                        log_y_fit = slope * log_x_fit + intercept

                        x_fit = np.exp(log_x_fit)
                        y_fit = np.exp(log_y_fit)

                        ax.plot(x_fit, y_fit, color=color, linewidth=1.2, alpha=0.9)
                else:
                    slope, intercept = np.polyfit(act1, act2, 1)
                    x_fit = np.linspace(act1.min(), act1.max(), 50)
                    y_fit = slope * x_fit + intercept
                    ax.plot(x_fit, y_fit, color=color)
                # Optionally, you might also compute correlation in log space or linear space
                # but we won't show that here unless you specifically need it.

            # If log scale, set it
            if use_log_scale:
                ax.set_xscale("log")
                ax.set_yscale("log")

            ax.grid(True)

            # Subplot title
            ax.set_title(f"{layer}, dmg={dmg_level}")
            # Optionally set labels only if top-left or some pattern
            if row_idx == (nrows - 1):
                ax.set_xlabel(f"{image1} activations")
            if col_idx == 0:
                ax.set_ylabel(f"{image2} activations")
            ax.legend()


    fig.suptitle(f"Scatter correlation grid: {image1} vs {image2}")
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    # Build a filename
    model_name = main_dir.strip("/").split("/")[-1]
    plot_name = (
        f"{model_name}_scatter_grid_{damage_type}_"
        f"{image1}_vs_{image2}_permut{n_permutations}"
    )
    if use_log_scale:
        plot_name += "_log"
    plot_name += ".png"

    save_path = os.path.join(plot_dir, plot_name)

    # Save or show
    if verbose == 1:
        save_plot = input(f"Save figure under {save_path}? (Y/N): ")
        if save_plot.strip().lower() == "y":
            plt.savefig(save_path, dpi=300)
        plt.show()
    elif verbose == 0:
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        raise ValueError(f"Invalid verbose value: {verbose}. Use 0 or 1.")


def damage_type_lineplot(
    layer,
    damage_types,
    main_dir="data/haupt_stim_activ/damaged/cornet_rt/",
    categories=["animal", "face", "object", "place", "total"],
    metric="observed_difference",
    subdir_regex=r"damaged_([\d\.]+)$",
    plot_dir="plots/",
    common_ylim=None,  # tuple or None
    verbose=0  # 0 or 1
):
    """
    1) Aggregates data for each damage_type & category.
    2) Plots one subplot *per* damage type.
    3) If exactly 2 damage types, plots a second figure with a dual x-axis
       (twinned axes) and a shared y-axis. 
       - Each axis is forced to start at 0 on the left corner.
       - The top and bottom axes have different colors and different param scales.
    4) If common_ylim is provided, it's applied to all subplots for consistent comparison.
    """

    # -------------
    # STEP 1: AGGREGATE DATA
    # -------------
    data = {}  # data[(damage_type, category)] -> { param_value : (mean, std) }

    for damage_type in damage_types:
        layer_path = os.path.join(main_dir, damage_type, layer, "selectivity")
        output_path = os.path.join(main_dir, damage_type, layer, "avg_selectivity")
        os.makedirs(output_path, exist_ok=True)

        if not os.path.isdir(layer_path):
            print(f"Warning: {layer_path} not found. Skipping {damage_type}.")
            continue

        # Initialize data structures
        for cat in categories:
            data[(damage_type, cat)] = {}

        # Check subdirectories
        for subdir_name in os.listdir(layer_path):
            subdir_path = os.path.join(layer_path, subdir_name)
            if not os.path.isdir(subdir_path):
                continue

            match = re.search(subdir_regex, subdir_name)
            if not match:
                continue  # Not a "damaged_xxx" folder

            fraction_raw = float(match.group(1))
            fraction_rounded = round(fraction_raw, 3)

            fraction_file_name = f"avg_selectivity_{fraction_rounded}.pkl"
            fraction_file_path = os.path.join(output_path, fraction_file_name)

            # If we haven't aggregated this fraction yet, do so
            if not os.path.exists(fraction_file_path):
                aggregated_data = {}  # aggregated_data[cat][metric] = []

                for fname in os.listdir(subdir_path):
                    if fname.lower().endswith(".pkl"):
                        pkl_path = os.path.join(subdir_path, fname)
                        with open(pkl_path, "rb") as f:
                            content = pickle.load(f)
                        if not isinstance(content, dict):
                            continue
                        for cat_name, metrics_dict in content.items():
                            if not isinstance(metrics_dict, dict):
                                continue
                            if cat_name not in aggregated_data:
                                aggregated_data[cat_name] = {}
                            for metric_name, val in metrics_dict.items():
                                aggregated_data[cat_name].setdefault(metric_name, [])
                                aggregated_data[cat_name][metric_name].append(val)

                # Compute mean & std
                stats_dict = {}
                for cat_name, metrics_map in aggregated_data.items():
                    stats_dict[cat_name] = {}
                    for metric_name, vals_list in metrics_map.items():
                        if len(vals_list) == 0:
                            continue
                        arr = np.array(vals_list, dtype=float)
                        mean_val = float(np.mean(arr))
                        std_val = float(np.std(arr))
                        stats_dict[cat_name][metric_name] = {
                            "mean": mean_val,
                            "std": std_val
                        }

                # Save aggregated stats
                with open(fraction_file_path, "wb") as f:
                    pickle.dump(stats_dict, f)

            # Load stats
            aggregated_content = safe_load_pickle(fraction_file_path)
            if not isinstance(aggregated_content, dict):
                continue

            for cat in categories:
                if cat in aggregated_content:
                    if (isinstance(aggregated_content[cat], dict) and 
                        metric in aggregated_content[cat]):
                        mean_val = aggregated_content[cat][metric]["mean"]
                        std_val = aggregated_content[cat][metric]["std"]
                        data[(damage_type, cat)][fraction_rounded] = (mean_val, std_val)

    # -------------
    # STEP 2: PLOT ONE SUBPLOT PER DAMAGE TYPE
    # -------------
    num_damage_types = len(damage_types)
    if num_damage_types == 0:
        print("No valid damage types found. Exiting.")
        return

    fig, axes = plt.subplots(
        1, num_damage_types, 
        figsize=(5 * num_damage_types, 5),
        squeeze=False
    )
    axes = axes[0]  # Flatten single row

    for i, damage_type in enumerate(damage_types):
        ax = axes[i]
        any_data_plotted = False

        for cat in categories:
            fraction_dict = data.get((damage_type, cat), {})
            if len(fraction_dict) == 0:
                continue

            x_sorted = sorted(fraction_dict.keys())
            y_means = [fraction_dict[x][0] for x in x_sorted]
            y_stds  = [fraction_dict[x][1] for x in x_sorted]

            ax.errorbar(
                x_sorted, y_means, yerr=y_stds,
                fmt='-o', capsize=3,
                label=f"{cat}"
            )
            any_data_plotted = True

        if not any_data_plotted:
            ax.text(0.5, 0.5, "No data found", ha="center", va="center", transform=ax.transAxes)

        # Try a heuristic label:
        if "fraction" in damage_type.lower():
            ax.set_xlabel("Fraction of units set to 0")
        elif "noise" in damage_type.lower() or "std" in damage_type.lower():
            ax.set_xlabel("Std of Gaussian noise")
        else:
            ax.set_xlabel(f"{damage_type} param")

        ax.set_ylabel(metric)
        ax.set_title(f"Layer {layer} - {damage_type}")
        ax.legend()

        # Common ylim if requested
        if common_ylim is not None:
            ax.set_ylim(common_ylim)

        # Force x-axis to start at 0 if that makes sense
        # (We'll do a quick check if all x>0)
        all_x = [x for c in categories for x in data.get((damage_type, c), {}).keys()]
        if len(all_x) > 0:
            min_x = min(all_x)
            max_x = max(all_x)
            if min_x >= 0:  # If the param is always >= 0
                ax.set_xlim(left=0)
            ax.set_xlim(right=max_x)

    fig.tight_layout()

    # Save the per-damage-type subplots figure
    os.makedirs(plot_dir, exist_ok=True)
    model_name = main_dir.split("/")[-2]  # e.g. "cornet_rt"
    plot_name = f"{model_name}_lineplot_{layer}_SUBPLOTS"
    for damage_type in damage_types:
        plot_name += f"_{damage_type}"
    for category in categories:
        plot_name += f"_{category[0]}"
    plot_name += f"_{metric}.png"
    save_path = os.path.join(plot_dir, plot_name)

    if verbose == 1:
        save_plot = input(f"Save subplot figure under {save_path}? Y/N: ")
        if save_plot.capitalize() == "Y":
            plt.savefig(save_path, dpi=300)
        plt.show()
    else:
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

    # -------------
    # STEP 3: IF EXACTLY TWO DAMAGE TYPES, CREATE A "TWINS" PLOT
    # -------------

    if num_damage_types == 2:
        ##### DUAL AXIS SECTION START #####
        d1, d2 = damage_types

        # Prepare figure
        fig2, ax_bottom = plt.subplots(figsize=(7, 5))
        # Create top axis
        ax_top = ax_bottom.twiny()

        # For color differentiation
        color_bottom = "C0"
        color_top    = "C1"

        # -------------
        # BOTTOM AXIS: damage_type d1
        # -------------
        any_data_bottom = False
        all_x_bottom = []

        for cat in categories:
            fraction_dict = data.get((d1, cat), {})
            if len(fraction_dict) == 0:
                continue

            x_sorted = sorted(fraction_dict.keys())
            y_means = [fraction_dict[x][0] for x in x_sorted]
            y_stds  = [fraction_dict[x][1] for x in x_sorted]
            line_bottom = ax_bottom.errorbar(
                x_sorted, y_means, yerr=y_stds,
                fmt='-o', capsize=3,
                color=color_bottom,
                label=f"{d1}:{cat}"
            )
            any_data_bottom = True
            all_x_bottom.extend(x_sorted)

        # If bottom data is >= 0, force x to start at 0
        if any_data_bottom and len(all_x_bottom) > 0:
            min_x1, max_x1 = min(all_x_bottom), max(all_x_bottom)
            if min_x1 >= 0:
                ax_bottom.set_xlim(left=0)
            ax_bottom.set_xlim(right=max_x1)

        # Label bottom axis
        ax_bottom.set_xlabel(
            f"{d1} parameter", 
            color=color_bottom
        )
        ax_bottom.tick_params(axis='x', labelcolor=color_bottom)
        ax_bottom.spines["bottom"].set_edgecolor(color_bottom)
        # The y-axis is shared
        ax_bottom.set_ylabel(metric)

        # -------------
        # TOP AXIS: damage_type d2
        # -------------
        any_data_top = False
        all_x_top = []

        for cat in categories:
            fraction_dict = data.get((d2, cat), {})
            if len(fraction_dict) == 0:
                continue

            x_sorted = sorted(fraction_dict.keys())
            y_means = [fraction_dict[x][0] for x in x_sorted]
            y_stds  = [fraction_dict[x][1] for x in x_sorted]
            line_top = ax_top.errorbar(
                x_sorted, y_means, yerr=y_stds,
                fmt='-s', capsize=3,
                color=color_top,
                label=f"{d2}:{cat}"
            )
            any_data_top = True
            all_x_top.extend(x_sorted)

        # If top data is >= 0, force x to start at 0
        if any_data_top and len(all_x_top) > 0:
            min_x2, max_x2 = min(all_x_top), max(all_x_top)
            if min_x2 >= 0:
                ax_top.set_xlim(left=0)
            ax_top.set_xlim(right=max_x2)

        # Label top axis
        ax_top.set_xlabel(
            f"{d2} parameter", 
            color=color_top
        )
        ax_top.tick_params(axis='x', labelcolor=color_top)
        ax_top.spines["top"].set_edgecolor(color_top)

        # Optionally set common y-limits
        if common_ylim is not None:
            ax_bottom.set_ylim(common_ylim)
            ax_top.set_ylim(common_ylim)

        # Combine legends
        lines_bottom, labels_bottom = ax_bottom.get_legend_handles_labels()
        lines_top, labels_top = ax_top.get_legend_handles_labels()
        ax_bottom.legend(
            lines_bottom + lines_top, 
            labels_bottom + labels_top,
            loc="best"
        )

        ax_bottom.set_title(f"Twinned Axes for Layer {layer}")

        fig2.tight_layout()

        # Save
        plot_name_twinned = f"{model_name}_lineplot_{layer}_TWINS"
        for damage_type in damage_types:
            plot_name_twinned += f"_{damage_type}"
        for category in categories:
            plot_name_twinned += f"_{category[0]}"
        plot_name_twinned += f"_{metric}.png"

        save_path_twinned = os.path.join(plot_dir, plot_name_twinned)

        if verbose == 1:
            save_plot = input(f"Save twinned figure under {save_path_twinned}? Y/N: ")
            if save_plot.capitalize() == "Y":
                plt.savefig(save_path_twinned, dpi=300)
            plt.show()
        else:
            plt.savefig(save_path_twinned, dpi=300)
            plt.close(fig2)
    else:
        print("Skipping twinned-axes plot, because we have != 2 damage types.")


def normalize_module_name(name):
    """
    Normalize a module name by removing any occurrence of the token '_modules'.
    For example, 'module._modules.IT._modules.output' becomes 'module.IT.output'.
    """
    tokens = name.split(".")
    tokens = [t for t in tokens if t != "_modules"]
    return ".".join(tokens)

def build_hierarchical_index(module, prefix="", current_index=None):
    """
    Recursively traverse the model to create a mapping from normalized full module names 
    to their hierarchical index. The index is a list of integers indicating the position 
    of the module at each level of the hierarchy.
    
    For example, a module with normalized full name "layer1.layer1-2.conv1" might get an index [0, 1, 0].
    """
    if current_index is None:
        current_index = []
    mapping = {}
    norm_prefix = normalize_module_name(prefix) if prefix else ""
    if norm_prefix:
        mapping[norm_prefix] = current_index
    for idx, (name, child) in enumerate(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        norm_full_name = normalize_module_name(full_name)
        child_index = current_index + [idx]
        mapping[norm_full_name] = child_index
        mapping.update(build_hierarchical_index(child, full_name, child_index))
    return mapping

def get_final_layers_to_hook(model, activation_layers_to_save, layer_paths_to_damage):
    """
    Determines which activation layers (given by their full module names) should be hooked,
    using a hierarchical index built from the model's architecture.
    
    The names (both from the model and the provided layer names) are first normalized to remove
    extra tokens (e.g. '_modules'). Then, for each activation layer, if its hierarchical index 
    (as a tuple) is lexicographically greater than that of any damage layer, the activation layer 
    is considered to be downstream of the damage and is included.
    
    For example, if a damage layer has index [0, 1]:
      - An activation layer with index [0, 1, 3] qualifies.
      - An activation layer with index [1, 0, 3] qualifies.
      - An activation layer with index [0, 0, 3] does not qualify.
      
    Parameters:
      model: A PyTorch model object.
      activation_layers_to_save: List of full module names (strings) where activations should be saved.
      layer_paths_to_damage: List of full module names (strings) where damage is applied.
      
    Returns:
      final_layers_to_hook: A list of activation layer names (as originally provided) that should be hooked.
    """
    # Build a mapping from normalized module names to hierarchical indices.
    hierarchy = build_hierarchical_index(model)
    
    # Normalize the provided layer names.
    norm_activation_layers = [normalize_module_name(name) for name in activation_layers_to_save]
    norm_damage_layers = [normalize_module_name(name) for name in layer_paths_to_damage]
    
    final_layers = []
    for orig_act, norm_act in zip(activation_layers_to_save, norm_activation_layers):
        act_index = hierarchy.get(norm_act, None)
        if act_index is None:
            # Could not find the activation layer in the model.
            continue
        # Check against each damage layer.
        for orig_dmg, norm_dmg in zip(layer_paths_to_damage, norm_damage_layers):
            dmg_index = hierarchy.get(norm_dmg, None)
            if dmg_index is None:
                # Could not find the damage layer in the model.
                continue
            # If the activation layer's index is lexicographically greater than the damage layer's index,
            # then it is considered downstream of the damage.
            if tuple(act_index) > tuple(dmg_index):
                final_layers.append(orig_act)
                break  # No need to check further if it qualifies for one damage layer.
    return final_layers


def run_svm_split(train_indices1, test_indices1,
                                      train_indices2, test_indices2,
                                      activations1, activations2):
    """
    Train and test SVM on the specific split using pre-scaled activations.
    Returns the test accuracy.
    """
    X_train = np.concatenate((activations1[np.array(train_indices1)],
                              activations2[np.array(train_indices2)]))
    y_train = np.concatenate((np.zeros(len(train_indices1), dtype=np.int64),
                              np.ones(len(train_indices2), dtype=np.int64)))

    X_test = np.concatenate((activations1[np.array(test_indices1)],
                             activations2[np.array(test_indices2)]))
    y_test = np.concatenate((np.zeros(len(test_indices1), dtype=np.int64),
                             np.ones(len(test_indices2), dtype=np.int64)))

    # Train SVM with a linear kernel
    clf = SVC(kernel="linear", random_state=42)
    clf.fit(X_train, y_train)

    preds_test = clf.predict(X_test)
    accuracy_test = np.mean(preds_test == y_test)
    return accuracy_test


def svm_process_split(train_idx1, train_idx2,
                  test_combinations1, test_combinations2,
                  act1_scaled, act2_scaled):
    """
    For a given pair of training indices, find the corresponding test indices
    and run the SVM, returning the test accuracy.
    """
    test_idx1 = test_combinations1[train_idx1]
    test_idx2 = test_combinations2[train_idx2]
    return run_svm_split(train_idx1, test_idx1,
                                             train_idx2, test_idx2,
                                             act1_scaled, act2_scaled)


def svm_process_file(pkl_file, training_samples=15, clip_val=1e6, max_permutations=None):
    """
    Load a .pkl file of activations (4 categories × 16 examples each = 64 rows),
    run SVM classification for all category pairs. 
    - If max_permutations < 256, we randomly sample that many permutations.
    - We prebuild arrays for each permutation to reduce overhead.

    Returns:
        A DataFrame with one row per SVM permutation, columns = each category pair,
        or None if the file is invalid (too few rows).
    """
    df = pd.read_pickle(pkl_file)
    df = df.drop("numeric_index", axis=1, errors="ignore")
    df.columns = df.columns.astype(str)

    if len(df) < 64:
        return None

    # Extract the four categories (16 rows each)
    cat1 = df.iloc[0:16].to_numpy()
    cat2 = df.iloc[16:32].to_numpy()
    cat3 = df.iloc[32:48].to_numpy()
    cat4 = df.iloc[48:64].to_numpy()

    categories = {
        "animal": cat1,
        "face":   cat2,
        "object":  cat3, 
        "place": cat4 
    }

    # Clip raw activations
    for key in categories:
        categories[key] = np.clip(categories[key], -clip_val, clip_val)

    pairs = list(combinations(categories.keys(), 2))
    pair_to_accuracies = {f"{p[0]}_vs_{p[1]}": [] for p in pairs}

    for (name1, name2) in pairs:
        act1 = categories[name1]
        act2 = categories[name2]

        # Removed Scaler fitting for now
        """# Fit RobustScaler once for the combined data of this pair
        combined = np.concatenate((act1, act2), axis=0)
        scaler = StandardScaler()
        scaler.fit(combined)
        min_epsilon = 1e-6
        scaler.scale_[scaler.scale_ < min_epsilon] = min_epsilon

        act1_scaled = scaler.transform(act1)
        act2_scaled = scaler.transform(act2)"""

        act1_scaled = np.clip(
            np.nan_to_num(act1, nan=0.0, posinf=clip_val, neginf=-clip_val),
            -clip_val, clip_val
        )
        act2_scaled = np.clip(
            np.nan_to_num(act2, nan=0.0, posinf=clip_val, neginf=-clip_val),
            -clip_val, clip_val
        )

        # Generate all train/test splits
        indices = np.arange(16)
        train_combos_1 = list(combinations(indices, training_samples))
        train_combos_2 = list(combinations(indices, training_samples))
        all_splits = list(product(train_combos_1, train_combos_2))

        if max_permutations is not None and max_permutations < len(all_splits):
            sampled_splits = random.sample(all_splits, max_permutations)
        else:
            sampled_splits = all_splits

        # Precompute test indices so we don't do it repeatedly
        test_combos_1 = {tc: tuple(np.setdiff1d(indices, tc)) for tc in train_combos_1}
        test_combos_2 = {tc: tuple(np.setdiff1d(indices, tc)) for tc in train_combos_2}

        # *** Prebuild all arrays into a single list ***
        all_data = []
        for (train_idx1, train_idx2) in sampled_splits:
            test_idx1 = test_combos_1[train_idx1]
            test_idx2 = test_combos_2[train_idx2]

            X_train = np.concatenate((act1_scaled[np.array(train_idx1)],
                                      act2_scaled[np.array(train_idx2)]))
            y_train = np.concatenate((np.zeros(len(train_idx1), dtype=np.int64),
                                      np.ones(len(train_idx2), dtype=np.int64)))

            X_test = np.concatenate((act1_scaled[np.array(test_idx1)],
                                     act2_scaled[np.array(test_idx2)]))
            y_test = np.concatenate((np.zeros(len(test_idx1), dtype=np.int64),
                                     np.ones(len(test_idx2), dtype=np.int64)))

            all_data.append((X_train, y_train, X_test, y_test))

        # Remove parallel processing for now
        """results = Parallel(n_jobs=1)(
            delayed(train_and_test_svm_arrays)(X_train, y_train, X_test, y_test)
            for (X_train, y_train, X_test, y_test) in all_data
        )"""
        results = (train_and_test_svm_arrays(X_train, y_train, X_test, y_test) for (X_train, y_train, X_test, y_test) in all_data)

        pair_key = f"{name1}_vs_{name2}"
        pair_to_accuracies[pair_key] = results

    # Build final DataFrame
    data_dict = {pair: pair_to_accuracies[pair] for pair in pair_to_accuracies}
    df_runs = pd.DataFrame(data_dict)
    return df_runs


def svm_process_directory(parent_dir, training_samples=15, allowed_subdirs=None, max_permutations=None):
    """
    Recursively walk through parent_dir. For any folder whose path
    includes "activations", we check the next directory after "activations"
    and only process it if it's in allowed_subdirs (if provided).

    - parent_dir: the root directory to start searching
    - training_samples: number of training samples per category
    - allowed_subdirs: optional list of subdirectory names to process
        e.g. ["V1", "IT"]. If None or empty, we process all subdirectories.

    We then gather all .pkl files in these valid subdirectories,
    run SVM permutations, and save results under a mirrored "svm_{training_samples}/"
    folder.
    """
    if allowed_subdirs is None:
        allowed_subdirs = []  # means "no filtering"

    # We'll collect all files that pass the subdir filter in this list
    pkl_file_paths = []

    for root, dirs, files in os.walk(parent_dir):
        # Check if "activations" is in the path
        if "activations" in root.split(os.sep):
            # Figure out the subdirectory immediately after "activations"
            parts = root.split(os.sep)
            try:
                idx = parts.index("activations")
            except ValueError:
                continue  # Shouldn't happen if "activations" is in path

            # subfolders after 'activations'
            subfolders_after_activations = parts[idx+1:]  # might be ["V1", "subsubdir"] if root is "parent/activations/V1/subsubdir"

            # If there's at least one subfolder after "activations",
            # we check whether subfolders_after_activations[0] is in allowed_subdirs.
            # If allowed_subdirs is empty, we skip the filter (process everything).
            if allowed_subdirs:
                if not subfolders_after_activations or subfolders_after_activations[0] not in allowed_subdirs:
                    # skip this directory
                    continue

            # If we reach here, it means we're either not filtering,
            # or the subdirectory is in allowed_subdirs.
            # Collect .pkl files
            for fname in files:
                if fname.lower().endswith(".pkl"):
                    pkl_file_paths.append((root, fname))

    # Now process each file with a progress bar
    from tqdm import tqdm
    for root, fname in tqdm(pkl_file_paths, desc="Processing PKL files", total=len(pkl_file_paths)):
        in_path = os.path.join(root, fname)
        # Mirror structure by replacing "activations" with "svm_{training_samples}"
        parts = root.split(os.sep)
        idx = parts.index("activations")
        activations_folder = os.path.join(*parts[:idx+1])
        rel_path = os.path.relpath(in_path, activations_folder)

        svm_dir = os.path.join(os.path.dirname(activations_folder), f"svm_{training_samples}")
        out_path = os.path.join(svm_dir, rel_path)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df_result = svm_process_file(in_path, training_samples=training_samples,max_permutations=max_permutations)
        if df_result is not None:
            df_result.to_pickle(out_path)


def train_and_test_svm_arrays(X_train, y_train, X_test, y_test):
    """
    Train and test a linear SVM given (X_train, y_train) and (X_test, y_test).
    Returns the test accuracy.
    """
    clf = SVC(kernel="linear", random_state=42)
    clf.fit(X_train, y_train)
    preds_test = clf.predict(X_test)
    return np.mean(preds_test == y_test)


def evaluate_imagenet_perclass(model, loader, device, topk=(1, 5)):
    """
    Works with ImageNet, ImageFolder *and* torch.utils.data.Subset.
    """
    # ---------------------------------------------------------------
    if hasattr(loader.dataset, "classes"):                 # normal case
        cls_names = loader.dataset.classes
    elif hasattr(loader.dataset, "dataset") and hasattr(loader.dataset.dataset, "classes"):
        cls_names = loader.dataset.dataset.classes        # Subset -> underlying
    else:
        raise AttributeError("Dataset has no 'classes' attribute.")
    n_cls = len(cls_names)
    # ---------------------------------------------------------------

    total = torch.zeros(n_cls, dtype=torch.long)
    corr1 = torch.zeros(n_cls, dtype=torch.long)
    corr5 = torch.zeros(n_cls, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            _, pred = logits.topk(max(topk), 1, True, True)
            pred = pred.t()
            matches = pred.eq(y.view(1, -1).expand_as(pred))

            for b in range(x.size(0)):
                cls = y[b].item()
                total[cls] += 1
                if matches[0, b]:
                    corr1[cls] += 1
                if matches[:5, b].any():
                    corr5[cls] += 1

    top1 = corr1.sum().item() / total.sum().item()
    top5 = corr5.sum().item() / total.sum().item()

    per_cls = {i: {"top1": corr1[i].item() / total[i].item(),
                   "top5": corr5[i].item() / total[i].item()}
               for i in range(n_cls)}
    return {"overall": {"top1": top1, "top5": top5},
            "classes": per_cls}


def run_damage_imagenet(
    model_info,
    pretrained,
    fraction_to_mask_params,
    noise_levels_params,
    layer_paths_to_damage,
    apply_to_all_layers,
    manipulation_method,          # "connections" | "noise"
    mc_permutations,
    layer_name,
    imagenet_root,
    only_conv,
    include_bias,
    masking_level="connections",
    batch_size=64,                # bigger batch, fits in 64 GB RAM
    num_workers=8,
    subset_pct=10,                # NEW – 10 % by default
    run_suffix=""
):
    # 1) damage-level list -----------------------------------------------
    if manipulation_method == "connections":
        damage_levels = generate_params_list(fraction_to_mask_params)
        noise_dict = None
    elif manipulation_method == "noise":
        damage_levels = generate_params_list(noise_levels_params)
        tmp_model, _ = load_model(model_info, pretrained, layer_name, layer_path=None)
        noise_dict = get_params_sd(tmp_model)
        del tmp_model
    else:
        raise ValueError("manipulation_method must be 'connections' or 'noise'.")

    # 2) naming pieces ----------------------------------------------------
    dir_tag = ("units" if manipulation_method == "connections" and masking_level == "units"
               else "connections" if manipulation_method == "connections"
               else "noise")
    time_steps = str(model_info.get("time_steps", ""))
    run_suffix = (("_c" if only_conv else "_all") + ("+b" if include_bias else "")) + run_suffix

    # 3) base dataset & transform ----------------------------------------
    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])
    full_dataset = get_val_dataset(imagenet_root, trans)
    n_full = len(full_dataset)

    # 4) outer loop -------------------------------------------------------
    total_iters = len(damage_levels) * mc_permutations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model, _ = load_model(model_info, pretrained, layer_name, layer_path=None)

    with tqdm(total=total_iters, desc="ImageNet damage eval") as pbar:
        for dmg in damage_levels:
            for perm in range(mc_permutations):
                # --- fresh balanced 10 % subset each permutation ----------
                subset = balanced_subset(full_dataset, subset_pct, seed=perm)
                loader = torch.utils.data.DataLoader(
                    subset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=torch.cuda.is_available()
                )

                # --- copy model & apply damage ---------------------------
                model = copy.deepcopy(base_model).to(device)
                if manipulation_method == "connections":
                    apply_masking(model, fraction_to_mask=dmg,
                                  layer_paths=layer_paths_to_damage,
                                  apply_to_all_layers=apply_to_all_layers,
                                  masking_level=masking_level,
                                  only_conv=only_conv,
                                  include_bias=include_bias)
                else:
                    apply_noise(model, noise_level=dmg, noise_dict=noise_dict,
                                layer_paths=layer_paths_to_damage,
                                apply_to_all_layers=apply_to_all_layers,
                                only_conv=only_conv,
                                include_bias=include_bias)

                # --- one evaluation -------------------------------------
                results = evaluate_imagenet_perclass(model, loader, device)

                # --- save -----------------------------------------------
                out_dir = (f"data/haupt_stim_activ/damaged/"
                           f"{model_info['name']}{time_steps}{run_suffix}/"
                           f"{dir_tag}/{layer_name}/imagenet/damaged_{round(dmg,3)}")
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, f"{perm}.pkl"), "wb") as f:
                    pickle.dump(results, f)

                pbar.update(1)
    print("ImageNet evaluation finished.")


def get_val_dataset(imagenet_root: str, transform):
    """
    Return a torchvision dataset for the ImageNet-1k validation set.

    • Tries `torchvision.datasets.ImageNet(root, split="val")`
      – works if the official metadata files are present.
    • Otherwise falls back to `torchvision.datasets.ImageFolder`,
      accepting either
          imagenet_root/val/<wnid>/*.JPEG   or
          imagenet_root/<wnid>/*.JPEG
    """
    try:
        return torchvision.datasets.ImageNet(imagenet_root,
                                             split="val",
                                             transform=transform)
    except (FileNotFoundError, RuntimeError):
        # look for a nested “val” dir first
        candidate = (os.path.join(imagenet_root, "val")
                     if os.path.isdir(os.path.join(imagenet_root, "val"))
                     else imagenet_root)
        return torchvision.datasets.ImageFolder(candidate, transform=transform)


def balanced_subset(dataset, pct, seed):
    """
    Return a torch.utils.data.Subset that contains `pct` percent of the
    dataset, keeping *≈ equal* counts per class.

    • pct must be in (0,100].
    • seed makes the draw reproducible (one seed per permutation is fine).
    """
    if not (0 < pct <= 100):
        raise ValueError("pct should be between 0 and 100.")
    g = torch.Generator().manual_seed(seed)

    cls_to_indices = {}
    if hasattr(dataset, "targets"):            # ImageFolder / ImageNet
        for idx, cls in enumerate(dataset.targets):
            cls_to_indices.setdefault(cls, []).append(idx)
    else:                                      # generic case
        for idx, (_, cls) in enumerate(dataset.samples):
            cls_to_indices.setdefault(cls, []).append(idx)

    per_cls = max(1, int(len(dataset) * pct / 100 / len(cls_to_indices)))
    subset_idx = []
    for lst in cls_to_indices.values():
        idx_tensor = torch.tensor(lst)
        perm = idx_tensor[torch.randperm(len(lst), generator=g)]
        subset_idx.extend(perm[:per_cls].tolist())
    return torch.utils.data.Subset(dataset, subset_idx)


DEP_OPTIONS = {"selectivity", "svm", "imagenet"}

# --------------------------------------------------------------------
def build_long_df(root_dir: str,
                  model_variants: list[str] | list[Mapping],
                  dep_cfg: dict,
                  merge_bias_into_base: bool = True,
                  use_bias_factor: bool = True) -> pd.DataFrame:
    """
    Return one tidy DataFrame that pools results from several model folders.

    NEW Each entry in *model_variants* may be either
        • a plain string  →  all three damage folders are accepted, OR
        • a dict {name: str, take: list[str]}  → only listed damage_types read.

    Example
    -------
    model_variants:
      - name: cornet_rt5_c
        take: [connections, noise]
      - name: cornet_rt5_c+b
        take: [units]
    """
    kind = dep_cfg["kind"]
    if kind not in DEP_OPTIONS:
        raise ValueError(f"dependent.kind must be one of {DEP_OPTIONS}")

    # -------------- helpers -----------------------------------------
    def _base_name(mv: str) -> str:
        return mv[:-2] if (merge_bias_into_base and mv.endswith("+b")) else mv

    def _walk_selectivity_dirs(mv_root: Path, allowed: set[str]):
        """Yield (dmg_type, dmg_layer, act_layer, damaged_folder Path)."""
        for dmg_type in ("units", "noise", "connections"):
            if allowed and dmg_type not in allowed:
                continue
            type_dir = mv_root / dmg_type
            if not type_dir.exists():
                continue
            for dmg_layer in type_dir.iterdir():
                sel_dir = dmg_layer / "selectivity"
                if not sel_dir.exists():
                    continue
                for act_layer in sel_dir.iterdir():
                    for damaged in act_layer.iterdir():
                        if damaged.name.startswith("damaged_"):
                            yield (dmg_type, dmg_layer.name,
                                   act_layer.name, damaged)

    def _gather_selectivity(mv_root: Path, mv_name: str,
                            bias_flag: int, allowed: set[str]):
        rows = []
        for dmg_type, dmg_layer, _, subdir in _walk_selectivity_dirs(mv_root, allowed):
            lvl = float(subdir.name.split("_")[-1])          # damaged_0.1 → 0.1
            for pkl in sorted(subdir.glob("*.pkl")):
                repl = int(pkl.stem)                         # 0.pkl → 0
                data = pd.read_pickle(pkl)                  # dict of categories
                for cat, metrics in data.items():
                    rows.append(dict(
                        value        = metrics[dep_cfg["metric"]],
                        model_variant= mv_name,
                        include_bias = bias_flag,
                        damage_type  = dmg_type,
                        damage_layer = dmg_layer,
                        damage_level = lvl,
                        category     = cat,
                        replicate    = repl,
                    ))
        return rows

    # -------------- root loop ---------------------------------------
    long_rows = []
    for entry in model_variants:
        if isinstance(entry, str):
            mv_name_on_disk = entry
            allowed_types   = set()          # empty → accept all
        else:                                # dict with filtering
            mv_name_on_disk = entry["name"]
            allowed_types   = set(entry["take"])
        mv_root   = Path(root_dir) / mv_name_on_disk
        mv_logical= _base_name(mv_name_on_disk)
        bias_flag = int(mv_name_on_disk.endswith("+b"))

        if kind == "selectivity":
            long_rows += _gather_selectivity(
                mv_root, mv_logical, bias_flag, allowed_types
            )
        elif kind == "svm":
            long_rows += _gather_svm(      # ← your old helper
                mv_root, mv_logical, bias_flag,
                dep_cfg["metric"], dep_cfg.get("svm_train_samples", 15),
                allowed_types
            )
        else:                              # imagenet
            long_rows += _gather_imagenet( # ← your old helper
                mv_root, mv_logical, bias_flag,
                dep_cfg["metric"], allowed_types
            )

    df = pd.DataFrame(long_rows)

    if not use_bias_factor and "include_bias" in df.columns:
        df = df.drop(columns="include_bias")

    # z-scores (unchanged)
    df["value_z"] = df.groupby("model_variant")["value"].transform(
        lambda v: (v - v.mean()) / v.std(ddof=0)
    )
    df["damage_scaled"] = df.groupby(
        ["model_variant", "damage_type"]
    )["damage_level"].transform(
        lambda v: (v - v.mean()) / v.std(ddof=0)
    )
    return df


def aggregate_selectivity(
    df: pd.DataFrame,
    by: Sequence[str] = ("damage_type", "damage_layer", "damage_level", "category"),
    value_col: str = "observed_difference",
) -> pd.DataFrame:
    """Collapse replicate‑level data into mean ± SD."""
    out = (
        df.groupby(list(by))[value_col]
        .agg(mean_selectivity="mean", sd_selectivity="std", n="size")
        .reset_index()
    )
    return out


def fit_selectivity_glm(
    df: pd.DataFrame,
    formula: str = (
        "observed_difference ~ damage_level * damage_type * damage_layer * category"
    ),
):
    """Fit a (full‑factorial) OLS model and return the result object."""
    return smf.ols(formula, data=df).fit(cov_type='HC3')


def _iter_selectivity_pkls(root_dir: str):
    _DMG_DIR_RE = re.compile(r"damaged_([\d\.]+)$")
    """Yield information about every selectivity‑pkl found under *root_dir*."""
    for damage_type in sorted(os.listdir(root_dir)):
        type_dir = os.path.join(root_dir, damage_type)
        if not os.path.isdir(type_dir):
            continue

        for damage_layer in sorted(os.listdir(type_dir)):
            layer_dir = os.path.join(type_dir, damage_layer, "selectivity")
            if not os.path.isdir(layer_dir):
                continue

            for activation_layer in sorted(os.listdir(layer_dir)):
                act_dir = os.path.join(layer_dir, activation_layer)
                if not os.path.isdir(act_dir):
                    continue

                for dmg_sub in os.listdir(act_dir):
                    m = _DMG_DIR_RE.match(dmg_sub)
                    if not m:
                        continue
                    dmg_level = float(m.group(1))
                    subdir = os.path.join(act_dir, dmg_sub)

                    for fname in os.listdir(subdir):
                        if fname.endswith(".pkl"):
                            yield (
                                damage_type,
                                damage_layer,
                                activation_layer,
                                dmg_level,
                                os.path.join(subdir, fname),
                            )


def _gather_svm(mv_root: Path,
                mv_name: str,
                bias_flag: int,
                metric: str,
                train_samples: int,
                allowed: set[str]):
    """
    Collect SVM accuracy pickles and return a list[dict] of long rows.

    metric:
        "overall"  – mean across *all* pairwise columns (alias "score")
        "by_category" – mean of columns that involve each focal category
    """
    if metric == "score":          # backward-compat
        metric = "overall"
    if metric not in ("overall", "by_category"):
        raise ValueError("dependent.metric for SVM must be "
                         "'overall', 'score', or 'by_category'")

    rows, n_files = [], 0
    folder_name = f"svm_{train_samples}"

    for dmg_type in ("units", "noise", "connections"):
        if allowed and dmg_type not in allowed:
            continue
        type_dir = mv_root / dmg_type
        if not type_dir.exists():
            print(f"[skip] {type_dir} (missing)")
            continue

        for dmg_layer in type_dir.iterdir():
            svm_dir = dmg_layer / folder_name
            if not svm_dir.exists():
                print(f"[skip] {svm_dir} (missing)")
                continue

            for act_layer in svm_dir.iterdir():
                for damaged in act_layer.iterdir():
                    if not damaged.name.startswith("damaged_"):
                        continue
                    lvl = float(damaged.name.split("_")[-1])

                    for pkl in damaged.glob("*.pkl"):
                        n_files += 1
                        repl = int(pkl.stem)
                        try:
                            df = pd.read_pickle(pkl)
                        except Exception as exc:
                            print(f"[warn] could not read {pkl}: {exc}")
                            continue
                        if not isinstance(df, pd.DataFrame):
                            print(f"[warn] {pkl} is not a DataFrame; skipped")
                            continue
                        df = df.select_dtypes("number")
                        if df.empty:
                            print(f"[warn] {pkl} has no numeric columns; skipped")
                            continue

                        if metric == "overall":
                            val = float(df.mean(axis=1).mean())
                            rows.append(dict(
                                value        = val,
                                model_variant= mv_name,
                                include_bias = bias_flag,
                                damage_type  = dmg_type,
                                damage_layer = dmg_layer.name,
                                damage_level = lvl,
                                category     = "overall",
                                replicate    = repl,
                            ))
                        else:  # by_category
                            cats = sorted({p for col in df.columns.str.lower()
                                             for p in col.split("_vs_")})
                            for cat in cats:
                                mask = df.columns.str.contains(cat, case=False, regex=False)
                                if mask.any():
                                    val = float(df.loc[:, mask].mean(axis=1).mean())
                                    rows.append(dict(
                                        value        = val,
                                        model_variant= mv_name,
                                        include_bias = bias_flag,
                                        damage_type  = dmg_type,
                                        damage_layer = dmg_layer.name,
                                        damage_level = lvl,
                                        category     = cat,
                                        replicate    = repl,
                                    ))

    if n_files == 0:
        raise RuntimeError(f"No SVM pickle files found under {mv_root}")
    if not rows:
        raise RuntimeError("SVM pickles were found but produced no usable rows. "
                           "Check 'metric' or file contents.")
    return rows



def _gather_imagenet(mv_root: Path,
                     mv_name: str,
                     bias_flag: int,
                     metric: str,
                     allowed: set[str]):
    """
    Walk   …/<damage_type>/<damage_layer>/imagenet/damaged_xxx/*.pkl

    Each Pickle is a dict  {"overall":{"top1":…,"top5":…}, … }
    """
    if metric not in ("top1", "top5"):
        raise ValueError("dependent.metric for ImageNet must be 'top1' or 'top5'")

    rows = []
    for dmg_type in ("units", "noise", "connections"):
        if allowed and dmg_type not in allowed:
            continue
        type_dir = mv_root / dmg_type
        if not type_dir.exists():
            continue
        for dmg_layer in type_dir.iterdir():
            imagenet_dir = dmg_layer / "imagenet"
            if not imagenet_dir.exists():
                continue
            for damaged in imagenet_dir.iterdir():
                if not damaged.name.startswith("damaged_"):
                    continue
                lvl = float(damaged.name.split("_")[-1])
                for pkl in sorted(damaged.glob("*.pkl")):
                    repl = int(pkl.stem)
                    d = pd.read_pickle(pkl)
                    val = float(d["overall"][metric])
                    rows.append(dict(
                        value        = val,
                        model_variant= mv_name,
                        include_bias = bias_flag,
                        damage_type  = dmg_type,
                        damage_layer = dmg_layer.name,
                        damage_level = lvl,
                        category     = "overall",
                        replicate    = repl,
                    ))
    return rows


def get_top_unit_indices(
    selectivity_dir: str,
    category: str,
    layer_name: str,
    top_frac: float,
    fmap_shape: Tuple[int, int, int]
) -> List[int]:
    """
    1. Look up the per-unit selectivity file for `category` (try .pkl first, then .csv).
    2. Load it into a DataFrame.
    3. Filter to rows where df.layer_name == layer_name.
    4. Take the top `top_frac` fraction by 'scaled_activation'.
    5. Given fmap_shape = (C, H, W), convert each (c,y,x) into a flat index.
    """
    base = os.path.join(selectivity_dir, f"{category}_unit_selectivity_all_units")
    df = None

    # 1) Try pickle, then CSV
    pkl_path = base + ".pkl"
    csv_path = base + ".csv"
    if os.path.isfile(pkl_path):
        df = pd.read_pickle(pkl_path)
    elif os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"No selectivity file found for category '{category}'.\n"
            f"Expected one of:\n  {pkl_path}\n  {csv_path}"
        )

    # 2) Filter to the correct layer
    layer_rows = df[df["layer_name"] == layer_name]
    if layer_rows.empty:
        raise ValueError(
            f"No units found for layer '{layer_name}' in selectivity file for '{category}'."
        )

    # 3) Pick top k units
    k = max(1, int(len(layer_rows) * top_frac))
    top = layer_rows.nlargest(k, "scaled_activation")

    # 4) Convert (channel, y, x) → flat index
    C, H, W = fmap_shape
    indices: List[int] = []
    for _, r in top.iterrows():
        # unit_id looks like "layer:channel:y:x"
        _, c_str, y_str, x_str = r["unit_id"].split(":")
        c, y, x = int(c_str), int(y_str), int(x_str)
        flat_idx = c * (H * W) + y * W + x
        indices.append(flat_idx)

    return indices


def generate_category_selective_RDMs(
    activations_root: str, # root of damage type directory e.g., .../connections
    layer_name: str,
    top_frac: float,
    categories: Sequence[str] = ("faces","places","objects","animals"),
    damage_levels: Sequence[str] = None,
    selection_mode: str = "percentage",  # "percentage" or "percentile"
    selectivity_file: str = "unit_selectivity/all_layers_units_mannwhitneyu.pkl",
    damage_layer: str = "V1",
    activation_layer: str = "IT"
):
    """
    Build category-selective RDMs from per-image activation pickles, using flat unit indices.
    selectivity_file: path to the selectivity .pkl or .csv file with columns: layer,unit,mw_animals,...
    """
    from pathlib import Path

    selectivity_path = Path(selectivity_file)
    # Load selectivity table
    if selectivity_path.suffix == ".pkl":
        sel_df = pd.read_pickle(selectivity_path)
    elif selectivity_path.suffix == ".csv":
        sel_df = pd.read_csv(selectivity_path)
    else:
        raise ValueError("Selectivity file must be .pkl or .csv")

    # 1) For each category, select top unit indices for the given layer
    idxs_by_cat = {}
    for cat in categories:
        cat_key = cat if f"mw_{cat}" in sel_df.columns else f"{cat}s"  # handle plural/singular
        mw_col = f"mw_{cat_key}"
        layer_rows = sel_df[sel_df["layer"] == "module." + layer_name]
        if mw_col not in layer_rows.columns:
            raise ValueError(f"Column {mw_col} not found in selectivity file.")
        if selection_mode == "percentage":
            k = max(1, int(len(layer_rows) * top_frac))
            top = layer_rows.nlargest(k, mw_col)
        elif selection_mode == "percentile":
            cutoff = np.percentile(layer_rows[mw_col], top_frac)
            top = layer_rows[layer_rows[mw_col] >= cutoff]
        else:
            raise ValueError("selection_mode must be 'percentage' or 'percentile'")
        idxs = top["unit"].astype(int).values
        idxs_by_cat[cat] = np.array(idxs, dtype=int)

    # 2) determine damage levels if not provided
    activ_root = Path(activations_root) / damage_layer / "activations" / activation_layer
    if damage_levels is None:
        damage_levels = sorted(
            d for d in os.listdir(activ_root)
            if (activ_root/d).is_dir()
        )

    # 3) process each category × damage level
    root_out = Path(activations_root) / damage_layer / f"RDM_{top_frac:.2f}_{selection_mode}" / activation_layer
    for cat, idxs in idxs_by_cat.items():
        for dmg in damage_levels:
            in_dir  = activ_root / dmg
            out_dir = root_out / (cat.strip("s")+"_selective") / dmg
            out_dir.mkdir(parents=True, exist_ok=True)

            for pkl_fname in sorted(in_dir.glob("*.pkl")):
                # load activations [n_images, n_units]
                with open(pkl_fname, "rb") as f:
                    A = pickle.load(f)
                    image_names = list(A.index)
                    A= np.asarray(A)
                # restrict to top-selective units (flat indices)
                A_sub = A[:, idxs]                   # [n_images, n_top_units]
                # build RDM: Pearson corr across the rows
                R = np.corrcoef(A_sub)               # [n_images, n_images]
                # save
                with open(out_dir / pkl_fname.name, "wb") as f:
                    pickle.dump({'RDM': R, 'image_names': image_names}, f)
