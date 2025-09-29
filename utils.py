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
from typing import Sequence, Mapping, Tuple, List, Iterator, Any
import zarr
import numcodecs


def get_layer_from_path(model, path: str):
    """
    Resolve dotted module paths across models (VGG/ResNet/CORnet/etc).
    Tolerates:
      - 'features.0' (Sequential/ModuleList int indexing)
      - 'features._modules.0' (explicit modules dict)
      - attribute access (e.g., 'layer1.0.conv1', 'V4', 'IT')
    """
    # 1) Normalize away explicit _modules segments
    try:
        norm = normalize_module_name(path)
    except NameError:
        # fallback if normalize_module_name isn't defined above this function yet
        norm = path.replace("._modules.", ".").replace("._modules", ".").strip(".")

    steps = [s for s in norm.split(".") if s]  # remove empty tokens
    current = model

    for step in steps:
        # A) direct attribute?
        if hasattr(current, step):
            current = getattr(current, step)
            continue

        # B) integer index (Sequential/ModuleList)?
        if step.isdigit():
            idx = int(step)
            # try __getitem__(int) first (Sequential/ModuleList)
            try:
                current = current[idx]
                continue
            except (TypeError, IndexError, KeyError):
                pass
            # then try ModuleDict/_modules dict with string key
            if hasattr(current, "_modules"):
                moddict = current._modules
                if str(idx) in moddict:
                    current = moddict[str(idx)]
                    continue
            # finally: plain dict-like with string keys
            if isinstance(current, dict) and str(idx) in current:
                current = current[str(idx)]
                continue

        # C) plain dict-like with string key
        if isinstance(current, dict) and step in current:
            current = current[step]
            continue
        if hasattr(current, "_modules") and step in current._modules:
            current = current._modules[step]
            continue

        raise KeyError(f"Cannot resolve step '{step}' in path '{path}' (normalized='{norm}') at object: {type(current)}")
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


def is_cornet(model_info: dict) -> bool:
    name = (model_info.get("name") or "").lower()
    return name.startswith("cornet")


def short_module_tag(path: str) -> str:
    # Uses your existing normalize_module_name if present
    try:
        norm = normalize_module_name(path)
    except NameError:
        norm = path.replace("._modules.", ".").replace("._modules", ".").strip(".")
    return norm.replace(".", "_")


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
            model = torch.hub.load(model_repo, model_name)
        else:
            model = torch.hub.load(model_repo, model_name, weights=model_weights)

        # Assign model to device
        model.cuda() if torch.cuda.is_available() else model.cpu()
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


def apply_noise(model, noise_level, noise_dict, layer_paths, apply_to_all_layers, only_conv=True, include_bias=False):
    targets = get_all_weight_layers(model, only_conv=only_conv) if apply_to_all_layers else _as_list(layer_paths)

    for w_path in targets:
        npath = normalize_module_name(w_path)
        layer = get_layer_from_path(model, npath)

        # respect only_conv
        if only_conv and not is_conv_like(layer):
            continue

        # WEIGHT
        if hasattr(layer, "weight") and layer.weight is not None:
            key_w = f"{npath}.weight"
            sd_w = noise_dict[key_w] if key_w in noise_dict else float(layer.weight.detach().std().item())
            with torch.no_grad():
                layer.weight.add_(torch.randn_like(layer.weight) * (sd_w * noise_level))

        # BIAS (optional)
        if include_bias and hasattr(layer, "bias") and layer.bias is not None:
            key_b = f"{npath}.bias"
            sd_b = noise_dict[key_b] if key_b in noise_dict else float(layer.bias.detach().std().item())
            with torch.no_grad():
                layer.bias.add_(torch.randn_like(layer.bias) * (sd_b * noise_level))


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
    # Activation-level unit masking 
    if masking_level == "unit_activations":
        # Ignore include_bias for activation masking (no bias at this stage).
        # Enumerate target layers
        target_paths = []
        if apply_to_all_layers:
            # If want to add support "all layers", call get_all_conv_layers(model, "")
            raise NotImplementedError("unit_activations with apply_to_all_layers=True not implemented")
        else:
            for path in layer_paths:
                base = get_layer_from_path(model, path)
                if only_conv:
                    target_paths.extend(get_all_conv_layers(base, path))
                else:
                    # You can also support Linear here via get_all_weight_layers
                    target_paths.extend(get_all_weight_layers(base, path))

        # Register forward hooks that zero whole output channels/features.
        for w_path in target_paths:
            layer = get_layer_from_path(model, w_path)

            # Determine number of units from WEIGHT shape if available (out_channels/out_features),
            # fall back to an attribute if needed.
            if hasattr(layer, "weight") and layer.weight is not None:
                num_units = layer.weight.shape[0]
            elif hasattr(layer, "out_channels"):
                num_units = layer.out_channels
            elif hasattr(layer, "out_features"):
                num_units = layer.out_features
            else:
                continue  # skip layers we can’t size

            k = int(fraction_to_mask * num_units)
            if k <= 0:
                # nothing to mask for this layer at this damage level
                continue

            # Draw unit indices once per layer/permutation (like unit weight masking).
            # We store a 1D mask; we’ll move it to the right device/dtype inside the hook.
            unit_idx = torch.randperm(num_units)[:k]
            base_mask_1d = torch.ones(num_units, dtype=torch.float32)
            base_mask_1d[unit_idx] = 0.0

            def _act_unit_mask_hook(module, _inputs, output, mask_1d=base_mask_1d):
                # Handle tensor or tuple outputs
                def _apply(o):
                    # Expect o shape [N, C, ...] for conv or [N, F] for linear
                    if not torch.is_tensor(o) or o.dim() < 2:
                        return o
                    m = mask_1d.to(device=o.device, dtype=o.dtype)
                    # Build view: [1, C, 1, 1, ...] to broadcast across batch & spatial dims
                    view = [1, o.shape[1]] + [1] * (o.dim() - 2)
                    return o * m.view(*view)

                if isinstance(output, tuple):
                    new_first = _apply(output[0])
                    return (new_first,) + output[1:]
                else:
                    return _apply(output)

            layer.register_forward_hook(_act_unit_mask_hook)

        # Done: hooks are registered; nothing else to do for this mode.
        return

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


def get_params_sd(model):
    sds = {}
    for path in get_all_weight_layers(model):  # your existing enumerator
        # canonicalize path like 'features.0' (strip any '._modules.' etc.)
        npath = normalize_module_name(path)
        submodule = get_layer_from_path(model, npath)

        if hasattr(submodule, "weight") and submodule.weight is not None:
            sds[f"{npath}.weight"] = float(submodule.weight.detach().std().item())
        if hasattr(submodule, "bias") and submodule.bias is not None:
            sds[f"{npath}.bias"] = float(submodule.bias.detach().std().item())
    return sds


def run_damage(
    model_info,
    pretrained,
    fraction_to_mask_params,
    noise_levels_params,
    groupnorm_scaling_params,
    layer_paths_to_damage,
    apply_to_all_layers,
    manipulation_method,
    mc_permutations,
    layer_name,
    activation_layers_to_save,
    image_dir,
    only_conv,
    include_bias,
    groupnorm_scaling_targets,
    gain_control_noise=0.0,
    masking_level="connections",
    eccentricity_layer_path=None,  # Specific layer path. The output will be used for damage
    eccentricity_bands: list[list[float]] | None = [[0.60, 1.00]],  # [min,max] normalized eccentricity bands
    ecc_fraction_to_mask_params=[0, 0, 0],
    run_suffix="",
    # Graded eccentricity params
    ecc_profile: str = "linear",
    ecc_mode: str = "dropout",
    ecc_per_channel: bool = False,
    ecc_poly_deg: float = 2.0,
    ecc_exp_k: float = 4.0,
    ecc_reverse: bool = False,
    ):
    """
    A merged run_damage that handles multiple manipulation methods.
    """
    # Determine time_steps for saving
    if "time_steps" in model_info:
        time_steps = str(model_info['time_steps'])
    elif model_info['name'] == "cornet_rt":
        time_steps = "5"
    else:
        time_steps = ""

    # Keep original run_suffix logic
    run_suffix = (("_c" if only_conv else "_all") + ("+b" if include_bias else "")) + run_suffix

    # ------------------------------------------------------------
    # A) Damage-level list (+ noise SDs if needed) — model-agnostic
    # ------------------------------------------------------------
    if manipulation_method == "connections":
        damage_levels_list = generate_params_list(fraction_to_mask_params)
        noise_dict = None
    elif manipulation_method == "noise":
        damage_levels_list = generate_params_list(noise_levels_params)
        # Build a model WITHOUT hooks to compute parameter SDs
        _tmp_model, _ = load_model(model_info, pretrained=pretrained, layer_name="temp", layer_path="")
        noise_dict = get_params_sd(_tmp_model)                                 # uses generic weight-walker
        del _tmp_model
    elif manipulation_method == "groupnorm_scaling":
        damage_levels_list = generate_params_list(groupnorm_scaling_params)
        noise_dict = None
    elif manipulation_method == "eccentricity":
        damage_levels_list = generate_params_list(ecc_fraction_to_mask_params)
        noise_dict = None
    elif manipulation_method == "eccentricity_gradual":
        damage_levels_list = generate_params_list(ecc_fraction_to_mask_params)
        noise_dict = None
    else:
        raise ValueError(f"manipulation_method '{manipulation_method}' is not recognized.")
    # (The old CORnet/VGG block-order heuristic was unused downstream; safely omitted.)  :contentReference[oaicite:1]{index=1}

    # ------------------------------------------------------------
    # B) Decide final hooks robustly (lists ok; any model ok)
    #     1) load once with NO hooks just to inspect structure
    #     2) compute final_layers_to_hook
    # ------------------------------------------------------------
    # Always treat inputs as lists from here on
    activation_layers_list = _as_list(activation_layers_to_save)
    damage_layers_list     = _as_list(layer_paths_to_damage)

    # Load a no-hook model to build the hierarchy
    model, _ = load_model(model_info, pretrained=pretrained, layer_path="")
    # Use your existing helper to select downstream activation hooks
    final_layers_to_hook = get_final_layers_to_hook(model, activation_layers_list, damage_layers_list)
    # Fallback: if the filter returns empty, just use requested activation layers
    if not final_layers_to_hook:
        final_layers_to_hook = activation_layers_list
    print("Activations to be saved for ", damage_layers_list, ": ", final_layers_to_hook)

    # ------------------------------------------------------------
    # C) Directory naming (unchanged)
    # ------------------------------------------------------------
    if manipulation_method == "noise":
        dir_tag = "noise"
    elif manipulation_method == "connections":
        if masking_level == "units":
            dir_tag = "units"
        elif masking_level == "unit_activations":
            dir_tag = "unit_activations"
        else:
            dir_tag = "connections"
    elif manipulation_method == "groupnorm_scaling":
        _map = {"groupnorm": "g", "conv": "c"}
        sel = [_map[t] for t in sorted(set(groupnorm_scaling_targets))]
        targ_tag = "+".join(sel)
        dir_tag = f"groupnorm_scaling_{targ_tag}_noise{gain_control_noise:.3f}"
    else:
        dir_tag = manipulation_method

    # ------------------------------------------------------------
    # D) Total iterations
    # ------------------------------------------------------------
    total_iterations = len(damage_levels_list) * mc_permutations * len(eccentricity_bands)

    # ------------------------------------------------------------
    # E) Main loops (reload WITH hooks; apply damage; save)
    # ------------------------------------------------------------
    with tqdm(total=total_iterations, desc=f"Running {manipulation_method} alteration") as pbar:
        for r_start, r_end in eccentricity_bands:

            # Clean folder naming for eccentricity modes (unchanged)
            if manipulation_method == "eccentricity_gradual":
                sweep_start = damage_levels_list[0]
                if sweep_start > 0.0:
                    band_label = f"band{r_start:.2f}-{r_end:.2f}_min_{sweep_start:.2f}"
                else:
                    band_label = f"band{r_start:.2f}-{r_end:.2f}"
                dir_tag = f"eccentricity_gradual_{band_label}"
            elif manipulation_method == "eccentricity":
                dir_tag = f"eccentricity_{r_start:.2f}-{r_end:.2f}"

            # ----- damage sweep -----
            for damage_level in damage_levels_list:
                for permutation_index in range(mc_permutations):

                    # 1) Load fresh model WITH ALL hooks at once
                    model, activations = load_model(
                        model_info=model_info,
                        pretrained=pretrained,
                        layer_name=layer_name,
                        layer_path=final_layers_to_hook     # LIST is okay
                    )
                    model.eval()

                    # 2) Apply chosen damage to ALL requested target blocks
                    if manipulation_method == "connections":
                        apply_masking(
                            model, fraction_to_mask=damage_level,
                            layer_paths=damage_layers_list,
                            apply_to_all_layers=apply_to_all_layers,
                            masking_level=masking_level,
                            only_conv=only_conv,
                            include_bias=include_bias
                        )

                    elif manipulation_method == "noise":
                        apply_noise(
                            model, noise_level=damage_level,
                            noise_dict=noise_dict,
                            layer_paths=damage_layers_list,
                            apply_to_all_layers=apply_to_all_layers,
                            only_conv=only_conv,
                            include_bias=include_bias
                        )

                    elif manipulation_method == "groupnorm_scaling":
                        apply_groupnorm_scaling(
                            model, scaling_factor=damage_level,
                            layer_paths=damage_layers_list,
                            apply_to_all_layers=apply_to_all_layers,
                            include_bias=include_bias,
                            targets=groupnorm_scaling_targets,
                            gain_control_noise=gain_control_noise
                        )

                    elif manipulation_method == "eccentricity":
                        apply_eccentricity_mask(
                            model,
                            layer_path=eccentricity_layer_path,
                            r_min=r_start,
                            r_max=r_end,
                            fraction=damage_level,
                            per_channel=False
                        )

                    elif manipulation_method == "eccentricity_gradual":
                        apply_eccentricity_graded(
                            model,
                            layer_path=eccentricity_layer_path,
                            r_start=r_start,
                            r_end=r_end,
                            p_min=0.0,
                            p_max=damage_level,
                            mode=ecc_mode,
                            profile=ecc_profile,
                            per_channel=ecc_per_channel,
                            poly_deg=ecc_poly_deg,
                            exp_k=ecc_exp_k,
                            reverse=ecc_reverse
                        )

                    # 3) Extract activations
                    per_layer_data = {lp: [] for lp in final_layers_to_hook}
                    image_files = sorted([
                        f for f in os.listdir(image_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                    ])

                    for image_file in image_files:
                        img_path = os.path.join(image_dir, image_file)
                        input_tensor = preprocess_image(img_path)
                        with torch.no_grad():
                            model(input_tensor)
                        for lp in final_layers_to_hook:
                            out_flat = activations[lp].flatten()
                            per_layer_data[lp].append(
                                out_flat.cpu().numpy() if torch.is_tensor(out_flat) else out_flat
                            )

                    # 4) Save outputs — robust name tags
                    for lp in final_layers_to_hook:
                        arr_2d = np.stack(per_layer_data[lp], axis=0)
                        activations_df = pd.DataFrame(arr_2d, index=image_files)
                        activations_df_sorted = sort_activations_by_numeric_index(activations_df)

                        # SAFE replacement for lp.split('.')[2]
                        lp_name = lp.split(".")[2] if is_cornet(model_info) else short_module_tag(lp)

                        # Activations
                        activation_dir = (
                            f"data/haupt_stim_activ/damaged/{model_info['name']}{time_steps}{run_suffix}/"
                            f"{dir_tag}/{layer_name}/activations/{lp_name}/damaged_{round(damage_level,3)}"
                        )
                        os.makedirs(activation_dir, exist_ok=True)
                        append_activation_to_zarr(activations_df_sorted.astype(np.float16),
                                                  activation_dir, perm_idx=permutation_index)

                        # RDM
                        correlation_matrix, sorted_image_names = compute_correlations(activations_df_sorted)
                        corrmat_dir = (
                            f"data/haupt_stim_activ/damaged/{model_info['name']}{time_steps}{run_suffix}/"
                            f"{dir_tag}/{layer_name}/RDM/{lp_name}/damaged_{round(damage_level,3)}"
                        )
                        os.makedirs(corrmat_dir, exist_ok=True)
                        append_activation_to_zarr(
                            pd.DataFrame(correlation_matrix.astype("float32")),
                            corrmat_dir,
                            perm_idx=permutation_index
                        )

                        # Within-between selectivity
                        categories_array = assign_categories(sorted_image_names)
                        results = convert_np_to_native(calc_within_between(correlation_matrix, categories_array))

                        selectivity_dir = (
                            f"data/haupt_stim_activ/damaged/{model_info['name']}{time_steps}{run_suffix}/"
                            f"{dir_tag}/{layer_name}/selectivity/{lp_name}/damaged_{round(damage_level,3)}"
                        )
                        os.makedirs(selectivity_dir, exist_ok=True)
                        with open(os.path.join(selectivity_dir, f"{permutation_index}.pkl"), "wb") as f:
                            pickle.dump(results, f)

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


def _load_svm_scores(path, categories):
    """
    Load one SVM replicate file and return *scalars*:

        {'animal': 0.92, 'face': 0.88, 'object': 0.89, 'place': 0.90,
         'overall': 0.897, 'total': 0.897}

    Accepted inputs
    ----------------
    •  *.pkl  containing a DataFrame, ndarray, or legacy dict
    •  *.zarr directory produced by `svm_process_*`

    Notes
    -----
    * The four base categories are **always** computed so the downstream
      cache files keep the full set of statistics, even if the caller only
      asked for `categories=['total']`.
    * Missing categories → value = np.nan.
    """
    import os, pickle, numpy as np, pandas as pd, zarr

    BASE_CATS = ("animal", "face", "object", "place")
    want_cats = set(cat.lower() for cat in categories) | set(BASE_CATS)

    # ---------- helper -------------------------------------------------
    def _df_to_scores(df: pd.DataFrame, cat_set):
        df = df.apply(pd.to_numeric, errors="coerce")
        out = {}
        for cat in cat_set:
            mask = [cat in str(col).lower() for col in df.columns]
            vals = df.loc[:, mask].values.flatten()
            vals = vals[~np.isnan(vals)]
            out[cat] = float(np.mean(vals)) if len(vals) else np.nan

        all_vals = df.values.flatten()
        all_vals = all_vals[~np.isnan(all_vals)]
        grand = float(np.mean(all_vals)) if len(all_vals) else np.nan
        out["overall"] = grand
        out["total"]   = grand          # alias wanted by plotting scripts
        return out
    # -------------------------------------------------------------------

    # ---------- Pickle branch ------------------------------------------
    if str(path).lower().endswith(".pkl"):
        if not (os.path.exists(path) and os.path.getsize(path) > 0):
            return {}
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
        except EOFError:
            return {}

        if isinstance(obj, pd.DataFrame):
            return _df_to_scores(obj, want_cats)

        if isinstance(obj, np.ndarray):
            return _df_to_scores(pd.DataFrame(obj), want_cats)

        if isinstance(obj, dict):                   # legacy formats
            # try modern nested {"animal":{"score":{"mean":…}}}
            maybe = {}
            for k, v in obj.items():
                if isinstance(v, dict) and "score" in v and "mean" in v["score"]:
                    maybe[k.lower()] = float(v["score"]["mean"])
                else:                               # very old {"animal": 0.93}
                    try:
                        maybe[k.lower()] = float(v)
                    except Exception:
                        continue
            df = pd.DataFrame([maybe])              # 1-row DF for helper
            return _df_to_scores(df, want_cats)

        return {}                                   # unknown pickle content

    # ---------- Zarr branch --------------------------------------------
    try:
        root = zarr.open(str(path), mode="r")
        arr  = root["activ"][0]                     # first permutation
        # try to recover column names (added by append_activation_to_zarr fix)
        cols = root.attrs.get("column_names")
        df   = pd.DataFrame(arr, columns=cols)
        return _df_to_scores(df, want_cats)
    except Exception:
        return {}

def categ_corr_lineplot(
    damage_layers,
    activations_layers,
    damage_type,
    main_dir="data/haupt_stim_activ/damaged/cornet_rt5_all/",
    categories=("overall",),
    metric="observed_difference",
    subdir_regex=r"damaged_([\d\.]+)(?:_|/|$)",   # handles damaged_1.0, damaged_1.0_123, damaged_1.0/...
    plot_dir="plots/",
    data_type="selectivity",
    scatter=False,
    verbose=0,
    ylim=None,
    percentage=False,
    selectivity_fraction: float|None = None,
    selection_mode: str = "percentage",
    selectivity_file: str|None   = "unit_selectivity/all_layers_units_mannwhitneyu.pkl",
    flip_x_axis=False,  # Used for e.g., gain control plots
    model_tag: str | None = None,
):
    """
    Aggregate replicate files into mean±std curves.

    Enhancement:
      - If selectivity RDMs are missing at RDM_{fraction}_{selection_mode}/<act>/<cat>_selective/,
        compute them from activation .zarr files on the fly, save to disk, and proceed.
    """

    # ------------ debug helper --------------------------------
    def _dbg(msg, level=1):
        if isinstance(verbose, int) and verbose >= level:
            print(msg)

    # ------------ helper loaders (pkl / zarr / activ) ----------
    def _is_zarr_dir(p):
        return os.path.isdir(p) and p.lower().endswith(".zarr")


    def _have_selective_rdms(rdm_dir, cats):
        for c in cats:
            cdir = rdm_dir / f"{c}_selective"
            if not cdir.exists():
                return False
            # require at least one damaged_* subdir with at least one pickle inside
            ok = False
            for d in cdir.glob("damaged_*"):
                if d.is_dir() and any(d.glob("*.pkl")):
                    ok = True; break
            if not ok:
                return False
        return True


    def _safe_load_pickle_file(p):
        try:
            obj = safe_load_pickle(p)
            if obj is not None:
                return obj
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def _compute_rdm_from_activ(activ_2d: np.ndarray):
        """
        Build an RDM from activations (n_images x n_units).
        Uses correlation distance across images: 1 - corr(rows).
        """
        X = activ_2d
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("Need 2D (n_images x n_units>=2) activations to compute an RDM.")
        # Center each image (row), then cosine similarity across images equals correlation
        X = X - X.mean(axis=1, keepdims=True)
        denom = np.linalg.norm(X, axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        Xn = X / denom
        C = (Xn @ Xn.T) / X.shape[1]
        return 1.0 - C

    def _load_rdm_record(pathlike):
        """
        Return {'RDM': 2D array, 'image_names': list[str] or None}
        Works for .pkl dicts (keys 'RDM' and optional 'image_names') and .zarr stores
        that already contain an RDM (not common in your pipeline, but supported).
        """
        if isinstance(pathlike, str) and pathlike.lower().endswith(".pkl"):
            content = _safe_load_pickle_file(pathlike)
            if isinstance(content, dict) and "RDM" in content:
                return {"RDM": np.array(content["RDM"]),
                        "image_names": content.get("image_names")}
            return None

        if _is_zarr_dir(pathlike):
            try:
                import zarr, numpy as _np
                root = zarr.open_group(pathlike, mode="r")
                for k in ("RDM", "rdm", "D", "distance"):
                    if k in root:
                        R = _np.array(root[k])
                        img_names = None
                        try:
                            attrs = dict(root[k].attrs)
                            for nkey in ("image_names", "images", "stimuli", "image_ids", "ids"):
                                if nkey in attrs:
                                    img_names = list(map(str, list(attrs[nkey])))
                                    break
                        except Exception:
                            pass
                        return {"RDM": R, "image_names": img_names}
            except Exception:
                return None
        return None

    # ---------- selectivity helpers (top-k units per category) ----------
    def _get_top_units_from_selectivity(sel_obj, layer, act, cat, frac, mode):
        """
        Returns np.ndarray[int] of unit indices for (layer, act, cat).
        Supports multiple sel_obj shapes: nested dicts or a DataFrame with columns:
            'layer','unit','mw_animals','mw_faces','mw_objects','mw_places'
        """
        cat_map = {"animal":"mw_animals","face":"mw_faces","object":"mw_objects","place":"mw_places",
                   "animals":"mw_animals","faces":"mw_faces","objects":"mw_objects","places":"mw_places"}
        col = cat_map.get(cat, None)

        # Nested dicts
        if isinstance(sel_obj, dict) and col is None:
            # Try several nestings
            for attempt in (
                lambda: sel_obj.get(layer, {}).get(act, {}).get(cat, None),
                lambda: sel_obj.get(act, {}).get(cat, None),
                lambda: sel_obj.get(cat, None),
            ):
                v = None
                try:
                    v = attempt()
                except Exception:
                    v = None
                if v is not None:
                    return np.asarray(v, dtype=int)

        # DataFrame-like
        try:
            if hasattr(sel_obj, "columns"):
                df = sel_obj
                # Require an exact match on layer column (or adapt this to your true schema)
                if "layer" in df.columns:
                    mask = (df["layer"].astype(str) == layer) | (df["layer"].astype(str).str.endswith(f"/{layer}"))
                    if not mask.any():
                        return None  # <- don't proceed with all rows
                    df = df[mask]
                if col not in df.columns or "unit" not in df.columns:
                    return None
                df = df.sort_values(col, ascending=False)
                top_n = max(1, int(round(frac * len(df)))) if mode == "percentage" else max(1, int(frac))
                return df["unit"].to_numpy()[:top_n].astype(int)
        except Exception:
            pass

        return None

    # ---------- write an RDM pickle ----------
    def _write_rdm_pickle(out_path, R, image_names):
        rec = {"RDM": R}
        if image_names is not None:
            rec["image_names"] = list(map(str, image_names))
        with open(out_path, "wb") as f:
            pickle.dump(rec, f)

    # ------------ 1. choose data sub-folder --------------------
    if data_type in ("selectivity",) or data_type.startswith("svm"):
        data_subfolder = data_type
    elif data_type == "imagenet":
        data_subfolder = "imagenet"
        if metric not in ("top1", "top5"):
            raise ValueError("metric must be 'top1' or 'top5' when data_type='imagenet'")
    else:
        raise ValueError(f"unknown data_type '{data_type}'")

    _dbg(f"[SETUP] data_type={data_type} metric={metric} categories={categories}", 1)
    _dbg(f"[SETUP] subdir_regex={subdir_regex!r} main_dir={main_dir}", 2)

    # ------------ 2. helper for cache filename ----------------
    def agg_fname(frac):
        if data_type == "imagenet":
            return f"avg_imagenet_{metric}_{frac}.pkl"
        else:
            return f"avg_{data_type}_{frac}.pkl"

    # ------------ 3. containers -------------------------------
    data       = {}   # (layer, act_key, cat) -> {frac:(mean,std,n)}
    raw_points = {}   # same keys -> {frac:[replicas]}

    # ===========================================================
    # Inner: PRECOMPUTE selective RDMs if missing
    # ===========================================================
    activ_dir = ("activations")  # we only compute from zarr in non-RDM dirs

    def _precompute_selective_rdms_if_missing(layer, act, categories_rdm, rdm_dir):
        """
        If rdm_dir/<cat>_selective is missing or empty, try to compute RDMs from activation .zarr
        and write them under the expected selective RDM tree.
        """
        # quick check: if all cats exist and have any files, nothing to do
        all_exist = True
        for cat in categories_rdm:
            cdir = rdm_dir / f"{cat}_selective"
            if not cdir.exists():
                all_exist = False; break
            if not any(cdir.rglob("*.pkl")) and not any(p for p in cdir.rglob("*.zarr") if p.is_dir()):
                all_exist = False; break
        if all_exist:
            _dbg("[PRECOMPUTE] selective RDMs already present — skipping build.", 2)
            return

        # locate activations
        layer_root = Path(main_dir) / damage_type / layer
        activ_roots = []
        for base in activ_dir:
            cand = layer_root / base / act
            if cand.exists():
                activ_roots.append(cand)
        # Prefer non-RDM roots for activations
        activ_roots = [r for r in activ_roots if r.name.lower() != "rdm"] + [r for r in activ_roots if r.name.lower() == "rdm"]

        if not activ_roots:
            _dbg(f"[PRECOMPUTE-MISS] No activation roots found under {layer_root} for act={act}. "
                 f"Tried {activ_dir}.", 1)
            return

        # load selectivity map
        sel_obj = _safe_load_pickle_file(selectivity_file) if selectivity_file else None
        if sel_obj is None:
            _dbg(f"[PRECOMPUTE-MISS] Could not load selectivity_file: {selectivity_file}", 1)
            return

        # gather top unit idx per category
        percat_units = {}
        for cat in categories_rdm:
            idx = _get_top_units_from_selectivity(sel_obj, layer, act, cat, selectivity_fraction, selection_mode)
            if idx is None or len(idx) == 0:
                raise RuntimeError(f"No selective units for {layer}/{act}/{cat} (frac={selectivity_fraction}, mode={selection_mode}). Check selectivity_file and labels.")
                percat_units[cat] = None
            else:
                percat_units[cat] = np.asarray(idx, dtype=int)
                _dbg(f"[PRECOMPUTE] {cat}: top {len(percat_units[cat])} units", 2)

        # walk damage folders and build RDMs
        dmg_re = re.compile(subdir_regex)
        built_any = False

        for aroot in activ_roots:
            # expect damaged_* subdirs
            dmg_dirs = [d for d in sorted(aroot.iterdir()) if d.is_dir() and dmg_re.search(d.name)]
            if not dmg_dirs:
                _dbg(f"[PRECOMPUTE-MISS] No damaged_* subdirs under {aroot}", 1)
                continue

            for dmg_dir in dmg_dirs:
                m = dmg_re.search(dmg_dir.name)
                dmg_level = m.group(1) if m else dmg_dir.name

                # find zarr replicates
                zarrs = [z for z in sorted(dmg_dir.iterdir()) if _is_zarr_dir(str(z))]
                if not zarrs:
                    _dbg(f"[PRECOMPUTE-MISS] No .zarr replicates under {dmg_dir}", 2)
                    continue

                for z in zarrs:
                    try:
                        import zarr
                        root = zarr.open(str(z), mode="r")
                        A = None; akey_used = None
                        for akey in ("activ", "activations", "A", "X"):
                            if akey in root:
                                A = np.array(root[akey]); akey_used = akey
                                break
                        if A is None or A.ndim != 2 or A.shape[0] < 2 or A.shape[1] < 1:
                            _dbg(f"[PRECOMPUTE-SKIP] Bad activ array in {z.name} (shape={None if A is None else A.shape})", 2)
                            continue

                        # get image names if present
                        img_names = None
                        try:
                            attrs = dict(root[akey_used].attrs)
                            for nkey in ("image_names", "images", "stimuli", "image_ids", "ids"):
                                if nkey in attrs:
                                    img_names = list(map(str, list(attrs[nkey])))
                                    break
                        except Exception:
                            pass

                        # compute & write per-category RDM
                        for cat in categories_rdm:
                            keep = percat_units.get(cat, None)
                            if keep is None or keep.size == 0:
                                continue
                            # bounds
                            keep = keep[(keep >= 0) & (keep < A.shape[1])]
                            if keep.size == 0:
                                continue

                            A_sel = A[:, keep]
                            R = _compute_rdm_from_activ(A_sel)

                            # out path: RDM_{frac}_{mode}/<act>/<cat>_selective/damaged_x.y/rep_*.pkl
                            out_cat_dmg = rdm_dir / f"{cat}_selective" / dmg_dir.name
                            out_cat_dmg.mkdir(parents=True, exist_ok=True)
                            rep_name = Path(z).stem + ".pkl"  # e.g., replicate name
                            out_path = out_cat_dmg / rep_name
                            if not out_path.exists():
                                _write_rdm_pickle(out_path, R, img_names)
                                built_any = True
                                _dbg(f"[PRECOMPUTE-WRITE] {out_path}", 2)
                    except Exception as e:
                        _dbg(f"[PRECOMPUTE-ERROR] failed on {z}: {e}", 1)
                        continue

        if not built_any:
            _dbg("[PRECOMPUTE-EMPTY] No selective RDMs were written (missing zarrs or unit indices?).", 1)
        else:
            _dbg("[PRECOMPUTE-DONE] Selective RDMs created.", 1)

    # ------------ 4. crawl the directory tree -----------------
    for layer in damage_layers:
        for act in activations_layers:
            act_key = act

            # ===========================
            # SELECTIVITY (fractioned)  |
            # ===========================
            if data_type == "selectivity" and selectivity_fraction is not None:
                # categories to pull RDMs for (precomputed per-category)
                if "total" in categories:
                    categories_rdm = ["face", "object", "animal", "place"]
                else:
                    categories_rdm = list(categories)

                # 1) Discover the selective RDM directory
                rdm_dir = Path(main_dir) / damage_type / layer / f"RDM_{selectivity_fraction:.2f}_{selection_mode}" / act
                _dbg(f"[DISCOVER] layer={layer} act={act}", 1)
                _dbg(f"[DISCOVER] target RDM dir = {rdm_dir}", 1)

                # NEW: try to build them if missing
                if not rdm_dir.exists() or not any(rdm_dir.rglob("*.pkl")):
                    _dbg("[DISCOVER] selective RDMs missing or empty — attempting on-the-fly build from activations.", 1)
                    activ_root = Path(main_dir) / damage_type / layer / "activations" / act
                    _dbg(f"[PRECOMPUTE] activ_root = {activ_root}", 1)

                    if not _have_selective_rdms(rdm_dir, categories_rdm):
                        generate_category_selective_RDMs(
                            activations_root = Path(main_dir) / damage_type,
                            layer_name       = layer,         # matches selectivity table’s layer label
                            top_frac         = float(selectivity_fraction),
                            categories       = categories_rdm,   # no "total" here
                            selection_mode   = selection_mode,
                            selectivity_file = selectivity_file,
                            damage_layer     = layer,
                            activation_layer = act,
                            model_tag        = model_tag
                        )
                    if not _have_selective_rdms(rdm_dir, categories_rdm):
                        raise RuntimeError(f"No selective RDMs built under {rdm_dir} for cats {categories_rdm}")

                # 2) Prepare output directory for averages
                out_base = Path(main_dir) / damage_type / layer / f"avg_selectivity_top{selectivity_fraction:.2f}_{selection_mode}" / act
                out_base.mkdir(parents=True, exist_ok=True)
                _dbg(f"[OUT] averages will be written under: {out_base}", 2)

                # 3) Iterate categories (face/object/animal/place or user subset)
                for cat in categories_rdm:
                    data[(layer, act_key, cat)] = {}
                    raw_points[(layer, act_key, cat)] = {}

                    # expected precomputed category subfolder
                    cat_dir = rdm_dir / f"{cat}_selective"
                    _dbg(f"[CAT] cat='{cat}' → {cat_dir}", 2)

                    if not cat_dir.exists():
                        alt = rdm_dir / "scene_selective"
                        alt_exists = alt.exists()
                        _dbg(f"[MISS] {cat_dir} does not exist."
                             f"{' Found scene_selective instead.' if alt_exists else ''}", 1)
                        continue

                    # 4) Crawl damage subdirs
                    subdirs = [d for d in sorted(cat_dir.iterdir()) if d.is_dir()]
                    if not subdirs:
                        _dbg(f"[MISS] no damage subfolders found under {cat_dir}", 1)
                        continue
                    _dbg(f"[HAVE] {len(subdirs)} damage subfolders under {cat_dir}: {[d.name for d in subdirs]}", 2)

                    dmg_re = re.compile(subdir_regex)

                    for dmg in subdirs:
                        m = dmg_re.search(dmg.name)
                        if not m:
                            _dbg(f"[SKIP] subdir name did not match subdir_regex: {dmg.name}  (regex={subdir_regex})", 1)
                            continue
                        try:
                            dmg_level = float(m.group(1))
                        except Exception:
                            _dbg(f"[SKIP] could not cast damage fraction from '{m.group(1)}' in {dmg.name}", 1)
                            continue

                        avg_file = out_base / f"avg_selectivity_{cat}_{dmg_level}.pkl"
                        if avg_file.exists():
                            try:
                                with open(avg_file, "rb") as f:
                                    stats = pickle.load(f)
                                data[(layer, act_key, cat)][dmg_level] = (stats.get("mean", np.nan),
                                                                           stats.get("std", 0.0),
                                                                           stats.get("n", 0))
                                raw_points[(layer, act_key, cat)][dmg_level] = stats.get("vals", [])
                                _dbg(f"[CACHE] hit {avg_file} → n={stats.get('n',0)}", 2)
                                continue
                            except Exception:
                                _dbg(f"[CACHE] corrupt {avg_file}; will rebuild.", 1)

                        # scan files inside this damage-level folder
                        files = []
                        for f in sorted(dmg.iterdir()):
                            if f.is_file() and f.suffix.lower() == ".pkl":
                                files.append(f)
                            elif f.is_dir() and f.name.lower().endswith(".zarr"):
                                files.append(f)
                        if not files:
                            _dbg(f"[MISS] no *.pkl or *.zarr files under {dmg}", 1)
                            continue
                        _dbg(f"[SCAN] {dmg.name}: {len(files)} files", 2)

                        selectivities = []
                        reasons_skipped = {"unreadable":0, "no_RDM":0, "no_names":0, "calc_error":0, "shape_mismatch":0}

                        for f in files:
                            rec = _load_rdm_record(str(f))
                            if rec is None:
                                reasons_skipped["unreadable"] += 1
                                _dbg(f"   [SKIP] unreadable file: {f.name}", 3)
                                continue
                            if "RDM" not in rec or rec["RDM"] is None:
                                reasons_skipped["no_RDM"] += 1
                                _dbg(f"   [SKIP] no RDM in file: {f.name}", 3)
                                continue

                            R = np.array(rec["RDM"])
                            if R.ndim != 2 or R.shape[0] != R.shape[1]:
                                reasons_skipped["shape_mismatch"] += 1
                                _dbg(f"   [SKIP] RDM not square (shape={R.shape}) in {f.name}", 2)
                                continue

                            img_names = rec.get("image_names", None)
                            if not img_names:
                                reasons_skipped["no_names"] += 1
                                _dbg(f"   [SKIP] missing image_names in {f.name}", 2)
                                continue

                            try:
                                cats = assign_categories(img_names)
                            except Exception as e:
                                reasons_skipped["calc_error"] += 1
                                _dbg(f"   [SKIP] assign_categories failed for {f.name}: {e}", 2)
                                continue

                            try:
                                sel = calc_within_between(R, cats)[cat]["observed_difference"]
                                selectivities.append(float(sel))
                            except Exception as e:
                                reasons_skipped["calc_error"] += 1
                                _dbg(f"   [SKIP] calc_within_between failed for {f.name}: {e}", 2)
                                continue

                        if not selectivities:
                            _dbg(f"[EMPTY] no usable replicas at {dmg} for cat={cat}. "
                                 f"Reasons: {reasons_skipped}", 1)
                            continue

                        mean_sel = float(np.mean(selectivities))
                        std_sel  = float(np.std(selectivities, ddof=1)) if len(selectivities) > 1 else 0.0
                        stats    = {"mean": mean_sel, "std": std_sel, "n": len(selectivities),
                                    "vals": [float(x) for x in selectivities]}
                        try:
                            with open(avg_file, "wb") as f:
                                pickle.dump(stats, f)
                            _dbg(f"[WRITE] {avg_file.name}: n={stats['n']} mean={mean_sel:.4f} std={std_sel:.4f}", 2)
                        except Exception as e:
                            _dbg(f"[ERROR] failed to write {avg_file}: {e}", 1)

                        data[(layer, act_key, cat)][dmg_level] = (mean_sel, std_sel, len(selectivities))
                        raw_points[(layer, act_key, cat)][dmg_level] = list(selectivities)

                #synthesize "total" from per-replicate means across base categories
                if ("total" in categories):
                    base_cats = ("animal", "face", "object", "place")
                    have_all = all((layer, act_key, bc) in data for bc in base_cats)
                    if have_all:
                        # union of all damage fractions present across cats
                        all_fracs = set()
                        for bc in base_cats:
                            all_fracs |= set(raw_points[(layer, act_key, bc)].keys())

                        # init "total"
                        data[(layer, act_key, "total")] = {}
                        raw_points[(layer, act_key, "total")] = {}

                        for frac in sorted(all_fracs):
                            lists = []
                            ok = True
                            for bc in base_cats:
                                rp = raw_points[(layer, act_key, bc)].get(frac, [])
                                if not rp:
                                    ok = False; break
                                lists.append(list(map(float, rp)))
                            if not ok:
                                continue

                            # align on min replicate count
                            L = min(len(lst) for lst in lists)
                            if L == 0:
                                continue

                            per_rep_means = [float(np.nanmean([lists[0][i], lists[1][i], lists[2][i], lists[3][i]]))
                                             for i in range(L)]
                            mean_total = float(np.nanmean(per_rep_means))
                            std_total  = float(np.nanstd(per_rep_means, ddof=1)) if L > 1 else 0.0  # SD of means

                            data[(layer, act_key, "total")][frac] = (mean_total, std_total, L)
                            raw_points[(layer, act_key, "total")][frac] = per_rep_means
            
            else:
                # =======================================
                # All other data_type paths (original)  |
                # =======================================
                if data_type == "imagenet":
                    layer_path = os.path.join(main_dir, damage_type, layer, "imagenet")
                    out_base   = os.path.join(main_dir, damage_type, layer, "avg_imagenet")
                    act_key    = "imagenet"
                else:
                    layer_path = os.path.join(main_dir, damage_type, layer, data_subfolder, act)
                    out_base   = os.path.join(main_dir, damage_type, layer, f"avg_{data_type}", act)
                    act_key    = act
                _dbg(f"[PATH] layer={layer} act={act_key}", 1)
                _dbg(f"[PATH] layer_path={layer_path}", 2)
                _dbg(f"[PATH] out_base={out_base}", 2)

                if not os.path.isdir(layer_path):
                    _dbg(f"[MISS] layer_path does not exist: {layer_path}", 1)
                    continue
                os.makedirs(out_base, exist_ok=True)

                # init dict slots
                for cat in categories:
                    data[(layer, act_key, cat)] = {}
                    raw_points[(layer, act_key, cat)] = {}

                # scan damaged_* subdirs
                for subdir in os.listdir(layer_path):
                    subdir_path = os.path.join(layer_path, subdir)
                    if not os.path.isdir(subdir_path):
                        continue
                    m = re.search(subdir_regex, subdir)
                    if not m:
                        _dbg(f"[SKIP] regex miss: {subdir}", 2)
                        continue
                    frac = round(float(m.group(1)), 3)
                    cache = os.path.join(out_base, agg_fname(frac))
                    _dbg(f"[FRACTION] {subdir} -> frac={frac}  cache={'HIT' if os.path.exists(cache) else 'MISS'}", 1)

                    # ---------- build cache if missing ----------
                    if not os.path.exists(cache):
                        agg = {}                      # cat -> list[values]
                        _dbg(f"[BUILD] scanning files in {subdir_path}", 2)

                        for fname in os.listdir(subdir_path):
                            p = os.path.join(subdir_path, fname)
                            is_pkl  = fname.lower().endswith(".pkl")
                            is_zarr = fname.lower().endswith(".zarr") and os.path.isdir(p)
                            if not (is_pkl or is_zarr):
                                continue

                            # ---- IMAGE NET MODE ----
                            if data_type == "imagenet":
                                if is_zarr:
                                    continue
                                content = safe_load_pickle(p)
                                if content is None:
                                    try:
                                        with open(p, "rb") as f:
                                            content = pickle.load(f)
                                    except Exception:
                                        continue
                                for cat in categories:
                                    if str(cat).lower() == "overall":
                                        val = content["overall"][metric]
                                    elif str(cat).isdigit():
                                        cls = int(cat)
                                        val = content["classes"].get(cls, {}).get(metric, np.nan)
                                    else:
                                        continue
                                    agg.setdefault(cat, []).append(float(val))

                            # ---- SELECTIVITY MODE (no fraction) ----
                            elif data_type == "selectivity":
                                if is_zarr:
                                    continue
                                content = safe_load_pickle(p)
                                if content is None:
                                    try:
                                        with open(p, "rb") as f:
                                            content = pickle.load(f)
                                    except Exception:
                                        continue
                                if not isinstance(content, dict):
                                    _dbg(f"[SKIP] not a dict: {fname}", 2)
                                    continue
                                _dbg(f"[FILE] {fname} keys={list(content.keys())[:6]}", 2)

                                base_cats = ["animal", "face", "object", "place"]

                                if "total" in categories:
                                    if ("total" in content) and (metric in content["total"]):
                                        agg.setdefault("total", []).append(float(content["total"][metric]))
                                        _dbg(f"[APPEND] direct 'total' from {fname}: {metric}", 2)
                                    else:
                                        _dbg(f"[FALLBACK] synthesize total from base cats in {fname}", 2)
                                        for cat_name, met_dict in content.items():
                                            if (cat_name in base_cats) and (metric in met_dict):
                                                agg.setdefault(cat_name, []).append(float(met_dict[metric]))
                                else:
                                    for cat_name, met_dict in content.items():
                                        if (cat_name in categories) and (metric in met_dict):
                                            agg.setdefault(cat_name, []).append(float(met_dict[metric]))

                            # ---- SVM MODE ----
                            else:   # data_type startswith "svm"
                                scores = _load_svm_scores(p, categories)
                                if not scores:
                                    continue
                                for cat_name, val in scores.items():
                                    agg.setdefault(cat_name, []).append(val)

                        if data_type == "selectivity" and ("total" in categories) and ("total" not in agg):
                            base_cats = ("animal", "face", "object", "place")
                            if all(bc in agg for bc in base_cats) and all(len(agg[bc]) for bc in base_cats):
                                L = min(len(agg[bc]) for bc in base_cats)
                                per_rep_means = [float(np.nanmean([agg["animal"][i], agg["face"][i], agg["object"][i], agg["place"][i]]))
                                                 for i in range(L)]
                                _dbg(f"[SYNTH] total from per-replicate means, n={len(per_rep_means)}", 1)
                                agg["total"] = per_rep_means

                        packed = {
                            c: {
                                metric: {
                                    "mean": float(np.mean(v)) if len(v) > 0 else np.nan,
                                    "std":  float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                                    "n":    len(v),
                                    "vals": [float(x) for x in v]
                                }
                            } for c, v in agg.items()
                        }
                        try:
                            with open(cache, "wb") as f:
                                pickle.dump(packed, f)
                        except Exception as e:
                            _dbg(f"[ERROR] failed to write cache {cache}: {e}", 1)

                    # ---------- read cache and populate data ----------
                    agg_content = safe_load_pickle(cache) or {}
                    if (data_type == "selectivity"
                        and ("total" in categories)
                        and ("total" not in agg_content)):
                        base = ("animal","face","object","place")
                        have_all = all(
                            (b in agg_content)
                            and (metric in agg_content[b])
                            and ("vals" in agg_content[b][metric])
                            for b in base
                        )
                        if have_all:
                            lists = [list(map(float, agg_content[b][metric]["vals"])) for b in base]
                            if all(len(lst) for lst in lists):
                                L = min(len(lst) for lst in lists)
                                per_rep_means = [float(np.nanmean([lists[0][i], lists[1][i], lists[2][i], lists[3][i]]))
                                                 for i in range(L)]
                                agg_content["total"] = {
                                    metric: {
                                        "mean": float(np.nanmean(per_rep_means)),
                                        "std":  float(np.nanstd(per_rep_means, ddof=1)) if L > 1 else 0.0,
                                        "n":    L,
                                        "vals": per_rep_means
                                    }
                                }
                                try:
                                    with open(cache, "wb") as f:
                                        pickle.dump(agg_content, f)
                                except Exception:
                                    pass

                    for cat in categories:
                        if cat not in agg_content or metric not in agg_content[cat]:
                            continue
                        rec = agg_content[cat][metric]
                        mean = rec.get("mean", np.nan)
                        std  = rec.get("std", 0.0)
                        n    = rec.get("n", 0)
                        vals = rec.get("vals", [])
                        data[(layer, act_key, cat)][frac] = (mean, std, n)
                        raw_points[(layer, act_key, cat)][frac] = vals

    # ------------ 5. optional percentage scaling --------------
    if percentage:
        def _baseline_fraction(frac_keys, dmg_type):
            fracs = sorted(frac_keys)
            if dmg_type == "groupnorm_scaling":
                return 1.0 if 1.0 in fracs else fracs[-1]
            return fracs[0]

        for key in list(data.keys()):
            frac_dict = data[key]
            raw_dict  = raw_points[key]
            if not frac_dict:
                _dbg(f"[WARN] no aggregated points for {key}", 1)
                continue

            base_frac = _baseline_fraction(frac_dict.keys(), damage_type)
            if base_frac not in raw_dict or len(raw_dict[base_frac]) == 0:
                _dbg(f"[WARN] percentage=True but no baseline replicas at frac={base_frac} for {key}", 1)
                continue

            base_vals = np.asarray(raw_dict[base_frac], dtype=float)

            for frac in list(frac_dict.keys()):
                cur_vals = np.asarray(raw_dict.get(frac, []), dtype=float)
                min_len  = min(len(base_vals), len(cur_vals))
                if min_len == 0:
                    _dbg(f"[WARN] cannot scale frac={frac} for {key} (no raw points)", 2)
                    continue
                ratio = 100.0 * (cur_vals[:min_len] / (base_vals[:min_len] + 1e-12))
                raw_points[key][frac] = list(map(float, ratio))
                m, s = float(np.mean(ratio)), float(np.std(ratio, ddof=1)) if len(ratio) > 1 else 0.0
                data[key][frac] = (m, s, len(ratio))

    # ------------ 6. plotting ----------------------
    for (layer, act_key, cat), frac_dict in data.items():
        if not frac_dict or cat not in categories:
            continue
        xs = sorted(frac_dict.keys())
        ys = [frac_dict[x][0] for x in xs]
        err= [frac_dict[x][1] for x in xs]
        lbl = f"{layer}-{act_key}-{cat}"

        if data_type == "imagenet":
            color_cat = "total"
            color_act = "IT"
        else:
            color_cat = cat
            color_act = act_key

        try:
            color = get_color_for_triple(layer, color_act, str(color_cat))
            plt.errorbar(xs, ys, yerr=err, fmt='-o', capsize=4, label=lbl, color=color)
        except Exception:
            plt.errorbar(xs, ys, yerr=err, fmt='-o', capsize=4, label=lbl)

        if scatter:
            for x in xs:
                jitter = np.random.normal(0, 0.005, size=len(raw_points[(layer, act_key, cat)].get(x, [])))
                if len(jitter):
                    try:
                        plt.scatter([x + j for j in jitter],
                                    raw_points[(layer, act_key, cat)].get(x, []),
                                    alpha=0.5, s=10, color=color)
                    except Exception:
                        plt.scatter([x + j for j in jitter],
                                    raw_points[(layer, act_key, cat)].get(x, []),
                                    alpha=0.5, s=10)

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

    if flip_x_axis:
        plt.gca().invert_xaxis()

    plt.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    name_parts = [main_dir.strip("/").split("/")[-1],
                  damage_type.replace("/", "-"),
                  "-".join(damage_layers),
                  "-".join(activations_layers)]
    categ_letters = ("total" if "total" in categories
                     else ("overall" if "overall" in categories
                           else "".join(cat[0] for cat in categories)))
    name_parts.append(categ_letters)
    if percentage:
        name_parts.append("percentage")
    if selectivity_fraction is not None:
        name_parts.append(f"top{selectivity_fraction:.2f}-{selection_mode}")
    plot_path = os.path.join(plot_dir, "_".join(name_parts) + ".png")
    plt.savefig(plot_path, dpi=500)
    if verbose:
        print(f"[PLOT] saved to {plot_path}")
        plt.show()
    else:
        plt.close()


def plot_avg_corr_mat(
    layers,
    damage_type,
    *,
    image_dir="stimuli/",
    output_dir="average_RDMs",
    subdir_regex=r"damaged_([\d\.]+)$",
    damage_levels=None,
    main_dir="data/haupt_stim_activ/damaged/cornet_rt/",
    vmax=1.0,
    plot_dir="plots/",
    verbose=0,
):
    """
    Build & plot average RDMs using **all permutations** in every file.
    Works with both *.pkl* and *.zarr* correlation stores.
    """
    fraction_to_matrix: dict[tuple[float, str], np.ndarray] = {}
    sorted_image_names = get_sorted_filenames(image_dir)
    n_img = len(sorted_image_names)

    for layer in layers:
        rdm_root   = Path(main_dir) / damage_type / layer / "RDM"
        cache_root = Path(main_dir) / damage_type / layer / output_dir
        cache_root.mkdir(parents=True, exist_ok=True)
        if not rdm_root.exists():
            print(f"[warn] missing {rdm_root}")
            continue

        for sub in rdm_root.iterdir():
            if not sub.is_dir():
                continue
            if damage_levels:
                if not any(sub.name.endswith(f"damaged_{lvl}") for lvl in damage_levels):
                    continue
                frac = float(extract_string_numeric_parts(sub.name)[1])
            else:
                m = re.search(subdir_regex, sub.name)
                if not m:
                    continue
                frac = float(m.group(1))
            frac = round(frac, 3)

            cache_file = cache_root / f"avg_RDM_{frac}.pkl"
            if cache_file.exists():
                avg_mat = np.asarray(pd.read_pickle(cache_file), dtype=np.float32)
            else:
                mats = []
                for item in sub.iterdir():
                    if item.suffix.lower() in (".pkl", ".zarr"):
                        try:
                            mats.extend(load_all_corr_mats(item))
                        except Exception as exc:
                            print(f"[skip] {item}: {exc}")
                if not mats:
                    continue
                avg_mat = np.mean(mats, axis=0).astype(np.float32)
                pd.to_pickle(avg_mat.tolist(), cache_file)

            fraction_to_matrix[(frac, layer)] = avg_mat

    if not fraction_to_matrix:
        print("No matrices found – nothing to plot.")
        return

    # ---------- plotting ----------
    keys = sorted(fraction_to_matrix)
    n   = len(keys)
    n_c = int(np.ceil(n ** 0.5))
    n_r = int(np.ceil(n / n_c))
    fig, axes = plt.subplots(n_r, n_c, figsize=(4*n_c, 4*n_r), squeeze=False)
    axes = axes.ravel()

    for i, (frac, layer) in enumerate(keys):
        R   = fraction_to_matrix[(frac, layer)]
        ax  = axes[i]
        im  = ax.imshow(R, cmap="viridis", vmin=0, vmax=vmax)
        ax.set_xticks(range(n_img))
        ax.set_yticks(range(n_img))
        ax.set_xticklabels(sorted_image_names, rotation=90, fontsize=4)
        ax.set_yticklabels(sorted_image_names, fontsize=4)
        ax.set_title(f"{layer} — dmg={frac}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[i+1:]:
        ax.axis("off")

    plt.suptitle(damage_type)
    plt.tight_layout()

    plot_dir = Path(plot_dir); plot_dir.mkdir(parents=True, exist_ok=True)
    model_name = Path(main_dir).parts[-2]  # “…/damaged/<model>/…”
    n_lvls = len(damage_levels) if damage_levels else len(keys)
    tag_layers = "_".join(layers)
    out_png = plot_dir / f"{model_name}_RDM_{damage_type}_{tag_layers}_{n_lvls}-levels.png"

    if verbose == 1:
        if input(f"Save plot to {out_png}? [Y/n] ").strip().lower() in ("", "y"):
            plt.savefig(out_png, dpi=500)
        plt.show()
    elif verbose == 0:
        plt.savefig(out_png, dpi=500)
        plt.close(fig)
    else:
        raise ValueError("verbose must be 0 or 1")


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
    # ---------------- NEW COMMON IO HELPERS --------------------------

    def load_svm_dataframes(item_path: str | os.PathLike) -> list[pd.DataFrame]:
        """
        Return one DataFrame per permutation.

        Accepts .pkl, .zarr, or a directory containing either.
        """
        dfs = []

        if os.path.isfile(item_path):
            if str(item_path).endswith(".pkl"):
                df = pd.read_pickle(item_path)
                if not isinstance(df, pd.DataFrame):
                    raise ValueError(f"{item_path} did not contain a DataFrame.")
                dfs.append(df)

            elif str(item_path).endswith(".zarr"):
                with zarr.open(str(item_path), mode="r") as zr:
                    for perm in zr.attrs["perm_indices"]:
                        df = load_activations_zarr(item_path, perm=perm)
                        dfs.append(df)

            else:
                raise ValueError(f"Unsupported file type: {item_path}")

        elif os.path.isdir(item_path):
            for fn in sorted(os.listdir(item_path)):
                if fn.endswith((".pkl", ".zarr")):
                    dfs += load_svm_dataframes(os.path.join(item_path, fn))
        else:
            raise FileNotFoundError(f"{item_path} is neither file nor directory.")

        if not dfs:
            raise FileNotFoundError(f"No SVM DataFrames found in {item_path}")
        return dfs
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
                        mats = load_all_corr_mats(item_path)
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
                df = load_activations_zarr(pkl_path)

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
    categories=("animal", "face", "object", "place", "total"),
    metric="observed_difference",
    subdir_regex=r"damaged_([\d\.]+)$",
    plot_dir="plots/",
    common_ylim=None,
    verbose=0,
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
    # ------------------------------------------------------------------
    # Helper : read **all permutations** from either a Pickle or a Zarr
    # ------------------------------------------------------------------
    def _load_selectivity_dicts(path: str | os.PathLike) -> list[dict]:
        out = []
        if os.path.isfile(path):
            if str(path).endswith(".pkl"):
                try:
                    with open(path, "rb") as f:
                        d = pickle.load(f)
                    if isinstance(d, dict):
                        out.append(d)
                except Exception as exc:
                    print(f"[warn] could not read {path}: {exc}")

            elif str(path).endswith(".zarr"):
                try:
                    root = zarr.open(str(path), mode="r")
                    for perm in root.attrs.get("perm_indices", []):
                        df = load_activations_zarr(path, perm=perm)
                        # expect a 1‑row DataFrame whose single cell is the dict
                        if df.shape[0] == 1 and isinstance(df.iloc[0, 0], dict):
                            out.append(df.iloc[0, 0])
                        else:
                            print(f"[warn] {path} did not contain a dict payload")
                except Exception as exc:
                    print(f"[warn] could not read {path}: {exc}")
        elif os.path.isdir(path):
            for fn in sorted(os.listdir(path)):
                if fn.endswith((".pkl", ".zarr")):
                    out += _load_selectivity_dicts(os.path.join(path, fn))
        return out
    # ------------------------------------------------------------------

    # ------------- STEP 1 – aggregate ---------------------------------
    data = { (dt, cat): {} for dt in damage_types for cat in categories }

    for dmg_type in damage_types:
        layer_path   = os.path.join(main_dir, dmg_type, layer, "selectivity")
        output_path  = os.path.join(main_dir, dmg_type, layer, "avg_selectivity")
        os.makedirs(output_path, exist_ok=True)

        if not os.path.isdir(layer_path):
            print(f"[warn] {layer_path} not found – skipping '{dmg_type}'")
            continue

        for subdir in os.listdir(layer_path):
            m = re.search(subdir_regex, subdir)
            if not (m and os.path.isdir(os.path.join(layer_path, subdir))):
                continue                                    # not a “damaged_x” dir

            frac = round(float(m.group(1)), 3)
            cache = os.path.join(output_path, f"avg_selectivity_{frac}.pkl")

            # -- build cache if missing --------------------------------
            if not os.path.exists(cache):
                agg: dict[str, list[float]] = {c: [] for c in categories}

                for d in _load_selectivity_dicts(os.path.join(layer_path, subdir)):
                    for cat, met in d.items():
                        if cat in categories and metric in met:
                            agg[cat].append(float(met[metric]))

                stats = {
                    c: {
                        "mean": float(np.mean(v)) if v else np.nan,
                        "std":  float(np.std (v)) if v else np.nan,
                    } for c, v in agg.items()
                }
                with open(cache, "wb") as f:
                    pickle.dump(stats, f)

            # -- read cache & fill master dict -------------------------
            cached = safe_load_pickle(cache) or {}
            for cat in categories:
                if (isinstance(cached.get(cat), dict) and
                        metric in cached[cat]):
                    μ = cached[cat][metric]["mean"]
                    σ = cached[cat][metric]["std"]
                    data[(dmg_type, cat)][frac] = (μ, σ)

    # ------------- STEP 2 – single‑axis sub‑plots ---------------------
    n = len(damage_types)
    if n == 0:
        print("No valid damage types → nothing to plot.")
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    axes = axes[0]

    for ax, dmg_type in zip(axes, damage_types):
        plotted = False
        for cat in categories:
            pts = data[(dmg_type, cat)]
            if not pts:
                continue
            x = sorted(pts)
            y = [pts[v][0] for v in x]
            e = [pts[v][1] for v in x]
            ax.errorbar(x, y, yerr=e, fmt="-o", capsize=3, label=cat)
            plotted = True

        if not plotted:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)

        if "fraction" in dmg_type.lower():
            ax.set_xlabel("Fraction of units 0")
        elif "noise" in dmg_type.lower() or "std" in dmg_type.lower():
            ax.set_xlabel("Noise σ")
        else:
            ax.set_xlabel(f"{dmg_type} param")

        ax.set_ylabel(metric)
        ax.set_title(f"{layer} — {dmg_type}")
        ax.legend()
        if common_ylim: ax.set_ylim(common_ylim)

        xs = [v for cat in categories for v in data[(dmg_type, cat)]]
        if xs and min(xs) >= 0:
            ax.set_xlim(left=0, right=max(xs))

    fig.tight_layout()

    os.makedirs(plot_dir, exist_ok=True)
    mdl = main_dir.rstrip("/").split("/")[-1]
    fn  = f"{mdl}_lineplot_{layer}_SUBPLOTS_" + "_".join(damage_types) + f"_{metric}.png"
    plt.savefig(os.path.join(plot_dir, fn), dpi=300)
    if verbose: plt.show()
    else:       plt.close(fig)

    # ------------- STEP 3 – twin‑axes plot (exactly two damage types) -
    if len(damage_types) != 2:
        print("Twinned‑axis plot skipped (need exactly 2 damage types).")
        return

    d1, d2 = damage_types
    fig2, ax_bot = plt.subplots(figsize=(7, 5))
    ax_top = ax_bot.twiny()

    def _plot(axis, dmg, color):
        xs_all = []
        for cat in categories:
            pts = data[(dmg, cat)]
            if not pts: continue
            x = sorted(pts)
            y = [pts[v][0] for v in x]
            e = [pts[v][1] for v in x]
            axis.errorbar(x, y, yerr=e, fmt="-o", capsize=3,
                          color=color, label=f"{dmg}:{cat}")
            xs_all += x
        if xs_all and min(xs_all) >= 0:
            axis.set_xlim(left=0, right=max(xs_all))

    _plot(ax_bot, d1, "C0"); ax_bot.set_xlabel(f"{d1} param", color="C0")
    _plot(ax_top, d2, "C1"); ax_top.set_xlabel(f"{d2} param", color="C1")
    for ax, c in [(ax_bot, "C0"), (ax_top, "C1")]:
        ax.tick_params(axis="x", labelcolor=c)
        ax.spines[("bottom" if ax is ax_bot else "top")].set_edgecolor(c)
    ax_bot.set_ylabel(metric)
    if common_ylim: ax_bot.set_ylim(common_ylim)
    ln1, lb1 = ax_bot.get_legend_handles_labels()
    ln2, lb2 = ax_top.get_legend_handles_labels()
    ax_bot.legend(ln1 + ln2, lb1 + lb2, loc="best")
    fig2.tight_layout()

    fn_tw = f"{mdl}_lineplot_{layer}_TWINS_{d1}_{d2}_{metric}.png"
    plt.savefig(os.path.join(plot_dir, fn_tw), dpi=300)
    if verbose: plt.show()
    else:       plt.close(fig2)


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


def _iter_activation_dfs(src: str | os.PathLike) -> Iterator[pd.DataFrame]:
    """
    Yield one activation DataFrame per permutation contained in *src*.

    *src* can be
    • a legacy Pickle file  – yields exactly one DataFrame
    • a “.zarr” directory   – yields **all** permutations found inside
                              (order preserved as in root.attrs['perm_indices'])
    """
    path = Path(src)
    if path.suffix == ".pkl":
        df = pd.read_pickle(path)
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"{path} does not contain a pandas DataFrame.")
        yield df

    elif path.suffix == ".zarr" and path.is_dir():
        root = zarr.open(str(path), mode="r")
        perm_ids = root.attrs.get("perm_indices", list(range(root["activ"].shape[0])))
        for perm in perm_ids:
            yield load_activations_zarr(path, perm=perm)

    else:
        raise FileNotFoundError(f"{src} is neither a .pkl file nor a .zarr store.")


def train_and_test_svm_arrays(X_train, y_train, X_test, y_test):
    clf = SVC(kernel="linear", random_state=42)
    clf.fit(X_train, y_train)
    return np.mean(clf.predict(X_test) == y_test)


def svm_process_file(
    activ_path,
    training_samples: int = 15,
    clip_val: float = 1e6,
    max_permutations: int | None = None,
):
    """
    For every permutation in *activ_path* run a 1‑vs‑1 SVM on each category pair.
    Returns a DataFrame whose **rows = permutations** and
    **columns = 'animal_vs_face', …**.
    """
    rows = []                       # one dict per permutation

    for df in _iter_activation_dfs(activ_path):
        df = df.drop("numeric_index", axis=1, errors="ignore")
        df.columns = df.columns.astype(str)

        if len(df) < 64:            # 4 cats × 16 images => 64 rows
            continue

        # -------- split into the four 16‑row category blocks ----------
        blocks = {
            "animal": df.iloc[0:16].to_numpy(dtype=float),
            "face":   df.iloc[16:32].to_numpy(dtype=float),
            "object": df.iloc[32:48].to_numpy(dtype=float),
            "place":  df.iloc[48:64].to_numpy(dtype=float),
        }
        for k in blocks:
            blocks[k] = np.clip(
                np.nan_to_num(blocks[k], nan=0.0, posinf=clip_val, neginf=-clip_val),
                -clip_val, clip_val
            )

        pairs = list(combinations(blocks.keys(), 2))
        perm_row = {}

        for (c1, c2) in pairs:
            a1, a2 = blocks[c1], blocks[c2]
            idx = np.arange(16)

            train_cmb1 = list(combinations(idx, training_samples))
            train_cmb2 = list(combinations(idx, training_samples))
            all_splits = list(product(train_cmb1, train_cmb2))

            if max_permutations and max_permutations < len(all_splits):
                all_splits = random.sample(all_splits, max_permutations)

            accs = []
            for tr1, tr2 in all_splits:
                tst1 = tuple(np.setdiff1d(idx, tr1))
                tst2 = tuple(np.setdiff1d(idx, tr2))

                X_tr = np.vstack((a1[list(tr1)], a2[list(tr2)]))
                y_tr = np.hstack((np.zeros(len(tr1)), np.ones(len(tr2))))
                X_te = np.vstack((a1[list(tst1)], a2[list(tst2)]))
                y_te = np.hstack((np.zeros(len(tst1)), np.ones(len(tst2))))

                accs.append(train_and_test_svm_arrays(X_tr, y_tr, X_te, y_te))

            perm_row[f"{c1}_vs_{c2}"] = np.mean(accs)   # average over splits

        rows.append(perm_row)

    return None if not rows else pd.DataFrame(rows)


def svm_process_directory(
    parent_dir,
    training_samples: int = 15,
    allowed_subdirs: list[str] | None = None,
    max_permutations: int | None = None,
):
    """
    Walk *parent_dir*; for every activation Pickle **or** Zarr found under an
    `.../activations/...` path (optionally filtered by *allowed_subdirs*)
    run `svm_process_file()` and save the results in a mirrored
        .../svm_<training_samples>/ path as **.zarr**.
    """
    allowed_subdirs = allowed_subdirs or []
    activ_files: list[tuple[str, str]] = []    # [(root, fname_or_dir), ...]

    for root, dirs, files in os.walk(parent_dir):
        if "activations" not in root.split(os.sep):
            continue

        # Identify which “activation layer” we are in (first folder after 'activations')
        parts = root.split(os.sep)
        try:
            act_idx = parts.index("activations")
        except ValueError:
            continue

        sub_after = parts[act_idx + 1] if len(parts) > act_idx + 1 else None
        if allowed_subdirs and sub_after not in allowed_subdirs:
            continue

        # Pickle files
        for f in files:
            if f.endswith(".pkl"):
                activ_files.append((root, f))
        # Zarr directories at this level
        for d in list(dirs):        # copy since we may mutate dirs
            if d.endswith(".zarr"):
                activ_files.append((root, d))
                dirs.remove(d)      # do not descend into the .zarr folder

    from tqdm import tqdm
    for root, name in tqdm(activ_files, desc="SVM processing"):
        in_path = os.path.join(root, name)

        # Mirror the path, swapping "activations" → "svm_<training_samples>"
        parts = root.split(os.sep)
        act_idx = parts.index("activations")
        act_root = os.path.join(*parts[: act_idx + 1])
        rel = os.path.relpath(in_path, act_root)
        svm_dir = os.path.join(os.path.dirname(act_root), f"svm_{training_samples}")
        out_path = os.path.join(svm_dir, rel)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df = svm_process_file(
            in_path,
            training_samples=training_samples,
            max_permutations=max_permutations,
        )
        if df is not None:
            append_activation_to_zarr(df, out_path, perm_idx=0)


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
    Collect SVM accuracy results (either .pkl or .zarr) and return a
    list[dict] of long‑form rows compatible with build_long_df().
    """
    if metric == "score":                 # legacy alias
        metric = "overall"
    if metric not in ("overall", "by_category"):
        raise ValueError("metric must be 'overall', 'score', or 'by_category'")

    rows, n_files = [], 0
    folder = f"svm_{train_samples}"

    for dmg_type in ("units", "noise", "connections"):
        if allowed and dmg_type not in allowed:
            continue
        for dmg_layer in (mv_root / dmg_type).iterdir():
            svm_dir = dmg_layer / folder
            if not svm_dir.exists():
                continue

            for act_layer in svm_dir.iterdir():
                for damaged in act_layer.iterdir():
                    if not damaged.name.startswith("damaged_"):
                        continue
                    lvl = float(damaged.name.split("_")[-1])

                    for item in damaged.iterdir():
                        is_pkl  = item.suffix.lower() == ".pkl"
                        is_zarr = item.suffix.lower() == ".zarr" and item.is_dir()
                        if not (is_pkl or is_zarr):
                            continue
                        n_files += 1

                        scores = _load_svm_scores(item, categories=("animal","face","object","place","overall"))
                        if not scores:
                            continue

                        if metric == "overall":
                            val = scores.get("overall", np.nan)
                            rows.append(dict(
                                value        = val,
                                model_variant= mv_name,
                                include_bias = bias_flag,
                                damage_type  = dmg_type,
                                damage_layer = dmg_layer.name,
                                damage_level = lvl,
                                category     = "overall",
                                replicate    = int(item.stem) if is_pkl else 0,
                            ))
                        else:  # by_category
                            for cat, val in scores.items():
                                if cat == "overall":
                                    continue
                                rows.append(dict(
                                    value        = val,
                                    model_variant= mv_name,
                                    include_bias = bias_flag,
                                    damage_type  = dmg_type,
                                    damage_layer = dmg_layer.name,
                                    damage_level = lvl,
                                    category     = cat,
                                    replicate    = int(item.stem) if is_pkl else 0,
                                ))

    if n_files == 0:
        raise RuntimeError(f"No SVM files found under {mv_root}")
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
    fmap_shape: Tuple[int, int, int],
    model_tag: str | None = None,
) -> List[int]:
    """
    1. Look up the per-unit selectivity file for `category` (try .pkl first, then .csv).
    2. Load it into a DataFrame.
    3. Filter to rows where df.layer_name == layer_name.
    4. Take the top `top_frac` fraction by 'scaled_activation'.
    5. Given fmap_shape = (C, H, W), convert each (c,y,x) into a flat index.
    """
    def _try(basename: str):
        pkl_path = os.path.join(selectivity_dir, basename + ".pkl")
        csv_path = os.path.join(selectivity_dir, basename + ".csv")
        if os.path.isfile(pkl_path): return pd.read_pickle(pkl_path)
        if os.path.isfile(csv_path): return pd.read_csv(csv_path)
        return None

    df = None
    if model_tag:
        df = _try(f"{model_tag}_{category}_unit_selectivity_all_units")
    if df is None:
        df = _try(f"{category}_unit_selectivity_all_units")
    if df is None:
        raise FileNotFoundError(
            f"No selectivity file for '{category}' in '{selectivity_dir}'. "
            f"Tried model-tagged and legacy basenames."
        )

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
    activations_root: str | Path,                 # e.g. Path(main_dir) / damage_type
    layer_name: str,                              # kept for compat; NOT used for selection
    top_frac: float,
    categories: Sequence[str] = ("faces", "places", "objects", "animals"),
    damage_levels: Sequence[str] | None = None,   # auto-discover if None
    selection_mode: str = "percentage",           # or "percentile"
    selectivity_file: str = "unit_selectivity/all_layers_units_mannwhitneyu.pkl",
    damage_layer: str = "V1",
    activation_layer: str = "IT",
    verbose: int = 1,
    model_tag: str | None = None,
):
    """
    Build category-selective RDMs under the SAME damage-type subtree the plotter expects.

    Reads:
        <activations_root>/<damage_layer>/activations/<activation_layer>/
            damaged_*/[<rep>/] {*.zarr|*.pkl}

    Writes:
        <activations_root>/<damage_layer>/RDM_<top_frac:0.2f>_<selection_mode>/<activation_layer>/
            <cat>_selective/<damaged_*>/<rep_or_store>.pkl

    RDM = correlation distance across images (1 - corr). Strict errors; no irrelevant fallbacks.
    """

    # ---- tiny logger ----
    def log(lvl: int, msg: str):
        if verbose >= lvl:
            print(f"[RDM:{lvl}] {msg}")

    # ---- normalize categories ----
    def norm_cat(c: str) -> str:
        c = c.lower().strip()
        if c in ("face", "faces"):     return "faces"
        if c in ("object", "objects"): return "objects"
        if c in ("animal", "animals"): return "animals"
        if c in ("place", "places"):   return "places"
        if c == "total":               return "total"
        raise ValueError(f"Unknown category '{c}'")

    cats = [norm_cat(c) for c in categories]
    if "total" in cats:  # expand to the four atomic groups
        cats = ["faces", "objects", "animals", "places"]

    col_map = {
        "faces":   "mw_faces",
        "objects": "mw_objects",
        "animals": "mw_animals",
        "places":  "mw_places",
    }
    out_dirname = {  # how the plotter expects folders to be named
        "faces":   "face_selective",
        "objects": "object_selective",
        "animals": "animal_selective",
        "places":  "place_selective",
    }

    activ_root = Path(activations_root) / damage_layer / "activations" / activation_layer
    out_root   = Path(activations_root) / damage_layer / f"RDM_{top_frac:.2f}_{selection_mode}" / activation_layer
    log(1, f"activations_root = {Path(activations_root)}")
    log(1, f"  activ_root     = {activ_root}")
    log(1, f"  out_root       = {out_root}")
    if not activ_root.is_dir():
        raise FileNotFoundError(f"Activation root not found: {activ_root}")

    # ---- selectivity table ----
    sel_path = resolve_selectivity_table(selectivity_file, model_tag=model_tag)
    if sel_path.suffix.lower() == ".pkl":
        sel_df = pd.read_pickle(sel_path)
    elif sel_path.suffix.lower() == ".csv":
        sel_df = pd.read_csv(sel_path)
    else:
        raise ValueError("selectivity_file must be .pkl or .csv")

    layer_key = activation_layer
    if "layer" not in sel_df.columns or "unit" not in sel_df.columns:
        raise ValueError("Selectivity file must include 'layer' and 'unit' columns.")

    # Filter to activation layer rows (strict, then a suffix fallback)
    rows = sel_df[sel_df["layer"] == layer_key]
    if rows.empty:
        layer_key = layer_key.replace("_", ".")
        print("trying key: ", layer_key)
    if rows.empty:
        layer_key = "module." + activation_layer
        rows = sel_df[sel_df["layer"] == layer_key]
    if rows.empty:
        raise ValueError(f"No rows in selectivity file for layer '{layer_key}' "
                         f"(or suffix '/{activation_layer}').")

    missing_cols = [c for c in col_map.values() if c not in sel_df.columns]
    if missing_cols:
        raise ValueError(f"Selectivity file missing columns: {missing_cols}")

    log(1, f"selectivity_file = {sel_path}")
    log(1, f"rows for {layer_key}: n={len(rows)} (total rows={len(sel_df)})")
    log(2, f"selectivity columns: {list(sel_df.columns)}")

    # ---- discover damage levels ----
    if damage_levels is None:
        damage_levels = [d.name for d in sorted(activ_root.iterdir())
                         if d.is_dir() and d.name.startswith("damaged_")]
    log(1, f"found damage_levels: {damage_levels[:6]}{' …' if len(damage_levels)>6 else ''}")
    if not damage_levels:
        raise FileNotFoundError(f"No damaged_* folders under {activ_root}")

    # ---- helper: find replicates under a damage folder ----
    def find_reps(dmg_dir: Path) -> Tuple[List[Path], List[Path]]:
        """
        Returns (zarr_stores, pkl_files) discovered anywhere under dmg_dir,
        but ignores anything under an 'RDM_' subfolder (to avoid reading our outputs).
        """
        zarrs: List[Path] = []
        pkls:  List[Path] = []
        # Prefer shallow first, then deeper
        # shallow: direct children
        for p in sorted(dmg_dir.iterdir()):
            if p.is_dir() and p.suffix == ".zarr":
                zarrs.append(p)
            elif p.is_file() and p.suffix == ".pkl":
                pkls.append(p)
        # deeper: one or two levels (rep dirs, etc.), excluding RDM_ trees
        for p in sorted(dmg_dir.rglob("*")):
            if out_root.name in str(p):    # skip outputs
                continue
            if "RDM_" in str(p):
                continue
            if p.is_dir() and p.suffix == ".zarr":
                zarrs.append(p)
            elif p.is_file() and p.suffix == ".pkl":
                pkls.append(p)
        # de-dup while preserving order
        def dedup(seq):
            seen=set(); out=[]
            for x in seq:
                if x in seen: continue
                seen.add(x); out.append(x)
            return out
        return dedup(zarrs), dedup(pkls)

    # ---- determine n_units (from first usable rep) ----
    n_units = None
    n_images_hint = None
    first_sample = None
    for dmg in damage_levels:
        in_dir = activ_root / dmg
        if not in_dir.is_dir(): 
            continue
        zarrs, pkls = find_reps(in_dir)
        log(2, f"{dmg}: discovered {len(zarrs)} zarr(s), {len(pkls)} pkl(s).")
        # try zarr first
        for zstore in zarrs:
            try:
                root = zarr.open(zstore, mode="r")
                if "activ" in root:
                    n_units = int(root["activ"].shape[2])
                    n_images_hint = int(root["activ"].shape[1])
                    first_sample = zstore
                    break
            except Exception as e:
                log(2, f"  zarr open failed {zstore}: {e}")
        if n_units is None:
            for pkl_path in pkls:
                try:
                    df = pd.read_pickle(pkl_path)
                    if isinstance(df, pd.DataFrame) and df.ndim == 2:
                        n_units = int(df.shape[1])
                        n_images_hint = int(df.shape[0])
                        first_sample = pkl_path
                        break
                except Exception as e:
                    log(2, f"  pickle read failed {pkl_path}: {e}")
        if n_units is not None:
            break

    if n_units is None:
        # Rich context on why we think it's empty
        children = [p.name for p in activ_root.iterdir()] if activ_root.exists() else []
        raise FileNotFoundError(
            "Could not read any activation (zarr or pickle).\n"
            f"  activ_root: {activ_root}\n"
            f"  exists: {activ_root.exists()}  children(sample): {children[:10]}\n"
            "  Reason: no usable '.zarr' with dataset 'activ' or 2-D pickle DataFrames found under any damaged_*."
        )
    log(1, f"n_units determined from {'zarr' if isinstance(first_sample, Path) and first_sample.suffix=='.zarr' else 'pickle'}: "
           f"{n_units} (n_images≈{n_images_hint}) at {first_sample}")

    # ---- pick top units per category; validate vs n_units ----
    idxs_by_cat: dict[str, np.ndarray] = {}
    for cat in cats:
        col = col_map[cat]
        if selection_mode == "percentage":
            k = max(1, int(round(len(rows) * float(top_frac))))
            top = rows.nlargest(k, col)
        elif selection_mode == "percentile":
            thr = float(np.percentile(rows[col].to_numpy(), float(top_frac)))
            top = rows[rows[col] >= thr]
        else:
            raise ValueError("selection_mode must be 'percentage' or 'percentile'")
        idx = top["unit"].astype(int).to_numpy()
        if idx.size == 0:
            raise ValueError(f"No units selected for '{cat}' in layer '{layer_key}'.")
        max_idx = int(idx.max())
        log(1, f"[SELECT] {cat}: picked {len(idx)} units (top_frac={top_frac}, mode={selection_mode}), "
               f"max_idx={max_idx}, n_units={n_units}")
        if max_idx >= n_units:
            # Surface exactly what's wrong and how to check it
            raise ValueError(
                f"Selected unit index {max_idx} for '{cat}' exceeds activations' n_units ({n_units}).\n"
                f"  Hints:\n"
                f"    • Ensure selectivity file rows are filtered to the same layer: layer == '{layer_key}' "
                f"(or endswith '/{activation_layer}').\n"
                f"    • Confirm activations you’re reading are from '{activation_layer}' and not another layer.\n"
                f"    • First sample used to gauge n_units: {first_sample}"
            )
        idxs_by_cat[cat] = idx



    # ---- build selective RDMs ----
    built = 0
    for dmg in damage_levels:
        in_dir = activ_root / dmg
        if not in_dir.is_dir():
            log(1, f"[SKIP] not a dir: {in_dir}")
            continue

        zarrs, pkls = find_reps(in_dir)
        log(1, f"{dmg}: will process {len(zarrs)} zarr(s), {len(pkls)} pkl(s)")

        # ZARR replicates (one store == one replicate)
        for zstore in zarrs:
            # discover image names early for clear errors
            root = zarr.open(zstore, mode="r")
            img_names = list(map(str, root.attrs.get("image_names", [])))
            if not img_names and "activ" in root and hasattr(root["activ"], "attrs"):
                img_names = list(map(str, root["activ"].attrs.get("image_names", [])))
            # load full matrix via helper
            try:
                df = load_activations_zarr(zstore, perm=None)
            except Exception as e:
                raise RuntimeError(f"load_activations_zarr failed for {zstore} (perm=None): {e}")
            if not isinstance(df, pd.DataFrame) or df.ndim != 2:
                raise ValueError(f"Zarr loader did not return 2-D DataFrame for {zstore}")
            if not img_names:
                # trust DF index if attrs absent
                img_names = [str(x) for x in df.index.tolist()]
                if not img_names:
                    raise ValueError(f"No image names in attrs or DF index for {zstore}")

            A_full = df.to_numpy(dtype=np.float32)  # (n_images x n_units)
            if A_full.shape[1] != n_units:
                log(1, f"[WARN] n_units changed at {zstore}: {A_full.shape[1]} (expected {n_units}) — "
                        "will still proceed if indices fit.")

            for cat, idx in idxs_by_cat.items():
                A = A_full[:, idx]
                R = np.corrcoef(A)
                out_dir = out_root / out_dirname[cat] / dmg
                out_dir.mkdir(parents=True, exist_ok=True)
                out_pkl = out_dir / f"{zstore.stem}.pkl"
                with open(out_pkl, "wb") as f:
                    pickle.dump({"RDM": R, "image_names": img_names}, f)
                built += 1
            log(2, f"  wrote {len(idxs_by_cat)} RDM(s) for {zstore}")

        # Legacy pickle replicates
        for pkl_path in pkls:
            if pkl_path.name.endswith(".pkl") and "RDM_" in str(pkl_path):
                continue  # safety
            df = pd.read_pickle(pkl_path)
            if not isinstance(df, pd.DataFrame) or df.ndim != 2:
                raise ValueError(f"Activation pickle is not a 2-D DataFrame: {pkl_path}")
            img_names = [str(x) for x in df.index.tolist()]
            if not img_names:
                raise ValueError(f"No image names (DataFrame index) in activation pickle: {pkl_path}")
            A_full = df.to_numpy(dtype=np.float32)
            for cat, idx in idxs_by_cat.items():
                A = A_full[:, idx]
                R = np.corrcoef(A)
                out_dir = out_root / out_dirname[cat] / dmg
                out_dir.mkdir(parents=True, exist_ok=True)
                out_pkl = out_dir / f"{pkl_path.stem}.pkl"
                with open(out_pkl, "wb") as f:
                    pickle.dump({"RDM": R, "image_names": img_names}, f)
                built += 1
            log(2, f"  wrote {len(idxs_by_cat)} RDM(s) for {pkl_path}")

    if built == 0:
        # dump a quick inventory to explain *why*
        inv = []
        for dmg in damage_levels:
            in_dir = activ_root / dmg
            if not in_dir.is_dir(): 
                inv.append((dmg, "MISSING_DIR", 0, 0))
                continue
            zarrs, pkls = find_reps(in_dir)
            inv.append((dmg, "OK_DIR", len(zarrs), len(pkls)))
        lines = "\n".join(f"    {d}: {flag} — zarrs={nz}, pkls={npk}" for d, flag, nz, npk in inv)
        raise RuntimeError(
            f"No selective RDMs were written under {out_root}.\n"
            f"Inventory:\n{lines}\n"
            "Check that damaged_* folders actually contain zarr stores (with dataset 'activ') "
            "or 2-D pickle DataFrames and that selectivity indices fit n_units."
        )

    log(1, f"Done. Wrote {built} selective RDM file(s) under {out_root}")
    return True


def get_all_groupnorm_layers(model, base_path=""):
    """
    Recursively find all submodules under `model` that are nn.GroupNorm,
    returning a list of layer paths.

    Parameters:
        model (nn.Module): The model or module to search.
        base_path (str): The starting path.

    Returns:
        List[str]: A list of dot-separated paths to each GroupNorm module.
    """
    gn_layers = []

    # If the current module itself is a GroupNorm, record its path
    if isinstance(model, nn.GroupNorm):
        gn_layers.append(base_path)
        # We don't need to recurse further into a GroupNorm layer
        return gn_layers

    # Otherwise, recurse into children
    for name, submodule in model._modules.items():
        if submodule is None:
            continue
        if base_path == "":
            new_path = name
        else:
            # Note: Preserving the original pathing style from your code
            new_path = base_path + "._modules." + name

        gn_layers.extend(get_all_groupnorm_layers(submodule, new_path))

    return gn_layers


def apply_groupnorm_scaling(
    model: nn.Module,
    scaling_factor: float,
    *,
    layer_paths: Sequence[str] | None = None,
    apply_to_all_layers: bool = False,
    include_bias: bool = False,
    targets: Sequence[str] = ("groupnorm",),
    gain_control_noise: float = 0.0,  # Noise level to be used while scaling
) -> None:
    """
    Multiply weights (and optional biases) by *scaling_factor*.
    Add Gaussian noise (before scaling) with std = gain_control_noise * weight.std().
    """
    do_gn  = "groupnorm" in targets
    do_conv= "conv"      in targets
    if not (do_gn or do_conv):
        return

    def _collect_targets(root_mod: nn.Module, root_path: str):
        paths = []
        if do_gn:
            paths += get_all_groupnorm_layers(root_mod, root_path)
        if do_conv:
            paths += get_all_conv_layers(root_mod, root_path, include_bias)
        return paths

    with torch.no_grad():
        if apply_to_all_layers:
            for p in _collect_targets(model, ""):
                _scale_module_(get_layer_from_path(model, p),
                               scaling_factor, include_bias, gain_control_noise)
        else:
            for block in (layer_paths or []):
                submod = get_layer_from_path(model, block)
                for p in _collect_targets(submod, block):
                    _scale_module_(get_layer_from_path(model, p),
                                   scaling_factor, include_bias, gain_control_noise)

def _scale_module_(module: nn.Module,
                   factor: float,
                   include_bias: bool = False,
                   gain_control_noise: float = 0.0) -> None:
    """
    Add Gaussian noise to weights/biases (before scaling), then scale.
    """
    if hasattr(module, "weight") and module.weight is not None:
        if gain_control_noise > 0:
            sd = module.weight.data.std().item()
            noise = torch.randn_like(module.weight.data) * (gain_control_noise * sd)
            module.weight.data.add_(noise)
        module.weight.data.mul_(factor)
    if include_bias and hasattr(module, "bias") and module.bias is not None:
        if gain_control_noise > 0:
            sd = module.bias.data.std().item()
            noise = torch.randn_like(module.bias.data) * (gain_control_noise * sd)
            module.bias.data.add_(noise)
        module.bias.data.mul_(factor)


# ── compressor used for every Zarr store ─────────────────────────────
_COMP = numcodecs.Blosc(cname="zstd",
                        clevel=7,
                        shuffle=numcodecs.Blosc.BITSHUFFLE)


def _unique_store(base: Path, tag: str, shape_tail: tuple[int, ...]) -> Path:
    """
    Return a collision‑free Zarr path such as

        myfile__activ_25089.zarr
        myfile__rdm_64.zarr

    so that objects with different shapes never share the same store.
    """
    suffix = "_".join(map(str, shape_tail))
    return base.with_suffix("").with_name(f"{base.stem}__{tag}_{suffix}.zarr")


def _init_store(zarr_path: Path, n_imgs: int, n_feat: int,
                chunks: tuple[int,int] | None = None):
    """Create an empty growable array  (None × n_imgs × n_feat)."""
    if chunks is None:
        chunks = (1, n_imgs, n_feat)          # appending one perm at a time
    root = zarr.open(zarr_path, mode="w")
    root.zeros("activ",
               shape=(0, n_imgs, n_feat),           # 0 → growable
               chunks=chunks, dtype="f2",
               compressor=_COMP, overwrite=True)
    root.attrs["image_names"]  = []
    root.attrs["perm_indices"] = []


def load_activations_zarr(source: str | Path,
                          perm: int | None = None) -> pd.DataFrame:
    """
    Return one permutation (default: the **first**) as a DataFrame identical
    to what the old Pickle contained.  If *perm* is None and >1 perms exist
    the caller must disambiguate.
    """
    root  = zarr.open(Path(source).with_suffix(".zarr"), mode="r")
    imset = root.attrs["image_names"]
    perms = root.attrs["perm_indices"]
    if perm is None:
        perm = perms[0]
    try:
        i = perms.index(perm)
    except ValueError:
        raise KeyError(f"perm {perm} not in {perms}")
    arr = root["activ"][i]                   # lazy read one chunk
    return pd.DataFrame(arr, index=imset)


def list_zarr_files(dir_path: str | Path) -> List[Path]:
    """Return *sorted* list of foo.zarr directories in *dir_path*."""
    p = Path(dir_path)
    return sorted([d for d in p.iterdir() if d.suffix == ".zarr" and d.is_dir()])


def load_matrix_zarr(path: str | Path, perm: int = 0) -> np.ndarray:
    """Load one permutation (default: 0) from a correlation‑matrix zarr store."""
    df = load_activations_zarr(path, perm=perm)  # <- your existing helper
    return df.to_numpy(dtype=np.float32)


def load_all_corr_mats(item: str | Path) -> list[np.ndarray]:
    """
    Yield **all permutations** found in *item*.

    • If *item* is  *.zarr  ➜ read root['activ'] → (P, N, N) and split.  
    • If *item* is  *.pkl   ➜ single matrix → length‑1 list.

    Returned matrices are float32.
    """
    p = Path(item)
    if p.suffix.lower() == ".zarr" and p.is_dir():
        root = zarr.open(p, mode="r")
        perms = root["activ"][:]                    # (P, N, N)
        return [m.astype(np.float32, copy=False) for m in perms]
    elif p.suffix.lower() == ".pkl":
        with open(p, "rb") as f:
            mat = pickle.load(f)
        return [np.asarray(mat, dtype=np.float32)]
    else:
        raise ValueError(f"Unsupported correlation file: {p}")


def _coerce_to_2d(arr_like: Any) -> np.ndarray:
    """
    • DataFrame / Series → values
    • 1‑D → (1,‑)
    • 2‑D → unchanged
    Raises if ndim > 2.
    """
    if isinstance(arr_like, pd.DataFrame):
        arr = arr_like.values
    elif isinstance(arr_like, pd.Series):
        arr = arr_like.values[None, :]          # → (1, nFeat)
    else:
        arr = np.asarray(arr_like)

    if arr.ndim == 1:
        arr = arr[None, :]                      # promote to (1,‑)
    if arr.ndim != 2:
        raise ValueError(f"cannot coerce shape {arr.shape} to 2‑D")
    return arr


def append_matrix_to_zarr(mat: np.ndarray,
                          target: str | Path,
                          perm_idx: int):
    """
    Append one square correlation matrix into a Zarr store.
    Each distinct matrix size lives in its own store.
    """
    assert mat.ndim == 2 and mat.shape[0] == mat.shape[1], "RDM must be square"

    zarr_path = _unique_store(Path(target), "rdm", (mat.shape[0],))

    if not zarr_path.exists():
        root = zarr.open(zarr_path, mode="w")
        root.zeros("activ",
                   shape=(0, *mat.shape),
                   chunks=(1, *mat.shape),
                   dtype="f4",
                   compressor=_COMP)
        root.attrs["perm_indices"] = []

    root = zarr.open(zarr_path, mode="a")
    z = root["activ"]

    tmp = f"tmp_{os.getpid()}_{perm_idx}"
    z.store[tmp] = mat.astype("float32", copy=False)[None, ...]

    try:
        z.append(z.store[tmp])
    except ValueError:
        alt = _unique_store(zarr_path.with_suffix(""), "rdm_alt", (mat.shape[0],))
        root_alt = zarr.open(alt, mode="w")
        root_alt.zeros("activ",
                       shape=(0, *mat.shape),
                       chunks=(1, *mat.shape),
                       dtype="f4",
                       compressor=_COMP)
        root_alt["activ"].append(mat[None, ...])

    del z.store[tmp]
    root.attrs["perm_indices"] = root.attrs["perm_indices"] + [perm_idx]


def append_activation_to_zarr(df: pd.DataFrame,
                              folder: os.PathLike,
                              perm_idx: int) -> None:
    """
    Append a (N_img × N_feat) activation tensor *df* for one permutation
    to a grow-able Zarr store living inside *folder*.

    One permutation  →  one store.
    """
    import numpy as np, zarr, os, shutil
    from pathlib import Path

    n_img, n_feat = df.shape
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1) Strictly one store per permutation:   "<perm>__activ_<feat>.zarr"
    # ------------------------------------------------------------------ #
    zarr_name = f"{perm_idx}__activ_{n_feat}.zarr"
    zarr_path = folder / zarr_name

    # If the file already exists *delete it* so we never get shape clashes
    if zarr_path.exists():
        shutil.rmtree(zarr_path)

    root = zarr.open(zarr_path, mode="w")
    z = root.create_dataset(
        "activ",
        shape=(0, n_img, n_feat),
        chunks=(1, n_img, n_feat),
        dtype=np.float16,
        compressor=zarr.Blosc(cname="zstd", clevel=3),
        overwrite=True,
    )
    z.append(df.to_numpy()[None, ...])         # (1, N_img, N_feat)

    root.attrs.update(
        image_names=df.index.tolist(),
        perm_indices=[perm_idx],
        column_names=list(df.columns),         # QoL: restore original DF
    )


def _build_radial_map(H: int, W: int, device=None) -> torch.Tensor:
    """Return an (H×W) tensor of *normalised* radii belongs to [0, 1].

    0 ≡ centre pixel, 1 ≡ farthest corner. Works for rectangular maps.
    """
    yy = torch.arange(H, device=device).view(H, 1).float()
    xx = torch.arange(W, device=device).view(1, W).float()
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    rr = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return rr / rr.max()


def _gen_band_mask(shape: Tuple[int, int, int, int],
                   r_min: float,
                   r_max: float,
                   frac: float,
                   *,
                   per_channel: bool = False,
                   device=None) -> torch.Tensor:
    """Create a Bernoulli mask that zeros *frac* of units with r belongs to [r_min, r_max)."""
    _, C, H, W = shape
    rmap = _build_radial_map(H, W, device)
    band = (rmap >= r_min) & (rmap < r_max)

    if per_channel:
        keep = torch.bernoulli(torch.full((C, H, W), 1 - frac, device=device))
        mask = torch.where(band, keep, torch.ones((C, H, W), device=device))
        return mask.unsqueeze(0)            # → (1, C, H, W)
    else:
        keep = torch.bernoulli(torch.full((H, W), 1 - frac, device=device))
        mask = torch.where(band, keep, torch.ones((H, W), device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # → (1, 1, H, W)


def apply_eccentricity_mask(model: torch.nn.Module,
                            *,
                            layer_path: str,
                            r_min: float,
                            r_max: float,
                            fraction: float,
                            per_channel: bool = False,
                            ) -> None:
    """Register a forward‑hook that randomly zeros `fraction` of activations
    in the eccentricity ring [r_min, r_max).

    Parameters
    ----------
    model        : the nn.Module holding the target layer.
    layer_path   : dotted path from `model` to the layer to hook.
    r_min, r_max : lower / upper bound **normalised to [0, 1]**.
    fraction     : proportion of units inside the band to delete (0–1).
    per_channel  : if *True* draw a separate Bernoulli for every (c,x,y), else
                   share one spatial mask for all channels.
    """
    assert 0.0 <= r_min < r_max <= 1.0, "r_min/r_max must satisfy 0 ≤ r_min < r_max ≤ 1"
    assert 0.0 <= fraction <= 1.0,      "fraction must be in [0, 1]"

    target_layer = get_layer_from_path(model, layer_path)

    # Cache one mask per unique output shape – allows variable batch sizes ------
    _mask_cache: dict[Tuple[int, int, int, int], torch.Tensor] = {}

    def _hook(_module, _input, output):
        key = tuple(output.shape)
        if key not in _mask_cache:
            _mask_cache[key] = _gen_band_mask(
                key, r_min, r_max, fraction,
                per_channel=per_channel,
                device=output.device,
            )
        return output * _mask_cache[key]

    target_layer.register_forward_hook(_hook)


def _gen_graded_mask(
    shape: Tuple[int, int, int, int],
    *,
    r_start: float,
    r_end: float,
    p_min: float,
    p_max: float,
    mode: str = "dropout",      # "dropout" (stochastic) or "scale" (deterministic attenuation)
    profile: str = "linear",    # "linear" | "cosine" | "exp" | "poly"
    per_channel: bool = False,
    poly_deg: float = 2.0,
    exp_k: float = 4.0,
    reverse: bool = False,      # if True, swap p_min/p_max (fovea-heavier)
    device=None
) -> torch.Tensor:
    """
    Create a *graded* spatial mask based on normalised eccentricity.
    Returns a multiplicative mask with shape (1,C,H,W) or (1,1,H,W).
    - mode='dropout': entries are 0/1 (Bernoulli keep), prob( drop ) = p(r).
    - mode='scale'  : entries are continuous in (0,1], scale = 1 - p(r).
    """
    _, C, H, W = shape
    rmap = _build_radial_map(H, W, device=device)  # [H,W], 0 centre → 1 corner

    # normalise eccentricity into [0,1] over [r_start, r_end]
    denom = max(1e-6, (r_end - r_start))
    t = torch.clamp((rmap - r_start) / denom, 0.0, 1.0)

    # choose profile for how p(r) grows with t
    if profile == "linear":
        w = t
    elif profile == "cosine":
        w = 0.5 - 0.5 * torch.cos(math.pi * t)
    elif profile == "exp":
        w = (torch.exp(exp_k * t) - 1.0) / (torch.exp(exp_k) - 1.0)
    elif profile == "poly":
        w = t**poly_deg
    else:
        raise ValueError(f"Unknown profile '{profile}'")

    if reverse:
        p_min, p_max = p_max, p_min  # fovea-heavier when reverse=True

    p = p_min + (p_max - p_min) * w  # [H,W]

    if mode == "dropout":
        # keep prob = 1 - p
        if per_channel:
            keep = torch.bernoulli((1.0 - p).unsqueeze(0).repeat(C, 1, 1).to(device))
            return keep.unsqueeze(0)                          # (1,C,H,W)
        else:
            keep = torch.bernoulli((1.0 - p).to(device))
            return keep.unsqueeze(0).unsqueeze(0)             # (1,1,H,W)
    elif mode == "scale":
        scale = (1.0 - p).to(device)                          # (H,W)
        if per_channel:
            return scale.unsqueeze(0).repeat(1, C, 1, 1)      # (1,C,H,W)
        else:
            return scale.unsqueeze(0).unsqueeze(0)            # (1,1,H,W)
    else:
        raise ValueError(f"Unknown mode '{mode}'")


def apply_eccentricity_graded(
    model,
    *,
    layer_path: str,
    r_start: float,
    r_end: float,
    p_min: float,
    p_max: float,
    mode: str = "dropout",
    profile: str = "linear",
    per_channel: bool = False,
    poly_deg: float = 2.0,
    exp_k: float = 4.0,
    reverse: bool = False
):
    """
    Forward hook that applies graded eccentricity damage.
    """
    assert 0.0 <= r_start < r_end <= 1.0
    assert 0.0 <= p_min <= 1.0 and 0.0 <= p_max <= 1.0

    target = get_layer_from_path(model, layer_path)
    _cache = {}

    def _hook(_m, _in, out):
        key = tuple(out.shape)
        if key not in _cache:
            _cache[key] = _gen_graded_mask(
                key, r_start=r_start, r_end=r_end, p_min=p_min, p_max=p_max,
                mode=mode, profile=profile, per_channel=per_channel,
                poly_deg=poly_deg, exp_k=exp_k, reverse=reverse, device=out.device
            )
        return out * _cache[key]

    target.register_forward_hook(_hook)

def _as_list(x):
    if x is None:
        return []
    return x if isinstance(x, (list, tuple)) else [x]


def is_conv_like(layer):
    """
    Returns True if the given layer is a convolutional layer.
    Supports 1D, 2D, and 3D convs.
    """
    return isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d))


def resolve_selectivity_table(path_or_dir: str | Path,
                              model_tag: str | None = None) -> Path:
    """
    Resolve the combined selectivity table path.

    Accepts:
      • direct file path (.pkl/.csv) -> returned as Path
      • a directory (e.g. 'unit_selectivity/') -> tries:
          <model_tag>_all_layers_units_mannwhitneyu.pkl
          <model_tag>_all_layers_units_mannwhitneyu.csv
          all_layers_units_mannwhitneyu.pkl
          all_layers_units_mannwhitneyu.csv
    Raises FileNotFoundError if nothing is found.
    """
    p = Path(path_or_dir)

    # direct file
    if p.is_file() and p.suffix.lower() in (".pkl", ".csv"):
        return p

    # directory search
    if p.is_dir():
        candidates: list[Path] = []
        if model_tag:
            candidates += [p / f"{model_tag}_all_layers_units_mannwhitneyu.pkl",
                           p / f"{model_tag}_all_layers_units_mannwhitneyu.csv"]
        candidates += [p / "all_layers_units_mannwhitneyu.pkl",
                       p / "all_layers_units_mannwhitneyu.csv"]
        for c in candidates:
            if c.exists():
                return c

        raise FileNotFoundError(
            f"No selectivity table found under '{p}'. "
            "Tried model-tagged and legacy filenames."
        )

    # last resort: treat as a path that should exist
    if p.exists():
        return p
    raise FileNotFoundError(f"Selectivity path '{p}' not found.")