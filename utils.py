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


def get_all_weight_layers(model, base_path=""):
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
    # Check if current module has a 'weight' parameter
    if hasattr(model, 'weight') and model.weight is not None:
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


def get_all_conv_layers(model, base_path=""):
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
        if getattr(model, 'weight', None) is not None:
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


def load_model(model_info: dict, pretrained=True, layer_name='IT', layer_path=""):
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
            model = cornet_rt(pretrained=pretrained, map_location=(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")))

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

    # Access the target layer and register forward hook
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


def apply_noise(model, noise_level, layer_paths=None, apply_to_all_layers=False):
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
                weight_layer_paths = get_all_conv_layers(target_layer, path)
                for w_path in weight_layer_paths:
                    w_layer = get_layer_from_path(model, w_path)
                    if hasattr(w_layer, 'weight') and w_layer.weight is not None:
                        noise = torch.randn_like(w_layer.weight) * noise_level
                        w_layer.weight += noise
                    else:
                        raise AttributeError(f"layer {w_path} does not have weights")


def apply_masking(model, fraction_to_mask, layer_paths=None, apply_to_all_layers=False, masking_level='connections'):
    """
    Apply masking to either units or connections.
    If masking_level='connections', randomly zero out a fraction of weights individually.
    If masking_level='units', randomly zero out entire units.
    """
    param_masks = {}

    with torch.no_grad():
        if apply_to_all_layers:
            for name, param in model.named_parameters():
                if 'weight' in name and param.requires_grad:
                    mask = create_mask(param, fraction_to_mask, masking_level=masking_level)
                    param_masks[name] = mask
        else:
            # For each specified path, find all sub-layers that have weights
            for path in layer_paths:
                target_layer = get_layer_from_path(model, path)
                weight_layer_paths = get_all_conv_layers(target_layer, path)
            
                # weight_layer_paths is a list of layer-path strings
                for w_path in weight_layer_paths:
                    w_layer = get_layer_from_path(model, w_path)
                    if hasattr(w_layer, 'weight') and w_layer.weight is not None:
                        mask = create_mask(w_layer.weight, fraction_to_mask, masking_level=masking_level)
                        param_masks[w_path] = mask
                    else:
                        raise AttributeError(f"layer {w_path} does not have weights")
    # For simplicity if apply_to_all_layers is complex, let's just handle layer_paths case well.
    if not apply_to_all_layers:
        for path in layer_paths:
            # Get all weight-bearing sub-layers under this path
            weight_layer_paths = get_all_conv_layers(get_layer_from_path(model, path), path)
        
            for w_path in weight_layer_paths:
                layer = get_layer_from_path(model, w_path)
                mask = param_masks[w_path]  # Access the mask using w_path, not path
            
                def layer_mask_hook(module, input, mask=mask):
                    if hasattr(module, 'weight'):
                        module.weight.data = module.weight.data * mask
                    return None

                layer.register_forward_pre_hook(layer_mask_hook)
    else:
        # Global masking approach for all layers:
        param_to_mask = {}
        for name, param in model.named_parameters():
            if name in param_masks:
                param_to_mask[param] = param_masks[name]

        def global_mask_hook(module, input):
            for name, param in module.named_parameters(recurse=False):
                if param in param_to_mask:
                    param.data = param.data * param_to_mask[param]
            return None

        for m in model.modules():
            if len(list(m.parameters(recurse=False))) > 0:
                m.register_forward_pre_hook(global_mask_hook)


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

       
def run_damage(
    model_info,
    pretrained,
    fraction_to_mask_params,
    noise_levels_params,
    layer_paths_to_damage,
    apply_to_all_layers,
    manipulation_method,   # "connections" or "noise"
    mc_permutations,
    layer_name,
    layer_path,
    image_dir
):
    """
    Loads a model, damages it (masking or noise) multiple times,
    computes and saves correlation matrices + within/between metrics.

    Parameters:
    -----------
    model_info : dict
    pretrained : bool
    fraction_to_mask_params : list  # [start, length, step] if "connections"
    noise_levels_params : list      # [start, length, step] if "noise"
    layer_paths_to_damage : list    # Which layers to damage
    apply_to_all_layers : bool
    manipulation_method : str       # "connections" or "noise"
    mc_permutations : int           # Number of Monte Carlo permutations
    layer_name : str                # For hooking activations
    layer_path : str                # For hooking activations
    image_dir : str                 # Directory of images
    """

    # Determine the list of damage levels
    if manipulation_method == "connections":
        damage_levels_list = generate_params_list(fraction_to_mask_params)
    elif manipulation_method == "noise":
        damage_levels_list = generate_params_list(noise_levels_params)
    else:
        raise ValueError("manipulation_method must be 'connections' or 'noise'.")

    # We'll have len(damage_levels_list) * mc_permutations total loops
    total_iterations = len(damage_levels_list) * mc_permutations

    with tqdm(total=total_iterations, desc="Running alteration") as pbar:
        # Outer loop over each damage level (fraction or noise STD)
        for damage_level in damage_levels_list:
            # Inner loop over Monte Carlo permutations
            for permutation_index in range(mc_permutations):
                # 1) Load fresh model & activations dict
                model, activations = load_model(
                    model_info,
                    pretrained=pretrained,
                    layer_name=layer_name,
                    layer_path=layer_path
                )

                # 2) Apply the chosen damage
                if manipulation_method == "connections":
                    apply_masking(
                        model,
                        fraction_to_mask=damage_level,
                        layer_paths=layer_paths_to_damage,
                        apply_to_all_layers=apply_to_all_layers,
                        masking_level='connections'
                    )
                else:  # "noise"
                    apply_noise(
                        model,
                        noise_level=damage_level,
                        layer_paths=layer_paths_to_damage,
                        apply_to_all_layers=apply_to_all_layers
                    )

                # 3) Forward pass on images to get activations
                activations_df = extract_activations(
                    model, activations, image_dir, layer_name=layer_name
                )
                activations_df_sorted = sort_activations_by_numeric_index(activations_df)

                # 3.5) Save activations
                activation_dir = (
                    f"data/haupt_stim_activ/damaged/{model_info['name']}/"
                    f"{manipulation_method}/{layer_name}/activations/damaged_{round(damage_level,3)}"
                )
                os.makedirs(activation_dir, exist_ok=True)
                activation_dir_path = os.path.join(activation_dir, f"{permutation_index}.pkl")

                # 1) Convert to float16 (optional)
                reduced_df = activations_df_sorted.astype(np.float16)
                reduced_df.to_pickle(activation_dir_path)

                # 4) Compute correlation matrix
                correlation_matrix, sorted_image_names = compute_correlations(activations_df_sorted)

                # 5) Save correlation matrix
                corrmat_dir = (
                    f"data/haupt_stim_activ/damaged/{model_info['name']}/"
                    f"{manipulation_method}/{layer_name}/RDM/damaged_{round(damage_level,3)}"
                )
                os.makedirs(corrmat_dir, exist_ok=True)
                corrmat_path = os.path.join(corrmat_dir, f"{permutation_index}.pkl")

                with open(corrmat_path, "wb") as f:
                    pickle.dump(correlation_matrix.tolist(), f)

                # 6) Compute within-between metrics
                categories_array = assign_categories(sorted_image_names)
                results = calc_within_between(correlation_matrix, categories_array)
                results = convert_np_to_native(results)
                # 7) Save within-between metrics
                selectivity_dir = (
                    f"data/haupt_stim_activ/damaged/{model_info['name']}/"
                    f"{manipulation_method}/{layer_name}/selectivity/damaged_{round(damage_level,3)}"
                )
                os.makedirs(selectivity_dir, exist_ok=True)
                selectivity_path = os.path.join(selectivity_dir, f"{permutation_index}.pkl")

                with open(selectivity_path, "wb") as f:
                    pickle.dump(results, f)

                """#  print summary
                print(f"[Damage: {damage_level}, Perm: {permutation_index}] -> "
                      f"RDM saved: {corrmat_path} | Selectivity saved: {selectivity_path}")
"""
                # Update the progress bar
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
    layers,
    damage_type,
    main_dir="data/haupt_stim_activ/damaged/cornet_rt/",
    categories=["animal", "face", "object", "place", "total"],
    metric="observed_difference",
    subdir_regex=r"damaged_([\d\.]+)$",
    plot_dir="plots/",
    verbose=0 # 0 or 1
):
    """
    1. For each layer in `layers`, build the directory path:
         {main_dir}{damage_type}/{layer}/selectivity/
    2. In that directory, for each subdirectory matching subdir_regex (e.g. "damaged_0.1"),
       parse the numeric fraction (e.g., 0.1).
    3. If an aggregated stats file "avg_selectivity_{fraction}.yaml" does not exist for
       that fraction, compute *all* categories & metrics found in the raw .yaml files.
       Save them to "avg_selectivity_{fraction}.yaml".
    4. For plotting, read back those aggregated stats, but only use the specified
       `categories` and single `metric` to populate `data[(layer, cat)]`.
    5. Plot them on a single figure, with one line per (layer, category).
    """

    # data[(layer, category)] -> { fraction_value : (mean, std) }
    data = {}

    # -- Loop over each layer to gather data --
    for layer in layers:
        # Construct the directory for this layer
        layer_path = os.path.join(main_dir, damage_type, layer, "selectivity")
        # Construct output file path to save aggregated Pickle
        output_path = os.path.join(main_dir, damage_type, layer, "avg_selectivity")
        os.makedirs(output_path, exist_ok=True)

        if not os.path.isdir(layer_path):
            # If the path doesn't exist, skip this layer
            continue

        # Initialize data dict for each (layer, cat) combination
        for cat in categories:
            data[(layer, cat)] = {}

        # 1) Loop over all subdirectories in layer_path
        for subdir_name in os.listdir(layer_path):
            subdir_path = os.path.join(layer_path, subdir_name)
            if not os.path.isdir(subdir_path):
                continue

            # Extract the numeric portion from the subdirectory name
            match = re.search(subdir_regex, subdir_name)
            if not match:
                continue  # subdir doesn't match, skip it

            fraction_raw = float(match.group(1))
            fraction_rounded = round(fraction_raw, 3)

            # The aggregated stats file name for this fraction
            fraction_file_name = f"avg_selectivity_{fraction_rounded}.pkl"
            fraction_file_path = os.path.join(output_path, fraction_file_name)

            # Check if the aggregated file for this fraction exists
            if not os.path.exists(fraction_file_path):
                # If it doesn't exist, we aggregate *all* categories & metrics
                # found in the .pkl files in this subdirectory.
                aggregated_data = {}  # aggregated_data[cat][metric] = list of values

                # 2) Look for .pkl files in the subdirectory
                for fname in os.listdir(subdir_path):
                    if fname.lower().endswith(".pkl"):
                        pkl_path = os.path.join(subdir_path, fname)
                        with open(pkl_path, "rb") as f:
                            content = pickle.load(f)

                        # Go through every category and metric in this file
                        # and store them in aggregated_data
                        if not isinstance(content, dict):
                            continue
                        for cat_name, metrics_dict in content.items():
                            if not isinstance(metrics_dict, dict):
                                continue
                            # Initialize if needed
                            if cat_name not in aggregated_data:
                                aggregated_data[cat_name] = {}
                            for metric_name, val in metrics_dict.items():
                                # Ensure we have a list to accumulate values
                                aggregated_data[cat_name].setdefault(metric_name, [])
                                aggregated_data[cat_name][metric_name].append(val)

                # Now compute the mean & std for every category and metric encountered
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

                # Write these aggregated stats to the fraction_file_path
                with open(fraction_file_path, "wb") as f:
                    pickle.dump(stats_dict, f)

            aggregated_content = safe_load_pickle(fraction_file_path)
            """# 3) Now read back the aggregated file (either just written or previously existing)
            with open(fraction_file_path, "rb") as f:
                aggregated_content = pickle.load(f)"""

            # 4) For plotting, we only care about the user-specified categories & single metric
            if not isinstance(aggregated_content, dict):
                continue

            for cat in categories:
                if cat in aggregated_content:
                    # The aggregated file might have multiple metrics; we only use `metric`
                    if (
                        isinstance(aggregated_content[cat], dict)
                        and metric in aggregated_content[cat]
                    ):
                        mean_val = aggregated_content[cat][metric]["mean"]
                        std_val = aggregated_content[cat][metric]["std"]
                        data[(layer, cat)][fraction_rounded] = (mean_val, std_val)

    # 5) Prepare a single figure with one line per (layer, category)
    plt.figure(figsize=(8, 6))

    for (layer, cat), fraction_dict in data.items():
        if len(fraction_dict) == 0:
            # This (layer, category) had no data at all, skip plotting
            continue

        # Sort the fractions in ascending order
        fractions_sorted = sorted(fraction_dict.keys())

        x_vals = []
        y_means = []
        y_stds = []

        # Extract the (mean, std) directly
        for frac in fractions_sorted:
            mean_val, std_val = fraction_dict[frac]
            x_vals.append(frac)
            y_means.append(mean_val)
            y_stds.append(std_val)

        # Plot error bars for this (layer, category)
        plt.errorbar(
            x_vals, 
            y_means, 
            yerr=y_stds, 
            fmt='-o', 
            capsize=4, 
            label=f"{layer} - {cat} ({metric})"
        )

    plt.xlabel("Damage Parameter Value")
    plt.ylabel(metric)
    plt.title("Correlation metric across damage parameter values")
    plt.legend()
    plt.tight_layout()

    # Saving plot
    os.makedirs(plot_dir, exist_ok=True)

    # Build a unique plot name
    model_name = main_dir.split("/")[-2] # Assuming that there is a slash after the model name ("/cornet/")
    plot_name = f"{model_name}_lineplot_{damage_type}"
    for layer in layers:
        plot_name += f"_{layer}"
    for category in categories:
        plot_name += f"_{category[0]}"
    plot_name += f"_{metric}.png"

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
    main_dir="data/haupt_stim_activ/damaged/cornet_rt/",
    damage_type="connections",
    layers=["V1","V2","V4","IT"],
    image_dir="stimuli/",
    mode='dirs',
    file_prefix='damaged_',
    damage_levels=["0"],
    comparison=False,
    plot_dir="plots/",
    verbose=0
):
    """
    Plot mean (with std error bars) of within-vs-between-category correlations,
    excluding the diagonal. Each "item" becomes a row in the final plot
    unless comparison=True, in which case all items are plotted together.

    Parameters
    ----------
    main_dir : str
        Base directory. Default is "data/haupt_stim_activ/damaged_cornet_rt/".
    damage_type : str
        Subdirectory under main_dir specifying a particular damage type.
    layers : list of str
        List of layer names to search within. Each layer becomes another 
        subdirectory under damage_type.
    image_dir : str
        Directory containing image filenames (used to determine categories).
    mode : {'files', 'dirs'}
        If 'files', each matching file is one item. If 'dirs', each matching 
        directory is one item.
    file_prefix : str
        Only consider files/subdirs that start with this prefix.
    damage_levels : list of str
        Only consider files/subdirs ending with these suffixes (e.g. multiple damage fractions).
    comparison : bool
        If False (default), each (layer, damage_level) combination is plotted 
        as its own row of subplots (the original behavior).
        If True, all (layer, damage_level) combinations share the same subplots, 
        so there is only one subplot per category (grouped bar chart).

    Raises
    ------
    FileNotFoundError
        If directories or files do not exist.
    ValueError
        If any correlation matrix is non-square or doesn't match the image count.
    """

    # ------------- HELPER FUNCTIONS -------------
    def extract_string_numeric_parts(fname):
        """Extracts leading string and trailing numeric part from filename."""
        import re
        match = re.match(r"([^\d]+)([\d\.]*)", os.path.splitext(fname)[0])
        if match:
            return (match.group(1), match.group(2))
        else:
            return (fname, "")

    def get_sorted_filenames(folder):
        """Return sorted list of filenames, ignoring subdirectories."""
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Image directory '{folder}' does not exist.")
        files = [
            f for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))
        ]
        return sorted(files)

    # --------------------------------------------

    if layers is None:
        layers = []

    # -------------------------------------------------------------------------
    # 1. Load and sort all image filenames, then group by category
    # -------------------------------------------------------------------------
    if image_dir is None or not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory '{image_dir}' does not exist.")

    sorted_filenames = get_sorted_filenames(image_dir)
    n_files = len(sorted_filenames)

    # Create a dict: {category: [filenames]}
    categories = {}
    for fname in sorted_filenames:
        cat = extract_string_numeric_parts(fname)[0]
        categories.setdefault(cat, []).append(fname)

    categories_list = sorted(categories.keys())
    n_categories = len(categories_list)

    # -------------------------------------------------------------------------
    # 2. Helpers to validate and load correlation matrices
    # -------------------------------------------------------------------------
    def _validate_matrix(mat, path_str):
        """Ensures 'mat' is square and matches the number of sorted_filenames."""
        if len(mat) != n_files:
            raise ValueError(
                f"Matrix in '{path_str}' has {len(mat)} rows, "
                f"but there are {n_files} image files."
            )
        for row in mat:
            if len(row) != n_files:
                raise ValueError(f"Non-square row in '{path_str}'.")

    def load_matrices_from_item(item_path):
        """
        If 'item_path' is a file -> load that single YAML file, return [one matrix].
        If 'item_path' is a directory -> load all matching YAML files, return list of matrices.
        """
        if os.path.isfile(item_path):
            # Single file
            with open(item_path, 'rb') as f:
                mat = pickle.load(f)
            _validate_matrix(mat, item_path)
            return [mat]

        elif os.path.isdir(item_path):
            # Possibly multiple .pkl files in this directory
            allfiles = sorted(os.listdir(item_path))
            matching = [os.path.join(item_path, x) for x in allfiles if x.endswith(".pkl")]
            matrices = []
            for mfile in matching:
                with open(mfile, 'rb') as f:
                    mat = pickle.load(f)
                _validate_matrix(mat, mfile)
                matrices.append(mat)
            return matrices

        else:
            raise FileNotFoundError(f"'{item_path}' is neither a file nor directory.")

    # -------------------------------------------------------------------------
    # 3. Compute within-vs-between differences for a list of correlation matrices
    # -------------------------------------------------------------------------
    def compute_differences_across_matrices(matrices):
        """
        For a list of correlation matrices, compute the within-vs-between difference
        for each category (excluding diagonals), then return mean +- 1.96*std.
        
        Returns: dict[category] = (other_cats, mean_diffs, std_diffs)
        """
        accum = {
            cat: {oc: [] for oc in categories_list if oc != cat}
            for cat in categories_list
        }

        for mat in matrices:
            single_res = {}
            for cat in categories_list:
                within = []
                between = {oc: [] for oc in categories_list if oc != cat}

                for r_i, row in enumerate(mat):
                    cat_r = extract_string_numeric_parts(sorted_filenames[r_i])[0]
                    for c_i, val in enumerate(row):
                        if r_i == c_i:
                            continue  # exclude diagonal
                        cat_c = extract_string_numeric_parts(sorted_filenames[c_i])[0]

                        if cat_r == cat and cat_c == cat:
                            within.append(val)
                        elif cat_r == cat and cat_c != cat:
                            between[cat_c].append(val)
                        elif cat_c == cat and cat_r != cat:
                            between[cat_r].append(val)

                w_avg = np.mean(within) if within else 0.0
                b_avg = {k: (np.mean(v) if v else 0.0) for k, v in between.items()}
                oc_list = [x for x in categories_list if x != cat]
                diffs = [w_avg - b_avg[x] for x in oc_list]
                single_res[cat] = (oc_list, diffs)

            # Accumulate
            for cat, (ocs, diffvals) in single_res.items():
                for oc, dv in zip(ocs, diffvals):
                    accum[cat][oc].append(dv)

        # Convert to mean 1.96*std
        result = {}
        for cat in accum:
            other_cats = sorted(accum[cat].keys())
            mean_vals = []
            std_vals = []
            for oc in other_cats:
                arr = np.array(accum[cat][oc])
                mean_vals.append(arr.mean())
                std_vals.append(arr.std() * 1.96)  # 95% CI
            result[cat] = (other_cats, mean_vals, std_vals)
        return result

    # -------------------------------------------------------------------------
    # 4. Gather all (layer, suffix, item_path, diffs_dict)
    # -------------------------------------------------------------------------
    if not layers:
        raise ValueError("No layers specified. Provide at least one layer.")

    all_results = []  # will hold tuples of (layer, suffix, item_path, diffs_dict)

    for layer in layers:
        base_path = os.path.join(main_dir, damage_type, layer, "RDM")
        if not os.path.isdir(base_path):
            raise FileNotFoundError(
                f"Layer directory '{base_path}' does not exist. "
                f"Check your 'damage_type' or 'layers' argument."
            )

        # Now loop over each suffix in damage_levels
        for suffix in damage_levels:
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
                raise ValueError("mode must be either 'files' or 'dirs'.")

            if not items:
                raise FileNotFoundError(
                    f"No matching items found in '{base_path}' "
                    f"for layer='{layer}', suffix='{suffix}', "
                    f"prefix='{file_prefix}', mode='{mode}'."
                )

            for item_path in items:
                mats = load_matrices_from_item(item_path)
                diffs_dict = compute_differences_across_matrices(mats)
                all_results.append((layer, suffix, item_path, diffs_dict))

    # -------------------------------------------------------------------------
    # 5. Plot the results
    # -------------------------------------------------------------------------
    if not comparison:
        # ------------------------------------------------
        # Original behavior: one row per (layer, suffix, item_path),
        # one column per category
        # ------------------------------------------------
        num_rows = len(all_results)
        fig, axes = plt.subplots(
            num_rows, n_categories,
            figsize=(3*n_categories, 3*num_rows),
            sharey=True
        )
        axes = np.array(axes, ndmin=2)  # ensure 2D array

        for i, (layer, suffix, item_path, diffs_dict) in enumerate(all_results):
            label = os.path.basename(item_path.rstrip("/\\"))

            for j, cat in enumerate(categories_list):
                ax = axes[i, j]
                if cat not in diffs_dict:
                    ax.set_visible(False)
                    continue

                other_cats, mean_vals, std_vals = diffs_dict[cat]
                x_pos = np.arange(len(other_cats))

                ax.bar(
                    x_pos, mean_vals,
                    yerr=std_vals,
                    color='skyblue',
                    edgecolor='black',
                    capsize=4
                )
                ax.set_xticks(x_pos)
                ax.set_xticklabels(other_cats, rotation=45, ha='right')

                # Show layer, suffix, and the item label in the subplot title
                ax.set_title(f"{layer}, Damage={suffix}\n{cat}")
                if j == 0:
                    ax.set_ylabel("Avg(Within) - Avg(Between)")

        plt.tight_layout()
        # Saving plot
        os.makedirs(plot_dir, exist_ok=True)
        model_name = main_dir.split("/")[-2] # Assuming that there is a slash after the model name ("/cornet/")
        plot_name = f"{model_name}_categ-diff_{damage_type}"

        for layer in layers:
            plot_name = plot_name + f"_{layer}"

        n_damages = len(damage_levels)
    
        plot_name = plot_name + f"_{n_damages}-levels"

        if comparison:
            plot_name = plot_name + "comparison"

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

    else:
        # ------------------------------------------------
        # comparison=True: 
        # Plot all (layer, suffix, item_path) combos in the SAME subplots,
        # so there is only ONE subplot per category.
        # We'll create grouped bars for each "other cat" across all combos.
        # ------------------------------------------------
        # 1) Build a dictionary: results_by_cat[cat][(layer, suffix, item_path)] = (other_cats, mean_vals, std_vals)
        results_by_cat = {
            cat: {} for cat in categories_list
        }

        # Store the combos in a stable order
        combos = []  # list of (layer, suffix, item_path) to keep a consistent legend
        for (layer, suffix, item_path, diffs_dict) in all_results:
            combo_key = (layer, suffix, item_path)
            if combo_key not in combos:
                combos.append(combo_key)
            # Insert into results_by_cat
            for cat in categories_list:
                if cat in diffs_dict:
                    results_by_cat[cat][combo_key] = diffs_dict[cat]
                else:
                    # Possibly no data if cat wasn't in diffs_dict
                    results_by_cat[cat][combo_key] = ([], [], [])

        # 2) Create one subplot per category
        fig, axes = plt.subplots(
            1, n_categories,
            figsize=(4*n_categories, 4),
            sharey=True
        )
        if n_categories == 1:
            axes = [axes]  # ensure iterable if only one category

        for j, cat in enumerate(categories_list):
            ax = axes[j]
            ax.set_title(f"Comparison for Category: {cat}")
            ax.set_ylabel("Avg(Within) - Avg(Between)")

            # For this category, we have results_by_cat[cat], which is a dict:
            #   {(layer, suffix, item_path) -> (other_cats, mean_vals, std_vals)}
            # We'll assume the same "other_cats" in the same order for each combo,
            # though they might differ if some combos have different categories.  
            # We'll look at the first non-empty set to get the other_cats ordering.
            # If everything is empty, we skip plotting.

            # 2a) Find a consistent ordering of "other_cats"
            all_other_cats = None
            for combo_key in combos:
                (oc_list, mvals, svals) = results_by_cat[cat][combo_key]
                if oc_list:  # found a non-empty one
                    all_other_cats = oc_list
                    break

            if not all_other_cats:
                # No data found for this category
                ax.text(0.5, 0.5, f"No data for category '{cat}'", 
                        ha='center', va='center', transform=ax.transAxes)
                continue

            x_pos = np.arange(len(all_other_cats))  # base x positions for each "other cat"
            bar_width = 0.8 / len(combos)           # narrower bars for grouping
            # We'll have each combo's bar slightly offset

            for i, combo_key in enumerate(combos):
                (oc_list, mean_vals, std_vals) = results_by_cat[cat][combo_key]
                # It's possible oc_list might be empty if that combo had no data
                # or if it didn't compute differences for that category.
                # So we must handle that carefully:
                if len(oc_list) != len(all_other_cats):
                    # We'll fill with zeros if missing
                    # or skip plotting. Here we'll fill with zeros for demonstration:
                    yvals = np.zeros(len(all_other_cats))
                    yerrs = np.zeros(len(all_other_cats))
                else:
                    yvals = mean_vals
                    yerrs = std_vals

                # Create an offset for this combo:
                offset = i * bar_width
                ax.bar(
                    x_pos + offset, 
                    yvals, 
                    yerr=yerrs, 
                    width=bar_width, 
                    label=f"{combo_key[0]}, dmg={combo_key[1]}", 
                    capsize=4
                )

            ax.set_xticks(x_pos + bar_width*(len(combos)/2 - 0.5))
            ax.set_xticklabels(all_other_cats, rotation=45, ha='right')

        ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        # Saving plot
        os.makedirs(plot_dir, exist_ok=True)
        model_name = main_dir.split("/")[-2] # Assuming that there is a slash after the model name ("/cornet/")
        plot_name = f"{model_name}_categ-diff_{damage_type}"

        for layer in layers:
            plot_name = plot_name + f"_{layer}"

        n_damages = len(damage_levels)
    
        plot_name = plot_name + f"_{n_damages}-levels"

        if comparison:
            plot_name = plot_name + "comparison"

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


def pair_corr_scatter_grid(
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
                ax.scatter(act1, act2, alpha=0.4, color=color, s=10)

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

    fig.suptitle(f"Scatter correlation grid: {image1} vs {image2}")
    ax.set_xlim(left=lower_limit)
    ax.set_ylim(bottom=lower_limit)
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

