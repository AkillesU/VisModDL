import os
import re
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from torchinfo import summary
import yaml


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
            model = cornet_z(pretrained=pretrained)

        elif model_name == "cornet_s":
            from cornet import cornet_s
            model = cornet_s(pretrained=pretrained)

        elif model_name == "cornet_rt":
            from cornet import cornet_rt
            model = cornet_rt(pretrained=pretrained)

        else:
            raise ValueError(f"CORnet model {model_name} not found. Check config file.")

    elif model_source == "pytorch_hub":
        if model_weights == "":
            model = torch.hub.load(model_repo, model_name)
        else:
            model = torch.hub.load(model_repo, model_name, weights=model_weights)
    else:
        raise ValueError(f"Check model source: {model_source}")
    
    # Print model summary
    print(model)

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
                weight_layer_paths = get_all_weight_layers(target_layer, path)
            
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
            weight_layer_paths = get_all_weight_layers(get_layer_from_path(model, path), path)
        
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
    save_path = f"figures/haupt_stim_activ/{model_name}/{layer_name}.png"
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
        save_path = f"figures/haupt_stim_activ/{model_name}/{layer_name}_within-between.yaml"
    else:
        save_path = output_path

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 4) Dump the *native* version of results to YAML
    with open(save_path, "w") as f:
        yaml.safe_dump(native_results, f, sort_keys=False)


def run_alteration(
    model_info, 
    pretrained,
    fraction_to_mask_list, 
    layer_paths_to_mask,
    apply_to_all_layers,
    masking_level,
    n_bootstrap,
    layer_name,
    layer_path,
    image_dir,
    vmax
):
    """
    Run the masking experiment for a list of fractions to mask, 
    and generate RDMs + within/between correlations.
    """
    from utils import (
        load_model,
        extract_activations,
        sort_activations_by_numeric_index,
        compute_correlations,
        assign_categories,
        bootstrap_correlations,
        print_within_between,
        apply_masking
    )

    num_permutations = len(fraction_to_mask_list)
    fig, axes = plt.subplots(1, num_permutations, figsize=(5 * num_permutations, 5))

    for i, fraction_to_mask in enumerate(fraction_to_mask_list):
        # Load a fresh model for each mask level
        model, activations = load_model(model_info, pretrained=pretrained, 
                                        layer_name=layer_name, layer_path=layer_path)

        if fraction_to_mask > 0.0:
            apply_masking(
                model, 
                fraction_to_mask, 
                layer_paths=layer_paths_to_mask, 
                apply_to_all_layers=apply_to_all_layers, 
                masking_level=masking_level
            )

        # Extract activations
        activations_df = extract_activations(model, activations, image_dir, layer_name=layer_name)
        activations_df_sorted = sort_activations_by_numeric_index(activations_df)
        correlation_matrix, sorted_image_names = compute_correlations(activations_df_sorted)

        # Plot RDM for this fraction_to_mask
        ax = axes[i] if num_permutations > 1 else axes
        im = ax.imshow(correlation_matrix, cmap="viridis", vmax=vmax, vmin=0)
        ax.set_title(f"RDM (Fraction={fraction_to_mask})")
        ax.set_xticks(range(len(sorted_image_names)))
        ax.set_yticks(range(len(sorted_image_names)))
        ax.set_xticklabels(sorted_image_names, rotation=90, fontsize=6)
        ax.set_yticklabels(sorted_image_names, fontsize=6)
        ax.set_xlabel("Images")
        ax.set_ylabel("Images")

        # Compute within-between metrics
        categories_array = assign_categories(sorted_image_names)
        results = bootstrap_correlations(correlation_matrix, categories_array, n_bootstrap=n_bootstrap)

        # Print & save results
        yaml_filename = (
            f"figures/haupt_stim_activ/damaged/{model_info['name']}/"
            f"{layer_name}_within-between_{fraction_to_mask}_{masking_level}.yaml"
        )
        print_within_between(results, layer_name=layer_name, model_name=model_info["name"], output_path=yaml_filename)

    # Final figure formatting
    plt.tight_layout()
    figure_path = (
        f"figures/haupt_stim_activ/damaged/{model_info['name']}/{layer_name}_RDMs.png"
    )
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    plt.savefig(figure_path, dpi=300)
    plt.show()