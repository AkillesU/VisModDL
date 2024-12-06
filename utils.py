import os
import re
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

def load_model(model_info, pretrained=True, layer_name='IT'):
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
            raise ValueError(f"CORnet model {model_info["name"]} not found. Check config file.")

    elif model_source == "pytorch_hub":
        if model_weights == "":
            model = torch.hub.load(model_repo, model_name)
        else:
            model = torch.hub.load(model_repo, model_name, weights=model_weights)
    else:
        raise ValueError(f"Check model source: {model_source}")

    model.eval()

    activations = {}

    # Hook function to capture layer outputs
    def hook_fn(module, input, output):
        activations[layer_name] = output.cpu().detach().numpy()

    # Access the target layer and register forward hook
    target_layer = model.module._modules[layer_name]._modules['pool']
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

def sort_activations_by_numeric_index(activations_df):
    """
    Extract a numeric index from image names, sort the DataFrame by that index, and return sorted DataFrame.

    Parameters:
        activations_df (pd.DataFrame): DataFrame with image names as index.

    Returns:
        activations_df_sorted (pd.DataFrame): DataFrame sorted by extracted numeric index.
    """
    # Ensure index is string
    activations_df.index = activations_df.index.astype(str)
    # Extract numeric part of the filename
    activations_df['numeric_index'] = activations_df.index.str.extract(r'^(\d+)', expand=False)
    # Convert to int and handle any NaN
    activations_df['numeric_index'] = activations_df['numeric_index'].astype(int)
    activations_df_sorted = activations_df.sort_values('numeric_index')
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

def plot_correlation_heatmap(correlation_matrix, sorted_image_names, layer_name='IT', vmax=0.4):
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
                xticklabels=sorted_image_names, yticklabels=sorted_image_names, vmax=vmax)
    plt.title(f"Correlation of Activations Between Images (Layer: {layer_name})")
    plt.xlabel("Images")
    plt.ylabel("Images")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def assign_categories(sorted_image_names):
    """
    Assign category labels to images based on their numeric index.

    Parameters:
        sorted_image_names (list): Sorted list of image filenames.

    Returns:
        categories_array (np.ndarray): Array of category labels.
    """
    categories = []
    for image_name in sorted_image_names:
        image_number = int(image_name.split('.')[0])
        if 1 <= image_number <= 16:
            category = 1
        elif 17 <= image_number <= 32:
            category = 2
        elif 33 <= image_number <= 48:
            category = 3
        elif 49 <= image_number <= 64:
            category = 4
        else:
            raise ValueError(f"Image number {image_number} is out of expected range")
        categories.append(category)

    return np.array(categories)

def bootstrap_correlations(correlation_matrix, categories_array, n_bootstrap=10000):
    """
    Perform bootstrap analysis comparing within-category and between-category correlations.

    Parameters:
        correlation_matrix (np.ndarray): Correlation matrix of image activations.
        categories_array (np.ndarray): Array of category labels for each image.
        n_bootstrap (int): Number of bootstrap iterations.

    Returns:
        results (dict): Dictionary containing analysis results for each category.
    """
    results = {}

    for category_number in range(1, 5):
        # Indices of images in current category
        category_indices = np.where(categories_array == category_number)[0]
        other_indices = np.where(categories_array != category_number)[0]

        # Within-category correlations
        submatrix_within = correlation_matrix[np.ix_(category_indices, category_indices)]
        n_within = len(category_indices)
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

        results[category_number] = {
            'avg_within': avg_within,
            'avg_between': avg_between,
            'observed_difference': observed_difference,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }

    return results

def print_within_between(results):
    """
    Print the bootstrap results in a readable format.

    Parameters:
        results (dict): Dictionary containing analysis results for each category.
    """
    for category_number in range(1, 5):
        avg_within = results[category_number]['avg_within']
        avg_between = results[category_number]['avg_between']
        observed_difference = results[category_number]['observed_difference']
        p_value = results[category_number]['p_value']
        ci_lower = results[category_number]['ci_lower']
        ci_upper = results[category_number]['ci_upper']

        print(f"Category {category_number}:")
        print(f"  Average within-category correlation: {avg_within:.4f}")
        print(f"  Average between-category correlation: {avg_between:.4f}")
        print(f"  Observed difference (within - between): {observed_difference:.4f}")
        print(f"  95% Confidence interval for difference: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  P-value (one-tailed test): {p_value:.4f}\n")
