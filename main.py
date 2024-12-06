"""
This script loads a model specified in the config file and passes through a set of images in a specified folder.
The activation RDM from a specified layer is then visualised and the within-between category differences are printed.
"""

import sys
import yaml
import torch
from utils import (
    load_model, 
    extract_activations, 
    sort_activations_by_numeric_index,
    compute_correlations,
    plot_correlation_heatmap,
    assign_categories,
    bootstrap_correlations,
    print_within_between
)


def main():
    # Get config filename from arguments, or use default
    if len(sys.argv) > 1:
        config_filename = sys.argv[1]
    else:
        config_filename = "cornet_s_it.yaml"  # default config file

    # Prepend the configs directory
    config_path = f"configs/{config_filename}"

    # Load configuration parameters
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    image_dir = config.get("image_dir", "stimuli")
    layer_name = config.get("layer_name", "IT")
    n_bootstrap = config.get("n_bootstrap", 10000)
    model_info = {}
    model_info["source"] = config.get("model_source", "cornet")
    model_info["repo"] = config.get("model_repo", "-")
    model_info["name"] = config.get("model_name", "cornet_z")
    model_info["weights"] = config.get("weights", "")
    vmax = config.get("vmax", 0.4)
    pretrained = config.get("pretrained", True)

    # Load model and register hook
    model, activations = load_model(model_info, pretrained=pretrained, layer_name=layer_name)

    # Extract activations
    activations_df = extract_activations(model, activations, image_dir, layer_name=layer_name)

    # Sort activations by numeric index in filename
    activations_df_sorted = sort_activations_by_numeric_index(activations_df)

    # Compute correlations
    correlation_matrix, sorted_image_names = compute_correlations(activations_df_sorted)

    # Plot correlation heatmap
    plot_correlation_heatmap(correlation_matrix, sorted_image_names, layer_name=layer_name, vmax=vmax, model_name=model_info["name"])

    # Assign categories to images
    categories_array = assign_categories(sorted_image_names)

    # Bootstrap correlations for each category
    results = bootstrap_correlations(correlation_matrix, categories_array, n_bootstrap=n_bootstrap)

    # Print and save results 
    print_within_between(results, model_name=model_info["name"], layer_name=layer_name)


if __name__ == "__main__":
    main()