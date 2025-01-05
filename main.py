"""
This script loads a model specified in the config file and passes through a set of images in a specified folder.
The activation RDM from a specified layer is then visualised and the within-between category differences are printed.
"""

import matplotlib.pyplot as plt
import copy
import random
import os
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
    print_within_between,
    apply_masking,
    run_alteration
)

random.seed(1234)

def main():
    # Get config filename from arguments, or use default
    if len(sys.argv) > 1:
        config_filename = sys.argv[1]
    else:
        config_filename = "cornet_s_it.yaml"  # default config file

    # Define the config path with the subdirectory predefined
    config_path = f"configs/{config_filename}"

    # LOAD CONFIG PARAMS
    # ----------------------------------------------------------------------
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    image_dir = config.get("image_dir", "stimuli")
    layer_name = config.get("layer_name", "IT")
    layer_path = config.get("layer_path", "")
    n_bootstrap = config.get("n_bootstrap", 10000)
    model_info = {}
    model_info["source"] = config.get("model_source", "cornet")
    model_info["repo"] = config.get("model_repo", "-")
    model_info["name"] = config.get("model_name", "cornet_z")
    model_info["weights"] = config.get("model_weights", "")
    vmax = config.get("vmax", 0.4)
    pretrained = config.get("pretrained", True)

    manipulation_method = config.get("manipulation_method", None)
    fraction_to_mask_list = config.get("fraction_to_mask", 0.0)
    layer_paths_to_mask = config.get("layer_paths_to_mask", [])
    apply_to_all_layers = config.get("apply_to_all_layers", False)
    masking_level = config.get("masking_level", "connections")  # "units" or "connections"
    # -----------------------------------------------------------------------


    # Prepare a figure with subplots for RDMs, one subplot per fraction or permutation
    num_permutations = len(fraction_to_mask_list)
    fig, axes = plt.subplots(1, num_permutations, figsize=(5 * num_permutations, 5))

    # If we are masking, just call the new function
    if manipulation_method == "masking":
        run_alteration(
            model_info=model_info, 
            pretrained=pretrained,
            fraction_to_mask_list=fraction_to_mask_list,
            layer_paths_to_mask=layer_paths_to_mask,
            apply_to_all_layers=apply_to_all_layers,
            masking_level=masking_level,
            n_bootstrap=n_bootstrap,
            layer_name=layer_name,
            layer_path=layer_path,
            image_dir=image_dir,
            vmax=vmax
        )

    # Load model and register hook
    model, activations = load_model(model_info, pretrained=pretrained, layer_name=layer_name, layer_path=layer_path)

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