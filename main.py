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
    run_damage
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
    model_info = {}
    model_info["source"] = config.get("model_source", "cornet")
    model_info["repo"] = config.get("model_repo", "-")
    model_info["name"] = config.get("model_name", "cornet_z")
    model_info["weights"] = config.get("model_weights", "")
    if "model_time_steps" in config:
        model_info["time_steps"] = config["model_time_steps"]
    pretrained = config.get("pretrained", True)

    manipulation_method = config.get("manipulation_method", None)
    fraction_to_mask_params = config.get("fraction_to_mask", [0.05, 10, 0.05])
    layer_paths_to_damage = config.get("layer_paths_to_damage", [])
    apply_to_all_layers = config.get("apply_to_all_layers", False)
    masking_level = config.get("masking_level", "connections")  # "units" or "connections"
    mc_permutations = config.get("mc_permutations", 100) # N of Monte Carlo permutations
    noise_levels_params = config.get("noise_levels", [0.1, 0, 0.1])
    include_bias = config.get("include_bias", False)
    only_conv = config.get("only_conv", True)
    # -----------------------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_damage(model_info=model_info, 
        pretrained=pretrained,
        fraction_to_mask_params=fraction_to_mask_params,
        noise_levels_params=noise_levels_params,
        layer_paths_to_damage=layer_paths_to_damage,
        apply_to_all_layers=apply_to_all_layers,
        manipulation_method="noise",
        mc_permutations=mc_permutations,
        layer_name=layer_name,
        layer_path=layer_path,
        image_dir=image_dir,
        only_conv=only_conv,
        include_bias=include_bias)

    run_damage(model_info=model_info, 
        pretrained=pretrained,
        fraction_to_mask_params=fraction_to_mask_params,
        noise_levels_params=noise_levels_params,
        layer_paths_to_damage=layer_paths_to_damage,
        apply_to_all_layers=apply_to_all_layers,
        manipulation_method="connections",
        mc_permutations=mc_permutations,
        layer_name=layer_name,
        layer_path=layer_path,
        image_dir=image_dir,
        only_conv=only_conv,
        include_bias=include_bias)

if __name__ == "__main__":
    main()