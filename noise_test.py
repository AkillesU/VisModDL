"""
This script is for testing and determining the correct amount of noise to inject to model weights.
The aim is to have a range of SDs of gaussian noise which on the lower end keep within-between difference
intact and on the higher end change it to 0 or close to 0. 
"""

###
### Seems like the upper end is around 0.1 SD Gaussian noise.
### Lower end probably around 0.01 and with 0.01 increments

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


# Get config filename from arguments, or use default
if len(sys.argv) > 1:
    config_filename = sys.argv[1]
else:
    config_filename = "cornet_rt_noise_test.yaml"  # default config file

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
fraction_to_mask_params = config.get("fraction_to_mask", [0.05, 10, 0.05])
layer_paths_to_damage = config.get("layer_paths_to_damage", [])
apply_to_all_layers = config.get("apply_to_all_layers", False)
masking_level = config.get("masking_level", "connections")  # "units" or "connections"
mc_permutations = config.get("mc_permutations", 100) # N of Monte Carlo permutations
noise_levels_params = config.get("noise_levels", [0.1, 10, 0.1])
# -----------------------------------------------------------------------



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
    image_dir=image_dir)
