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
    # ... (existing config loading logic remains the same)
    if len(sys.argv) > 1:
        config_filename = sys.argv[1]
    else:
        config_filename = "cornet_s_it.yaml"

    config_path = f"configs/{config_filename}"

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
    masking_level = config.get("masking_level", "connections")
    mc_permutations = config.get("mc_permutations", 100)
    noise_levels_params = config.get("noise_levels", [0.1, 0, 0.1])
    include_bias = config.get("include_bias", False)
    only_conv = config.get("only_conv", True)
    # Add the new parameter for GroupNorm scaling
    groupnorm_scaling_params = config.get("groupnorm_scaling", [1.0, 1, 0.0])
    gain_control_noise = config.get("gain_control_noise", 0.0)
    groupnorm_scaling_targets = config.get(
    "groupnorm_scaling_targets",
    ["groupnorm"]                   
)
    # New eccentricity params
    eccentricity_layer_path       = config.get("eccentricity_layer_path", None)
    eccentricity_bands            = config.get("eccentricity_bands", [])   # e.g. [[0.60, 1.00]]
    ecc_fraction_to_mask_params          = config.get("ecc_fraction_to_mask",[0, 21, 0.05])
    # -----------------------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    common_kwargs = dict(                     
        model_info              = model_info,
        pretrained              = pretrained,
        fraction_to_mask_params = fraction_to_mask_params,
        noise_levels_params     = noise_levels_params,
        groupnorm_scaling_params= groupnorm_scaling_params,
        layer_paths_to_damage   = layer_paths_to_damage,
        apply_to_all_layers     = apply_to_all_layers,
        mc_permutations         = mc_permutations,
        layer_name              = layer_name,
        activation_layers_to_save = layer_path,
        image_dir               = image_dir,
        only_conv               = only_conv,
        include_bias            = include_bias,
        masking_level           = masking_level,
        groupnorm_scaling_targets = groupnorm_scaling_targets
    )

    # Noise ---------------------------------------------------------
    run_damage(**common_kwargs, manipulation_method="noise")

    # Connection / unit masking ------------------------------------
    run_damage(**common_kwargs, manipulation_method="connections")

    # GroupNorm scaling --------------------------------------------
    run_damage(
        **common_kwargs,
        manipulation_method="groupnorm_scaling",
        gain_control_noise=gain_control_noise
    )

    # Eccentricity-based activation thinning ------------------
    #     (creates sub-folders eccentricity_0.60-1.00/)
    run_damage(
        **common_kwargs,
        manipulation_method="eccentricity",
        eccentricity_layer_path = eccentricity_layer_path,
        eccentricity_bands      = eccentricity_bands,
        ecc_fraction_to_mask_params    = ecc_fraction_to_mask_params
    )

    # Eccentricity-based activation thinning (graded)
    run_damage(
        **common_kwargs,
        manipulation_method="eccentricity_gradual",
        eccentricity_layer_path = eccentricity_layer_path,
        eccentricity_bands      = config.get("eccentricity_bands_gradual", eccentricity_bands),
        ecc_fraction_to_mask_params = config.get("ecc_fraction_to_mask_gradual", ecc_fraction_to_mask_params),
        ecc_profile             = config.get("ecc_profile", "linear"),
        ecc_mode                = config.get("ecc_mode", "dropout"),
        ecc_per_channel         = config.get("ecc_per_channel", False),
        ecc_poly_deg            = config.get("ecc_poly_deg", 2.0),
        ecc_exp_k               = config.get("ecc_exp_k", 4.0),
        ecc_reverse             = config.get("ecc_reverse", False)
    )
    
if __name__ == "__main__":
    main()