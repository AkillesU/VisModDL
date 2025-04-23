"""
Run all damage jobs listed under `damage_jobs:` in a YAML.
Automatically honours include_bias per job.
"""

import os, copy, sys, yaml, torch
from utils import run_damage_imagenet

def main(cfg_file):
    cfg = yaml.safe_load(open(cfg_file))

    # global knobs
    global_keys = {k: v for k, v in cfg.items() if k != "damage_jobs"}
    model_info = {
        "source":  global_keys.get("model_source", "cornet"),
        "repo":    global_keys.get("model_repo", "-"),
        "name":    global_keys.get("model_name", "cornet_rt"),
        "weights": global_keys.get("model_weights", "")
    }
    if "model_time_steps" in global_keys:
        model_info["time_steps"] = global_keys["model_time_steps"]

    if "torch_threads" in global_keys:
        torch.set_num_threads(int(global_keys["torch_threads"]))

    for job in cfg["damage_jobs"]:
        # merge job-specific keys over the globals
        params = copy.deepcopy(global_keys)
        params.update(job)                 # job overrides/extends

        print("\n=== Running damage job:", job, "===")
        run_damage_imagenet(
            model_info           = model_info,
            pretrained           = params.get("pretrained", True),
            fraction_to_mask_params = params.get("fraction_to_mask", [0,0,0]),
            noise_levels_params     = params.get("noise_levels",    [0,0,0]),
            layer_paths_to_damage   = params["layer_paths_to_damage"],
            apply_to_all_layers     = params.get("apply_to_all_layers", False),
            manipulation_method     = params["method"],
            masking_level           = params.get("masking_level", "connections"),
            include_bias            = params.get("include_bias", False),
            mc_permutations         = params["mc_permutations"],
            layer_name              = params["layer_name"],
            imagenet_root           = params["imagenet_root"],
            only_conv              = params.get("only_conv", True),
            batch_size             = params["batch_size"],
            num_workers            = params["num_workers"],
            subset_pct             = params["subset_pct"],
        )

if __name__ == "__main__":
    main(sys.argv[1])
