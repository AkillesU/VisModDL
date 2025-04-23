"""
Run ImageNet accuracy for the model specified
in a YAML config file.

Usage
-----
python run_imagenet.py configs/imagenet/my_cfg.yaml
"""

import sys, yaml, torch, random
from utils import run_damage_imagenet

random.seed(1234)

def main():
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "configs/cornet_s_it.yaml"
    cfg = yaml.safe_load(open(cfg_file))

    # ---------- model info ----------
    model_info = {
        "source":  cfg.get("model_source", "cornet"),
        "repo":    cfg.get("model_repo", "-"),
        "name":    cfg.get("model_name", "cornet_s"),
        "weights": cfg.get("model_weights", "")
    }
    if "model_time_steps" in cfg:
        model_info["time_steps"] = cfg["model_time_steps"]

    # ---------- call runner ----------
    run_damage_imagenet(
        model_info=model_info,
        pretrained=cfg.get("pretrained", True),
        fraction_to_mask_params=cfg.get("fraction_to_mask", [0.05,10,0.05]),
        noise_levels_params=cfg.get("noise_levels",    [0.1, 10, 0.1]),
        layer_paths_to_damage=cfg.get("layer_paths_to_damage", []),
        apply_to_all_layers=cfg.get("apply_to_all_layers", False),
        manipulation_method=cfg.get("manipulation_method", "connections"),  # or "noise"
        mc_permutations=cfg.get("mc_permutations", 5),
        layer_name=cfg.get("layer_name", "IT"),
        imagenet_root=cfg["imagenet_root"],      # must be in YAML
        only_conv=cfg.get("only_conv", True),
        include_bias=cfg.get("include_bias", False),
        masking_level=cfg.get("masking_level", "connections"),
        batch_size=cfg.get("batch_size", 32),
        num_workers=cfg.get("num_workers", 8)
    )

if __name__ == "__main__":
    main()
