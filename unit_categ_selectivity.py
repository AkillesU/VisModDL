import sys
import pathlib
import yaml
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.stats import mannwhitneyu

def load_model(cfg, device):
    src, name, wts = cfg.get("source", "torchvision"), cfg["name"], cfg.get("weights", "pretrained")
    if src == "cornet":
        import cornet
        ctor = {
            "cornet_rt": cornet.cornet_rt,
            "cornet_s": cornet.cornet_s,
            "cornet_z": cornet.cornet_z,
        }[name.lower()]
        model = ctor(
            pretrained=(wts == "pretrained"),
            **({"times": cfg.get("time_steps")} if name == "cornet_rt" else {}),
        )
    elif src == "timm":
        import timm
        model = timm.create_model(name, pretrained=(wts == "pretrained"))
    else:
        if src == "pytorch_hub":
            model = torch.hub.load(cfg["repo"], name,
                                   weights=None if wts != "pretrained" else wts)
        else:
            import torchvision.models as tvm
            ctor = getattr(tvm, name)
            model = ctor(weights="IMAGENET1K_V1" if wts == "pretrained" else None)
    return model.to(device).eval()

def build_transform():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def iter_imgs(folder):
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            yield p.name, Image.open(p).convert("RGB")

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8-sig"))
    device = torch.device(
        cfg.get("device") if cfg.get("device") != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = load_model(cfg["model"], device)
    transform = build_transform()

    # 1) Find all modules with an 'output' submodule
    block_layers = [name for name, m in model.named_modules() if hasattr(m, "output")]
    print("Identified block output layers:", block_layers)
    act_dict = {layer: {} for layer in block_layers}  # layer -> {img_name: activation}

    # 2) Register hooks to capture block outputs
    modules = dict(model.named_modules())
    for layer in block_layers:
        def make_hook(layer_name):
            def hook(module, inp, out):
                # If out is a tuple, take the first element
                if isinstance(out, tuple):
                    out_tensor = out[0]
                else:
                    out_tensor = out
                act_dict[layer_name][current_img] = out_tensor.detach().cpu().numpy().flatten()
            return hook
        modules[layer].register_forward_hook(make_hook(layer))

    # 3) Process all images, save activations
    root = pathlib.Path(cfg["data_root"])
    categories = sorted([p.name for p in root.iterdir() if p.is_dir()])
    img_to_cat = {}
    all_imgs = []
    for cat in categories:
        for img_name, _ in iter_imgs(root / cat):
            img_to_cat[img_name] = cat
            all_imgs.append((cat, img_name))

    # For each image, run through model and collect activations
    global current_img
    for cat in categories:
        for img_name, img in tqdm(iter_imgs(root / cat), desc=f"Cat={cat}"):
            current_img = img_name
            x = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                model(x)

    # 4) Save activations for each block
    out_dir = pathlib.Path(cfg.get("out_dir", "results"))
    out_dir.mkdir(exist_ok=True)

    # 5) Compute selectivity metric for each unit and category
    combined_stats = []  # Collect all units across all layers
    all_unit_cat_stats = []      # collects per-unit/category MW stats   <-- add this

    # ──-- 1. per-layer bar ─-──
    for layer in tqdm(block_layers, desc="Layers"):
        df = pd.read_pickle(out_dir / f"{layer}_block_activations.pkl")
        unit_ids = df.columns
        results = []
        unit_cat_to_stat = {}

        # ──-- 2. per-unit bar inside each layer ─-──
        for unit in tqdm(unit_ids,
                         desc=f"{layer}: units",
                         leave=False,          # keep the outer bar visible
                         position=1):          # indent the nested bar one row
            # ──-- 3. optional per-category bar (comment out if noisy) ─-──
            for cat in categories:            # or wrap in tqdm(categories, …)
                target_vals = df.loc[[img for img in df.index
                                      if img_to_cat[img] == cat], unit].values
                other_vals  = df.loc[[img for img in df.index
                                      if img_to_cat[img] != cat], unit].values
                try:
                    stat, p = mannwhitneyu(target_vals, other_vals,
                                           alternative="two-sided")
                except Exception:
                    stat, p = np.nan, np.nan
                results.append((layer, unit, cat, stat, p,
                                target_vals.mean(), other_vals.mean()))
                unit_cat_to_stat.setdefault((layer, unit), {})[cat] = stat

        # save per-layer statistics
        res_df = pd.DataFrame(
            results,
            columns=["layer", "unit", "category",
                     "mannwhitneyu_stat", "p_value",
                     "target_mean", "other_mean"]
        )
        res_df.to_csv(out_dir / f"{layer}_block_selectivity_mw.csv", index=False)
        res_df.to_pickle(out_dir / f"{layer}_block_selectivity_mw.pkl")
        for (layer, unit), cat_stats in unit_cat_to_stat.items():
            all_unit_cat_stats.append((layer, unit, cat_stats))
        print(f"Wrote selectivity stats for {layer} ({len(unit_ids)} units)")

    # combined file (already had a bar; unchanged except for nicer label)
    # TODO: Add column with flag when "selective" category value is smaller than others! - Works without this but this has to be in place
    for layer, unit, cat_stats in tqdm(all_unit_cat_stats,
                                       desc="Combining all layers"):
        row = {"layer": layer, "unit": unit}
        for cat in categories:
            row[f"mw_{cat}"] = cat_stats.get(cat, np.nan)
        combined_stats.append(row)

    # Write combined file once
    all_stats_df = pd.DataFrame(combined_stats)
    all_stats_df.to_csv(out_dir / "all_layers_units_mannwhitneyu.csv", index=False)
    all_stats_df.to_pickle(out_dir / "all_layers_units_mannwhitneyu.pkl")
    print(f"Wrote combined Mann-Whitney stats for all units to {out_dir / 'all_layers_units_mannwhitneyu.csv'}")



if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python unit_categ_selectivity_all_units.py <config.yaml>")
    main(sys.argv[1])
