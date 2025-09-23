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

def load_model(cfg):
    src, name, wts = cfg.get("source", "torchvision"), cfg["name"], cfg.get("weights", "pretrained")
    if src == "cornet":
        import cornet
        nm = name.lower()
        ctor = {
            "cornet_rt": cornet.cornet_rt,
            "cornet_s": cornet.cornet_s,
            "cornet_z": cornet.cornet_z,
        }[nm]
        model = ctor(
            pretrained=(wts == "pretrained"), 
            map_location=(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")),
            **({"times": cfg.get("time_steps")} if nm == "cornet_rt" else {}),
        )
    elif src == "timm":
        import timm
        model = timm.create_model(name, pretrained=(wts == "pretrained"))
    else:
        if src == "pytorch_hub":
            # Most torch.hub models expect `pretrained=...`
            model = torch.hub.load(cfg["repo"], name, pretrained=(wts == "pretrained"), map_location=(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")))
        else:
            import torchvision.models as tvm
            ctor = getattr(tvm, name)
            model = ctor(weights="IMAGENET1K_V1" if wts == "pretrained" else None)
    return model.eval()

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

# --- NEW: Hedges' g with small-sample correction ---
def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Hedges' g for two independent samples x (target) and y (other).
    Uses unbiased small-sample correction J.
    Returns np.nan on degenerate cases.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = x.size, y.size
    if nx < 2 or ny < 2:
        return np.nan
    mx, my = np.mean(x), np.mean(y)
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    df = nx + ny - 2
    if df <= 0:
        return np.nan
    sp = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / df)
    if not np.isfinite(sp) or sp == 0:
        return np.nan
    d = (mx - my) / sp
    J = 1.0 - (3.0 / (4.0 * df - 1.0))  # Hedges & Olkin correction
    return J * d

def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8-sig"))

    # metrics selection (default to Mann–Whitney U) ---
    raw_metrics = cfg.get("metrics", ["mannwhitneyu"])
    if isinstance(raw_metrics, str):
        metrics = [raw_metrics.lower()]
    else:
        metrics = [m.lower() for m in raw_metrics]
    use_mw = "mannwhitneyu" in metrics
    use_hg = "hedgesg" in metrics

    model = load_model(cfg["model"])
    transform = build_transform()

    # 1) Find all modules with an 'output' submodule
    block_layers = [name for name, m in model.named_modules() if hasattr(m, "output")]
    print("Identified block output layers:", block_layers)
    act_dict = {layer: {} for layer in block_layers}  # layer -> {img_key: activation}

    # 2) Register hooks to capture block outputs
    modules = dict(model.named_modules())
    def make_hook(layer_name):
        def hook(module, inp, out):
            out_tensor = out[0] if isinstance(out, tuple) else out
            act_dict[layer_name][current_img_key] = out_tensor.detach().cpu().numpy().flatten()
        return hook
    for layer in block_layers:
        modules[layer].register_forward_hook(make_hook(layer))

    # 3) Process all images, save activations
    root = pathlib.Path(cfg["data_root"])
    categories = sorted([p.name for p in root.iterdir() if p.is_dir()])
    img_to_cat = {}
    # Build a map of unique keys to categories (avoid filename collisions)
    for cat in categories:
        for img_name, _ in iter_imgs(root / cat):
            img_key = f"{cat}/{img_name}"
            img_to_cat[img_key] = cat

    global current_img_key
    for cat in categories:
        for img_name, img in tqdm(iter_imgs(root / cat), desc=f"Cat={cat}"):
            current_img_key = f"{cat}/{img_name}"
            x = transform(img).unsqueeze(0).to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
            with torch.no_grad():
                model(x)

    # 4) Save activations for each block (materialize act_dict -> PKL/CSV)
    out_dir = pathlib.Path(cfg.get("out_dir", "results"))
    out_dir.mkdir(exist_ok=True)

    for layer in block_layers:
        img_keys = sorted(act_dict[layer].keys())
        X = np.stack([act_dict[layer][k] for k in img_keys], axis=0) if img_keys else np.empty((0,0))
        df = pd.DataFrame(X, index=img_keys)
        df.to_pickle(out_dir / f"{layer}_block_activations.pkl")
        df.to_csv(out_dir / f"{layer}_block_activations.csv")
        print(f"Wrote activations for {layer}: {df.shape}")

    # 5) Compute selectivity metrics for each unit and category
    combined_stats = []
    all_unit_cat_stats_mw = []  # MW for combined file
    all_unit_cat_stats_hg = []  # HG for combined file

    for layer in tqdm(block_layers, desc="Layers"):
        df = pd.read_pickle(out_dir / f"{layer}_block_activations.pkl")
        if df.empty:
            print(f"Warning: no activations for layer {layer}")
            continue

        unit_ids = df.columns
        results = []
        unit_cat_to_mw = {}  # for combined file
        unit_cat_to_hg = {}  

        for unit in tqdm(unit_ids, desc=f"{layer}: units", leave=False, position=1):
            for cat in categories:
                target_idx = [k for k in df.index if img_to_cat[k] == cat]
                other_idx  = [k for k in df.index if img_to_cat[k] != cat]
                target_vals = df.loc[target_idx, unit].values
                other_vals  = df.loc[other_idx,  unit].values

                mw_stat, mw_p = (np.nan, np.nan)
                hg_stat = np.nan
                if use_mw:
                    try:
                        mw_stat, mw_p = mannwhitneyu(target_vals, other_vals, alternative="two-sided")
                    except Exception:
                        mw_stat, mw_p = np.nan, np.nan
                if use_hg:
                    try:
                        hg_stat = hedges_g(target_vals, other_vals)
                    except Exception:
                        hg_stat = np.nan

                row = {
                    "layer": layer,
                    "unit": unit,
                    "category": cat,
                    "target_mean": float(np.mean(target_vals)) if target_vals.size else np.nan,
                    "other_mean":  float(np.mean(other_vals)) if other_vals.size else np.nan,
                }
                if use_mw:
                    row["mannwhitneyu_stat"] = mw_stat
                    row["p_value"] = mw_p
                    unit_cat_to_mw.setdefault((layer, unit), {})[cat] = mw_stat
                if use_hg:
                    row["hedgesg_stat"] = hg_stat
                    unit_cat_to_hg.setdefault((layer, unit), {})[cat] = hg_stat  # <-- ADD THIS

                results.append(row)

        # save per-layer file (unchanged)
        res_df = pd.DataFrame(results)
        res_df.to_csv(out_dir / f"{layer}_block_selectivity_mw.csv", index=False)
        res_df.to_pickle(out_dir / f"{layer}_block_selectivity_mw.pkl")
        print(f"Wrote selectivity stats for {layer} ({len(unit_ids)} units)")

        if use_mw:
            for (layer_name, unit), cat_stats in unit_cat_to_mw.items():
                all_unit_cat_stats_mw.append((layer_name, unit, cat_stats))
        if use_hg:
            for (layer_name, unit), cat_stats in unit_cat_to_hg.items():
                all_unit_cat_stats_hg.append((layer_name, unit, cat_stats))

    # Combined file: write mw_* and hg_* columns (whichever are enabled)
    if (use_mw and all_unit_cat_stats_mw) or (use_hg and all_unit_cat_stats_hg):
        # Build dicts keyed by (layer, unit) -> {cat: stat}
        mw_map = {}
        for layer, unit, cat_stats in all_unit_cat_stats_mw:
            mw_map[(layer, unit)] = cat_stats

        hg_map = {}
        for layer, unit, cat_stats in all_unit_cat_stats_hg:
            hg_map[(layer, unit)] = cat_stats

        # Union of keys from both metrics
        keys = set(mw_map.keys()) | set(hg_map.keys())

        combined_rows = []
        for (layer, unit) in tqdm(sorted(keys), desc="Combining all layers"):
            row = {"layer": layer, "unit": unit}
            # MW columns
            if use_mw:
                stats = mw_map.get((layer, unit), {})
                for cat in categories:
                    row[f"mw_{cat}"] = stats.get(cat, np.nan)
            # HG columns
            if use_hg:
                stats = hg_map.get((layer, unit), {})
                for cat in categories:
                    row[f"hg_{cat}"] = stats.get(cat, np.nan)
            combined_rows.append(row)

        all_stats_df = pd.DataFrame(combined_rows)
        # Keep legacy filename, now includes hg_* columns when enabled
        all_stats_df.to_csv(out_dir / "all_layers_units_mannwhitneyu.csv", index=False)
        all_stats_df.to_pickle(out_dir / "all_layers_units_mannwhitneyu.pkl")
        print(f"Wrote combined stats (mw_* and hg_* if enabled) to {out_dir / 'all_layers_units_mannwhitneyu.csv'}")
    else:
        print("Skipping combined stats (no data).")
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python unit_categ_selectivity_all_units.py <config.yaml>")
    main(sys.argv[1])
