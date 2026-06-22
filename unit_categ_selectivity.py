import re
import sys
import pathlib
import yaml
import torch
import torchvision.transforms as T
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.stats import mannwhitneyu
from utils import get_layer_from_path, load_model, normalize_module_name


def as_list(value) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v not in (None, "")]
    return [str(value)]


def load_yaml(path: pathlib.Path) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f) or {}


def dedupe_layers(layers: list[str]) -> list[str]:
    out = []
    seen = set()
    for layer in layers:
        norm = normalize_module_name(layer)
        if norm in seen:
            continue
        seen.add(norm)
        out.append(layer)
    return out


def model_name_key(name: str | None) -> str:
    name = str(name or "").lower()
    return name[:-3] if name.endswith("_ut") else name


def configs_match_model(reference: dict, candidate: dict) -> bool:
    if model_name_key(reference.get("model_name")) != model_name_key(candidate.get("model_name")):
        return False

    ref_source = reference.get("model_source")
    cand_source = candidate.get("model_source")
    if ref_source and cand_source and ref_source != cand_source:
        return False

    if bool(reference.get("pretrained", True)) != bool(candidate.get("pretrained", True)):
        return False

    ref_steps = reference.get("model_time_steps")
    cand_steps = candidate.get("model_time_steps")
    if ref_steps is not None and cand_steps is not None and str(ref_steps) != str(cand_steps):
        return False

    return True


def infer_layers_from_conv_configs(cfg: dict, cfg_path: pathlib.Path) -> list[str]:
    """
    Unit-selectivity configs often specify only the model. In that case, borrow the
    activation/read-out layer_path entries from configs/conv_layers, which are the
    configs main.py uses for saving activations.
    """
    if cfg.get("conv_layer_config"):
        conv_cfg_path = pathlib.Path(cfg["conv_layer_config"])
        if not conv_cfg_path.is_absolute():
            local_cfg_path = cfg_path.parent / conv_cfg_path
            conv_cfg_path = local_cfg_path if local_cfg_path.exists() else resolve_input_path(str(conv_cfg_path))
        if not conv_cfg_path.is_absolute():
            conv_cfg_path = pathlib.Path.cwd() / conv_cfg_path
        return as_list(load_yaml(conv_cfg_path).get("layer_path"))

    conv_dir = resolve_input_path(str(pathlib.Path(cfg.get("conv_layers_dir", "configs/conv_layers"))))
    if not conv_dir.is_absolute():
        conv_dir = pathlib.Path.cwd() / conv_dir
    if not conv_dir.is_dir():
        return []

    recursive = bool(cfg.get("conv_layers_recursive", False))
    pattern = "**/*.yaml" if recursive else "*.yaml"
    layers = []
    for path in sorted(conv_dir.glob(pattern)):
        if path.resolve() == cfg_path.resolve():
            continue
        conv_cfg = load_yaml(path)
        if configs_match_model(cfg, conv_cfg):
            layers.extend(as_list(conv_cfg.get("layer_path")))
    return dedupe_layers(layers)


def select_block_layers(model, cfg: dict, cfg_path: pathlib.Path) -> list[str]:
    layers = as_list(cfg.get("layer_path"))
    if not layers:
        layers = infer_layers_from_conv_configs(cfg, cfg_path)

    if not layers:
        model_tag = str(cfg.get("model_name", "")).lower()
        if "alexnet" in model_tag:
            layers = ["features.12"]
        elif "vgg16" in model_tag:
            layers = ["features.23", "features.30"]
        else:
            layers = [lname for lname, m in model.named_modules() if hasattr(m, "output")]

    return dedupe_layers(layers)


def build_model_tag(model_info: dict, pretrained: bool) -> str:
    name = str(model_info["name"])
    tag = name[:-3] if name.endswith("_ut") else name

    if tag.startswith(("cornet_rt", "cornet_s")) and "time_steps" in model_info:
        tag = f"{tag}{model_info['time_steps']}"

    if not pretrained and not tag.endswith("_ut"):
        tag = f"{tag}_ut"

    return tag


def safe_file_tag(tag: str) -> str:
    tag = normalize_module_name(str(tag)).replace("module.", "")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", tag).strip("_")


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

def run_config(cfg_path, output_prefix=None):
    cfg_path = pathlib.Path(cfg_path)
    cfg = load_yaml(cfg_path)
    if output_prefix is not None and "output_prefix" not in cfg:
        cfg["output_prefix"] = output_prefix

    # metrics selection (default to Mann–Whitney U) ---
    raw_metrics = cfg.get("metrics", ["mannwhitneyu"])
    if isinstance(raw_metrics, str):
        metrics = [raw_metrics.lower()]
    else:
        metrics = [m.lower() for m in raw_metrics]
    use_mw = "mannwhitneyu" in metrics
    use_hg = "hedgesg" in metrics

    # Load model
    model_info = {}
    model_info["source"] = cfg.get("model_source", "pytorch_hub")
    model_info["repo"] = cfg.get("model_repo", "-")
    model_info["name"] = cfg.get("model_name", "cornet_z")
    model_info["weights"] = cfg.get("model_weights", "")
    if "model_time_steps" in cfg:
        model_info["time_steps"] = cfg["model_time_steps"]
    pretrained = cfg.get("pretrained", True)


    model, _ = load_model(model_info=model_info, pretrained=pretrained)

    transform = build_transform()

    # Model tag for filenames
    model_tag = build_model_tag(model_info, pretrained)
    output_tag = safe_file_tag(cfg.get("output_prefix", model_tag))

    # Select layers to process
    block_layers = select_block_layers(model, cfg, cfg_path)
    if not block_layers:
        raise ValueError(
            f"No activation/read-out layers found for {cfg_path}. "
            "Set layer_path in the config or add a matching config under configs/conv_layers."
        )
    print(f"[{model_tag}] Activation/read-out layers from {cfg_path}: {block_layers}")
    act_dict = {layer: {} for layer in block_layers}
    layer_tags = {layer: safe_file_tag(layer) for layer in block_layers}

    # 2) Register hooks to capture block outputs
    current_img_key = None

    def make_hook(layer_name):
        def hook(module, inp, out):
            out_tensor = out[0] if isinstance(out, tuple) else out
            act_dict[layer_name][current_img_key] = out_tensor.detach().cpu().numpy().flatten()
        return hook

    hook_handles = []
    for layer in block_layers:
        try:
            target_layer = get_layer_from_path(model, layer)
        except Exception as exc:
            raise KeyError(
                f"Could not resolve layer_path '{layer}' from {cfg_path} "
                f"for model '{model_tag}'."
            ) from exc
        hook_handles.append(target_layer.register_forward_hook(make_hook(layer)))

    # 3) Process all images, save activations
    root = pathlib.Path(cfg.get("data_root", "categ_images"))
    if not root.is_dir():
        raise FileNotFoundError(
            f"Category image root '{root}' was not found. "
            "Set data_root to a directory containing category subfolders."
        )
    categories = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if not categories:
        raise ValueError(f"No category subfolders found in '{root}'.")
    img_to_cat = {}
    # Build a map of unique keys to categories (avoid filename collisions)
    for cat in categories:
        for img_name, _ in iter_imgs(root / cat):
            img_key = f"{cat}/{img_name}"
            img_to_cat[img_key] = cat

    for cat in categories:
        for img_name, img in tqdm(iter_imgs(root / cat), desc=f"Cat={cat}"):
            current_img_key = f"{cat}/{img_name}"
            x = transform(img).unsqueeze(0).to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
            with torch.no_grad():
                model(x)

    # 4) Save activations for each block (materialize act_dict -> PKL/CSV)
    out_dir = pathlib.Path(cfg.get("out_dir", "unit_selectivity"))
    out_dir.mkdir(parents=True, exist_ok=True)

    for layer in block_layers:
        img_keys = sorted(act_dict[layer].keys())
        X = np.stack([act_dict[layer][k] for k in img_keys], axis=0) if img_keys else np.empty((0,0))
        df = pd.DataFrame(X, index=img_keys)
        layer_tag = layer_tags[layer]
        df.to_pickle(out_dir / f"{output_tag}_{layer_tag}_block_activations.pkl")
        df.to_csv(out_dir / f"{output_tag}_{layer_tag}_block_activations.csv")

        print(f"Wrote activations for {layer}: {df.shape}")

    # 5) Compute selectivity metrics for each unit and category
    combined_stats = []
    all_unit_cat_stats_mw = []  # MW for combined file
    all_unit_cat_stats_hg = []  # HG for combined file

    for layer in tqdm(block_layers, desc="Layers"):
        layer_tag = layer_tags[layer]
        df = pd.read_pickle(out_dir / f"{output_tag}_{layer_tag}_block_activations.pkl")
        if df.empty:
            print(f"Warning: no activations for layer {layer} in {model_tag}")
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
        res_df.to_csv(out_dir / f"{output_tag}_{layer_tag}_block_selectivity_mw.csv", index=False)
        res_df.to_pickle(out_dir / f"{output_tag}_{layer_tag}_block_selectivity_mw.pkl")
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
        all_stats_df.to_csv(out_dir / f"{output_tag}_all_layers_units_mannwhitneyu.csv", index=False)
        all_stats_df.to_pickle(out_dir / f"{output_tag}_all_layers_units_mannwhitneyu.pkl")
        print(f"Wrote combined stats (mw_* and hg_* if enabled) to {out_dir /f'{output_tag}_all_layers_units_mannwhitneyu.csv'}")
    else:
        print("Skipping combined stats (no data).")

    for handle in hook_handles:
        handle.remove()


def resolve_input_path(raw_path: str) -> pathlib.Path:
    path = pathlib.Path(raw_path)
    if path.exists():
        return path

    # The repo directory is configs/, but it is easy to type config/.
    parts = path.parts
    if parts and parts[0] == "config":
        alt = pathlib.Path("configs", *parts[1:])
        if alt.exists():
            return alt

    return path


def discover_config_paths(input_path: pathlib.Path, recursive=False) -> list[pathlib.Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        pattern = "**/*.yaml" if recursive else "*.yaml"
        return sorted(input_path.glob(pattern))
    raise FileNotFoundError(f"Config path '{input_path}' was not found.")


def main(input_path, recursive=False):
    input_path = resolve_input_path(input_path)
    cfg_paths = discover_config_paths(input_path, recursive=recursive)
    if not cfg_paths:
        raise ValueError(f"No .yaml configs found in '{input_path}'.")

    multi_config = len(cfg_paths) > 1
    for cfg_path in cfg_paths:
        prefix = None
        if multi_config:
            prefix = safe_file_tag(f"{cfg_path.parent.name}_{cfg_path.stem}")
            if cfg_path.parent == input_path:
                prefix = safe_file_tag(cfg_path.stem)
        run_config(cfg_path, output_prefix=prefix)


if __name__ == "__main__":
    if len(sys.argv) not in {2, 3}:
        sys.exit("Usage: python unit_categ_selectivity.py <config.yaml|config_dir> [--recursive]")
    if len(sys.argv) == 3 and sys.argv[2] != "--recursive":
        sys.exit("Usage: python unit_categ_selectivity.py <config.yaml|config_dir> [--recursive]")
    main(sys.argv[1], recursive=(len(sys.argv) == 3 and sys.argv[2] == "--recursive"))
