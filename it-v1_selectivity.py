#!/usr/bin/env python3
"""
V1-to-input contribution analysis for top IT units.

For each configured top fraction of IT units, compute V1 gradient maps for
selective units and random unit samples, then save per-feature-map statistics,
summary plots, diagnostic histograms, and configured per-image overlays.
"""

from __future__ import annotations

import sys
import yaml
import pathlib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from scipy.stats import mannwhitneyu, ttest_ind
import os
import random
from statsmodels.stats.multitest import multipletests
from typing import List
import threading
import gc
from utils import (
    build_imagenet_transform,
    cliffs_delta,
    hedges_g,
    iter_image_paths,
    iter_rgb_images,
    load_rgb_image,
    load_model as load_project_model,
    mann_whitney_u_stat,
)

_thread_cache = threading.local()

GLOBAL_TFM = None
GLOBAL_DEVICE: torch.device | None = None


def _slice_rand_for_index(rand_all: np.ndarray, idx: int, N_expected: int | None) -> np.ndarray:
    """
    Return the per-image random tensor for index `idx` from a container that may be:
      [R, N, C, H, W]  -> returns [R, C, H, W]
      [N, C, H, W]     -> returns [C, H, W]
      [R, H, W, N]     -> returns [R, H, W]   (discouraged but handled)
      [N, H, W]        -> returns [H, W]      (fully channel-collapsed; caller should guard)
      [C, H, W]        -> returns [C, H, W]   (already per-image)
    We prefer to use `N_expected` (from the selective array shape) to disambiguate.
    """
    nd = rand_all.ndim

    if nd == 5:
        # [R, N, C, H, W]
        return rand_all[:, idx]

    if nd == 4:
        # Either [N, C, H, W] or [R, H, W, N]
        if N_expected is not None:
            if rand_all.shape[0] == N_expected:   # [N, C, H, W]
                return rand_all[idx]
            if rand_all.shape[-1] == N_expected:  # [R, H, W, N]
                return rand_all[..., idx]
        # Fallback: assume [N, C, H, W] if possible
        if idx < rand_all.shape[0]:
            return rand_all[idx]
        # else assume trailing N
        return rand_all[..., idx]

    if nd == 3:
        # Could be [N, H, W] (bad for overlays) or [C, H, W] (already per-image)
        if N_expected is not None and rand_all.shape[0] == N_expected:
            # [N, H, W] -> returns [H, W] (caller may reject)
            return rand_all[idx]
        # Assume [C, H, W]
        return rand_all

    # Any other shape: pass through (caller will validate / error)
    return rand_all


def load_model(mcfg, device):
    model_info = {
        "source": mcfg.get("source", "cornet"),
        "repo": mcfg.get("repo", "-"),
        "name": mcfg["name"],
        "weights": mcfg.get("weights", ""),
    }
    if "time_steps" in mcfg:
        model_info["time_steps"] = mcfg["time_steps"]

    pretrained = str(mcfg.get("weights", "pretrained")).lower() == "pretrained"
    model, _ = load_project_model(model_info, pretrained=pretrained)
    return model.to(device).eval()


def _as_list(value) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v not in (None, "")]
    return [str(value)]


def _category_stems(category: str) -> set[str]:
    category = str(category).strip().lower()
    stems = {category}
    if category.endswith("s"):
        stems.add(category[:-1])
    return {stem for stem in stems if stem}


def _matches_category(path: pathlib.Path, category: str) -> bool:
    stem = path.stem.lower()
    return any(stem.startswith(prefix) for prefix in _category_stems(category))


def _load_images_from_folder(folder: pathlib.Path, categories: list[str] | None = None) -> list[Image.Image]:
    paths = list(iter_image_paths(folder))
    if categories:
        paths = [
            path for path in paths
            if any(_matches_category(path, category) for category in categories)
        ]
    return [load_rgb_image(path) for path in paths]


def gather_images(cfg: dict) -> tuple[List[Image.Image], str]:
    """Return the configured image set and a compact category tag."""

    max_imgs = cfg.get("max_images", 20)

    if cfg.get("image_category_dirs"):
        folders = [pathlib.Path(d).expanduser().resolve() for d in cfg["image_category_dirs"]]
        categories = [folder.name for folder in folders]
        source_label = ", ".join(str(folder) for folder in folders)
        imgs = []
        for folder in folders:
            if not folder.exists():
                print(f"[WARN] Folder {folder} missing - skipped.")
                continue
            imgs.extend(iter_rgb_images(folder))
    else:
        base = pathlib.Path(cfg["category_images"]).expanduser().resolve()
        source_label = str(base)
        if not base.exists():
            raise FileNotFoundError(f"Image directory {base} does not exist")

        categories = _as_list(cfg.get("image_categories")) or [str(cfg["category"])]
        category_dirs = [base / category for category in categories if (base / category).is_dir()]
        if category_dirs:
            imgs = []
            for folder in category_dirs:
                imgs.extend(iter_rgb_images(folder))
        else:
            imgs = _load_images_from_folder(base, categories)

    cat_tag = "".join(sorted({category[0].lower() for category in categories if category}))

    if not imgs:
        raise RuntimeError(f"No images found for categories {categories} in {source_label}.")

    return imgs[:max_imgs], cat_tag


class HookBuffers:
    """Stores forward activations and V1 gradients for one model instance."""
    def __init__(self):
        self.out  : dict[str, torch.Tensor] = {}
        self.grad : dict[str, torch.Tensor] = {}

def setup_hooks(model: nn.Module) -> HookBuffers:
    """
    Register forward-hooks on module.IT and module.V1.nonlin_input *once*.
    Returns a HookBuffers object that will be filled every forward/backward.
    """
    bufs = HookBuffers()
    mods = dict(model.named_modules())

    v1 = mods["module.V1.nonlin_input"]
    target_names = {"module.IT", "module.V1.nonlin_input"}

    def make_fwd_hook(name):
        def f(_m, _in, out): bufs.out[name] = out
        return f

    for name, mod in mods.items():
        if name in target_names:
            mod.register_forward_hook(make_fwd_hook(name))

    def v1_grad_hook(grad): bufs.grad["module.V1.nonlin_input"] = grad
    def v1_fwd_hook(_m, _i, out):
        out.register_hook(v1_grad_hook)

    v1.register_forward_hook(v1_fwd_hook)

    return bufs


def flat_to_cyx(idx, C, H, W):
    c   = idx // (H * W)
    rem = idx %  (H * W)
    y   = rem // W
    x   = rem %  W
    return c, y, x


def grads_per_image(img: Image.Image,
                    model: nn.Module,
                    bufs: HookBuffers,
                    units: list[tuple[str,int,int,int]]) -> np.ndarray:
    """
    Returns an array of shape [len(units), C_v1, H_v1, W_v1] with the absolute
    gradient in module.V1.nonlin_input for each requested IT voxel.
    """
    out, grad = bufs.out, bufs.grad

    x = GLOBAL_TFM(img).unsqueeze(0).to(GLOBAL_DEVICE)


    # single forward pass
    model.zero_grad(set_to_none=True)
    _ = model(x)

    per_unit_grads = []
    for i, (lname, c, y, x_) in enumerate(units):
        act = out[lname]
        if isinstance(act, tuple):           # CORnet-RT returns a tuple
            act = act[0]
        loss = act[:, c, y, x_].sum()
        # retain graph for all but the last unit
        loss.backward(retain_graph=(i < len(units) - 1))
        g = grad["module.V1.nonlin_input"][0].abs().cpu().numpy()
        per_unit_grads.append(g)
        model.zero_grad(set_to_none=True)    # clear grads for next unit

    return np.stack(per_unit_grads)          # shape [n_units, C, H, W]


def get_thread_model_and_bufs(cfg):
    """
    Return (model, HookBuffers) that are **unique to the current executor
    thread**.  They are created the first time the thread calls this function
    and then reused for every subsequent image handled by that thread.
    """
    if not hasattr(_thread_cache, "model"):
        model = load_model(cfg["model"], GLOBAL_DEVICE)
        bufs  = setup_hooks(model)
        _thread_cache.model = model
        _thread_cache.bufs  = bufs
    return _thread_cache.model, _thread_cache.bufs


def collapse_random(arr: np.ndarray, method: str | None):
    """
    Collapse random repeats along axis 0 if requested.
    arr shape: [R, N_img, C, H, W]
    returns (collapsed_arr, collapsed_flag)
      - if collapsed: shape [N_img, C, H, W]
      - otherwise: unchanged input array
    """
    if method is None:
        return arr, False
    m = str(method).lower()
    if m in {"mean", "avg", "average"}:
        return arr.mean(axis=0), True
    if m in {"median", "med"}:
        return np.median(arr, axis=0), True
    raise ValueError(f"Unknown collapse_method: {method!r} (use None, 'mean', or 'median')")


def _resize_map_to_img(eff_map: np.ndarray, target_hw: tuple[int,int]) -> np.ndarray:
    """Resize a 2D float map to (H, W) using PIL bilinear."""
    Ht, Wt = target_hw
    pil = Image.fromarray(eff_map.astype(np.float32), mode="F")
    pil = pil.resize((Wt, Ht), resample=Image.BILINEAR)
    return np.array(pil, dtype=np.float32)

def _ensure_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Accept HxW, HxWx1, or HxWx3 data and return HxWx3 uint8."""
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (np.clip(img, 0, 1) * 255.0).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    return img


def _compute_effect_map_for_image(sel_img: np.ndarray,
                                  rand_img: np.ndarray,
                                  effect_size: str) -> np.ndarray:
    """
    sel_img:  [C, H, W]
    rand_img: [R, C, H, W]  (not collapsed)
              [C, H, W]     (collapsed across repeats)
              [H, W]        (fully collapsed across repeats and channels)
    Returns eff_map: [H, W]
    """
    if sel_img.ndim != 3:
        raise ValueError(f"Expected sel_img as [C,H,W], got shape {sel_img.shape}")
    C, H, W = sel_img.shape

    # build a function to compute the effect between two 1-D vectors
    def _eff(a: np.ndarray, b: np.ndarray) -> float:
        if effect_size == "hedges_g":
            return hedges_g(a, b, eps=1e-12)
        elif effect_size == "cliffs_delta":
            return cliffs_delta(a, b)
        elif effect_size == "mw_u":
            return mann_whitney_u_stat(a, b)
        else:
            raise ValueError(f"Unknown effect_size: {effect_size}")

    eff_map = np.empty((H, W), dtype=np.float32)

    if rand_img.ndim == 4:
        # [R, C, H, W]: flatten repeats and channels per voxel.
        for y in range(H):
            for x in range(W):
                a = sel_img[:, y, x]                      # len = C
                b = rand_img[:, :, y, x].reshape(-1)      # len = R*C
                eff_map[y, x] = _eff(a, b)

    elif rand_img.ndim == 3:
        # [C, H, W]: channel-wise comparison per voxel.
        if rand_img.shape[0] != C:
            raise ValueError(f"rand_img first dim (C={rand_img.shape[0]}) "
                             f"does not match sel_img C={C}")
        for y in range(H):
            for x in range(W):
                a = sel_img[:, y, x]          # len = C
                b = rand_img[:, y, x]         # len = C
                eff_map[y, x] = _eff(a, b)

    elif rand_img.ndim == 2:
        # [H, W]: compare each scalar voxel to the selective channel distribution.
        # This preserves directionality for Hedges g and MW U; Cliff's delta is defined
        # between two samples and still works with a constant sample for b.
        import warnings
        warnings.warn("[per_image_overlay] rand_img is 2-D [H,W]; "
                      "random repeats & channels appear fully collapsed. "
                      "Using per-voxel scalar as the 'b' sample (repeated).")
        for y in range(H):
            for x in range(W):
                a = sel_img[:, y, x]                 # len = C
                b0 = float(rand_img[y, x])
                b = np.full_like(a, b0, dtype=np.float32)  # len = C (constant)
                eff_map[y, x] = _eff(a, b)
    else:
        raise ValueError(f"Unexpected rand_img ndim={rand_img.ndim} shape={rand_img.shape}")

    return eff_map


def _normalise_for_viz(eff_map: np.ndarray,
                       viz_norm: str,
                       effect_size: str,
                       n_a: int | None = None,
                       n_b: int | None = None,
                       global_min: float | None = None,
                       global_max: float | None = None) -> tuple[np.ndarray, dict]:
    """
    Returns (map_for_plot, norm_kwargs) where norm_kwargs can hold vmin/vmax.
    """
    if viz_norm == "u_unit" and effect_size == "mw_u" and n_a and n_b:
        denom = float(n_a * n_b)
        mapped = eff_map / max(denom, 1.0)
        return mapped, dict(vmin=0.0, vmax=1.0)
    if viz_norm == "global" and global_min is not None and global_max is not None:
        return eff_map, dict(vmin=float(global_min), vmax=float(global_max))
    if viz_norm == "per_image":
        return eff_map, dict(vmin=float(np.nanmin(eff_map)), vmax=float(np.nanmax(eff_map)))
    # viz_norm == "none"
    return eff_map, {}


def run_per_image_overlays(
    all_v1_grads_sel: np.ndarray,             # [N, C, H, W]
    all_v1_grads_rand: np.ndarray,            # [R, N, C, H, W] or [N, C, H, W] if collapsed
    effect_size: str,                         # "hedges_g" | "cliffs_delta" | "mw_u"
    per_img_cfg: dict,
    outdir: Path,
    get_preprocessed_image=None,              # callable: idx -> np.ndarray (H,W,3) in 0..1 or 0..255
    cat_tag="faces",
    selectivity_frac=1,
):
    """
    Computes per-image effect maps and overlays them onto the corresponding preprocessed images.
    """
    indices = per_img_cfg.get("indices", [])
    if not indices:
        return

    alpha      = float(per_img_cfg.get("alpha", 0.45))
    cmap       = str(per_img_cfg.get("cmap", "magma"))
    viz_norm   = str(per_img_cfg.get("viz_norm", "per_image")).lower()
    save_raw   = bool(per_img_cfg.get("save_raw_maps", True))
    subdir     = str(per_img_cfg.get("out_subdir", "overlays"))

    overlay_dir = outdir / subdir
    overlay_dir.mkdir(parents=True, exist_ok=True)

    N, C, H, W = all_v1_grads_sel.shape
    overlay_collapse_method = per_img_cfg.get("collapse_method", "mean")
    rand_local = all_v1_grads_rand
    if rand_local.ndim == 5 and overlay_collapse_method is not None:
        m = str(overlay_collapse_method).lower()
        if m in {"mean", "avg", "average"}:
            rand_local = rand_local.mean(axis=0)           # [N,C,H,W]
        elif m in {"median", "med"}:
            rand_local = np.median(rand_local, axis=0)     # [N,C,H,W]
        else:
            raise ValueError(f"Unknown per-image collapse_method: {overlay_collapse_method!r}")
    else:
        # Keep as-is: [N,C,H,W] or [R,N,C,H,W]
        pass
    # Compute shared color limits before rendering when requested.
    global_min = None
    global_max = None
    if viz_norm == "global":
        mins = []
        maxs = []
        for idx in indices:
            if not (0 <= idx < N):
                continue
            sel_img  = all_v1_grads_sel[idx]                        # [C,H,W]
            rand_img = _slice_rand_for_index(rand_local, idx, N_expected=N)
            eff_map  = _compute_effect_map_for_image(sel_img, rand_img, effect_size)
            mins.append(np.nanmin(eff_map))
            maxs.append(np.nanmax(eff_map))
        if mins and maxs:
            global_min = float(np.nanmin(mins))
            global_max = float(np.nanmax(maxs))

    for idx in indices:
        if not (0 <= idx < N):
            print(f"[per_image_overlay] Skipping idx {idx} (out of range 0..{N-1})")
            continue

        sel_img  = all_v1_grads_sel[idx]                        # [C,H,W]
        rand_img = _slice_rand_for_index(rand_local, idx, N_expected=N)

        eff_map  = _compute_effect_map_for_image(sel_img, rand_img, effect_size)

        if save_raw:
            np.save(overlay_dir / f"img{idx:05d}_{effect_size}_map.npy", eff_map)

        n_a = sel_img.shape[0]  # C
        if rand_img.ndim == 4:          # [R, C, H, W]
            n_b = rand_img.shape[0] * rand_img.shape[1]  # R*C
        elif rand_img.ndim == 3:        # [C, H, W]
            n_b = rand_img.shape[0]                      # C
        else:
            raise ValueError(f"rand_img must be [R,C,H,W] or [C,H,W], got {rand_img.shape}")


        if get_preprocessed_image is not None:
            base_img = get_preprocessed_image(idx)
        else:
            raise RuntimeError(
                "Please provide `get_preprocessed_image` to `run_per_image_overlays` "
                "so we can overlay on the correctly preprocessed (aligned) image."
            )

        base_img = _ensure_uint8_rgb(np.asarray(base_img))
        H_img, W_img = base_img.shape[:2]

        if bool(per_img_cfg.get("make_greyscale", False)):
            base_img = np.dot(base_img[..., :3], [0.2989, 0.5870, 0.1140])  # luminance weighting
            base_img = np.stack([base_img] * 3, axis=-1).astype(np.uint8)

        eff_viz, norm_kwargs = _normalise_for_viz(
            eff_map, viz_norm=viz_norm, effect_size=effect_size,
            n_a=n_a, n_b=n_b, global_min=global_min, global_max=global_max
        )

        eff_up = _resize_map_to_img(eff_viz, (H_img, W_img))

        fig = plt.figure(figsize=(6, 6))
        plt.imshow(base_img)
        im = plt.imshow(eff_up, cmap=cmap, alpha=alpha, interpolation="bilinear", **norm_kwargs)
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label(effect_size)
        plt.axis("off")

        selectivity_frac_str = str(selectivity_frac)
        out_png = overlay_dir / f"{cat_tag}_img{idx:05d}_{effect_size}_overlay_{selectivity_frac_str}-sel.png"
        plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        if save_raw and (viz_norm in {"global", "u_unit"}):
            np.save(overlay_dir / f"img{idx:05d}_{effect_size}_map_vizscaled.npy", eff_viz)

        print(f"[per_image_overlay] Saved {out_png}")

def main(cfg_path: str | pathlib.Path):
    random.seed(1234)
    cfg = yaml.safe_load(open(cfg_path, 'r'))
    collapse_method = cfg.get("collapse_method", None)  # None, "mean", or "median"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_random_repeats = cfg.get("n_random_repeats", 100)
    effect_size = str(cfg.get("effect_size", "hedges_g")).lower()  # "hedges_g" | "cliffs_delta" | "mw_u"
    allowed_effects = {"hedges_g", "cliffs_delta", "mw_u"}
    if effect_size not in allowed_effects:
        raise ValueError(f"effect_size must be one of {allowed_effects}, got {effect_size!r}")

    global GLOBAL_DEVICE, GLOBAL_TFM
    GLOBAL_DEVICE = device
    GLOBAL_TFM = build_imagenet_transform()

    if torch.cuda.is_available() is False:
        try:
            num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
        except TypeError:
            num_threads = 1
        torch.set_num_threads(num_threads)
        print(f"PyTorch intra-op parallelism set to {torch.get_num_threads()} threads.")
    else:
        print("Using CUDA")

    model_name = cfg["model"].get("name", "cornet_rt").lower()
    is_pretrained = (cfg["model"].get("weights", "pretrained").lower() == "pretrained")
    ut_suffix = "_ut" if not is_pretrained else ""

    imgs, cat_tag = gather_images(cfg)
    print(f"Loaded {len(imgs)} images; category‑tag = '{cat_tag}'.")

    outdir = pathlib.Path(cfg.get("output_dir", "it2v1_analysis"))
    outdir.mkdir(parents=True, exist_ok=True)

    top_frac_cfg = cfg.get("top_frac", 0.1)
    if isinstance(top_frac_cfg, (float, int)):
        top_frac_list = [top_frac_cfg]
    else:
        top_frac_list = list(top_frac_cfg)

    for top_frac in top_frac_list:
        mode = cfg.get("top_unit_selection", "percentage").lower()
        cat = cfg["category"].lower()
        prefix = f"{cat}_{cat_tag}_{mode}_{top_frac}{ut_suffix}"
        activ_dir = outdir / f"activations_{prefix}_grad"
        diff_csv_path = outdir / f"{prefix}_v1_featuremap_selective_vs_random_grad.csv"

        print(f"\n=== Processing top_frac={top_frac} ===")
        
        sel_filename = f"{model_name}{ut_suffix}_all_layers_units_mannwhitneyu.pkl"
        sel_path = pathlib.Path(cfg["selectivity_csv_dir"]) / sel_filename
        sel      = pd.read_pickle(sel_path)
        mw_col   = f"mw_{cat}"
        if mw_col not in sel.columns:
            sys.exit(f"[ERROR] column {mw_col!r} not found in {sel_path}")

        it_rows = sel[sel["layer"] == "module.IT"]
        if len(it_rows) == 0:
            sys.exit("[ERROR] No rows found for layer 'module.IT' in selectivity file.")

        mode = cfg.get("top_unit_selection", "percentage").lower()
        top_units = []

        if mode == "percentage":
            k        = max(1, int(top_frac * len(it_rows)))
            top_rows = it_rows.nlargest(k, mw_col)
        elif mode == "percentile":
            pct = top_frac if isinstance(top_frac, (float, int)) else top_frac[0]
            cutoff = np.percentile(it_rows[mw_col], pct)
            top_rows = it_rows[it_rows[mw_col] >= cutoff]
        else:
            sys.exit(f"[ERROR] Unknown top_unit_selection mode: {mode}")

        it_layer_names = sorted(top_rows.layer.unique())
        model          = load_model(cfg["model"], GLOBAL_DEVICE)
        feature_shapes = {}
        mods = dict(model.named_modules())
        hooks = []
        def shape_hook(name):
            def fn(_m, _i, o):
                t = o[0] if isinstance(o, tuple) else o
                feature_shapes[name] = t.shape[1:]   # drop batch dim
            return fn
        for lname in it_layer_names:
            hooks.append(mods[lname].register_forward_hook(shape_hook(lname)))
        with torch.no_grad():
            _ = model(torch.zeros(1, 3, 224, 224, device=device))
        for h in hooks: h.remove()

        top_units = []
        for _, row in top_rows.iterrows():
            lay   = row.layer
            idx   = int(row.unit)
            C, H, W = feature_shapes[lay]
            c, y, x = flat_to_cyx(idx, C, H, W)
            top_units.append((lay, c, y, x))

        if mode == "percentile":
            print(f"Selected {len(top_units)} IT units at or above the {cfg.get('top_percentile', top_frac)}th percentile.")
        else:
            print(f"Selected {len(top_units)} IT units (top {top_frac:.0%}).")
        
        all_it_units = []
        for _, row in it_rows.iterrows():
            lay   = row.layer
            idx   = int(row.unit)
            C, H, W = feature_shapes[lay]
            c, y, x = flat_to_cyx(idx, C, H, W)
            all_it_units.append((lay, c, y, x))

        have_grad_files = activ_dir.exists()
        if os.path.exists(diff_csv_path):
            print(f"Found existing results at {diff_csv_path}, loading…")
            df = pd.read_csv(diff_csv_path)

            Hf = df["fy"].max() + 1
            Wf = df["fx"].max() + 1

            obs_map = df["mean_diff"].values.reshape(Hf, Wf)
            mean_diff_vec  = obs_map.ravel()  

            delta_map = None
            if effect_size == "mw_u" and "mw_u" in df.columns:
                delta_arr = pd.to_numeric(df["mw_u"], errors="coerce").astype(np.float32)
                delta_map = delta_arr.values.reshape(Hf, Wf)
            elif effect_size == "cliffs_delta" and "cliffs_delta" in df.columns:
                delta_arr = pd.to_numeric(df["cliffs_delta"], errors="coerce").astype(np.float32)
                delta_map = delta_arr.values.reshape(Hf, Wf)
            elif effect_size == "hedges_g" and "hedges_g" in df.columns:
                delta_arr = pd.to_numeric(df["hedges_g"], errors="coerce").astype(np.float32)
                delta_map = delta_arr.values.reshape(Hf, Wf)
            else:
                for col in ("mw_u", "cliffs_delta", "hedges_g"):
                    if col in df.columns:
                        delta_arr = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
                        delta_map = delta_arr.values.reshape(Hf, Wf)
                        break

            if "fdr_q_value" in df.columns:
                q_map = df["fdr_q_value"].values.reshape(Hf, Wf)

                if "significant_fdr" in df.columns:
                    sig_mask = df["significant_fdr"].astype(bool).values.reshape(Hf, Wf)
                else:
                    sig_mask = (q_map < 0.05)

            elif "bonferroni_p_value" in df.columns:
                q_map    = df["bonferroni_p_value"].values.reshape(Hf, Wf)
                sig_mask = (q_map < 0.05)

            else:
                raise ValueError(
                    "CSV exists but does not contain recognised significance columns "
                    "(expected 'fdr_q_value' or 'bonferroni_p_value'). "
                    "Delete the file and rerun to regenerate."
                )

        elif os.path.exists(activ_dir):
            print(f"Found existing activation gradients at {activ_dir}, loading...")
            all_v1_grads_sel = np.load(activ_dir / "grads_selective.npy")
            all_v1_grads_rand = [np.load(activ_dir / f"grads_random_{rep}.npy") for rep in range(n_random_repeats)]
            print(f"Loaded {len(all_v1_grads_rand)} random repeats.")
            all_v1_grads_rand = np.stack(all_v1_grads_rand)

            if not (activ_dir / "grads_random_stacked.npy").exists():
                print("No random stacked file found; stacking from individual repeats.")
                stacked_random = np.stack([
                    np.load(activ_dir / f"grads_random_{rep}.npy")
                    for rep in range(n_random_repeats)
                ])
                np.save(activ_dir / "grads_random_stacked.npy", stacked_random)

            N, C, Hf, Wf = all_v1_grads_sel.shape
            all_v1_grads_rand, _rand_collapsed = collapse_random(all_v1_grads_rand, collapse_method)

            if _rand_collapsed:
                sel_flat  = all_v1_grads_sel.reshape(-1, Hf * Wf)    # [N*C, H*W]
                rand_flat = all_v1_grads_rand.reshape(-1, Hf * Wf)   # [N*C, H*W]
            else:
                sel_flat  = all_v1_grads_sel.reshape(-1, Hf * Wf)    # [N*C, H*W]
                rand_flat = all_v1_grads_rand.reshape(-1, Hf * Wf)   # [R*N*C, H*W]


            mean_diff_vec = np.empty(Hf * Wf, dtype=np.float32)
            g_vec         = np.empty(Hf * Wf, dtype=np.float32)      # effect size
            p_vec         = np.empty(Hf * Wf, dtype=np.float64)      # Welch p‑values


            for i in tqdm(range(Hf * Wf), desc="voxel stats"):
                a = sel_flat[:, i]
                b = rand_flat[:, i]

                mean_diff_vec[i] = a.mean() - b.mean()
                if effect_size == "hedges_g":
                    g_vec[i] = hedges_g(a, b, eps=1e-12)
                elif effect_size == "mw_u":
                    g_vec[i] = mann_whitney_u_stat(a, b)
                else:
                    g_vec[i] = hedges_g(a, b, eps=1e-12)
                _, p_val   = ttest_ind(a, b, equal_var=False, nan_policy='omit')
                p_vec[i]   = p_val

            rej, q_vec, _, _ = multipletests(p_vec, alpha=0.05, method='fdr_bh')

            obs_map   = mean_diff_vec.reshape(Hf, Wf)
            effect_map = g_vec.reshape(Hf, Wf)
            delta_map  = effect_map
            q_map     = q_vec.reshape(Hf, Wf)
            sig_mask  = rej.reshape(Hf, Wf)

            df = pd.DataFrame({
                'fy'            : np.repeat(np.arange(Hf), Wf),
                'fx'            : np.tile  (np.arange(Wf), Hf),
                'mean_diff'     : mean_diff_vec,
                'hedges_g'      : g_vec,
                'mw_u'      : (g_vec if effect_size == "mw_u"      else np.full_like(g_vec, np.nan)),
                'welch_p_value' : p_vec,
                'fdr_q_value'   : q_vec,
                'significant'   : rej.astype(int)
            })

            df.to_csv(diff_csv_path, index=False)
            print("Saved feature-map statistics to", diff_csv_path)


        else:
            print("Computing mean |gradient| for selective IT units...")
            activ_dir.mkdir(parents=True, exist_ok=True)

            model, bufs = get_thread_model_and_bufs(cfg)

            all_v1_grads_sel = []
            for img in tqdm(imgs, desc="Images (selective)"):
                grads = grads_per_image(img, model, bufs, top_units).mean(axis=0)
                all_v1_grads_sel.append(grads)

            all_v1_grads_sel = np.stack(all_v1_grads_sel)
            np.save(str(activ_dir / "grads_selective.npy"), all_v1_grads_sel)
            del all_v1_grads_sel
            gc.collect()

            all_v1_grads_rand = []
            for rep in tqdm(range(n_random_repeats), desc="Random repeats"):
                rand_units = random.sample(all_it_units, len(top_units))

                rep_arr = []
                for img in imgs:
                    rep_arr.append(grads_per_image(img, model, bufs, rand_units).mean(axis=0))
                rep_arr = np.stack(rep_arr)

                np.save(str(activ_dir / f"grads_random_{rep}.npy"), rep_arr)
                del rep_arr
                gc.collect()

            # Load results
            all_v1_grads_sel = np.load(activ_dir / "grads_selective.npy")
            all_v1_grads_rand = [np.load(activ_dir / f"grads_random_{rep}.npy") for rep in range(n_random_repeats)]
            print(f"Loaded {len(all_v1_grads_rand)} random repeats.")
            all_v1_grads_rand = np.stack(all_v1_grads_rand)

            np.save(activ_dir / "grads_random_stacked.npy", all_v1_grads_rand)

            print("Mean selective gradient magnitude:", np.mean(all_v1_grads_sel))
            print("Mean random gradient magnitude:", np.mean(all_v1_grads_rand))

            all_v1_grads_rand, _rand_collapsed = collapse_random(all_v1_grads_rand, collapse_method)

            N, C, Hf, Wf = all_v1_grads_sel.shape
            if _rand_collapsed:
                sel_flat  = all_v1_grads_sel.reshape(-1, Hf * Wf)    # [N*C, H*W]
                rand_flat = all_v1_grads_rand.reshape(-1, Hf * Wf)   # [N*C, H*W]
            else:
                sel_flat  = all_v1_grads_sel.reshape(-1, Hf * Wf)    # [N*C, H*W]
                rand_flat = all_v1_grads_rand.reshape(-1, Hf * Wf)   # [R*N*C, H*W]


            mean_diff_vec  = np.empty(Hf * Wf, dtype=np.float32)
            cliff_vec      = np.empty(Hf * Wf, dtype=np.float32)
            p_vec          = np.empty(Hf * Wf, dtype=np.float64)

            for i in tqdm(range(Hf * Wf), desc="voxel stats"):
                a = sel_flat[:, i]
                b = rand_flat[:, i]

                mean_diff_vec[i] = a.mean() - b.mean()
                if effect_size == "cliffs_delta":
                    cliff_vec[i] = cliffs_delta(a, b)
                elif effect_size == "mw_u":
                    cliff_vec[i] = mann_whitney_u_stat(a, b)
                else:
                    cliff_vec[i] = cliffs_delta(a, b)
                p_vec[i]         = mannwhitneyu(a, b, alternative='two-sided').pvalue

            rej, q_vec, _, _ = multipletests(p_vec, alpha=0.05, method='fdr_bh')

            obs_map   = mean_diff_vec.reshape(Hf, Wf)
            delta_map = cliff_vec.reshape(Hf, Wf)
            q_map     = q_vec.reshape(Hf, Wf)
            sig_mask  = rej.reshape(Hf, Wf)

            df = pd.DataFrame({
                'fy'               : np.repeat(np.arange(Hf), Wf),
                'fx'               : np.tile  (np.arange(Wf), Hf),
                'mean_diff'        : mean_diff_vec,
                'cliffs_delta' : (cliff_vec if effect_size == "cliffs_delta" else np.full_like(cliff_vec, np.nan)),
                'mw_u'          : (cliff_vec if effect_size == "mw_u"         else np.full_like(cliff_vec, np.nan)),
                'mw_p_value'       : p_vec,
                'fdr_q_value'      : q_vec,
                'significant_fdr'  : rej.astype(int)
            })
            df.to_csv(diff_csv_path, index=False)
            print("Saved feature-map statistics to", diff_csv_path)

        plt.figure(figsize=(6,5))
        plt.imshow(obs_map, cmap="bwr", interpolation='nearest')
        plt.title(f"Mean |grad| diff  (sel−rand, top_frac={top_frac})")
        plt.axis('off')
        plt.colorbar(label="Δ |grad|")
        plt.savefig(outdir / f"{prefix}_A_mean_diff.png", dpi=300)

        delta_plot = np.where(sig_mask, delta_map, np.nan)
        plt.figure(figsize=(6,5))
        plt.imshow(delta_plot, cmap="bwr", interpolation='nearest')
        plt.axis('off')
        eff_label  = {"hedges_g": "Hedges’ g", "cliffs_delta": "Cliff’s δ", "mw_u": "Mann–Whitney U"}[effect_size]
        file_suffix = {"hedges_g": "hedges_g", "cliffs_delta": "cliffs_delta", "mw_u": "mw_u"}[effect_size]

        plt.title(eff_label)
        plt.colorbar(label=eff_label)
        plt.savefig(outdir / f"{prefix}_B_{file_suffix}.png", dpi=300)

        q = q_map
        q[~sig_mask] = np.nan
        plt.figure(figsize=(6,5))
        plt.imshow(q, cmap="viridis", interpolation='nearest')
        plt.title("FDR q")
        plt.axis('off')
        plt.colorbar(label="q")
        plt.savefig(outdir / f"{prefix}_C_q.png", dpi=300)

        if ("all_v1_grads_sel" not in locals() or all_v1_grads_sel is None) and have_grad_files:
            print("Loading raw gradient arrays for histogram plotting …")
            all_v1_grads_sel = np.load(activ_dir / "grads_selective.npy")
            all_v1_grads_rand = np.stack([
                np.load(activ_dir / f"grads_random_{rep}.npy")
                for rep in range(n_random_repeats)
            ])
        if "all_v1_grads_sel" in locals() and all_v1_grads_sel is not None:
            num_voxels_to_plot = 10
            flat_idx = np.argsort(-np.abs(mean_diff_vec))[:num_voxels_to_plot]
            coords_to_plot = [(idx // Wf, idx % Wf) for idx in flat_idx]

            for (y, x) in coords_to_plot:
                sel_vals  = all_v1_grads_sel[:, :, y, x].ravel()
                if all_v1_grads_rand.ndim == 4:
                    rand_vals = all_v1_grads_rand[:, :, y, x].ravel()
                else:
                    rand_vals = all_v1_grads_rand[:, :, :, y, x].ravel()


                eps = 1e-8  # avoid zeros on log‑scale
                sel_vals += eps
                rand_vals += eps

                plt.figure(figsize=(4,3))
                bins = np.logspace(
                    np.log10(min(sel_vals.min(), rand_vals.min())),
                    np.log10(max(sel_vals.max(), rand_vals.max())),
                    50
                )
                plt.hist(rand_vals, bins=bins, alpha=0.5, label="random",   density=True)
                plt.hist(sel_vals,  bins=bins, alpha=0.5, label="selective", density=True)
                plt.xscale("log")
                plt.xlabel("|grad|")
                plt.ylabel("density")
                plt.legend()
                plt.title(f"Voxel (y={y}, x={x})  top_frac={top_frac}")
                plt.tight_layout()

                fname = outdir / f"{prefix}_v1_hist_voxel_{y}_{x}.png"
                plt.savefig(fname, dpi=300)
                plt.close()

            print(f"Saved {num_voxels_to_plot} voxel histograms to {outdir}")

        per_img_cfg = cfg.get("per_image_overlay", {})
        if per_img_cfg.get("enabled", False) and per_img_cfg.get("indices"):
            if ("all_v1_grads_sel" not in locals()) or ("all_v1_grads_rand" not in locals()):
                if activ_dir.exists():
                    all_v1_grads_sel = np.load(activ_dir / "grads_selective.npy")
                    all_v1_grads_rand = np.stack([
                        np.load(activ_dir / f"grads_random_{rep}.npy")
                        for rep in range(n_random_repeats)
                    ])
                else:
                    raise RuntimeError(
                        "Per-image overlays requested but activation gradients not found. "
                        f"Expected arrays under {activ_dir}."
                    )

            def get_preprocessed_image(idx: int) -> np.ndarray:
                """
                Return the displayed image aligned with the transformed tensor.
                """
                pil_img = imgs[idx]
                x = GLOBAL_TFM(pil_img)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                x = (x * std + mean).clamp(0, 1)
                x = x.permute(1, 2, 0).cpu().numpy()
                return x

            if (activ_dir / "grads_random_stacked.npy").exists():
                all_v1_grads_rand_base = np.load(activ_dir / "grads_random_stacked.npy", mmap_mode="r")
            else:
                all_v1_grads_rand_base = np.stack([
                    np.load(activ_dir / f"grads_random_{rep}.npy")
                    for rep in range(n_random_repeats)
                ])
                np.save(activ_dir / "grads_random_stacked.npy", all_v1_grads_rand_base)
                print("Saved stacked random gradients:", activ_dir / "grads_random_stacked.npy")

            run_per_image_overlays(
                all_v1_grads_sel=all_v1_grads_sel,
                all_v1_grads_rand=all_v1_grads_rand_base,
                effect_size=effect_size,
                per_img_cfg=per_img_cfg,
                outdir=outdir,
                get_preprocessed_image=get_preprocessed_image,
                cat_tag=cat_tag,
                selectivity_frac=top_frac,
            )




       


if __name__=="__main__":
    if len(sys.argv)!=2:
        sys.exit("usage: python it2v1_featuremap_contrib.py <config.yaml>")
    main(sys.argv[1])
