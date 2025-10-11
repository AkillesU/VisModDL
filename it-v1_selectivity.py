#!/usr/bin/env python3
"""
it2v1_input_contrib.py  ·  V1-to-input contributon with PFI

For a top-fraction of IT units: 
1. Compute their activation-based loss.
2. Backpropagate to V1 feature-map outputs to get per-unit gradients.
3. Weight each V1 activation by its gradient (|grad * activation|) → per-unit importance.
4. Project each V1 spatial unit back to input-pixel coordinates via receptive-field centroids.
5. Splat per-unit importance into a 224×224 input-space importance map, normalize.
6. Compute Peripheral–Foveal Index of that map and plot a heatmap.
"""

from __future__ import annotations

import sys
import yaml
import pathlib
import math
import torch
import torch.nn as nn
import torchvision.transforms as T
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from scipy.stats import mannwhitneyu, ttest_ind
import os
from scipy.ndimage import gaussian_filter
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from statsmodels.stats.multitest import multipletests
from typing import List, Iterable
import threading
import gc, uuid

_thread_cache = threading.local()      # each executor thread gets its own slot

# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers – unchanged
# ──────────────────────────────────────────────────────────────────────────────
# ------------------------------------------------------------------
# Globals initialised once in main()
# ------------------------------------------------------------------
GLOBAL_TFM   : T.Compose | None = None   # image pre-processing
GLOBAL_DEVICE: torch.device | None = None


def permutation_test(sel_vals, rand_vals, n_permutations=1000, alternative='two-sided', random_state=None):
    """
    Permutation test for difference in means between two groups.
    Returns (observed_stat, p_value).
    """
    rng = np.random.default_rng(random_state)
    sel_vals = np.asarray(sel_vals)
    rand_vals = np.asarray(rand_vals)
    n_sel = len(sel_vals)
    n_rand = len(rand_vals)
    all_vals = np.concatenate([sel_vals, rand_vals])
    group_labels = np.array([0]*n_sel + [1]*n_rand)
    observed = np.mean(sel_vals) - np.mean(rand_vals)
    perm_stats = []
    for _ in range(n_permutations):
        rng.shuffle(group_labels)
        perm_sel = all_vals[group_labels == 0]
        perm_rand = all_vals[group_labels == 1]
        # If group sizes change due to shuffling, skip this permutation
        if len(perm_sel) != n_sel or len(perm_rand) != n_rand:
            continue
        stat = np.mean(perm_sel) - np.mean(perm_rand)
        perm_stats.append(stat)
    perm_stats = np.array(perm_stats)
    if alternative == 'two-sided':
        p = np.mean(np.abs(perm_stats) >= np.abs(observed))
    elif alternative == 'greater':
        p = np.mean(perm_stats >= observed)
    else:
        p = np.mean(perm_stats <= observed)
    return observed, p

# ------------------------------------------------------------------
def mw_u(a: np.ndarray, b: np.ndarray) -> float:
    """
    Mann–Whitney U statistic for the first sample (two-sided test call).
    Range: 0 .. (len(a)*len(b)). Larger ⇒ sample a tends to be greater than b.
    """
    return mannwhitneyu(a, b, alternative='two-sided').statistic

def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """
    Hedges’ g (bias‑corrected pooled‑SD Cohen’s d).
    Works with unequal sample sizes & variances.
    """
    n1, n2   = len(a), len(b)
    s1, s2   = a.std(ddof=1), b.std(ddof=1)
    sp       = math.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1 + n2 - 2) + 1e-12)
    d        = (a.mean() - b.mean()) / sp
    J        = 1 - (3 / (4*(n1 + n2) - 9))      # bias‑correction
    return d * J
# ------------------------------------------------------------------

# ──────────────────────────────────────────────────────────────────────────────
# Model & image helpers
# ──────────────────────────────────────────────────────────────────────────────


def _slice_rand_for_index(rand_all: np.ndarray, idx: int) -> np.ndarray:
    """
    Return a per-image random tensor for index idx in one of the supported shapes:
      [R,C,H,W]  or  [C,H,W]  or  [H,W]
    """
    if rand_all.ndim == 5:            # [R, N, C, H, W]
        return rand_all[:, idx]       # [R, C, H, W]
    if rand_all.ndim == 4:
        # Either [N,C,H,W]   → per-image [C,H,W]
        # or    [R,H,W,N]    (unlikely, but guard)
        if rand_all.shape[0] == all_v1_grads_sel.shape[0]:   # N
            return rand_all[idx]       # [C,H,W]
        elif rand_all.shape[-1] == all_v1_grads_sel.shape[0]:
            return rand_all[:, :, :, idx]  # [R,H,W]  → will be handled in 2-D branch below after flatten
        else:
            return rand_all[idx]       # best effort
    if rand_all.ndim == 3:
        # Could be [N,H,W] or [C,H,W]; prefer per-image first
        if rand_all.shape[0] == all_v1_grads_sel.shape[0]:   # N
            return rand_all[idx]       # [H,W]
        else:
            return rand_all            # [C,H,W] assumed
    return rand_all  # pass through (will be validated downstream)


def load_model(mcfg, device):
    import cornet
    ctor = {
        "cornet_rt": cornet.cornet_rt,
        "cornet_s" : cornet.cornet_s,
        "cornet_z" : cornet.cornet_z
    }[mcfg["name"].lower()]
    kwargs = {"pretrained": True}
    kwargs["map_location"] = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    if mcfg["name"].lower() == "cornet_rt":
        kwargs["times"] = mcfg.get("time_steps", 5)
    return ctor(**kwargs).to(device).eval()


def build_transform():
    return T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])


def iter_imgs(folder: pathlib.Path) -> Iterable[Image.Image]:
    """Yield all RGB images in *folder* (jpg/jpeg/png), sorted alphabetically."""
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            yield Image.open(p).convert("RGB")

# ──────────────────────────────────────────────────────────────────────────────
# NEW helper – gather images from (possibly) multiple categories
# ──────────────────────────────────────────────────────────────────────────────

def gather_images(cfg: dict) -> tuple[List[Image.Image], str]:
    """Return (images, category_tag) according to the new config logic."""

    max_imgs = cfg.get("max_images", 20)
    img_folders: List[pathlib.Path] = []

    # Priority 1 – explicit directory list
    if cfg.get("image_category_dirs"):
        img_folders = [pathlib.Path(d).expanduser() for d in cfg["image_category_dirs"]]
        categories = [f.name for f in img_folders]

    elif cfg.get("image_categories"):
        # Priority 2 – sibling category names
        base = pathlib.Path(cfg["category_images"]).expanduser().resolve()
        if not base.parent.exists():
            raise FileNotFoundError(f"Base directory {base.parent} not found")
        categories = list(cfg["image_categories"])
        img_folders = [base.parent / c for c in categories]

    else:
        # Fallback – original single folder
        base = pathlib.Path(cfg["category_images"]).expanduser().resolve()
        if not base.exists():
            raise FileNotFoundError(f"Image directory {base} does not exist")
        img_folders = [base]
        categories = [cfg["category"]]

    # Build a compact tag: first letter of each category, sorted & uniq‑preserve
    cat_tag = "".join(sorted({c[0].lower() for c in categories}))

    imgs: List[Image.Image] = []
    for folder in img_folders:
        if not folder.exists():
            print(f"[WARN] Folder {folder} missing – skipped.")
            continue
        imgs.extend(iter_imgs(folder))

    if not imgs:
        raise RuntimeError("No images found across the requested categories.")

    return imgs[:max_imgs], cat_tag

# ──────────────────────────────────────────────────────────────────────────────
# Receptive‑field utility (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def rf_px(fy, fx, stride, padding, kernel):
    # center = (fy * stride + (kernel-1)/2 - padding)
    cy = fy * stride + (kernel-1)/2 - padding
    cx = fx * stride + (kernel-1)/2 - padding
    return cy, cx


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

    v1 = mods["module.V1.nonlin_input"]    # will receive grad-hook later
    target_names = {"module.IT", "module.V1.nonlin_input"}

    # forward hooks – save activations
    def make_fwd_hook(name):
        def f(_m, _in, out): bufs.out[name] = out
        return f

    for name, mod in mods.items():
        if name in target_names:
            mod.register_forward_hook(make_fwd_hook(name))

    # gradient hook on V1 output
    def v1_grad_hook(grad): bufs.grad["module.V1.nonlin_input"] = grad
    def v1_fwd_hook(_m, _i, out):
        out.register_hook(v1_grad_hook)   # attach the gradient hook
        # DO NOT return anything → return value is None

    v1.register_forward_hook(v1_fwd_hook)

    return bufs       # <-- one buffers object per model


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
      - else: original arr
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
    """Accepts HxW, HxWx1, or HxWx3 in [0..1] or [0..255] → returns HxWx3 uint8."""
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
              [H, W]        (fully collapsed across repeats and channels)  ← defensive support
    Returns eff_map: [H, W]
    """
    # --- shapes ---
    if sel_img.ndim != 3:
        raise ValueError(f"Expected sel_img as [C,H,W], got shape {sel_img.shape}")
    C, H, W = sel_img.shape

    # build a function to compute the effect between two 1-D vectors
    def _eff(a: np.ndarray, b: np.ndarray) -> float:
        if effect_size == "hedges_g":
            return hedges_g(a, b)
        elif effect_size == "cliffs_delta":
            return cliffs_delta(a, b)
        elif effect_size == "mw_u":
            return mw_u(a, b)
        else:
            raise ValueError(f"Unknown effect_size: {effect_size}")

    eff_map = np.empty((H, W), dtype=np.float32)

    if rand_img.ndim == 4:
        # [R, C, H, W] → flatten repeats×channels per voxel
        for y in range(H):
            for x in range(W):
                a = sel_img[:, y, x]                      # len = C
                b = rand_img[:, :, y, x].reshape(-1)      # len = R*C
                eff_map[y, x] = _eff(a, b)

    elif rand_img.ndim == 3:
        # [C, H, W] → channel-wise comparison per voxel
        if rand_img.shape[0] != C:
            raise ValueError(f"rand_img first dim (C={rand_img.shape[0]}) "
                             f"does not match sel_img C={C}")
        for y in range(H):
            for x in range(W):
                a = sel_img[:, y, x]          # len = C
                b = rand_img[:, y, x]         # len = C
                eff_map[y, x] = _eff(a, b)

    elif rand_img.ndim == 2:
        # [H, W] → fully collapsed random map. We can only compare each voxel’s
        # scalar to the selective channel distribution by repeating that scalar.
        # This preserves directionality for Hedges g & MW U; Cliff’s δ is defined
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


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cliff’s δ for two 1‑D vectors (uses pairwise compare, still fast for <10⁴ items).
    Range −1‥+1.  Positive ⇒ selective > random.
    """
    a = a.ravel();  b = b.ravel()
    gt = (a[:, None] > b).sum()
    lt = (a[:, None] < b).sum()
    return (gt - lt) / (len(a) * len(b) + 1e-12)


def run_per_image_overlays(
    all_v1_grads_sel: np.ndarray,             # [N, C, H, W]
    all_v1_grads_rand: np.ndarray,            # [R, N, C, H, W] or [N, C, H, W] if collapsed
    effect_size: str,                         # "hedges_g" | "cliffs_delta" | "mw_u"
    per_img_cfg: dict,
    outdir: Path,
    get_preprocessed_image=None,              # callable: idx -> np.ndarray (H,W,3) in 0..1 or 0..255
    cat_tag="faces",
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
    rand_collapsed = (all_v1_grads_rand.ndim == 4)  # [N, C, H, W]
    if not rand_collapsed:
        R = all_v1_grads_rand.shape[0]
    else:
        R = 1  # semantically “one random sample set” for bookkeeping

    # ---------- Pass 1: compute global min/max if requested ----------
    global_min = None
    global_max = None
    if viz_norm == "global":
        mins = []
        maxs = []
        for idx in indices:
            if not (0 <= idx < N):
                continue
            sel_img  = all_v1_grads_sel[idx]                        # [C,H,W]
            rand_img = _slice_rand_for_index(all_v1_grads_rand, idx)
            print(f"[overlay] idx={idx}  sel_img={sel_img.shape}  rand_img={np.shape(rand_img)}")
            eff_map  = _compute_effect_map_for_image(sel_img, rand_img, effect_size)
            mins.append(np.nanmin(eff_map))
            maxs.append(np.nanmax(eff_map))
        if mins and maxs:
            global_min = float(np.nanmin(mins))
            global_max = float(np.nanmax(maxs))

    # ---------- Pass 2: render overlays ----------
    for idx in indices:
        if not (0 <= idx < N):
            print(f"[per_image_overlay] Skipping idx {idx} (out of range 0..{N-1})")
            continue

        # compute effect map (H x W)
        sel_img  = all_v1_grads_sel[idx]                        # [C,H,W]
        rand_img = all_v1_grads_rand[idx] if rand_collapsed else all_v1_grads_rand[:, idx]
        eff_map  = _compute_effect_map_for_image(sel_img, rand_img, effect_size)

        # save raw per-image map (before any visual scaling)
        if save_raw:
            np.save(overlay_dir / f"img{idx:05d}_{effect_size}_map.npy", eff_map)

        # figure out sample sizes (for U normalisation if chosen)
        if rand_collapsed:
            n_a, n_b = sel_img.shape[0], rand_img.shape[0]          # C, C
        else:
            n_a, n_b = sel_img.shape[0], rand_img.shape[0] * rand_img.shape[1]  # C, R*C

        # get the preprocessed image (must align with the gradients)
        if get_preprocessed_image is not None:
            base_img = get_preprocessed_image(idx)                  # expected HxWx3 in 0..1 or 0..255
        else:
            # ---- TODO: Replace with your project’s image-fetch logic ----
            # For example, if you keep preprocessed tensors in memory:
            #   base_img = (preprocessed_imgs[idx].permute(1,2,0).cpu().numpy())  # HxWx3 in [0..1]
            raise RuntimeError(
                "Please provide `get_preprocessed_image` to `run_per_image_overlays` "
                "so we can overlay on the correctly preprocessed (aligned) image."
            )

        base_img = _ensure_uint8_rgb(np.asarray(base_img))
        H_img, W_img = base_img.shape[:2]

        # visual normalisation (optional)
        eff_viz, norm_kwargs = _normalise_for_viz(
            eff_map, viz_norm=viz_norm, effect_size=effect_size,
            n_a=n_a, n_b=n_b, global_min=global_min, global_max=global_max
        )

        # resize effect map to the image dimensions
        eff_up = _resize_map_to_img(eff_viz, (H_img, W_img))

        # plot overlay
        fig = plt.figure(figsize=(6, 6))
        plt.imshow(base_img)
        im = plt.imshow(eff_up, cmap=cmap, alpha=alpha, interpolation="bilinear", **norm_kwargs)
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label(effect_size)
        plt.axis("off")

        out_png = overlay_dir / f"{cat_tag}_img{idx:05d}_{effect_size}_overlay.png"
        plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        # also store globally scaled map if requested and viz_norm==global/u_unit
        if save_raw and (viz_norm in {"global", "u_unit"}):
            np.save(overlay_dir / f"img{idx:05d}_{effect_size}_map_vizscaled.npy", eff_viz)

        print(f"[per_image_overlay] Saved {out_png}")

# ──────────────────────────────────────────────────────────────────────────────
#                                 MAIN
# ──────────────────────────────────────────────────────────────────────────────

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
    GLOBAL_TFM    = build_transform()        # replaces local tfm
    # Thread handling (unchanged)
    if torch.cuda.is_available() is False:
        try:
            num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
        except TypeError:
            num_threads = 1
        torch.set_num_threads(num_threads)
        print(f"✅ PyTorch intra‑op parallelism set to {torch.get_num_threads()} threads.")
    else:
        print("Using CUDA")

    # Figure out which image categories to evaluate --------------
    imgs, cat_tag = gather_images(cfg)
    print(f"Loaded {len(imgs)} images; category‑tag = '{cat_tag}'.")

    outdir = pathlib.Path(cfg.get("output_dir", "it2v1_analysis"))
    outdir.mkdir(parents=True, exist_ok=True)

    # Everything below remains *functionally* identical – only the name of the
    # `imgs` list changed from the original single‑folder logic.  ───────────

    top_frac_cfg = cfg.get("top_frac", 0.1)
    if isinstance(top_frac_cfg, (float, int)):
        top_frac_list = [top_frac_cfg]
    else:
        top_frac_list = list(top_frac_cfg)

    for top_frac in top_frac_list:
        mode = cfg.get("top_unit_selection", "percentage").lower()
        cat = cfg["category"].lower()
        prefix = f"{cat}_{cat_tag}_{mode}_{top_frac}"
        activ_dir = outdir / f"activations_{prefix}_grad"
        diff_csv_path = outdir / f"{prefix}_v1_featuremap_selective_vs_random_grad.csv"

        print(f"\n=== Processing top_frac={top_frac} ===")
        
        sel_path = pathlib.Path(cfg["selectivity_csv_dir"]) / "all_layers_units_mannwhitneyu.pkl"
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
        probe_model = model                 # keep a reference

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
        
        imgs = list(iter_imgs(pathlib.Path(cfg["category_images"])))[:cfg.get("max_images",20)]
        img_tuples  = [img for img in imgs]
        
        # --- get all IT units for random selection ---
        all_it_units = []
        for _, row in it_rows.iterrows():
            lay   = row.layer
            idx   = int(row.unit)
            C, H, W = feature_shapes[lay]
            c, y, x = flat_to_cyx(idx, C, H, W)
            all_it_units.append((lay, c, y, x))

        # === Check for existing results ===
        have_grad_files = activ_dir.exists()
        if os.path.exists(diff_csv_path):
            print(f"Found existing results at {diff_csv_path}, loading…")
            df = pd.read_csv(diff_csv_path)
            

            Hf = df["fy"].max() + 1
            Wf = df["fx"].max() + 1
            if 'hedges_g' in df.columns:
                delta_map = df['hedges_g'].values.reshape(Hf, Wf)
            elif 'cliffs_delta' in df.columns:      # legacy support
                delta_map = df['cliffs_delta'].values.reshape(Hf, Wf)
            else:
                delta_map = None
            

            # ---- mandatory column ----
            obs_map = df["mean_diff"].values.reshape(Hf, Wf)
            mean_diff_vec  = obs_map.ravel()  

            # ---- effect-size map selection (respect config if present) ----
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
                # backward-compatible fallback: try any known column, priority mw_u > cliffs_delta > hedges_g
                for col in ("mw_u", "cliffs_delta", "hedges_g"):
                    if col in df.columns:
                        delta_arr = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
                        delta_map = delta_arr.values.reshape(Hf, Wf)
                        break


            # ---- significance / p‑value handling -------------------------
            if "fdr_q_value" in df.columns:          # FDR‑corrected p values exist
                q_map = df["fdr_q_value"].values.reshape(Hf, Wf)

                if "significant_fdr" in df.columns:  # column present (new files)
                    sig_mask = df["significant_fdr"].astype(bool).values.reshape(Hf, Wf)
                else:                                # older files → derive the mask
                    sig_mask = (q_map < 0.05)

            elif "bonferroni_p_value" in df.columns: # very old schema
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
            N, C, Hf, Wf = all_v1_grads_sel.shape
            R = n_random_repeats

            # Optionally collapse random repeats so N matches selective
            all_v1_grads_rand, _rand_collapsed = collapse_random(all_v1_grads_rand, collapse_method)
            if _rand_collapsed:
                # Now shapes match: random is [N, C, H, W], same as selective
                sel_flat  = all_v1_grads_sel.reshape(-1, Hf * Wf)    # [N*C, H*W]
                rand_flat = all_v1_grads_rand.reshape(-1, Hf * Wf)   # [N*C, H*W]
            else:
                # Keep existing behavior (all repeats count as samples)
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
                    g_vec[i] = hedges_g(a, b)
                elif effect_size == "mw_u":
                    g_vec[i] = mw_u(a, b)
                else:
                    # keep parity if someone requests cliffs on parametric path, or raise
                    g_vec[i] = hedges_g(a, b)
                _, p_val   = ttest_ind(a, b, equal_var=False, nan_policy='omit') # Welch's t-test
                p_vec[i]   = p_val

            # FDR (Benjamini‑Hochberg) correction ------------------------------
            rej, q_vec, _, _ = multipletests(p_vec, alpha=0.05, method='fdr_bh')

            # reshape back to (H, W) ------------------------------------------
            obs_map   = mean_diff_vec.reshape(Hf, Wf)
            effect_map = g_vec.reshape(Hf, Wf)       # Hedges g map
            delta_map  = effect_map                  # keeps old variable name used in plots
            q_map     = q_vec.reshape(Hf, Wf)
            sig_mask  = rej.reshape(Hf, Wf)

            # ─────────── save CSV (so future runs can reuse) ────────────────
            df = pd.DataFrame({
                'fy'            : np.repeat(np.arange(Hf), Wf),
                'fx'            : np.tile  (np.arange(Wf), Hf),
                'mean_diff'     : mean_diff_vec,
                'hedges_g'      : g_vec,            # ← new 
                'mw_u'      : (g_vec if effect_size == "mw_u"      else np.full_like(g_vec, np.nan)),
                'welch_p_value' : p_vec,            # ← new column name
                'fdr_q_value'   : q_vec,
                'significant'   : rej.astype(int)
            })

            df.to_csv(diff_csv_path, index=False)
            print("Saved Cliff’s δ & FDR stats →", diff_csv_path)

            


        else:
            
            print("Computing mean |gradient| for selective IT units (parallelized)...")
            all_v1_grads_sel = []
            activ_dir.mkdir(parents=True, exist_ok=True)

            model, bufs = get_thread_model_and_bufs(cfg)

            all_v1_grads_sel = []
            for img in tqdm(imgs, desc="Images (selective)"):
                grads = grads_per_image(img, model, bufs, top_units).mean(axis=0) # Mean V1 gradient across IT units
                all_v1_grads_sel.append(grads)

            all_v1_grads_sel = np.stack(all_v1_grads_sel)
            np.save(str(activ_dir / "grads_selective.npy"), all_v1_grads_sel)
            del all_v1_grads_sel
            gc.collect()

            # -------- many random repeats ------------------
            all_v1_grads_rand = []
            for rep in tqdm(range(n_random_repeats), desc="Random repeats"):
                rand_units = random.sample(all_it_units, len(top_units))

                rep_arr = []
                for img in imgs:
                    rep_arr.append(grads_per_image(img, model, bufs, rand_units).mean(axis=0)) # Append mean IT unit gradient per V1 unit
                rep_arr = np.stack(rep_arr)                         # [N_img, S, C, H, W]

                np.save(str(activ_dir / f"grads_random_{rep}.npy"), rep_arr)
                del rep_arr
                gc.collect()

            # Load results
            all_v1_grads_sel = np.load(activ_dir / "grads_selective.npy")
            all_v1_grads_rand = [np.load(activ_dir / f"grads_random_{rep}.npy") for rep in range(n_random_repeats)]
            print(f"Loaded {len(all_v1_grads_rand)} random repeats.")
            all_v1_grads_rand = np.stack(all_v1_grads_rand)
            print("Mean selective gradient magnitude:", np.mean(all_v1_grads_sel))
            print("Mean random gradient magnitude:", np.mean(all_v1_grads_rand))

            all_v1_grads_rand, _rand_collapsed = collapse_random(all_v1_grads_rand, collapse_method)
            

            # ──────────────────────────────────────────────────────────────────
            # NEW voxel‑wise statistics: mean‑difference, Cliff’s δ, M‑W + FDR
            # ──────────────────────────────────────────────────────────────────
            

            def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
                """
                Cliff’s δ for two 1‑D vectors (uses pairwise compare, still fast for <10⁴ items).
                Range −1‥+1.  Positive ⇒ selective > random.
                """
                a = a.ravel();  b = b.ravel()
                gt = (a[:, None] > b).sum()
                lt = (a[:, None] < b).sum()
                return (gt - lt) / (len(a) * len(b) + 1e-12)

            N, C, Hf, Wf = all_v1_grads_sel.shape
            R = n_random_repeats
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
                    cliff_vec[i] = mw_u(a, b)
                else:
                    # default to cliffs on the NP branch
                    cliff_vec[i] = cliffs_delta(a, b)
                p_vec[i]         = mannwhitneyu(a, b, alternative='two-sided').pvalue

            # FDR (Benjamini‑Hochberg) correction ------------------------------
            rej, q_vec, _, _ = multipletests(p_vec, alpha=0.05, method='fdr_bh')

            # reshape back to (H, W) ------------------------------------------
            obs_map   = mean_diff_vec.reshape(Hf, Wf)
            delta_map = cliff_vec.reshape(Hf, Wf)
            q_map     = q_vec.reshape(Hf, Wf)
            sig_mask  = rej.reshape(Hf, Wf)

            # ─────────── save CSV (so future runs can reuse) ────────────────
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
            print("Saved Cliff’s δ & FDR stats →", diff_csv_path)

           

            print(f"Saved histograms ➜ {outdir}")

        # ──────────────── PLOTS ─────────────────────────────────────────
        plt.figure(figsize=(6,5))
        plt.imshow(obs_map, cmap="bwr", interpolation='nearest')
        plt.title(f"Mean |grad| diff  (sel−rand, top_frac={top_frac})")
        plt.axis('off')
        plt.colorbar(label="Δ |grad|")
        plt.savefig(outdir / f"{prefix}_A_mean_diff.png", dpi=300)

        # Cliff’s δ – show only significant voxels (others as NaN/white)
        delta_plot = np.where(sig_mask, delta_map, np.nan)
        plt.figure(figsize=(6,5))
        plt.imshow(delta_plot, cmap="bwr", interpolation='nearest')
        plt.axis('off')
        eff_label  = {"hedges_g": "Hedges’ g", "cliffs_delta": "Cliff’s δ", "mw_u": "Mann–Whitney U"}[effect_size]
        file_suffix = {"hedges_g": "hedges_g", "cliffs_delta": "cliffs_delta", "mw_u": "mw_u"}[effect_size]

        plt.title(eff_label)
        plt.colorbar(label=eff_label)
        plt.savefig(outdir / f"{prefix}_B_{file_suffix}.png", dpi=300)

        # –log10(q) significance map, same mask
        q = q_map
        q[~sig_mask] = np.nan
        plt.figure(figsize=(6,5))
        plt.imshow(q, cmap="viridis", interpolation='nearest')
        plt.title("FDR q")
        plt.axis('off')
        plt.colorbar(label="q")
        plt.savefig(outdir / f"{prefix}_C_q.png", dpi=300)

        # ───────── DIAGNOSTIC HISTOGRAMS ───────────────────────
        # How many voxels to visualise

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
                if all_v1_grads_rand.ndim == 4:   # collapsed → [N, C, H, W]
                    rand_vals = all_v1_grads_rand[:, :, y, x].ravel()
                else:                              # not collapsed → [R, N, C, H, W]
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

            print(f"Saved {num_voxels_to_plot} voxel histograms ➜ {outdir}")


                # ─────────────── PER-IMAGE EFFECT OVERLAYS (optional) ───────────────
        per_img_cfg = cfg.get("per_image_overlay", {})
        if per_img_cfg.get("enabled", False) and per_img_cfg.get("indices"):
            # Ensure we have the raw gradient arrays in memory (load if needed)
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

            # Apply the same collapse policy as the main analysis
            all_v1_grads_rand, _ = collapse_random(all_v1_grads_rand, collapse_method)

            # Provide a preprocessed-image getter that matches the exact transform
            def get_preprocessed_image(idx: int) -> np.ndarray:
                """
                Return the *preprocessed* image that aligns with the V1 maps for image idx.
                Output: HxWx3 in [0..1] float (uint8 also OK).
                """
                pil_img = imgs[idx]  # imgs is the list used for the analysis above
                # Use the *exact* transform used during forward pass (GLOBAL_TFM),
                # then de-normalize back to display space so colors look right.
                x = GLOBAL_TFM(pil_img)  # [3,224,224] normalized
                # De-normalize (ImageNet stats)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                x = (x * std + mean).clamp(0, 1)  # back to [0..1]
                x = x.permute(1, 2, 0).cpu().numpy()  # HxWx3
                return x

            # Run the overlay generator
            run_per_image_overlays(
                all_v1_grads_sel=all_v1_grads_sel,            # [N,C,H,W]
                all_v1_grads_rand=all_v1_grads_rand,          # [R,N,C,H,W] or [N,C,H,W] if collapsed
                effect_size=effect_size,                      # "hedges_g" | "cliffs_delta" | "mw_u"
                per_img_cfg=per_img_cfg,
                outdir=outdir,
                get_preprocessed_image=get_preprocessed_image,
                cat_tag=cat_tag
            )



       


if __name__=="__main__":
    if len(sys.argv)!=2:
        sys.exit("usage: python it2v1_featuremap_contrib.py <config.yaml>")
    main(sys.argv[1])
