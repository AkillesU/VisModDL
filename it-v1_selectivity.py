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

# ──────────────────────────────────────────────────────────────────────────────
#                                 MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main(cfg_path: str | pathlib.Path):
    random.seed(1234)
    cfg = yaml.safe_load(open(cfg_path, 'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_random_repeats = cfg.get("n_random_repeats", 100)
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

    # Figure out which image categories to evaluate (NEW logic) --------------
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

            # ---- effect‑size map -------------------------------------------------
            if "cliffs_delta" in df.columns:          # newer files
                delta_arr = pd.to_numeric(df["cliffs_delta"], errors="coerce").astype(np.float32)
                delta_map = delta_arr.values.reshape(Hf, Wf)
            elif "hedges_g" in df.columns:            # if stored Hedge's g instead
                delta_arr = pd.to_numeric(df["hedges_g"], errors="coerce").astype(np.float32)
                delta_map = delta_arr.values.reshape(Hf, Wf)
            else:                                     # legacy files – no effect size
                delta_map = None

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

            def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
                """
                Cliff’s δ for two 1‑D vectors (uses pairwise compare, still fast for <10⁴ items).
                Range −1‥+1.  Positive ⇒ selective > random.
                """
                a = a.ravel();  b = b.ravel()
                gt = (a[:, None] > b).sum()
                lt = (a[:, None] < b).sum()
                return (gt - lt) / (len(a) * len(b) + 1e-12)

            # ----- flatten sample axes so every voxel is a column -------------
            sel_flat  = all_v1_grads_sel.reshape(-1, Hf * Wf)      # [M_sel , H*W]
            rand_flat = all_v1_grads_rand.reshape(-1, Hf * Wf)      # [M_rand, H*W]

            mean_diff_vec = np.empty(Hf * Wf, dtype=np.float32)
            g_vec         = np.empty(Hf * Wf, dtype=np.float32)      # effect size
            p_vec         = np.empty(Hf * Wf, dtype=np.float64)      # Welch p‑values


            for i in tqdm(range(Hf * Wf), desc="voxel stats"):
                a = sel_flat[:, i]
                b = rand_flat[:, i]

                mean_diff_vec[i] = a.mean() - b.mean()
                g_vec[i]   = hedges_g(a, b)                               # effect size
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
                'hedges_g'      : g_vec,            # ← new column
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

            # 1. Prepare model instances and hooks for each thread
            max_workers = torch.get_num_threads()
            max_workers = 1
            img_args = [(img, top_units) for img in imgs]

            def thread_worker(img, units):
                model, bufs = get_thread_model_and_bufs(cfg)
                return grads_per_image(img, model, bufs, units)

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                all_v1_grads_sel = list(
                    tqdm(ex.map(lambda img: thread_worker(img, top_units), imgs),
                         total=len(imgs),
                         desc="Images (selective)")
                )

                all_v1_grads_sel = np.stack(all_v1_grads_sel)
                np.save(str(activ_dir / "grads_selective.npy"), all_v1_grads_sel)
                del all_v1_grads_sel
                gc.collect()

                # -------- many random repeats ------------------
                all_v1_grads_rand = []
                for rep in tqdm(range(n_random_repeats), desc="Random repeats"):
                    rand_units = random.sample(all_it_units, len(top_units))

                    rep_arr = list(
                        tqdm(ex.map(lambda img: thread_worker(img, rand_units), imgs),
                             total=len(imgs),
                             desc=f"Rep {rep+1}/{n_random_repeats}",
                             leave=False)
                    )
                    rep_arr = np.stack(rep_arr)           # shape [N_img, S, C, H, W]

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

            # ----- flatten sample axes so every voxel is a column -------------
            N, C, Hf, Wf = all_v1_grads_sel.shape
            R = n_random_repeats
            sel_flat  = all_v1_grads_sel.reshape(-1, Hf * Wf)      # [M_sel , H*W]
            rand_flat = all_v1_grads_rand.reshape(-1, Hf * Wf)      # [M_rand, H*W]

            mean_diff_vec  = np.empty(Hf * Wf, dtype=np.float32)
            cliff_vec      = np.empty(Hf * Wf, dtype=np.float32)
            p_vec          = np.empty(Hf * Wf, dtype=np.float64)

            for i in tqdm(range(Hf * Wf), desc="voxel stats"):
                a = sel_flat[:, i]
                b = rand_flat[:, i]

                mean_diff_vec[i] = a.mean() - b.mean()
                cliff_vec[i]     = cliffs_delta(a, b)
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
                'cliffs_delta'     : cliff_vec,
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
        plt.title("Hedges g")
        plt.axis('off')
        plt.colorbar(label="Hedges g")
        plt.savefig(outdir / f"{prefix}_B_cliffs_delta.png", dpi=300)

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


       


if __name__=="__main__":
    if len(sys.argv)!=2:
        sys.exit("usage: python it2v1_featuremap_contrib.py <config.yaml>")
    main(sys.argv[1])
