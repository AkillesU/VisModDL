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
from scipy.stats import mannwhitneyu
import os
from scipy.ndimage import gaussian_filter
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from statsmodels.stats.multitest import multipletests


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


# ─────────── helpers ────────────
def load_model(mcfg, device):
    import cornet
    ctor = {
        "cornet_rt": cornet.cornet_rt,
        "cornet_s": cornet.cornet_s,
        "cornet_z": cornet.cornet_z
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


def iter_imgs(folder):
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in {".jpg",".jpeg",".png"}:
            yield Image.open(p).convert("RGB")

# compute receptive-field center in input pixels from feature-map coords
def rf_px(fy, fx, stride, padding, kernel):
    # center = (fy * stride + (kernel-1)/2 - padding)
    cy = fy * stride + (kernel-1)/2 - padding
    cx = fx * stride + (kernel-1)/2 - padding
    return cy, cx

# ─────────── main ────────────────
def main(cfg_path):
    random.seed(1234)
    cfg = yaml.safe_load(open(cfg_path,'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_random_repeats = cfg.get("n_random_repeats", 100)

    if torch.cuda.is_available() is False:
        try:
            num_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
        except TypeError:
            # os.cpu_count() can return None, handle this case
            num_threads = 1

        torch.set_num_threads(num_threads)
        print(f"✅ PyTorch intra-op parallelism set to {torch.get_num_threads()} threads.")
    else:
        print("Using: ", torch.DeviceObjType)
    


    top_frac_cfg = cfg.get("top_frac", 0.1)
    if isinstance(top_frac_cfg, (float, int)):
        top_frac_list = [top_frac_cfg]
    else:
        top_frac_list = list(top_frac_cfg)

    for top_frac in top_frac_list:
        print(f"\n=== Processing top_frac={top_frac} ===")
        cat = cfg["category"].lower()
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
        model          = load_model(cfg["model"], device)
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

        def flat_to_cyx(idx, C, H, W):
            c   = idx // (H * W)
            rem = idx %  (H * W)
            y   = rem // W
            x   = rem %  W
            return c, y, x

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

        model = load_model(cfg["model"], device)
        v1 = dict(model.named_modules())["module.V1.nonlin_input"]
        it_v1_names = ["module.IT", "module.V1.nonlin_input"]
        layer_out = {}
        layer_grad = {}
        def save_out(name):
            def hook(_mod, _in, out):
                layer_out[name] = out
            return hook
        def save_grad(name):
            def hook(grad): layer_grad[name]=grad
            return hook

        for name,mod in model.named_modules():
            if name in it_v1_names:
                mod.register_forward_hook(save_out(name))
        def v1_hook(m, inp, out):
            out.register_hook(save_grad("module.V1.nonlin_input"))
        v1.register_forward_hook(v1_hook)
        
        tfm  = build_transform()
        imgs = list(iter_imgs(pathlib.Path(cfg["category_images"])))[:cfg.get("max_images",20)]

        outdir = cfg.get("output_dir", "it2v1_analysis")
        pathlib.Path(outdir).mkdir(exist_ok=True)
        # Modified file name to reflect the new metric (grad instead of actgrad)
        diff_csv_path = f"{outdir}/{cat}_{mode}_{top_frac}_v1_featuremap_selective_vs_random_grad.csv"

        # --- get all IT units for random selection ---
        all_it_units = []
        for _, row in it_rows.iterrows():
            lay   = row.layer
            idx   = int(row.unit)
            C, H, W = feature_shapes[lay]
            c, y, x = flat_to_cyx(idx, C, H, W)
            all_it_units.append((lay, c, y, x))

        # === Check for existing results ===
        if os.path.exists(diff_csv_path):
            print(f"Found existing results at {diff_csv_path}, loading…")
            df = pd.read_csv(diff_csv_path)
            Hf = df["fy"].max() + 1
            Wf = df["fx"].max() + 1
            activ_dir_path = pathlib.Path(outdir) / f"activations_{cat}_{top_frac}_{mode}_grad"
            have_grad_files = activ_dir_path.exists()

            # ---- mandatory column ----
            obs_map = df["mean_diff"].values.reshape(Hf, Wf)
            mean_diff_vec  = obs_map.ravel()  

            # ---- effect‑size map -----------------------------------------
            if "cliffs_delta" in df.columns:
                delta_map = df["cliffs_delta"].values.reshape(Hf, Wf)
            else:                       # legacy files had no Cliff’s δ
                delta_map = None

            # ---- significance / p‑value handling -------------------------
            if "fdr_q_value" in df.columns:             # new schema
                q_map    = df["fdr_q_value"].values.reshape(Hf, Wf)
                sig_mask = df["significant_fdr"].astype(bool).values.reshape(Hf, Wf)
            elif "bonferroni_p_value" in df.columns:    # very old schema
                q_map    = df["bonferroni_p_value"].values.reshape(Hf, Wf)
                sig_mask = (q_map < 0.05)
            else:
                raise ValueError(
                    "CSV exists but does not contain recognised significance columns "
                    "(expected 'fdr_q_value' or 'bonferroni_p_value'). "
                    "Delete the file and rerun to regenerate."
                )
        elif os.path.exists(f"{outdir}/activations_{cat}_{top_frac}_{mode}_grad/"):
            print(f"Found existing activation gradients at {outdir}/activations_{cat}_{top_frac}_{mode}_grad/, loading...")
            all_v1_grads_sel = np.load(f"{outdir}/activations_{cat}_{top_frac}_{mode}_grad/grads_selective.npy")
            all_v1_grads_rand = [np.load(f"{outdir}/activations_{cat}_{top_frac}_{mode}_grad/grads_random_{rep}.npy") for rep in range(n_random_repeats)]
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

            


        else:
            
            print("Computing mean |gradient| for selective IT units (parallelized)...")
            all_v1_grads_sel = []

            def compute_v1_grad_for_unit(args):
                img, x, model, layer_out, layer_grad, unit = args
                name, c, y, x_coord = unit
                layer_out.clear(); layer_grad.clear()
                model.zero_grad()
                _ = model(x)
                act = layer_out[name]
                if isinstance(act, tuple):
                    act = act[0]
                loss = act[:, c, y, x_coord].sum()
                loss.backward()
                v1_gr = layer_grad["module.V1.nonlin_input"][0].cpu().numpy()
                return np.abs(v1_gr)

            # 1. Prepare model instances and hooks for each thread
            max_workers = torch.get_num_threads()
            model_pool = [load_model(cfg["model"], device) for _ in range(max_workers)]

            def setup_hooks(model):
                v1 = dict(model.named_modules())["module.V1.nonlin_input"]
                it_v1_names = ["module.IT", "module.V1.nonlin_input"]
                layer_out = {}
                layer_grad = {}

                def save_out(name):
                    def hook(_mod, _in, out):
                        layer_out[name] = out
                    return hook
                def save_grad(name):
                    def hook(grad): layer_grad[name]=grad
                    return hook

                for name, mod in model.named_modules():
                    if name in it_v1_names:
                        mod.register_forward_hook(save_out(name))
                def v1_hook(m, inp, out):
                    out.register_hook(save_grad("module.V1.nonlin_input"))
                v1.register_forward_hook(v1_hook)
                return layer_out, layer_grad

            hook_pool = [setup_hooks(model) for model in model_pool]

            # 2. Define the image processing function
            def process_image(args):
                img, model, layer_out, layer_grad = args
                x = tfm(img).unsqueeze(0).to(device)
                single_img_v1_grads = []
                for (name, c, y, x_coord) in top_units:
                    layer_out.clear(); layer_grad.clear()
                    model.zero_grad()
                    _ = model(x)
                    act = layer_out[name]
                    if isinstance(act, tuple):
                        act = act[0]
                    loss = act[:, c, y, x_coord].sum()
                    loss.backward()
                    v1_gr = layer_grad["module.V1.nonlin_input"][0].cpu().numpy()
                    single_img_v1_grads.append(np.abs(v1_gr))
                mean_grad_map = np.mean(np.stack(single_img_v1_grads), axis=0)
                return mean_grad_map

            # 3. Assign each image to a thread/model
            img_args = []
            for i, img in enumerate(imgs):
                model_idx = i % max_workers
                layer_out, layer_grad = hook_pool[model_idx]
                img_args.append((img, model_pool[model_idx], layer_out, layer_grad))

            # 4. Parallel execution
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                all_v1_grads_sel = list(tqdm(executor.map(process_image, img_args), total=len(imgs), desc="Images (selective)"))

            all_v1_grads_sel = np.stack(all_v1_grads_sel)  # [N, C, H, W]

            print(f"Computing mean |gradient| for {n_random_repeats} random IT unit sets...")
            all_v1_grads_rand = []
            n_sel = len(top_units)
            for rep in tqdm(range(n_random_repeats), desc="Random repeats"):
                rand_units = random.sample(all_it_units, n_sel)
                rep_v1_grads = []
                for img in imgs:
                    img_tensor = tfm(img).unsqueeze(0).to(device)

                    # For each image, get gradients from each random IT unit
                    single_img_v1_grads_rand = []
                    for (name, c, y, x_coord) in rand_units:
                        layer_out.clear(); layer_grad.clear()
                        model.zero_grad()
                        _ = model(img_tensor)

                        act = layer_out[name]
                        if isinstance(act, tuple):
                            act = act[0]
                        loss = act[:, c, y, x_coord].sum()
                        loss.backward()
                        
                        v1_gr = layer_grad["module.V1.nonlin_input"][0].cpu().numpy()
                        single_img_v1_grads_rand.append(np.abs(v1_gr))
                    
                    # Aggregate gradients for this image by taking the mean
                    mean_grad_map_rand = np.mean(np.stack(single_img_v1_grads_rand), axis=0)
                    rep_v1_grads.append(mean_grad_map_rand)

                rep_v1_grads = np.stack(rep_v1_grads) # [N, C, H, W]
                all_v1_grads_rand.append(rep_v1_grads)
            all_v1_grads_rand = np.stack(all_v1_grads_rand)  # [R, N, C, H, W]


            print("Mean selective gradient magnitude:", np.mean(all_v1_grads_sel))
            print("Mean random gradient magnitude:", np.mean(all_v1_grads_rand))
            
            activ_dir = f"activations_{cat}_{top_frac}_{mode}_grad"
            activ_dir_path = pathlib.Path(outdir) / activ_dir
            activ_dir_path.mkdir(parents=True, exist_ok=True)

            np.save(str(activ_dir_path / "grads_selective.npy"), all_v1_grads_sel)
            for rep in range(len(all_v1_grads_rand)):
                np.save(str(activ_dir_path / f"grads_random_{rep}.npy"), all_v1_grads_rand[rep])

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

           

            print(f"Saved {num_voxels_to_plot} voxel histograms ➜ {outdir}")

        # ──────────────── PLOTS ─────────────────────────────────────────
        plt.figure(figsize=(6,5))
        plt.imshow(obs_map, cmap="bwr", interpolation='nearest')
        plt.title(f"Mean |grad| diff  (sel−rand, top_frac={top_frac})")
        plt.axis('off')
        plt.colorbar(label="Δ |grad|")
        plt.savefig(f"{outdir}/{cat}_{mode}_{top_frac}_A_mean_diff.png", dpi=300)

        # Cliff’s δ – show only significant voxels (others as NaN/white)
        delta_plot = np.where(sig_mask, delta_map, np.nan)
        plt.figure(figsize=(6,5))
        plt.imshow(delta_plot, cmap="bwr", vmin=-1, vmax=1, interpolation='nearest')
        plt.title("Cliff’s δ (FDR < 0.05)")
        plt.axis('off')
        plt.colorbar(label="Cliff’s δ")
        plt.savefig(f"{outdir}/{cat}_{mode}_{top_frac}_B_cliffs_delta.png", dpi=300)

        # –log10(q) significance map, same mask
        q = q_map
        q[~sig_mask] = np.nan
        plt.figure(figsize=(6,5))
        plt.imshow(q, cmap="viridis", interpolation='nearest')
        plt.title("FDR q")
        plt.axis('off')
        plt.colorbar(label="q")
        plt.savefig(f"{outdir}/{cat}_{mode}_{top_frac}_C_q.png", dpi=300)

        # ───────── DIAGNOSTIC HISTOGRAMS ───────────────────────
        # How many voxels to visualise

        if ("all_v1_grads_sel" not in locals() or all_v1_grads_sel is None) and have_grad_files:
                print("Loading raw gradient arrays for histogram plotting …")
                all_v1_grads_sel = np.load(activ_dir_path / "grads_selective.npy")
                all_v1_grads_rand = np.stack([
                    np.load(activ_dir_path / f"grads_random_{rep}.npy")
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

                fname = f"{outdir}/{cat}_{mode}_{top_frac}_v1_hist_voxel_{y}_{x}.png"
                plt.savefig(fname, dpi=300)
                plt.close()

            print(f"Saved {num_voxels_to_plot} voxel histograms ➜ {outdir}")


       


if __name__=="__main__":
    if len(sys.argv)!=2:
        sys.exit("usage: python it2v1_featuremap_contrib.py <config.yaml>")
    main(sys.argv[1])
