﻿#!/usr/bin/env python3
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

        def target_fn(units):
            val = torch.tensor(0., device=device, requires_grad=True)
            for name, c, y, x in units:
                act = layer_out[name]
                if isinstance(act, tuple):
                    act = act[0]
                val = val + act[:, c, y, x].sum() / act.size(0)
            return val

        tfm  = build_transform()
        imgs = list(iter_imgs(pathlib.Path(cfg["category_images"])))[:cfg.get("max_images",20)]

        outdir = cfg.get("output_dir", "it2v1_analysis")
        pathlib.Path(outdir).mkdir(exist_ok=True)
        diff_csv_path = f"{outdir}/{cat}_{mode}_{top_frac}_v1_featuremap_selective_vs_random.csv"

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
            print(f"Found existing results at {diff_csv_path}, loading...")
            df = pd.read_csv(diff_csv_path)
            Hf = df['fy'].max() + 1
            Wf = df['fx'].max() + 1
            if 'mean_diff' in df.columns and 'permutation_p_value' in df.columns:
                obs_map = df['mean_diff'].values.reshape(Hf, Wf)
                p_map = df['permutation_p_value'].values.reshape(Hf, Wf)
                p_map_bonf = df['bonferroni_p_value'].values.reshape(Hf, Wf)
            else:
                raise ValueError("Existing results file does not contain permutation test results. Please delete the file and rerun.")

        else:
            print("Computing |activation * gradient| for selective IT units...")
            all_v1_actgrads_sel = []
            for img in tqdm(imgs, desc="Images (selective)"):
                x = tfm(img).unsqueeze(0).to(device)
                layer_out.clear(); layer_grad.clear()
                model.zero_grad(); _ = model(x)
                loss = target_fn(top_units); loss.backward()
                v1_act = layer_out["module.V1.nonlin_input"].detach().cpu().numpy()[0]  # [C, H, W]
                v1_gr  = layer_grad["module.V1.nonlin_input"][0].cpu().numpy()         # [C, H, W]
                v1_actgrad = np.abs(v1_act * v1_gr)                                    # [C, H, W]
                all_v1_actgrads_sel.append(v1_actgrad)
            all_v1_actgrads_sel = np.stack(all_v1_actgrads_sel)  # [N, C, H, W]

            print(f"Computing |activation * gradient| for {n_random_repeats} random IT unit sets...")
            all_v1_actgrads_rand = []
            n_sel = len(top_units)
            for rep in tqdm(range(n_random_repeats), desc="Random repeats"):
                rand_units = random.sample(all_it_units, n_sel)
                rep_v1_actgrads = []
                for img in imgs:
                    x = tfm(img).unsqueeze(0).to(device)
                    layer_out.clear(); layer_grad.clear()
                    model.zero_grad(); _ = model(x)
                    loss = target_fn(rand_units); loss.backward()
                    v1_act = layer_out["module.V1.nonlin_input"].detach().cpu().numpy()[0]
                    v1_gr  = layer_grad["module.V1.nonlin_input"][0].cpu().numpy()
                    v1_actgrad = np.abs(v1_act * v1_gr)
                    rep_v1_actgrads.append(v1_actgrad)
                rep_v1_actgrads = np.stack(rep_v1_actgrads)  # [N, C, H, W]
                all_v1_actgrads_rand.append(rep_v1_actgrads)
            all_v1_actgrads_rand = np.stack(all_v1_actgrads_rand)  # [R, N, C, H, W]

            print("Mean selective activation gradients:", np.mean(all_v1_actgrads_sel))
            print("Mean random activation gradients:", np.mean(all_v1_actgrads_rand))

            # Build the activations directory name
            activ_dir = f"activations_{cat}_{top_frac}_{mode}"
            activ_dir_path = pathlib.Path(outdir) / activ_dir
            activ_dir_path.mkdir(parents=True, exist_ok=True)

            # Save selective activ*grad
            np.save(str(activ_dir_path / "activgrads_selective.npy"), all_v1_actgrads_sel)

            # Save each random permutation activ*grad
            for rep in range(len(all_v1_actgrads_rand)):
                np.save(str(activ_dir_path / f"activgrads_random_{rep}.npy"), all_v1_actgrads_rand[rep])


            N, C, Hf, Wf = all_v1_actgrads_sel.shape
            R = n_random_repeats

            n_permutations = 1000  # or as needed
            obs_map = np.zeros((Hf, Wf), dtype=np.float32)
            p_map = np.zeros((Hf, Wf), dtype=np.float32)

            def perm_test_at_loc(args):
                y, x = args
                sel_vals = all_v1_actgrads_sel[:, :, y, x].flatten()
                rand_vals = all_v1_actgrads_rand[:, :, :, y, x].flatten()
                obs, p = permutation_test(sel_vals, rand_vals, n_permutations=n_permutations, alternative='two-sided')
                return y, x, obs, p

            coords = [(y, x) for y in range(Hf) for x in range(Wf)]

            print("Running permutation test for each V1 unit (multi-threaded)...")
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(perm_test_at_loc, coord) for coord in coords]
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Permutation test"):
                    y, x, obs, p = fut.result()
                    obs_map[y, x] = obs
                    p_map[y, x] = p

            # Bonferroni correction
            n_tests = Hf * Wf
            p_map_bonf = np.minimum(p_map * n_tests, 1.0)
            rej_map_bonf = p_map_bonf < 0.05  # or your chosen alpha

            # Save results
            df = pd.DataFrame({
                'fy': np.repeat(np.arange(Hf), Wf),
                'fx': np.tile(np.arange(Wf), Hf),
                'mean_diff': obs_map.flatten(),
                'permutation_p_value': p_map.flatten(),
                'bonferroni_p_value': p_map_bonf.flatten(),
                'bonferroni_significant': rej_map_bonf.flatten()
            })
            df.to_csv(diff_csv_path, index=False)
            print("Saved permutation test feature-map and CSV to", diff_csv_path)



        

        # Visualization
        plt.figure(figsize=(6,5))
        plt.imshow(obs_map, cmap="bwr", interpolation='nearest')
        plt.title(f"V1 |act*grad| mean diff (selective - random, top_frac={top_frac})")
        plt.axis('off')
        plt.colorbar(label="Mean(|act*grad|) diff")
        plt.savefig(f"{outdir}/{cat}_{mode}_{top_frac}_v1_featuremap_actgrad_diff_perm.png", dpi=300)
        plt.show()

        plt.figure(figsize=(6,5))
        plt.imshow(p_map_bonf, cmap="viridis", interpolation='nearest', vmin=0, vmax=1)
        plt.title(f"V1 permutation p-value map (selective vs random, top_frac={top_frac})")
        plt.axis('off')
        plt.colorbar(label="Permutation p-value")
        plt.savefig(f"{outdir}/{cat}_{mode}_{top_frac}_v1_featuremap_perm_pval.png", dpi=300)
        plt.show()


if __name__=="__main__":
    if len(sys.argv)!=2:
        sys.exit("usage: python it2v1_featuremap_contrib.py <config.yaml>")
    main(sys.argv[1])
