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
    # load config
    cfg = yaml.safe_load(open(cfg_path,'r'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Accept top_frac as a list or single value
    top_frac_cfg = cfg.get("top_frac", 0.1)
    if isinstance(top_frac_cfg, (float, int)):
        top_frac_list = [top_frac_cfg]
    else:
        top_frac_list = list(top_frac_cfg)

    for top_frac in top_frac_list:
        print(f"\n=== Processing top_frac={top_frac} ===")
        # ── 1)  SELECT TOP-UNITS FROM NEW COMBINED PICKLE  ──────────────────────────
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

        # --- get feature shape, as before ---
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

        # 2) load model and hooks
        model = load_model(cfg["model"], device)
        v1 = dict(model.named_modules())["module.V1.nonlin_input"] # instead of conv_input
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

        def target_fn():
            val = torch.tensor(0., device=device, requires_grad=True)
            for name, c, y, x in top_units:
                act = layer_out[name]
                if isinstance(act, tuple):
                    act = act[0]
                val = val + act[:, c, y, x].sum() / act.size(0)
            return val

        tfm  = build_transform()
        imgs = list(iter_imgs(pathlib.Path(cfg["category_images"])))[:cfg.get("max_images",20)]

        outdir = cfg.get("output_dir", "it2v1_analysis")
        pathlib.Path(outdir).mkdir(exist_ok=True)
        csv_path = f"{outdir}/{cat}_{mode}_{top_frac}_v1_featuremap_mannwhitneyu.csv"

        if os.path.exists(csv_path):
            print(f"Found existing results at {csv_path}, loading...")
            df = pd.read_csv(csv_path)
            Hf = df['fy'].max() + 1
            Wf = df['fx'].max() + 1
            stat_map = df['mannwhitneyu_stat'].values.reshape(Hf, Wf)
            pval_map = df['p_value'].values.reshape(Hf, Wf)
        else:
            all_v1_grads = []
            for img in tqdm(imgs, desc="Images"):
                x = tfm(img).unsqueeze(0).to(device)
                layer_out.clear(); layer_grad.clear()
                model.zero_grad(); _ = model(x)
                loss = target_fn(); loss.backward()
                v1_gr = layer_grad["module.V1.nonlin_input"][0].cpu().numpy()  # [C, H, W]
                all_v1_grads.append(v1_gr)

            all_v1_grads = np.stack(all_v1_grads)  # [N, C, H, W]
            N, C, Hf, Wf = all_v1_grads.shape

            global_mean_grad = np.mean(all_v1_grads)
            u_stats_per_image = np.zeros((N, Hf, Wf))
            pvals_per_image = np.zeros((N, Hf, Wf))
            mean_grad_map = np.zeros((Hf, Wf))

            for img_idx in tqdm(range(N), desc="Images (MWU per image)"):
                grads = all_v1_grads[img_idx]  # [C, H, W]
                for y in tqdm(range(Hf), desc=f"Y (img {img_idx+1}/{N})", leave=False):
                    for x in tqdm(range(Wf), desc=f"X (img {img_idx+1}/{N}, y={y})", leave=False):
                        loc_vals = grads[:, y, x]  # [C]
                        mask = np.ones((Hf, Wf), dtype=bool)
                        mask[y, x] = False
                        other_vals = grads[:, mask].flatten()
                        try:
                            stat, p = mannwhitneyu(loc_vals, other_vals, alternative="two-sided")
                        except Exception:
                            stat, p = np.nan, np.nan
                        u_stats_per_image[img_idx, y, x] = stat
                        pvals_per_image[img_idx, y, x] = p

            stat_map = np.nanmean(u_stats_per_image, axis=0)
            pval_map = np.nanmean(pvals_per_image, axis=0)
            mean_grad_map = np.mean(all_v1_grads, axis=(0, 1))  # shape [Hf, Wf]

            df = pd.DataFrame({
                'fy': np.repeat(np.arange(Hf), Wf),
                'fx': np.tile(np.arange(Wf), Hf),
                'mannwhitneyu_stat': stat_map.flatten(),
                'p_value': pval_map.flatten(),
                'mean_gradient': mean_grad_map.flatten(),
                'global_mean_gradient': np.full(Hf * Wf, global_mean_grad)
            })
            df.to_csv(csv_path, index=False)
            print("Saved Mann-Whitney U feature-map and CSV to", outdir)

        stat_map_norm = (stat_map - np.nanmin(stat_map)) / (np.nanmax(stat_map) - np.nanmin(stat_map) + 1e-9)

        gaussian_sigma = cfg.get("gaussian_sigma", None)
        plot_map = stat_map_norm
        if gaussian_sigma and gaussian_sigma > 0:
            plot_map = gaussian_filter(stat_map_norm, sigma=gaussian_sigma)


        plt.figure(figsize=(6,5))
        plt.imshow(plot_map, cmap="hot", interpolation='nearest', vmin=0.3)
        plt.title(f"V1 Feature-Map: Mann-Whitney U Statistic (top_frac={top_frac})")
        plt.axis('off')
        plt.colorbar(label="Normalized U-statistic")
        plt.savefig(f"{outdir}/{cat}_{mode}_{top_frac}_v1_featuremap_mannwhitneyu.png", dpi=300)
        plt.show()



if __name__=="__main__":
    if len(sys.argv)!=2:
        sys.exit("usage: python it2v1_featuremap_contrib.py <config.yaml>")
    main(sys.argv[1])
