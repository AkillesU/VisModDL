#!/usr/bin/env python3
"""
pfi_band_pipeline.py

End-to-end pipeline to quantify foveal vs peripheral weighting using eccentricity bands, 
Cliff's delta as the effect-size metric, bootstrap CIs, and permutation tests.

Key choices (robust & fair):
- Images are the unit of independence. We summarise each image's map into K band-density values.
- Corners are kept; radial bins use Euclidean distance.
- Effect size per band: Cliff's Δ (non-parametric, unequal Ns ok).
- 95% CIs via cluster (row) bootstrap over images/maps.
- Primary significance: curve-level permutation (area under sel–rand diff).
- Secondary (optional): band-wise permutation + FDR.

Config YAML fields (example at bottom):
--------------------------------------
output_dir: it2v1_analysis
categories: [faces, scenes]
top_frac: [0.1, 0.2]
mode: percentage          # or percentile
nbins: 12
binning: equal_width      # or equal_area
bootstrap_iters: 10000
permutations: 10000
curve_stat: area          # area, l2, or max
area_normalize: true
alpha_fdr: 0.05

Run:
  python pfi_band_pipeline.py config.yaml
"""

import argparse
import pathlib
import yaml
import numpy as np
import pandas as pd
from numpy.random import default_rng
from tqdm import tqdm
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from matplotlib.cm import get_cmap
import hashlib

EPS = 1e-12

# ------------------------- Utils --------------------------------------------

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def derive_paths(cfg, category, top_frac, mode):
    outdir = pathlib.Path(cfg.get('output_dir', 'it2v1_analysis'))
    activ_dir = outdir / f"activations_{category}_{top_frac}_{mode}_grad"
    return outdir, activ_dir

# ------------------------- Eccentricity / Bins -------------------------------

def ecc_map(h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h-1)/2.0, (w-1)/2.0
    ecc = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    r_max = ecc.max()
    return ecc, r_max

def make_bins(ecc, r_max, nbins=12, binning='equal_width'):
    vals = ecc.ravel()
    if binning == 'equal_area':
        qs = np.linspace(0, 1, nbins+1)
        edges = np.quantile(vals, qs)
        edges[0] = 0.0
    else:
        edges = np.linspace(0, r_max, nbins+1)
    centers = 0.5*(edges[:-1] + edges[1:])
    return edges, centers

# ------------------------- Band densities ------------------------------------

def band_density(imp_map, ecc, edges, area_norm=True):
    flat_imp = imp_map.ravel()
    flat_ecc = ecc.ravel()
    idx = np.digitize(flat_ecc, edges) - 1
    nb = len(edges) - 1
    dens = np.zeros(nb)
    for b in range(nb):
        m = idx == b
        if m.any():
            val = flat_imp[m].sum()
            if area_norm:
                dens[b] = val / m.sum()
            else:
                dens[b] = val
    return dens

def stack_to_matrix(stack, ecc, edges, area_norm=True):
    """stack: [N, C, H, W] or [N, H, W] -> N x K band-density matrix"""
    if stack.ndim == 4:
        stack = stack.mean(axis=1)
    mats = []
    for i in range(stack.shape[0]):
        imp = np.abs(stack[i])
        imp /= imp.sum() + EPS
        dens = band_density(imp, ecc, edges, area_norm)
        mats.append(dens)
    return np.vstack(mats)

# ------------------------- Effect size (Cliff's Δ) ---------------------------

def cliffs_delta(a, b):
    # Efficient approximate Δ if arrays large: subsample b if massive
    a = np.asarray(a); b = np.asarray(b)
    n1, n2 = len(a), len(b)
    # Broadcasting can be heavy; if too big, subsample
    max_pairs = 2_000_000  # adjust if memory tight
    pairs = n1 * n2
    if pairs > max_pairs:
        rng = default_rng(0)
        k = max_pairs // n1
        idx = rng.integers(0, n2, size=k)
        b_sub = b[idx]
        diff = a[:, None] - b_sub[None, :]
        delta = (np.sum(diff > 0) - np.sum(diff < 0)) / (n1 * k)
        return delta
    diff = a[:, None] - b[None, :]
    return (np.sum(diff > 0) - np.sum(diff < 0)) / (n1 * n2)

# ------------------------- Bootstrap & Permutation ---------------------------

def bootstrap_effect(sel_mat, rand_mat, stat_fn, B=10000, rng=None):
    if rng is None:
        rng = default_rng(0)
    N_sel = sel_mat.shape[0]
    N_rand = rand_mat.shape[0]
    K = sel_mat.shape[1]
    boots = np.zeros((B, K), dtype=float)
    for b in range(B):
        si = rng.integers(0, N_sel, N_sel)
        ri = rng.integers(0, N_rand, N_rand)
        sS = sel_mat[si]
        rS = rand_mat[ri]
        for k in range(K):
            boots[b, k] = stat_fn(sS[:,k], rS[:,k])
    return boots

def curve_permutation(sel_mat, rand_mat, n_perm=10000, stat='area', rng=None):
    if rng is None:
        rng = default_rng(0)
    diff_curve = sel_mat.mean(0) - rand_mat.mean(0)
    if stat == 'area':
        obs = np.trapz(diff_curve)
    elif stat == 'l2':
        obs = np.sum(diff_curve**2)
    else:
        obs = np.max(np.abs(diff_curve))
    all_curves = np.vstack([sel_mat, rand_mat])
    nA = sel_mat.shape[0]
    stats = np.zeros(n_perm)
    for i in range(n_perm):
        rng.shuffle(all_curves)
        d = all_curves[:nA].mean(0) - all_curves[nA:].mean(0)
        if stat == 'area':
            stats[i] = np.trapz(d)
        elif stat == 'l2':
            stats[i] = np.sum(d**2)
        else:
            stats[i] = np.max(np.abs(d))
    p = (np.abs(stats) >= np.abs(obs)).mean()
    return obs, p

def band_permutation(sel_mat, rand_mat, n_perm=10000, stat_fn=cliffs_delta, rng=None):
    if rng is None:
        rng = default_rng(0)
    N_sel = sel_mat.shape[0]
    all_rows = np.vstack([sel_mat, rand_mat])
    labels = np.array([1]*N_sel + [0]*rand_mat.shape[0])
    K = sel_mat.shape[1]
    obs = np.array([stat_fn(sel_mat[:,k], rand_mat[:,k]) for k in range(K)])
    perm_p = np.zeros(K)
    for k in range(K):
        diffs = np.zeros(n_perm)
        for i in range(n_perm):
            rng.shuffle(labels)
            sel_idx = labels.astype(bool)
            diffs[i] = stat_fn(all_rows[sel_idx, k], all_rows[~sel_idx, k])
        perm_p[k] = (np.abs(diffs) >= np.abs(obs[k])).mean()
    return obs, perm_p

# ------------------------- Plotting ------------------------------------------

def plot_effect_curve_multi(centers, curves, cis_low, cis_high, colors, out_png):
    """
    curves: {label: delta_vec}
    cis_low/high: {label: low/high vec}
    colors: {label: color}
    """
    plt.figure(figsize=(7,5))
    plt.axhline(0, color='k', lw=0.6)
    for label, delta in curves.items():
        c = colors[label]
        lo = cis_low[label]; hi = cis_high[label]
        plt.plot(centers, delta, marker='o', color=c, label=label)
        plt.fill_between(centers, lo, hi, alpha=0.20, color=c)
    plt.xlabel('Eccentricity band center (px)')
    plt.ylabel("Cliff's Δ (sel − rand)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_effect_curve_single(centers, delta, ci_low, ci_high, out_png, color='C0', label=None):
    plt.figure(figsize=(6,4))
    plt.axhline(0, color='k', lw=0.6)
    plt.plot(centers, delta, marker='o', color=color, label=label)
    plt.fill_between(centers, ci_low, ci_high, alpha=0.25, color=color)
    plt.xlabel('Eccentricity band center (px)')
    plt.ylabel("Cliff's Δ (sel − rand)")
    if label:
        plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()


def stable_color(label, palette='tab20', seed=0, fixed=None):
    """
    Return a hex color for a label that is stable across runs.
    If 'fixed' dict provided and label in it, use that.
    Otherwise hash(label+seed) and index into palette.
    """
    if fixed and label in fixed:
        return fixed[label]
    h = hashlib.md5((str(seed)+label).encode()).hexdigest()
    idx = int(h[:8], 16)  # big int
    cmap = get_cmap(palette)
    # cmap.N is number of discrete colors if ListedColormap, else use 256
    ncols = getattr(cmap, 'N', 256)
    return cmap(idx % ncols)

# ------------------------- Main ---------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='PFI band analysis with Cliff\'s delta')
    ap.add_argument('config', type=str)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    cats = cfg.get('categories', [cfg.get('category', 'category').lower()])
    tf_cfg = cfg.get('top_frac', 0.1)
    top_fracs = tf_cfg if isinstance(tf_cfg, (list, tuple)) else [tf_cfg]
    mode = cfg.get('mode', cfg.get('top_unit_selection', 'percentage')).lower()
    nbins = cfg.get('nbins', 12)
    binning = cfg.get('binning', 'equal_width')
    B = cfg.get('bootstrap_iters', 10000)
    NPERM = cfg.get('permutations', 10000)
    curve_stat = cfg.get('curve_stat', 'area')
    area_norm = bool(cfg.get('area_normalize', True))
    alpha_fdr = cfg.get('alpha_fdr', 0.05)
    plot_all = bool(cfg.get('plot_all_together', False))
    palette = cfg.get('palette', 'tab20')
    color_seed = cfg.get('color_seed', 0)
    fixed_colors = cfg.get('fixed_colors', None)

    out_root = pathlib.Path(cfg.get('output_dir', 'it2v1_analysis')) / 'pfi_band'
    out_root.mkdir(parents=True, exist_ok=True)

    rng = default_rng(0)

    ecc = None
    edges = centers = None

    summary_rows = []
    all_curves = {}
    all_ci_low = {}
    all_ci_high = {}
    all_colors = {}

    for cat in cats:
        for tf in top_fracs:
            outdir, activ_dir = derive_paths(cfg, cat, tf, mode)
            sel_path = activ_dir / 'grads_selective.npy'
            rand_paths = sorted(activ_dir.glob('grads_random_*.npy'))
            if not sel_path.exists() or not rand_paths:
                print(f"Missing data for {cat} tf={tf}; skipping")
                continue

            print(f"Loading selective {sel_path}")
            sel = np.load(sel_path)
            rand = np.concatenate([np.load(p) for p in rand_paths], axis=0)

            H, W = sel.shape[2], sel.shape[3]
            if ecc is None:
                ecc, rmax = ecc_map(H, W)
                edges, centers = make_bins(ecc, rmax, nbins, binning)

            sel_mat = stack_to_matrix(sel, ecc, edges, area_norm)
            rand_mat = stack_to_matrix(rand, ecc, edges, area_norm)

            # Cliff's delta per band
            delta_hat = np.array([cliffs_delta(sel_mat[:,k], rand_mat[:,k]) for k in range(sel_mat.shape[1])])

            # Bootstrap CIs
            print(f"Bootstrapping ({B} resamples) for {cat} tf={tf} ...")
            boots = bootstrap_effect(sel_mat, rand_mat, cliffs_delta, B=B, rng=rng)
            ci_low = np.percentile(boots, 2.5, axis=0)
            ci_high = np.percentile(boots, 97.5, axis=0)

            # Curve-level permutation
            print(f"Permutation test ({NPERM}) on curve for {cat} tf={tf} ...")
            obs_curve, p_curve = curve_permutation(sel_mat, rand_mat, n_perm=NPERM, stat=curve_stat, rng=rng)

            # Band-wise permutation + FDR (optional, slower)
            print("Band-wise permutation tests ...")
            _, p_band = band_permutation(sel_mat, rand_mat, n_perm=NPERM, stat_fn=cliffs_delta, rng=rng)
            p_fdr = multipletests(p_band, method='fdr_bh', alpha=alpha_fdr)[1]

            # Save results
            label = f"{cat}_tf{tf}"
            df_delta = pd.DataFrame({
                'band_center': centers,
                'cliffs_delta': delta_hat,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'p_perm_band': p_band,
                'p_fdr_band': p_fdr,
            })
            df_delta.to_csv(out_root / f'delta_{label}.csv', index=False)

            # raw band densities
            pd.DataFrame(sel_mat, columns=[f'band_{i}' for i in range(len(centers))]).to_csv(out_root / f'sel_band_{label}.csv', index=False)
            pd.DataFrame(rand_mat, columns=[f'band_{i}' for i in range(len(centers))]).to_csv(out_root / f'rand_band_{label}.csv', index=False)

            # Decide color for this label
            col = stable_color(label, palette=palette, seed=color_seed, fixed=fixed_colors)
            all_curves[label]   = delta_hat
            all_ci_low[label]   = ci_low
            all_ci_high[label]  = ci_high
            all_colors[label]   = col

            # Single plot (keep existing behaviour)
            plot_effect_curve_single(centers, delta_hat, ci_low, ci_high, color=col, out_png=out_root / f'delta_{label}.png',label=label)


            summary_rows.append(dict(category=cat, top_frac=tf, obs_curve_stat=obs_curve, p_curve=p_curve))

    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        df_sum['p_curve_fdr'] = multipletests(df_sum['p_curve'], method='fdr_bh', alpha=alpha_fdr)[1]
        df_sum.to_csv(out_root / 'curve_permutation_summary.csv', index=False)


    if plot_all and all_curves:
        combined_name = cfg.get('combined_plot_name', 'all_curves.png')
        plot_effect_curve_multi(centers, all_curves, all_ci_low, all_ci_high, all_colors,
                                out_root / combined_name)
        # also save the color mapping for reproducibility
        pd.DataFrame([
            {'label': k, 'color': all_colors[k]}
            for k in all_colors]).to_csv(out_root / 'colors_used.csv', index=False)

    # Save meta
    meta = dict(nbins=nbins, binning=binning, bootstrap_iters=B, permutations=NPERM,
                curve_stat=curve_stat, area_normalize=area_norm, centers=centers.tolist())
    with open(out_root / 'meta.yaml', 'w') as f:
        yaml.safe_dump(meta, f)

    print('Done. Results in', out_root)

if __name__ == '__main__':
    main()

"""
# Example config.yaml
# -------------------
# output_dir: it2v1_analysis
# categories: [faces, scenes]
# top_frac: [0.1, 0.2]
# mode: percentage
# nbins: 12
# binning: equal_width  # or equal_area
# bootstrap_iters: 10000
# permutations: 10000
# curve_stat: area
# area_normalize: true
# alpha_fdr: 0.05
"""
