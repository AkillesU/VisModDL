#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    from torchvision.models import AlexNet_Weights, alexnet
except Exception:
    AlexNet_Weights = None
    alexnet = None


APP_TITLE = 'AlexNet selective-unit robustness app'
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


@dataclass
class LayerSpec:
    key: str
    label: str
    conv_idx: int
    relu_idx: int
    pool_idx: int
    in_channels: int
    out_channels: int
    kernel_size: int
    padding: int
    pool_kernel: int
    pool_stride: int
    gn_groups: int


LAYER_SPECS: Dict[str, LayerSpec] = {
    'first': LayerSpec(
        key='first',
        label='First block (features.0 / 1 / 2)',
        conv_idx=0,
        relu_idx=1,
        pool_idx=2,
        in_channels=3,
        out_channels=64,
        kernel_size=11,
        padding=2,
        pool_kernel=3,
        pool_stride=2,
        gn_groups=16,
    ),
    'final': LayerSpec(
        key='final',
        label='Final block (features.10 / 11 / 12)',
        conv_idx=10,
        relu_idx=11,
        pool_idx=12,
        in_channels=256,
        out_channels=256,
        kernel_size=3,
        padding=1,
        pool_kernel=3,
        pool_stride=2,
        gn_groups=32,
    ),
}


@st.cache_resource(show_spinner=False)
def load_model_and_preprocess() -> Tuple[nn.Module, object, object]:
    if alexnet is None:
        raise RuntimeError('torchvision is required for this app.')

    if AlexNet_Weights is not None:
        weights = AlexNet_Weights.DEFAULT
        model = alexnet(weights=weights).eval()
        preprocess = weights.transforms()
        categories = weights.meta.get('categories', None)
    else:
        model = alexnet(pretrained=True).eval()
        from torchvision import transforms as T
        preprocess = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        categories = None
    return model, preprocess, categories


def list_category_dirs(root: Path) -> List[Path]:
    if not root.exists() or not root.is_dir():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])


def list_images_in_dir(folder: Path) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted([p for p in folder.rglob('*') if p.suffix.lower() in IMAGE_SUFFIXES])


def build_image_table(root: Path, target_category: str, per_category_limit: int) -> pd.DataFrame:
    rows = []
    for cat_dir in list_category_dirs(root):
        paths = list_images_in_dir(cat_dir)
        if per_category_limit > 0:
            paths = paths[:per_category_limit]
        for p in paths:
            rows.append({'path': str(p), 'category': cat_dir.name, 'is_target': int(cat_dir.name == target_category)})
    return pd.DataFrame(rows)


def load_image_tensor(image_path: str, preprocess) -> torch.Tensor:
    img = Image.open(image_path).convert('RGB')
    return preprocess(img)


def read_table_any(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == '.csv':
        return pd.read_csv(path)
    if suffix in {'.tsv', '.txt'}:
        try:
            return pd.read_csv(path, sep='\t')
        except Exception:
            return pd.read_csv(path)
    if suffix == '.parquet':
        return pd.read_parquet(path)
    if suffix in {'.xlsx', '.xls'}:
        return pd.read_excel(path)
    raise ValueError(f'Unsupported selectivity file format: {suffix}')


def guess_selectivity_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = {c.lower(): c for c in df.columns}

    def find_any(candidates: Sequence[str]) -> Optional[str]:
        for cand in candidates:
            if cand in cols:
                return cols[cand]
        for raw in df.columns:
            low = raw.lower()
            for cand in candidates:
                if cand in low:
                    return raw
        return None

    return {
        'layer': find_any(['layer', 'layer_key', 'layer_name']),
        'unit': find_any(['unit', 'unit_idx', 'channel', 'channel_idx', 'feature', 'neuron']),
        'category': find_any(['category', 'class', 'target_category', 'label']),
        'score': find_any(['score', 'selectivity', 'dprime', 'd_prime', 'auc', 'metric']),
    }


def pick_units_from_selectivity(df: pd.DataFrame, target_category: str, layer_key: str, top_k: int) -> List[int]:
    cols = guess_selectivity_columns(df)
    if cols['unit'] is None:
        return []
    sub = df.copy()
    if cols['category'] is not None:
        sub = sub[sub[cols['category']].astype(str) == str(target_category)]
    if cols['layer'] is not None:
        layer_series = sub[cols['layer']].astype(str).str.lower()
        if layer_key == 'first':
            sub = sub[layer_series.str.contains(r'first|0|features\.0|conv1', regex=True)]
        else:
            sub = sub[layer_series.str.contains(r'final|10|12|features\.10|features\.12|conv5|pool5', regex=True)]
    if sub.empty:
        return []
    unit_col = cols['unit']
    sub[unit_col] = pd.to_numeric(sub[unit_col], errors='coerce')
    sub = sub.dropna(subset=[unit_col])
    if sub.empty:
        return []
    if cols['score'] is not None:
        sub[cols['score']] = pd.to_numeric(sub[cols['score']], errors='coerce')
        sub = sub.sort_values(cols['score'], ascending=False)
    units = [int(x) for x in sub[unit_col].tolist()]
    deduped = []
    seen = set()
    for u in units:
        if u not in seen:
            deduped.append(u)
            seen.add(u)
    return deduped[:top_k]


def forward_features(model: nn.Module, x: torch.Tensor) -> Dict[int, torch.Tensor]:
    feats = {}
    cur = x
    for idx, layer in enumerate(model.features):
        cur = layer(cur)
        feats[idx] = cur
    return feats


def get_layer_input(features: Dict[int, torch.Tensor], x: torch.Tensor, spec: LayerSpec) -> torch.Tensor:
    return x if spec.conv_idx == 0 else features[spec.conv_idx - 1]


def get_conv_layer(model: nn.Module, spec: LayerSpec) -> nn.Conv2d:
    layer = model.features[spec.conv_idx]
    assert isinstance(layer, nn.Conv2d)
    return layer


def continue_local_block(layer_input: torch.Tensor, conv: nn.Conv2d, spec: LayerSpec, target_channel: int,
                         branch_type: str, weight_mask: Optional[torch.Tensor] = None,
                         input_mask: Optional[torch.Tensor] = None, pre_relu_mask: Optional[torch.Tensor] = None
                         ) -> Dict[str, torch.Tensor]:
    x = layer_input.clone()
    if input_mask is not None:
        x = x * input_mask
    weight = conv.weight.detach().clone()
    bias = conv.bias.detach().clone() if conv.bias is not None else None
    if weight_mask is not None:
        weight[target_channel] = weight[target_channel] * weight_mask
    pre = F.conv2d(x, weight, bias=bias, stride=conv.stride, padding=conv.padding)
    if pre_relu_mask is not None:
        pre = pre * pre_relu_mask
    relu = F.relu(pre)
    pool = F.max_pool2d(relu, kernel_size=spec.pool_kernel, stride=spec.pool_stride)
    gn = F.group_norm(relu, num_groups=spec.gn_groups, weight=None, bias=None, eps=1e-5)
    branch = pool if branch_type == 'maxpool' else gn
    return {'pre': pre, 'relu': relu, 'pool': pool, 'gn': gn, 'branch': branch}


def global_channel_response(branch_map: torch.Tensor, target_channel: int) -> torch.Tensor:
    return branch_map[:, target_channel].amax(dim=(-1, -2))


def global_channel_argmax(branch_map: torch.Tensor, target_channel: int) -> Tuple[torch.Tensor, torch.Tensor]:
    m = branch_map[:, target_channel]
    w = m.shape[-1]
    idx = m.reshape(m.shape[0], -1).argmax(dim=1)
    return idx // w, idx % w


def pool_cell_to_prepool_window(py: int, px: int, spec: LayerSpec, pre_h: int, pre_w: int) -> Tuple[int, int, int, int]:
    y0 = py * spec.pool_stride
    x0 = px * spec.pool_stride
    return y0, min(y0 + spec.pool_kernel, pre_h), x0, min(x0 + spec.pool_kernel, pre_w)


def pick_prepool_winner(relu_map: torch.Tensor, channel: int, py: int, px: int, spec: LayerSpec) -> Tuple[int, int]:
    y0, y1, x0, x1 = pool_cell_to_prepool_window(py, px, spec, relu_map.shape[-2], relu_map.shape[-1])
    local = relu_map[channel, y0:y1, x0:x1]
    idx = int(local.reshape(-1).argmax().item())
    ww = local.shape[1]
    return y0 + idx // ww, x0 + idx % ww


def extract_input_patch(layer_input_single: torch.Tensor, y: int, x: int, conv: nn.Conv2d) -> torch.Tensor:
    k = conv.kernel_size[0]
    p = conv.padding[0]
    padded = F.pad(layer_input_single.unsqueeze(0), (p, p, p, p))
    return padded[:, :, y:y + k, x:x + k].squeeze(0)


def apply_local_input_patch_mask(layer_input_single: torch.Tensor, y: int, x: int, conv: nn.Conv2d,
                                 local_mask: torch.Tensor) -> torch.Tensor:
    k = conv.kernel_size[0]
    p = conv.padding[0]
    padded = F.pad(layer_input_single.unsqueeze(0), (p, p, p, p))
    padded[:, :, y:y + k, x:x + k] *= local_mask.unsqueeze(0)
    if p > 0:
        return padded[:, :, p:-p, p:-p].squeeze(0)
    return padded.squeeze(0)


def random_binary_mask(shape: Tuple[int, ...], zero_fraction: float, rng: np.random.Generator) -> torch.Tensor:
    total = int(np.prod(shape))
    n_zero = int(round(zero_fraction * total))
    arr = np.ones(total, dtype=np.float32)
    if n_zero > 0:
        arr[rng.choice(total, size=n_zero, replace=False)] = 0.0
    return torch.tensor(arr.reshape(shape), dtype=torch.float32)


def hoyer_sparsity(x: np.ndarray) -> float:
    x = np.abs(np.asarray(x, dtype=np.float64)).reshape(-1)
    n = len(x)
    if n == 0:
        return float('nan')
    l1 = np.sum(x)
    l2 = np.sqrt(np.sum(x ** 2))
    if l2 < 1e-12:
        return 0.0
    return float((np.sqrt(n) - (l1 / l2)) / (np.sqrt(n) - 1.0 + 1e-12))


def pairwise_cosine_mean(arrs: List[np.ndarray]) -> float:
    if len(arrs) < 2:
        return float('nan')
    mats = [a.reshape(-1).astype(np.float64) for a in arrs]
    vals = []
    for i, ai in enumerate(mats):
        nai = np.linalg.norm(ai)
        if nai < 1e-12:
            continue
        for aj in mats[i + 1:]:
            naj = np.linalg.norm(aj)
            if naj < 1e-12:
                continue
            vals.append(float(np.dot(ai, aj) / (nai * naj)))
    return float(np.nanmean(vals)) if vals else float('nan')


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def dprime(pos: np.ndarray, neg: np.ndarray) -> float:
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    denom = math.sqrt(0.5 * (np.var(pos) + np.var(neg)) + 1e-12)
    return float((np.mean(pos) - np.mean(neg)) / denom)


def compute_selectivity_from_data(model: nn.Module, batch: torch.Tensor, targets: np.ndarray, spec: LayerSpec) -> pd.DataFrame:
    with torch.no_grad():
        feats = forward_features(model, batch)
    responses = feats[spec.pool_idx].amax(dim=(-1, -2)).cpu().numpy()
    rows = []
    for ch in range(responses.shape[1]):
        pos = responses[targets == 1, ch]
        neg = responses[targets == 0, ch]
        if len(pos) and len(neg):
            rows.append({'unit': ch, 'mean_target': float(np.mean(pos)), 'mean_other': float(np.mean(neg)),
                         'dprime': dprime(pos, neg), 'gap': float(np.mean(pos) - np.mean(neg))})
    return pd.DataFrame(rows).sort_values(['dprime', 'gap'], ascending=False)


def analyze_intact_examples(model: nn.Module, batch: torch.Tensor, metadata: pd.DataFrame, spec: LayerSpec,
                            target_channel: int) -> Tuple[pd.DataFrame, List[np.ndarray], List[np.ndarray]]:
    conv = get_conv_layer(model, spec)
    rows, pref_patches, other_patches = [], [], []
    with torch.no_grad():
        features = forward_features(model, batch)
        layer_input = get_layer_input(features, batch, spec)
        out = continue_local_block(layer_input, conv, spec, target_channel, branch_type='maxpool')
        relu, pool = out['relu'], out['pool']
        pool_y, pool_x = global_channel_argmax(pool, target_channel)
        for i in range(batch.shape[0]):
            py, px = int(pool_y[i].item()), int(pool_x[i].item())
            wy, wx = pick_prepool_winner(relu[i], target_channel, py, px, spec)
            inp_patch_np = extract_input_patch(layer_input[i], wy, wx, conv).detach().cpu().numpy()
            y0, y1, x0, x1 = pool_cell_to_prepool_window(py, px, spec, relu.shape[-2], relu.shape[-1])
            prepool_window = relu[i, target_channel, y0:y1, x0:x1].detach().cpu().numpy()
            row = {
                'image_path': metadata.iloc[i]['path'],
                'category': metadata.iloc[i]['category'],
                'is_target': int(metadata.iloc[i]['is_target']),
                'response_pool_max': float(pool[i, target_channel].amax().item()),
                'response_relu_max': float(relu[i, target_channel].amax().item()),
                'input_patch_hoyer': hoyer_sparsity(inp_patch_np),
                'input_patch_l0_gt1pctmax': float(np.mean(inp_patch_np > (0.01 * np.max(inp_patch_np) if np.max(inp_patch_np) > 0 else 0))),
                'prepool_window_hoyer': hoyer_sparsity(prepool_window),
            }
            rows.append(row)
            (pref_patches if row['is_target'] == 1 else other_patches).append(inp_patch_np)
    return pd.DataFrame(rows), pref_patches, other_patches


def run_damage_experiment(model: nn.Module, batch: torch.Tensor, metadata: pd.DataFrame, spec: LayerSpec,
                          target_channel: int, lesion_fractions: List[float], n_repeats: int) -> pd.DataFrame:
    conv = get_conv_layer(model, spec)
    rows = []
    with torch.no_grad():
        intact_features = forward_features(model, batch)
        intact_layer_input = get_layer_input(intact_features, batch, spec)
    for branch_type in ['maxpool', 'groupnorm']:
        with torch.no_grad():
            intact = continue_local_block(intact_layer_input, conv, spec, target_channel, branch_type=branch_type)
            intact_branch = intact['branch']
            intact_response = global_channel_response(intact_branch, target_channel).cpu().numpy()
            intact_y, intact_x = global_channel_argmax(intact_branch, target_channel)
        for damage_mode in ['incoming_weights', 'input_activations', 'pre_relu_units']:
            for frac in lesion_fractions:
                for rep in range(n_repeats):
                    rng = np.random.default_rng(1000 * rep + int(frac * 1000) + abs(hash((spec.key, branch_type, damage_mode))) % 1000)
                    if damage_mode == 'incoming_weights':
                        weight_mask = random_binary_mask(tuple(conv.weight[target_channel].shape), frac, rng)
                        with torch.no_grad():
                            damaged = continue_local_block(intact_layer_input, conv, spec, target_channel, branch_type=branch_type,
                                                           weight_mask=weight_mask)
                            damaged_response = global_channel_response(damaged['branch'], target_channel).cpu().numpy()
                    else:
                        damaged_values = []
                        for i in range(batch.shape[0]):
                            single_input = intact_layer_input[i]
                            iy, ix = int(intact_y[i].item()), int(intact_x[i].item())
                            if branch_type == 'maxpool':
                                wy, wx = pick_prepool_winner(intact['relu'][i], target_channel, iy, ix, spec)
                            else:
                                wy, wx = iy, ix
                            if damage_mode == 'input_activations':
                                local_mask = random_binary_mask((spec.in_channels, spec.kernel_size, spec.kernel_size), frac, rng)
                                masked_input = apply_local_input_patch_mask(single_input, wy, wx, conv, local_mask)
                                out = continue_local_block(masked_input.unsqueeze(0), conv, spec, target_channel, branch_type=branch_type)
                            else:
                                with torch.no_grad():
                                    intact_single = continue_local_block(single_input.unsqueeze(0), conv, spec, target_channel, branch_type=branch_type)
                                pre = intact_single['pre']
                                if branch_type == 'maxpool':
                                    y0, y1, x0, x1 = pool_cell_to_prepool_window(iy, ix, spec, pre.shape[-2], pre.shape[-1])
                                else:
                                    y0, y1 = max(0, wy - 1), min(pre.shape[-2], wy + 2)
                                    x0, x1 = max(0, wx - 1), min(pre.shape[-1], wx + 2)
                                mask = torch.ones((1, spec.out_channels, pre.shape[-2], pre.shape[-1]), dtype=torch.float32)
                                local_mask = random_binary_mask((y1 - y0, x1 - x0), frac, rng)
                                mask[0, target_channel, y0:y1, x0:x1] = local_mask
                                out = continue_local_block(single_input.unsqueeze(0), conv, spec, target_channel, branch_type=branch_type,
                                                           pre_relu_mask=mask)
                            damaged_values.append(float(global_channel_response(out['branch'], target_channel).item()))
                        damaged_response = np.asarray(damaged_values, dtype=np.float64)
                    for i, val in enumerate(damaged_response):
                        rows.append({
                            'layer': spec.key,
                            'branch': branch_type,
                            'damage_mode': damage_mode,
                            'lesion_fraction': frac,
                            'repeat': rep,
                            'image_path': metadata.iloc[i]['path'],
                            'category': metadata.iloc[i]['category'],
                            'is_target': int(metadata.iloc[i]['is_target']),
                            'intact_response': float(intact_response[i]),
                            'damaged_response': float(val),
                        })
    return pd.DataFrame(rows)


def summarize_damage(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(['layer', 'branch', 'damage_mode', 'lesion_fraction', 'repeat']):
        layer, branch, damage_mode, lesion_fraction, repeat = keys
        pos_intact = sub[sub['is_target'] == 1]['intact_response'].values
        neg_intact = sub[sub['is_target'] == 0]['intact_response'].values
        pos_dam = sub[sub['is_target'] == 1]['damaged_response'].values
        neg_dam = sub[sub['is_target'] == 0]['damaged_response'].values
        rows.append({
            'layer': layer, 'branch': branch, 'damage_mode': damage_mode, 'lesion_fraction': lesion_fraction, 'repeat': repeat,
            'target_retention': float(np.mean(pos_dam) / (np.mean(pos_intact) + 1e-12)),
            'other_retention': float(np.mean(neg_dam) / (np.mean(neg_intact) + 1e-12)),
            'selectivity_gap_intact': float(np.mean(pos_intact) - np.mean(neg_intact)),
            'selectivity_gap_damaged': float(np.mean(pos_dam) - np.mean(neg_dam)),
            'target_vs_other_dprime_intact': dprime(pos_intact, neg_intact),
            'target_vs_other_dprime_damaged': dprime(pos_dam, neg_dam),
            'pattern_corr': safe_corr(sub['intact_response'].values, sub['damaged_response'].values),
        })
    return pd.DataFrame(rows)


def plot_example_images(df: pd.DataFrame, target_category: str, max_per_group: int = 4) -> plt.Figure:
    target_df = df[df['is_target'] == 1].head(max_per_group)
    other_df = df[df['is_target'] == 0].head(max_per_group)
    cols = max(len(target_df), len(other_df), 1)
    fig, axes = plt.subplots(2, cols, figsize=(3 * cols, 6))
    if cols == 1:
        axes = np.array(axes).reshape(2, 1)
    for row, subset in enumerate([target_df, other_df]):
        for col in range(cols):
            ax = axes[row, col]
            ax.axis('off')
            if col < len(subset):
                ax.imshow(Image.open(subset.iloc[col]['path']).convert('RGB'))
                ax.set_title(subset.iloc[col]['category'], fontsize=9)
    fig.suptitle(f'Target: {target_category} (top row) vs non-target (bottom row)', fontsize=12)
    plt.tight_layout()
    return fig


def plot_sparsity_consistency(layer_results: Dict[str, Tuple[pd.DataFrame, List[np.ndarray], List[np.ndarray]]]) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for col, metric in enumerate(['input_patch_hoyer', 'prepool_window_hoyer']):
        ax = axes[0, col]
        for layer_key, (df, _, _) in layer_results.items():
            ax.hist(df[df['is_target'] == 1][metric].values, bins=20, alpha=0.4, label=f'{layer_key} target')
            ax.hist(df[df['is_target'] == 0][metric].values, bins=20, alpha=0.4, label=f'{layer_key} other')
        ax.set_title(metric)
        ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    rows = []
    for layer_key, (_, pref_patches, other_patches) in layer_results.items():
        rows.append((layer_key, 'target', pairwise_cosine_mean(pref_patches)))
        rows.append((layer_key, 'other', pairwise_cosine_mean(other_patches)))
    labels = [f'{lk}\n{grp}' for lk, grp, _ in rows]
    vals = [v for _, _, v in rows]
    ax.bar(np.arange(len(vals)), vals)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title('Local input consistency\n(pairwise cosine of winning input patches)')

    ax = axes[1, 1]
    for layer_key, (df, _, _) in layer_results.items():
        ax.hist(df[df['is_target'] == 1]['response_pool_max'].values, bins=20, alpha=0.4, label=f'{layer_key} target')
        ax.hist(df[df['is_target'] == 0]['response_pool_max'].values, bins=20, alpha=0.4, label=f'{layer_key} other')
    ax.set_title('Intact pooled responses')
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle('Sparsity and consistency by layer', fontsize=13)
    plt.tight_layout()
    return fig


def plot_damage_summary(summary_df: pd.DataFrame, value_col: str, title: str) -> plt.Figure:
    damage_modes = ['incoming_weights', 'input_activations', 'pre_relu_units']
    fig, axes = plt.subplots(2, len(damage_modes), figsize=(15, 8), sharex=True)
    for row, layer_key in enumerate(['first', 'final']):
        for col, damage_mode in enumerate(damage_modes):
            ax = axes[row, col]
            sub = summary_df[(summary_df['layer'] == layer_key) & (summary_df['damage_mode'] == damage_mode)]
            for branch in ['maxpool', 'groupnorm']:
                s = sub[sub['branch'] == branch]
                if s.empty:
                    continue
                x = np.sort(s['lesion_fraction'].unique())
                means, lo, hi = [], [], []
                for frac in x:
                    vals = s[s['lesion_fraction'] == frac][value_col].values
                    means.append(np.nanmean(vals))
                    lo.append(np.nanpercentile(vals, 25))
                    hi.append(np.nanpercentile(vals, 75))
                ax.plot(x, means, label=branch)
                ax.fill_between(x, lo, hi, alpha=0.2)
            ax.set_title(f'{layer_key} | {damage_mode}')
            ax.grid(alpha=0.25)
            if row == 1:
                ax.set_xlabel('lesion fraction')
            if col == 0:
                ax.set_ylabel(value_col)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    return fig


def run_sanity_checks(model: nn.Module) -> pd.DataFrame:
    rows = []
    for layer_key, spec in LAYER_SPECS.items():
        conv = get_conv_layer(model, spec)
        x = torch.randn(2, spec.in_channels, 31, 31)
        pre = F.conv2d(x, conv.weight, bias=conv.bias, stride=conv.stride, padding=conv.padding)
        relu = F.relu(pre)
        pool = F.max_pool2d(relu, kernel_size=spec.pool_kernel, stride=spec.pool_stride)
        manual_pool = F.max_pool2d(relu, kernel_size=spec.pool_kernel, stride=spec.pool_stride)
        gn = F.group_norm(relu, num_groups=spec.gn_groups, weight=None, bias=None, eps=1e-5)
        g = spec.gn_groups
        reshaped = gn.view(gn.shape[0], g, gn.shape[1] // g, gn.shape[2], gn.shape[3])
        means = reshaped.mean(dim=(2, 3, 4))
        vars_ = reshaped.var(dim=(2, 3, 4), unbiased=False)
        rows.append({'layer': layer_key, 'maxpool_ok': bool(torch.allclose(pool, manual_pool, atol=1e-6)),
                     'groupnorm_group_mean_maxabs': float(means.abs().max().item()),
                     'groupnorm_group_var_maxabs_error': float((vars_ - 1.0).abs().max().item())})
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout='wide')
    st.title(APP_TITLE)
    st.caption('Uses pretrained AlexNet and your real image folders to compare category-selective units in the first and final blocks, including a GroupNorm replacement branch for the local pooling stage.')

    with st.sidebar:
        st.header('Paths')
        root_dir = Path(st.text_input('Root image directory', value='./images'))
        selectivity_path_str = st.text_input('Selectivity file path (optional)', value='')
        selectivity_path = Path(selectivity_path_str) if selectivity_path_str.strip() else None
        cat_dirs = list_category_dirs(root_dir)
        cat_names = [p.name for p in cat_dirs]
        target_category = st.selectbox('Target category', options=cat_names if cat_names else [''], index=0 if cat_names else None)

        st.header('Analysis options')
        per_category_limit = st.slider('Max images per category', min_value=5, max_value=200, value=40, step=5)
        top_k_units = st.slider('Top selective units to test per layer', min_value=1, max_value=10, value=3, step=1)
        manual_first = st.text_input('Manual first-layer unit indices (comma-separated, optional)', value='')
        manual_final = st.text_input('Manual final-layer unit indices (comma-separated, optional)', value='')

        st.header('Lesions')
        n_repeats = st.slider('Repeats per lesion fraction', min_value=1, max_value=50, value=8, step=1)
        lesion_max = st.slider('Max lesion fraction', min_value=0.1, max_value=1.0, value=0.9, step=0.05)
        lesion_steps = st.slider('Number of lesion steps', min_value=3, max_value=15, value=8, step=1)
        lesion_fractions = np.round(np.linspace(0.0, lesion_max, lesion_steps), 3).tolist()
        run_button = st.button('Run analysis', type='primary', use_container_width=True)

    if not cat_dirs:
        st.warning('No category subdirectories found yet.')
        return

    image_df = build_image_table(root_dir, target_category, per_category_limit)
    if image_df.empty:
        st.warning('No images found in the supplied directory tree.')
        return

    st.write(f'Loaded {len(image_df)} images across {image_df["category"].nunique()} categories.')
    st.pyplot(plot_example_images(image_df, target_category), clear_figure=True)

    model, preprocess, _ = load_model_and_preprocess()
    st.dataframe(run_sanity_checks(model), use_container_width=True)

    batch = torch.stack([load_image_tensor(p, preprocess) for p in image_df['path'].tolist()], dim=0)
    targets = image_df['is_target'].values.astype(int)

    selectivity_df = None
    if selectivity_path is not None and selectivity_path.exists():
        try:
            selectivity_df = read_table_any(selectivity_path)
            st.success(f'Loaded selectivity file with {len(selectivity_df)} rows.')
        except Exception as e:
            st.warning(f'Could not parse selectivity file: {e}')

    if run_button:
        selected_units: Dict[str, List[int]] = {}
        computed_selectivity_tables: Dict[str, pd.DataFrame] = {}
        for layer_key, spec in LAYER_SPECS.items():
            units: List[int] = []
            manual = manual_first if layer_key == 'first' else manual_final
            if manual.strip():
                try:
                    units = [int(x.strip()) for x in manual.split(',') if x.strip()]
                except Exception:
                    units = []
            if not units and selectivity_df is not None:
                units = pick_units_from_selectivity(selectivity_df, target_category, layer_key, top_k_units)
            if not units:
                rank_df = compute_selectivity_from_data(model, batch, targets, spec)
                computed_selectivity_tables[layer_key] = rank_df
                units = rank_df['unit'].head(top_k_units).astype(int).tolist()
            selected_units[layer_key] = units

        st.subheader('Selected units')
        st.dataframe(pd.DataFrame([{'layer': lk, 'unit': u} for lk, units in selected_units.items() for u in units]), use_container_width=True)

        for lk, rank_df in computed_selectivity_tables.items():
            with st.expander(f'Computed selectivity ranking for {lk} layer'):
                st.dataframe(rank_df, use_container_width=True)

        all_damage_rows, all_summary_rows = [], []
        for layer_key, spec in LAYER_SPECS.items():
            for unit in selected_units[layer_key]:
                intact_df, pref_patches, other_patches = analyze_intact_examples(model, batch, image_df, spec, unit)
                damage_df = run_damage_experiment(model, batch, image_df, spec, unit, lesion_fractions, n_repeats)
                damage_df['unit'] = unit
                summary_df = summarize_damage(damage_df)
                summary_df['unit'] = unit
                all_damage_rows.append(damage_df)
                all_summary_rows.append(summary_df)

                st.markdown(f'### {spec.label} | unit {unit}')
                c1, c2 = st.columns(2)
                with c1:
                    st.dataframe(intact_df.groupby('is_target', as_index=False)[['response_pool_max', 'input_patch_hoyer', 'prepool_window_hoyer']].mean(), use_container_width=True)
                with c2:
                    st.write({'target_patch_consistency': pairwise_cosine_mean(pref_patches),
                              'other_patch_consistency': pairwise_cosine_mean(other_patches),
                              'intact_dprime': dprime(intact_df[intact_df['is_target'] == 1]['response_pool_max'].values,
                                                     intact_df[intact_df['is_target'] == 0]['response_pool_max'].values)})
                st.pyplot(plot_sparsity_consistency({f'{layer_key}-u{unit}': (intact_df, pref_patches, other_patches)}), clear_figure=True)
                st.pyplot(plot_damage_summary(summary_df, 'target_vs_other_dprime_damaged', f'Selectivity robustness | {layer_key} unit {unit}'), clear_figure=True)
                st.pyplot(plot_damage_summary(summary_df, 'target_retention', f'Target-response retention | {layer_key} unit {unit}'), clear_figure=True)

        if all_summary_rows:
            all_damage_df = pd.concat(all_damage_rows, ignore_index=True)
            all_summary_df = pd.concat(all_summary_rows, ignore_index=True)
            st.subheader('Combined summaries')
            st.dataframe(all_summary_df, use_container_width=True)
            grouped = all_summary_df.groupby(['layer', 'branch', 'damage_mode', 'lesion_fraction', 'repeat'], as_index=False)[['target_vs_other_dprime_damaged', 'target_retention', 'pattern_corr']].mean()
            st.pyplot(plot_damage_summary(grouped, 'target_vs_other_dprime_damaged', 'Mean selectivity robustness across selected units'), clear_figure=True)
            st.pyplot(plot_damage_summary(grouped, 'target_retention', 'Mean target retention across selected units'), clear_figure=True)
            st.download_button('Download all_damage_rows.csv', data=all_damage_df.to_csv(index=False).encode('utf-8'), file_name='all_damage_rows.csv', mime='text/csv')
            st.download_button('Download all_summary_rows.csv', data=all_summary_df.to_csv(index=False).encode('utf-8'), file_name='all_summary_rows.csv', mime='text/csv')

        st.markdown(
            '**Interpretation notes**\n'
            '- `response_pool_max` is the global max over the target channel after the native AlexNet max-pool branch.\n'
            '- The GroupNorm branch is a counterfactual replacement applied to the post-ReLU feature map with no learned affine parameters.\n'
            '- `input_patch_hoyer` measures sparsity of the local input tensor feeding the winning pre-pool unit.\n'
            '- Patch consistency is the mean pairwise cosine similarity between those winning local input patches across images.\n'
            '- `target_vs_other_dprime_damaged` tracks whether the unit still separates target from other categories under lesioning.'
        )


if __name__ == '__main__':
    main()
