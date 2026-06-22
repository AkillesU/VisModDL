#!/usr/bin/env python3
"""
Selective-unit and connection-damage diagnostics across VGG16, AlexNet, and CORnet-RT.

This script is intentionally adapted to the conventions already present in this folder:
- uses utils.load_model / preprocess_image / apply_masking / get_all_conv_layers
- tolerates DataParallel/module-wrapped paths
- resolves selectivity files from either a direct file path or a selectivity directory
- tries to work with both channel-level and row-level selectivity tables

Main analyses
-------------
1) Top selective unit robustness to incoming-weight deletion
2) Distribution of top category-selective units across layers
3) Effect of severing a single connection in each conv layer on the final readout layer
4) Activation histograms / sparsity summaries across conv layers

Notes
-----
- For VGG16/AlexNet the "unit layer" defaults to the final conv layer (block5 proxy).
- For CORnet-RT the "unit layer" defaults to IT.conv1 and the propagation readout defaults to IT.output.
- Post-damage selectivity is recomputed as a normalized Mann–Whitney effect score
  U / (n_target * n_other). This is not guaranteed to be numerically identical to any
  precomputed selectivity column in your pickles, but it is directly interpretable and
  monotonically related to MW-based category separation.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to sys.path so we can import utils from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import copy
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
try:
    import seaborn as sns
    HAS_SNS = True
except Exception:
    HAS_SNS = False

try:
    from scipy.stats import mannwhitneyu
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

from utils import (
    load_model,
    preprocess_image,
    get_layer_from_path,
    assign_categories,
    apply_masking,
    get_all_conv_layers,
    normalize_module_name,
)

try:
    from utils import resolve_selectivity_table
except Exception:
    resolve_selectivity_table = None

import utils as _utils


# -----------------------------------------------------------------------------
# Robust layer-path resolver (adapted from your uploaded diagnostics scripts)
# -----------------------------------------------------------------------------

def _robust_get_layer_from_path(model: nn.Module, path: str):
    try:
        norm = normalize_module_name(path)
    except Exception:
        norm = str(path).replace("._modules.", ".").replace("._modules", ".").strip(".")

    steps = [s for s in norm.split(".") if s]
    current = model

    if hasattr(current, "module"):
        current = current.module

    if steps and steps[0] == "module":
        steps = steps[1:]

    def _resolve_step(obj, step: str):
        if hasattr(obj, step):
            return getattr(obj, step), True
        if step.isdigit():
            idx = int(step)
            try:
                return obj[idx], True
            except Exception:
                pass
            if hasattr(obj, "_modules") and str(idx) in obj._modules:
                return obj._modules[str(idx)], True
        if hasattr(obj, "_modules") and step in obj._modules:
            return obj._modules[step], True
        if isinstance(obj, dict) and step in obj:
            return obj[step], True
        return obj, False

    for step in steps:
        nxt, ok = _resolve_step(current, step)
        if ok:
            current = nxt
            continue
        if hasattr(current, "module"):
            unwrapped = current.module
            nxt, ok = _resolve_step(unwrapped, step)
            if ok:
                current = nxt
                continue
        avail = list(getattr(current, "_modules", {}).keys())[:40] if hasattr(current, "_modules") else []
        raise KeyError(
            f"Cannot resolve step '{step}' in path '{path}' (normalized='{norm}'). "
            f"Current object type: {type(current)}. Available child modules (first 40): {avail}"
        )
    return current


_utils.get_layer_from_path = _robust_get_layer_from_path
get_layer_from_path = _robust_get_layer_from_path


# -----------------------------------------------------------------------------
# Model defaults
# -----------------------------------------------------------------------------

MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "vgg16": {
        "model_info": {"source": "pytorch_hub", "repo": "pytorch/vision", "name": "vgg16", "weights": "DEFAULT"},
        "unit_measure_layer": "features.30",
        "unit_weight_layer": "features.28",
        "unit_layer": "features.30",
        "propagation_hook": "features.30",
        "friendly_layers": {
            "features.0": "block1_conv1",
            "features.1": "block1_relu1",
            "features.2": "block1_conv2",
            "features.3": "block1_relu2",
            "features.5": "block2_conv1",
            "features.6": "block2_relu1",
            "features.7": "block2_conv2",
            "features.8": "block2_relu2",
            "features.10": "block3_conv1",
            "features.11": "block3_relu1",
            "features.12": "block3_conv2",
            "features.13": "block3_relu2",
            "features.14": "block3_conv3",
            "features.15": "block3_relu3",
            "features.17": "block4_conv1",
            "features.18": "block4_relu1",
            "features.19": "block4_conv2",
            "features.20": "block4_relu2",
            "features.21": "block4_conv3",
            "features.22": "block4_relu3",
            "features.24": "block5_conv1",
            "features.25": "block5_relu1",
            "features.26": "block5_conv2",
            "features.27": "block5_relu2",
            "features.28": "block5_conv3",
            "features.29": "block5_relu3",
            "features.30": "block5_output",
        },
    },
    "alexnet": {
        "model_info": {"source": "pytorch_hub", "repo": "pytorch/vision", "name": "alexnet", "weights": "DEFAULT"},
        "unit_measure_layer": "features.11",
        "unit_weight_layer": "features.10",
        "unit_layer": "features.11",
        "propagation_hook": "features.12",
        "friendly_layers": {
            "features.0": "block1_conv1",
            "features.1": "block1_relu1",
            "features.3": "block2_conv2",
            "features.4": "block2_relu2",
            "features.6": "block3_conv3",
            "features.7": "block3_relu3",
            "features.8": "block4_conv4",
            "features.9": "block4_relu4",
            "features.10": "block5_conv5",
            "features.11": "block5_relu5",
            "features.12": "block5_output",
        },
    },
    "cornet_rt": {
        "model_info": {"source": "cornet", "repo": "-", "name": "cornet_rt", "weights": "", "time_steps": 5},
        "unit_measure_layer": "module.IT.output",
        "unit_weight_layer": "module.IT.conv1",
        "unit_layer": "module.IT.output",
        "propagation_hook": "module.IT.output",
        "friendly_layers": {
            "V1.nonlin_input": "V1_nonlin_input",
            "V1.nonlin1": "V1_nonlin1",
            "V1.output": "V1_output",
            "V2.nonlin_input": "V2_nonlin_input",
            "V2.nonlin1": "V2_nonlin1",
            "V2.output": "V2_output",
            "V4.nonlin_input": "V4_nonlin_input",
            "V4.nonlin1": "V4_nonlin1",
            "V4.output": "V4_output",
            "IT.nonlin_input": "IT_nonlin_input",
            "module.IT.nonlin_input": "IT_nonlin_input",
            "IT.nonlin1": "IT_nonlin1",
            "module.IT.nonlin1": "IT_nonlin1",
            "IT.output": "IT_output",
            "module.IT.output": "IT_output",
            "V1.conv_input": "V1_conv_input",
            "V1.conv1": "V1_conv1",
            "V2.conv_input": "V2_conv_input",
            "V2.conv1": "V2_conv1",
            "V2.conv2": "V2_conv2",
            "V2.conv3": "V2_conv3",
            "V4.conv_input": "V4_conv_input",
            "V4.conv1": "V4_conv1",
            "V4.conv2": "V4_conv2",
            "V4.conv3": "V4_conv3",
            "IT.conv_input": "IT_conv_input",
            "module.IT.conv_input": "IT_conv_input",
            "IT.conv1": "IT_conv1",
            "module.IT.conv1": "IT_conv1",
            "IT.conv2": "IT_conv2",
            "IT.conv3": "IT_conv3",
        },
    },
}


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _mkdir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sanitize(text: Any) -> str:
    return "".join(c if str(c).isalnum() or c in "-_" else "_" for c in str(text))



def _normalize_layer_label(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    for pref in ("model.", "net.", "module.", "wrapped."):
        if s.startswith(pref):
            s = s[len(pref):]
    s = s.replace("module.", "")
    try:
        s = normalize_module_name(s)
    except Exception:
        s = s.replace("._modules.", ".").replace("._modules", ".")
    return s.lower().strip(".")



def _hook_to_selectivity_layer(hook: str) -> str:
    hook = _normalize_layer_label(hook)
    if not hook:
        return hook
    return hook.split(".")[0] if hook.endswith(".output") else hook


CORNET_BLOCK_TOKENS = {"v1", "v2", "v4", "it"}


def _base_normalize_layer_path(layer_name: str) -> str:
    if layer_name is None:
        return ""
    norm = str(layer_name).strip()
    for pref in ("model.", "net.", "module.", "wrapped."):
        if norm.startswith(pref):
            norm = norm[len(pref):]
    try:
        norm = normalize_module_name(norm)
    except Exception:
        norm = norm.replace("._modules.", ".").replace("._modules", ".")
    return norm.strip(".")


def _canonicalize_measure_layer_for_model(layer_name: str, model_name: Optional[str] = None) -> str:
    """Canonicalize a layer path for *reading out* activations."""
    norm = _base_normalize_layer_path(layer_name)
    if not norm:
        return norm
    if str(model_name or "").lower() == "cornet_rt":
        last = norm.split(".")[-1]
        if last.lower() in CORNET_BLOCK_TOKENS and not norm.lower().endswith(".output"):
            return f"{norm}.output"
    return norm


def _canonicalize_weight_layer_for_model(layer_name: str, model_name: Optional[str] = None) -> str:
    """Canonicalize a layer path for *incoming weights*.

    For CORnet blocks, selectivity rows are stored on block outputs (e.g. module.IT), but
    the incoming weights we want to damage live on that block's conv1 module.
    """
    norm = _base_normalize_layer_path(layer_name)
    if not norm:
        return norm
    if str(model_name or "").lower() == "cornet_rt":
        if norm.lower().endswith(".output"):
            return f"{norm[:-len('.output')]}.conv1"
        last = norm.split(".")[-1]
        if last.lower() in CORNET_BLOCK_TOKENS:
            return f"{norm}.conv1"
    return norm


def _canonicalize_layer_for_model(layer_name: str, model_name: Optional[str] = None) -> str:
    """Backward-compatible canonicalizer used for layer-location summaries.

    This keeps CORnet block labels aligned with selectivity files by interpreting bare block
    names like IT/V1/V2/V4 as their output readouts.
    """
    return _canonicalize_measure_layer_for_model(layer_name, model_name=model_name)


def _layer_aliases(layer_name: str, model_name: Optional[str] = None) -> List[str]:
    norm = _normalize_layer_label(layer_name)
    aliases: List[str] = []
    if norm:
        aliases.append(norm)
    model_key = str(model_name or "").lower()
    if model_key == "cornet_rt" and norm:
        if norm.endswith(".output"):
            aliases.append(norm[: -len(".output")])
        last = norm.split(".")[-1]
        if last in CORNET_BLOCK_TOKENS and not norm.endswith(".output"):
            aliases.append(f"{norm}.output")
    elif model_key == "alexnet" and norm:
        # Allow selectivity files saved on the post-pool block output to match the
        # post-ReLU measurement layer, and vice versa.
        if norm == "features.11":
            aliases.append("features.12")
        elif norm == "features.12":
            aliases.append("features.11")
    elif model_key == "vgg16" and norm:
        # Final block selectivity may have been saved either after the last ReLU or after
        # the last max-pool. The measurement layer remains features.30 by default.
        if norm == "features.30":
            aliases.append("features.29")
    seen = set()
    deduped: List[str] = []
    for a in aliases:
        if a and a not in seen:
            seen.add(a)
            deduped.append(a)
    return deduped



def _first_tensor(out: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)):
        for v in out:
            t = _first_tensor(v)
            if t is not None:
                return t
    if isinstance(out, dict):
        for v in out.values():
            t = _first_tensor(v)
            if t is not None:
                return t
    return None



def _attach_hook(model: nn.Module, layer_path: str, store: Dict[str, Any], keep_grad: bool = False):
    layer = get_layer_from_path(model, layer_path)

    def _hook(_m, _inp, _out):
        t = _first_tensor(_out)
        if t is None:
            return
        if keep_grad:
            store[layer_path] = t
            try:
                t.retain_grad()
            except Exception:
                pass
        else:
            store[layer_path] = t.detach().cpu().numpy()

    return layer.register_forward_hook(_hook)



def _channel_summary(arr: np.ndarray | torch.Tensor) -> np.ndarray:
    if torch.is_tensor(arr):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim < 2:
        raise ValueError(f"Expected activation tensor with at least 2 dims, got {arr.shape}")
    if arr.ndim == 2:
        return arr
    axes = tuple(range(2, arr.ndim))
    return arr.mean(axis=axes)



def _flatten_except_batch(arr: np.ndarray | torch.Tensor) -> np.ndarray:
    if torch.is_tensor(arr):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    return arr.reshape(arr.shape[0], -1)



def _to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device, non_blocking=torch.cuda.is_available())



def _concat_tensors(items: Sequence[torch.Tensor]) -> torch.Tensor:
    if not items:
        raise ValueError("Cannot concatenate an empty sequence of tensors.")
    return torch.cat(list(items), dim=0)



def _pick_layer_column(df: pd.DataFrame) -> str:
    candidates = ["layer", "layer_name", "layer_path", "module", "block", "region", "area"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        cl = c.lower()
        if "layer" in cl or "path" in cl or "block" in cl:
            return c
    raise ValueError(f"Could not find a layer column. Columns: {list(df.columns)[:50]}")



def _pick_unit_column(df: pd.DataFrame) -> Optional[str]:
    for c in ("unit", "unit_idx", "unit_index", "channel", "filter", "neuron"):
        if c in df.columns:
            return c
    if "unit_id" in df.columns:
        return None
    int_cols = [c for c in df.columns if pd.api.types.is_integer_dtype(df[c])]
    return int_cols[0] if int_cols else None



def _pick_selectivity_column(df: pd.DataFrame, target_cat: str) -> str:
    target_cat = (target_cat or "").lower()
    cat_to_col = {
        "face": ["mw_faces", "faces", "face"],
        "faces": ["mw_faces", "faces", "face"],
        "object": ["mw_objects", "objects", "object"],
        "objects": ["mw_objects", "objects", "object"],
        "animal": ["mw_animals", "animals", "animal"],
        "animals": ["mw_animals", "animals", "animal"],
        "place": ["mw_places", "places", "place"],
        "places": ["mw_places", "places", "place"],
    }
    for cand in cat_to_col.get(target_cat, []):
        if cand in df.columns:
            return cand
    for c in df.columns:
        if target_cat and target_cat in c.lower():
            return c
    preferred = [c for c in df.columns if c.lower().startswith("mw_") and pd.api.types.is_numeric_dtype(df[c])]
    if preferred:
        return preferred[0]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise ValueError(f"Could not find a numeric selectivity column. Columns: {list(df.columns)[:50]}")
    return num_cols[0]



def _extract_channel_index(row: pd.Series, unit_col: Optional[str]) -> int:
    """Best-effort legacy extractor for a channel/filter index only.

    Prefer parsing unit_id strings like ``layer:channel:y:x`` when available.
    This helper is kept for compatibility, but the main robustness analysis now uses
    ``_resolve_measurement_address`` so that post-ReLU/post-pool spatial unit indices
    can be measured correctly while still mapping to the right incoming conv channel.
    """
    if "unit_id" in row.index:
        parts = str(row["unit_id"]).split(":")
        if len(parts) >= 2:
            try:
                return int(float(parts[1]))
            except Exception:
                pass
    if unit_col is not None and unit_col in row.index:
        return int(row[unit_col])
    raise ValueError("Could not extract a channel/filter index from selectivity row.")



def _parse_unit_id(unit_id: Any) -> Optional[Tuple[int, Optional[Tuple[int, ...]]]]:
    """Parse strings like 'layer:channel:y:x' -> (channel, (y, x))."""
    if unit_id is None:
        return None
    s = str(unit_id)
    if ":" not in s:
        return None
    parts = s.split(":")
    if len(parts) < 2:
        return None
    try:
        ints = [int(float(v)) for v in parts[1:]]
    except Exception:
        return None
    if not ints:
        return None
    channel = int(ints[0])
    spatial = tuple(int(v) for v in ints[1:]) if len(ints) > 1 else None
    return channel, spatial



def _resolve_measurement_address(
    row: pd.Series,
    unit_col: Optional[str],
    activation_shape: Sequence[int],
) -> Dict[str, Any]:
    """Resolve how to *measure* the selective unit and which conv channel feeds it.

    Returns a dict with:
      - measure_mode: 'flat' or 'channel'
      - flat_index: index into flattened post-nonlinearity activations, if applicable
      - channel_index: output channel / filter index for incoming weights
      - spatial_coords: tuple or None
      - activation_shape: tuple(C, ...) excluding batch

    Logic:
      1) If unit_id provides spatial coordinates, measure that exact flattened unit.
      2) Else, if the numeric unit index fits inside the full flattened map but exceeds
         channel count, treat it as a flattened unit index.
      3) Else, treat it as a channel-level index and measure the channel summary.
    """
    act_shape = tuple(int(v) for v in activation_shape)
    if len(act_shape) < 1:
        raise ValueError(f"Invalid activation shape: {activation_shape}")
    C = int(act_shape[0])
    spatial_size = int(np.prod(act_shape[1:])) if len(act_shape) > 1 else 1
    total_units = int(np.prod(act_shape))

    parsed = _parse_unit_id(row.get("unit_id", None)) if isinstance(row, pd.Series) else None
    if parsed is not None:
        channel_index, spatial_coords = parsed
        if channel_index < 0 or channel_index >= C:
            raise IndexError(
                f"Parsed channel index {channel_index} from unit_id is out of bounds for activation shape {act_shape}"
            )
        if len(act_shape) == 1:
            return {
                "measure_mode": "channel",
                "flat_index": int(channel_index),
                "channel_index": int(channel_index),
                "spatial_coords": None,
                "activation_shape": act_shape,
            }
        if spatial_coords is not None and len(spatial_coords) >= (len(act_shape) - 1):
            coords = tuple(int(v) for v in spatial_coords[: len(act_shape) - 1])
            for dim_i, (coord, dim) in enumerate(zip(coords, act_shape[1:])):
                if coord < 0 or coord >= dim:
                    raise IndexError(
                        f"Spatial coordinate {coord} at dim {dim_i} out of bounds for activation shape {act_shape}"
                    )
            flat_index = int(np.ravel_multi_index((channel_index, *coords), act_shape))
            return {
                "measure_mode": "flat",
                "flat_index": flat_index,
                "channel_index": int(channel_index),
                "spatial_coords": coords,
                "activation_shape": act_shape,
            }
        return {
            "measure_mode": "channel",
            "flat_index": None,
            "channel_index": int(channel_index),
            "spatial_coords": None,
            "activation_shape": act_shape,
        }

    raw_index = None
    if unit_col is not None and unit_col in row.index and pd.notna(row[unit_col]):
        raw_index = int(row[unit_col])

    if raw_index is None:
        raise ValueError("Could not extract any usable unit index from the selectivity row.")

    if len(act_shape) == 1:
        if raw_index < 0 or raw_index >= C:
            raise IndexError(f"Unit index {raw_index} is out of bounds for activation shape {act_shape}")
        return {
            "measure_mode": "channel",
            "flat_index": int(raw_index),
            "channel_index": int(raw_index),
            "spatial_coords": None,
            "activation_shape": act_shape,
        }

    unit_col_l = str(unit_col or "").lower()
    explicit_channel = unit_col_l in {"channel", "filter"}
    if explicit_channel or (0 <= raw_index < C):
        return {
            "measure_mode": "channel",
            "flat_index": None,
            "channel_index": int(raw_index),
            "spatial_coords": None,
            "activation_shape": act_shape,
        }

    if 0 <= raw_index < total_units:
        channel_index = int(raw_index // spatial_size)
        return {
            "measure_mode": "flat",
            "flat_index": int(raw_index),
            "channel_index": int(channel_index),
            "spatial_coords": None,
            "activation_shape": act_shape,
        }

    raise IndexError(
        f"Unit index {raw_index} could not be mapped onto activation shape {act_shape}. "
        f"Channel count={C}, flattened unit count={total_units}."
    )



def _load_selectivity_df(path_or_dir: str | Path, model_tag: Optional[str] = None) -> pd.DataFrame:
    p = Path(path_or_dir)

    if resolve_selectivity_table is not None:
        try:
            resolved = resolve_selectivity_table(p, model_tag=model_tag)
            if str(resolved).lower().endswith(".csv"):
                return pd.read_csv(resolved)
            return pd.read_pickle(resolved)
        except Exception:
            pass

    if p.is_file():
        return pd.read_csv(p) if p.suffix.lower() == ".csv" else pd.read_pickle(p)

    candidates: List[Path] = []
    if model_tag:
        candidates.extend([
            p / f"{model_tag}_all_layers_units_mannwhitneyu.pkl",
            p / f"{model_tag}_all_layers_units_mannwhitneyu.csv",
        ])
    candidates.extend([
        p / "all_layers_units_mannwhitneyu.pkl",
        p / "all_layers_units_mannwhitneyu.csv",
    ])
    for c in candidates:
        if c.exists():
            return pd.read_csv(c) if c.suffix.lower() == ".csv" else pd.read_pickle(c)
    raise FileNotFoundError(f"Could not resolve a selectivity table under: {p}")



def _resolve_model_spec(model_name: str, cornet_time_steps: int = 5) -> Dict[str, Any]:
    key = model_name.lower()
    if key not in MODEL_SPECS:
        raise ValueError(f"Unknown model '{model_name}'. Available: {sorted(MODEL_SPECS)}")
    spec = copy.deepcopy(MODEL_SPECS[key])
    if key == "cornet_rt":
        spec["model_info"]["time_steps"] = int(cornet_time_steps)
    return spec



def _load_model_with_fallbacks(model_info: Dict[str, Any], hook_paths: Optional[List[str]] = None):
    hook_paths = hook_paths or []
    try:
        return load_model(model_info, pretrained=True, layer_path=hook_paths if hook_paths else None)
    except Exception as e:
        source = str(model_info.get("source", "")).lower()
        name = str(model_info.get("name", "")).lower()
        if source == "pytorch_hub":
            try:
                import torchvision.models as tvm
                if name == "vgg16":
                    weights = getattr(tvm, "VGG16_Weights", None)
                    model = tvm.vgg16(weights=(weights.DEFAULT if weights is not None else None))
                elif name == "alexnet":
                    weights = getattr(tvm, "AlexNet_Weights", None)
                    model = tvm.alexnet(weights=(weights.DEFAULT if weights is not None else None))
                else:
                    raise
                model.eval()
                activations: Dict[str, Any] = {}
                for hp in hook_paths:
                    layer = get_layer_from_path(model, hp)
                    layer.register_forward_hook(lambda _m, _i, _o, name=hp: activations.__setitem__(name, _first_tensor(_o).detach().cpu().numpy()))
                return model, activations
            except Exception:
                raise RuntimeError(
                    f"utils.load_model failed and torchvision fallback also failed for {model_info}. Original error: {e}"
                ) from e
        raise RuntimeError(f"Could not load model {model_info}. Error: {e}") from e



def get_stimuli_batches(image_dir: str | Path, device: torch.device) -> Tuple[Dict[str, torch.Tensor], List[str], np.ndarray]:
    image_dir = Path(image_dir)
    files = sorted([f.name for f in image_dir.iterdir() if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg")])
    if not files:
        raise FileNotFoundError(f"No images found in {image_dir}")
    categories = assign_categories(files)
    unique_cats = np.unique(categories)
    batches: Dict[str, torch.Tensor] = {}
    for cat in unique_cats:
        idx = np.where(categories == cat)[0]
        tensors = [preprocess_image(image_dir / files[i]) for i in idx]
        batches[str(cat)] = _to_device(torch.cat(tensors, dim=0), device)
    return batches, files, categories



def _collect_all_images_batch(batches: Dict[str, torch.Tensor], exclude_keys: Optional[Iterable[str]] = None) -> torch.Tensor:
    exclude = set(exclude_keys or [])
    keep = [v for k, v in batches.items() if k not in exclude]
    return _concat_tensors(keep)



def _run_hooked_activations(model: nn.Module, hook_path: str, x: torch.Tensor) -> np.ndarray:
    store: Dict[str, Any] = {}
    handle = _attach_hook(model, hook_path, store, keep_grad=False)
    with torch.no_grad():
        model(x)
    handle.remove()
    if hook_path not in store:
        raise RuntimeError(f"Hook '{hook_path}' did not fire.")
    return np.asarray(store[hook_path])



def _normalized_mw_effect(target_vals: np.ndarray, other_vals: np.ndarray) -> float:
    target_vals = np.asarray(target_vals).astype(float)
    other_vals = np.asarray(other_vals).astype(float)
    if target_vals.size == 0 or other_vals.size == 0:
        return float("nan")
    if HAS_SCIPY:
        u = mannwhitneyu(target_vals, other_vals, alternative="two-sided").statistic
        return float(u / (len(target_vals) * len(other_vals)))
    # fallback: probability of superiority with tie handling
    wins = 0.0
    total = 0.0
    for t in target_vals:
        diffs = t - other_vals
        wins += np.sum(diffs > 0) + 0.5 * np.sum(diffs == 0)
        total += len(other_vals)
    return float(wins / total) if total > 0 else float("nan")



def _fraction_affected(clean: np.ndarray, damaged: np.ndarray, eps: float = 1e-4) -> Tuple[float, float]:
    diff = np.abs(clean - damaged)
    thr = eps * (np.abs(clean).mean() + 1e-12)
    affected = diff > thr
    nonzero = np.abs(clean) > thr
    frac_all = float(np.mean(affected))
    frac_nz = float(np.mean(affected & nonzero) / (np.mean(nonzero) + 1e-12))
    return frac_all, frac_nz



def _friendly_layer_name(model_spec: Dict[str, Any], layer_path: str) -> str:
    norm = normalize_module_name(layer_path)
    return model_spec.get("friendly_layers", {}).get(norm, norm)


def _layer_exists(model: nn.Module, layer_path: str) -> bool:
    try:
        get_layer_from_path(model, layer_path)
        return True
    except Exception:
        return False


def _selectivity_rows_for_layer(df: pd.DataFrame, layer_name: str, model_name: Optional[str] = None) -> pd.DataFrame:
    layer_col = _pick_layer_column(df)
    aliases = _layer_aliases(layer_name, model_name=model_name)
    layer_norm = df[layer_col].astype(str).map(_normalize_layer_label)

    sub = df.iloc[0:0].copy()
    for alias in aliases:
        sub = df[layer_norm == alias]
        if not sub.empty:
            return sub.copy()

    for alias in aliases:
        sub = df[layer_norm.str.contains(alias, regex=False, na=False)]
        if not sub.empty:
            return sub.copy()

    if aliases:
        token_sets = [[t for t in alias.replace("/", ".").split(".") if t] for alias in aliases]
        for target_tokens in token_sets:
            mask = pd.Series(True, index=df.index)
            for tok in target_tokens:
                mask &= layer_norm.str.contains(tok, regex=False, na=False)
            sub = df[mask]
            if not sub.empty:
                return sub.copy()
    return sub.copy()


# -----------------------------------------------------------------------------
# Analysis 1: top selective unit robustness to incoming-weight deletion
# -----------------------------------------------------------------------------

def pick_top_selective_unit(selectivity_df: pd.DataFrame, unit_layer: str, target_cat: str, model_name: Optional[str] = None) -> Dict[str, Any]:
    sub = _selectivity_rows_for_layer(selectivity_df, unit_layer, model_name=model_name)
    if sub.empty:
        raise ValueError(f"No selectivity rows matched unit layer '{unit_layer}'.")
    score_col = _pick_selectivity_column(sub, target_cat)
    unit_col = _pick_unit_column(sub)
    top_row = sub.sort_values(score_col, ascending=False).iloc[0]
    return {
        "score_col": score_col,
        "unit_col": unit_col,
        "row": top_row,
        "stored_selectivity_score": float(top_row[score_col]),
    }



def damage_single_unit_incoming_weights(model: nn.Module, layer_path: str, unit_idx: int, frac: float, rng: np.random.Generator):
    layer = get_layer_from_path(model, layer_path)
    if not hasattr(layer, "weight") or layer.weight is None:
        raise ValueError(f"Layer '{layer_path}' has no weight tensor.")
    with torch.no_grad():
        w = layer.weight.data
        if unit_idx >= w.shape[0]:
            raise IndexError(f"unit_idx {unit_idx} out of bounds for layer '{layer_path}' with shape {tuple(w.shape)}")
        flat = w[unit_idx].reshape(-1)
        n = flat.numel()
        k = int(round(float(frac) * n))
        if k <= 0:
            return
        idx = torch.as_tensor(rng.choice(n, size=min(k, n), replace=False), device=flat.device, dtype=torch.long)
        flat[idx] = 0.0



def analyze_top_unit_robustness(
    model_name: str,
    model_spec: Dict[str, Any],
    image_dir: str | Path,
    selectivity_root: str | Path,
    target_cat: str,
    fractions: Sequence[float],
    n_permutations: int,
    output_dir: str | Path,
) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unit_measure_layer = _canonicalize_measure_layer_for_model(
        model_spec.get("unit_measure_layer", model_spec.get("unit_layer")),
        model_name=model_name,
    )
    unit_weight_layer = _canonicalize_weight_layer_for_model(
        model_spec.get("unit_weight_layer", unit_measure_layer),
        model_name=model_name,
    )

    model, _ = _load_model_with_fallbacks(model_spec["model_info"], hook_paths=[])
    model = model.to(device)

    batches, _, _ = get_stimuli_batches(image_dir, device)
    if target_cat not in batches:
        raise ValueError(f"Target category '{target_cat}' not found in image_dir. Found: {sorted(batches.keys())}")
    target_batch = batches[target_cat]
    other_batch = _collect_all_images_batch({k: v for k, v in batches.items() if k != target_cat})
    all_batch = _collect_all_images_batch(batches)

    sel_df = _load_selectivity_df(selectivity_root, model_tag=model_name)
    top_info = pick_top_selective_unit(sel_df, unit_measure_layer, target_cat, model_name=model_name)

    clean_target_raw = np.asarray(_run_hooked_activations(model, unit_measure_layer, target_batch))
    clean_other_raw = np.asarray(_run_hooked_activations(model, unit_measure_layer, other_batch))
    clean_all_raw = np.asarray(_run_hooked_activations(model, unit_measure_layer, all_batch))

    address = _resolve_measurement_address(
        top_info["row"],
        top_info["unit_col"],
        clean_target_raw.shape[1:],
    )
    unit_channel_idx = int(address["channel_index"])
    measured_flat_idx = address["flat_index"]
    measure_mode = str(address["measure_mode"])

    # Sanity-check that the corresponding weighted layer really has this many output channels.
    weight_layer_obj = get_layer_from_path(model, unit_weight_layer)
    if not hasattr(weight_layer_obj, "weight") or weight_layer_obj.weight is None:
        raise ValueError(f"Weight layer '{unit_weight_layer}' has no weights.")
    out_channels = int(weight_layer_obj.weight.shape[0])
    if unit_channel_idx < 0 or unit_channel_idx >= out_channels:
        raise IndexError(
            f"Resolved channel index {unit_channel_idx} is out of bounds for weight layer '{unit_weight_layer}' "
            f"with shape {tuple(weight_layer_obj.weight.shape)}"
        )

    if measure_mode == "flat":
        clean_target = _flatten_except_batch(clean_target_raw)
        clean_other = _flatten_except_batch(clean_other_raw)
        clean_all = _flatten_except_batch(clean_all_raw)
        if measured_flat_idx is None:
            raise ValueError("measure_mode='flat' but flat_index is None")
        if measured_flat_idx < 0 or measured_flat_idx >= clean_target.shape[1]:
            raise IndexError(
                f"Measured flat index {measured_flat_idx} out of bounds for measure layer '{unit_measure_layer}' "
                f"with flattened size {clean_target.shape[1]}"
            )
        baseline_target_act = clean_target[:, measured_flat_idx]
        baseline_other_act = clean_other[:, measured_flat_idx]
        baseline_all_act = clean_all[:, measured_flat_idx]
    elif measure_mode == "channel":
        clean_target = _channel_summary(clean_target_raw)
        clean_other = _channel_summary(clean_other_raw)
        clean_all = _channel_summary(clean_all_raw)
        if unit_channel_idx < 0 or unit_channel_idx >= clean_target.shape[1]:
            raise IndexError(
                f"Resolved channel index {unit_channel_idx} out of bounds for measure layer '{unit_measure_layer}' "
                f"with summary shape {clean_target.shape}"
            )
        baseline_target_act = clean_target[:, unit_channel_idx]
        baseline_other_act = clean_other[:, unit_channel_idx]
        baseline_all_act = clean_all[:, unit_channel_idx]
    else:
        raise ValueError(f"Unknown measure_mode: {measure_mode}")

    baseline_mw = _normalized_mw_effect(baseline_target_act, baseline_other_act)

    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(12345)

    for frac in fractions:
        for perm in range(int(n_permutations)):
            model_dmg = copy.deepcopy(model).to(device)
            damage_single_unit_incoming_weights(model_dmg, unit_weight_layer, unit_channel_idx, frac, rng)
            dmg_target_raw = np.asarray(_run_hooked_activations(model_dmg, unit_measure_layer, target_batch))
            dmg_other_raw = np.asarray(_run_hooked_activations(model_dmg, unit_measure_layer, other_batch))
            dmg_all_raw = np.asarray(_run_hooked_activations(model_dmg, unit_measure_layer, all_batch))

            if measure_mode == "flat":
                dmg_target = _flatten_except_batch(dmg_target_raw)
                dmg_other = _flatten_except_batch(dmg_other_raw)
                dmg_all = _flatten_except_batch(dmg_all_raw)
                damaged_target_act = dmg_target[:, measured_flat_idx]
                damaged_other_act = dmg_other[:, measured_flat_idx]
                damaged_all_act = dmg_all[:, measured_flat_idx]
            else:
                dmg_target = _channel_summary(dmg_target_raw)
                dmg_other = _channel_summary(dmg_other_raw)
                dmg_all = _channel_summary(dmg_all_raw)
                damaged_target_act = dmg_target[:, unit_channel_idx]
                damaged_other_act = dmg_other[:, unit_channel_idx]
                damaged_all_act = dmg_all[:, unit_channel_idx]

            frac_all, frac_nz = _fraction_affected(baseline_all_act, damaged_all_act)
            rows.append({
                "model": model_name,
                "unit_measure_layer": unit_measure_layer,
                "unit_weight_layer": unit_weight_layer,
                "friendly_unit_measure_layer": _friendly_layer_name(model_spec, unit_measure_layer),
                "friendly_unit_weight_layer": _friendly_layer_name(model_spec, unit_weight_layer),
                "target_category": target_cat,
                "unit_measure_mode": measure_mode,
                "unit_channel_index": unit_channel_idx,
                "measured_flat_index": measured_flat_idx,
                "spatial_coords": None if address.get("spatial_coords") is None else str(tuple(address.get("spatial_coords"))),
                "stored_selectivity_score": top_info["stored_selectivity_score"],
                "baseline_mw_effect": baseline_mw,
                "damage_fraction": float(frac),
                "perm_idx": perm,
                "baseline_mean_target_activation": float(np.mean(baseline_target_act)),
                "damaged_mean_target_activation": float(np.mean(damaged_target_act)),
                "activation_ratio_vs_baseline": float(np.mean(damaged_target_act) / (np.mean(baseline_target_act) + 1e-12)),
                "activation_abs_change": float(np.mean(np.abs(damaged_target_act - baseline_target_act))),
                "activation_signed_change": float(np.mean(damaged_target_act - baseline_target_act)),
                "mw_effect_after_damage": _normalized_mw_effect(damaged_target_act, damaged_other_act),
                "unit_frac_affected": frac_all,
                "unit_frac_affected_nonzero": frac_nz,
            })
    df = pd.DataFrame(rows)
    df.to_csv(Path(output_dir) / f"{model_name}__top_unit_robustness.csv", index=False)
    _plot_top_unit_robustness(df, output_dir, model_name)
    return df



def _plot_top_unit_robustness(df: pd.DataFrame, output_dir: str | Path, model_name: str):
    out = _mkdir(Path(output_dir) / "plots_top_unit")
    agg = (
        df.groupby("damage_fraction", as_index=False)
          .agg(
              mean_activation_ratio=("activation_ratio_vs_baseline", "mean"),
              sd_activation_ratio=("activation_ratio_vs_baseline", "std"),
              mean_abs_change=("activation_abs_change", "mean"),
              sd_abs_change=("activation_abs_change", "std"),
              mean_mw=("mw_effect_after_damage", "mean"),
              sd_mw=("mw_effect_after_damage", "std"),
          )
    )

    def _save(fig: plt.Figure, name: str, expl: str):
        fig.text(0.5, 0.01, expl, ha="center", va="bottom", wrap=True, fontsize=9)
        fig.tight_layout(rect=[0, 0.08, 1, 1])
        fig.savefig(out / name, dpi=220)
        plt.close(fig)

    fig = plt.figure(figsize=(8.2, 4.6))
    ax = plt.gca()
    ax.errorbar(agg["damage_fraction"], agg["mean_activation_ratio"], yerr=agg["sd_activation_ratio"].fillna(0.0), marker="o", capsize=4)
    ax.set_xlabel("Fraction of incoming weights deleted for the top selective unit")
    ax.set_ylabel("Mean target-driven activation / baseline")
    ax.set_title(f"{model_name}: robustness of top selective unit activation")
    _save(fig, f"{model_name}__top_unit_activation_ratio.png",
          "Each point shows how much the top category-selective unit's mean response to its preferred-category images remains after deleting a fraction of its incoming weights. Error bars are SD across random deletion permutations.")

    fig = plt.figure(figsize=(8.2, 4.6))
    ax = plt.gca()
    ax.errorbar(agg["damage_fraction"], agg["mean_mw"], yerr=agg["sd_mw"].fillna(0.0), marker="o", capsize=4)
    ax.set_xlabel("Fraction of incoming weights deleted for the top selective unit")
    ax.set_ylabel("Recomputed MW effect score")
    ax.set_title(f"{model_name}: top unit category selectivity after damage")
    _save(fig, f"{model_name}__top_unit_mw_effect.png",
          "This plot recomputes a normalized Mann–Whitney effect score for the same unit after damage, comparing responses to the preferred category versus all other categories. Higher values indicate stronger category separation.")


# -----------------------------------------------------------------------------
# Analysis 2: distribution of category-selective units across layers
# -----------------------------------------------------------------------------

def analyze_selective_unit_locations(
    model_name: str,
    model_spec: Dict[str, Any],
    selectivity_root: str | Path,
    target_cat: str,
    top_global_frac: float,
    output_dir: str | Path,
) -> pd.DataFrame:
    sel_df = _load_selectivity_df(selectivity_root, model_tag=model_name)
    layer_col = _pick_layer_column(sel_df)
    score_col = _pick_selectivity_column(sel_df, target_cat)

    work = sel_df.copy()
    work["_layer_norm"] = work[layer_col].astype(str).map(lambda x: _canonicalize_layer_for_model(x, model_name=model_name))
    work = work[np.isfinite(pd.to_numeric(work[score_col], errors="coerce"))].copy()
    work["_score"] = pd.to_numeric(work[score_col], errors="coerce")
    work = work.sort_values("_score", ascending=False).reset_index(drop=True)
    work["global_rank"] = np.arange(1, len(work) + 1)

    top_n = max(1, int(round(len(work) * float(top_global_frac))))
    top = work.iloc[:top_n].copy()
    top["rank_weight"] = 1.0 / top["global_rank"].astype(float)

    total_by_layer = work.groupby("_layer_norm", as_index=False).size().rename(columns={"size": "n_units_total"})
    top_by_layer = top.groupby("_layer_norm", as_index=False).agg(
        n_top_units=("_score", "size"),
        mean_selectivity=("_score", "mean"),
        weighted_rank_mass=("rank_weight", "sum"),
    )
    merged = total_by_layer.merge(top_by_layer, on="_layer_norm", how="left").fillna(0)
    merged["prop_top_units"] = merged["n_top_units"] / merged["n_units_total"].replace(0, np.nan)
    merged["weighted_rank_mass_per_unit"] = merged["weighted_rank_mass"] / merged["n_units_total"].replace(0, np.nan)
    merged["model"] = model_name
    merged["target_category"] = target_cat

    # order layers using actual conv-layer order when possible
    try:
        model, _ = _load_model_with_fallbacks(model_spec["model_info"], hook_paths=[])
        named_mods = [(_canonicalize_layer_for_model(n, model_name=model_name), i) for i, (n, _m) in enumerate(model.named_modules()) if n]
        order_map = {}
        for nm, i in named_mods:
            order_map.setdefault(nm, i)
        merged["layer_order"] = merged["_layer_norm"].map(lambda x: order_map.get(x, 10**6))
    except Exception:
        merged["layer_order"] = np.arange(len(merged))
    merged = merged.sort_values(["layer_order", "_layer_norm"]).reset_index(drop=True)
    merged["friendly_layer"] = merged["_layer_norm"].map(lambda x: _friendly_layer_name(model_spec, x))

    merged.to_csv(Path(output_dir) / f"{model_name}__selective_unit_locations.csv", index=False)
    return merged



def plot_selective_unit_locations(all_location_df: pd.DataFrame, output_dir: str | Path, target_cat: str):
    out = _mkdir(Path(output_dir) / "plots_locations")
    if all_location_df.empty:
        return

    for metric, ylabel, fname, expl in [
        (
            "prop_top_units",
            "Proportion of globally top-selective units in layer",
            "selective_unit_location__proportion.png",
            "For each model and layer, this shows what fraction of that layer's units fall into the globally most category-selective set. Larger values indicate that the model concentrates top category-selective units in that part of the network.",
        ),
        (
            "weighted_rank_mass_per_unit",
            "Weighted top-rank mass per unit",
            "selective_unit_location__weighted_rank_mass.png",
            "This score gives more weight to higher-ranked units and then normalizes by the number of units in the layer. It highlights whether the very strongest category-selective units are concentrated earlier or later in the model.",
        ),
    ]:
        fig = plt.figure(figsize=(11.0, 5.2))
        ax = plt.gca()
        for model_name in sorted(all_location_df["model"].unique().tolist()):
            sub = all_location_df[all_location_df["model"] == model_name].sort_values("layer_order")
            xs = np.arange(len(sub))
            ax.plot(xs, sub[metric], marker="o", label=model_name)
        # use the longest x tick sequence from the last model plotted purely for labels
        max_rows = all_location_df.groupby("model").size().max()
        label_source = all_location_df.sort_values(["model", "layer_order"]).drop_duplicates(["model", "layer_order"])
        tick_labels = label_source[label_source["model"] == sorted(all_location_df["model"].unique().tolist())[0]]["friendly_layer"].tolist()
        if tick_labels:
            ax.set_xticks(np.arange(len(tick_labels)))
            ax.set_xticklabels(tick_labels, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Layer order")
        ax.set_title(f"Where are the strongest {target_cat}-selective units across model depth?")
        ax.legend(title="Model")
        fig.text(0.5, 0.01, expl, ha="center", va="bottom", wrap=True, fontsize=9)
        fig.tight_layout(rect=[0, 0.10, 1, 1])
        fig.savefig(out / fname, dpi=220)
        plt.close(fig)


# -----------------------------------------------------------------------------
# Analysis 3: sever one connection and measure downstream affected units
# -----------------------------------------------------------------------------

def sever_single_random_connection(model: nn.Module, layer_path: str, rng: np.random.Generator) -> Tuple[int, float]:
    layer = get_layer_from_path(model, layer_path)
    if not hasattr(layer, "weight") or layer.weight is None:
        raise ValueError(f"Layer '{layer_path}' has no weights.")
    with torch.no_grad():
        flat = layer.weight.data.view(-1)
        idx = int(rng.integers(0, flat.numel()))
        old = float(flat[idx].item())
        flat[idx] = 0.0
    return idx, old



def analyze_single_connection_propagation(
    model_name: str,
    model_spec: Dict[str, Any],
    image_dir: str | Path,
    target_cat: Optional[str],
    n_permutations: int,
    eps: float,
    output_dir: str | Path,
) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hook_path = _canonicalize_measure_layer_for_model(model_spec["propagation_hook"], model_name=model_name)
    model, _ = _load_model_with_fallbacks(model_spec["model_info"], hook_paths=[])
    model = model.to(device)

    batches, _, _ = get_stimuli_batches(image_dir, device)
    if target_cat is None or target_cat == "all":
        eval_batch = _collect_all_images_batch(batches)
        eval_label = "all_categories"
    else:
        if target_cat not in batches:
            raise ValueError(f"Category '{target_cat}' not found in image dir.")
        eval_batch = batches[target_cat]
        eval_label = target_cat

    baseline = _channel_summary(_run_hooked_activations(model, hook_path, eval_batch))
    conv_layers = [normalize_module_name(p) for p in get_all_conv_layers(model, "")]

    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(777)
    for layer_path in conv_layers:
        for perm in range(int(n_permutations)):
            model_dmg = copy.deepcopy(model).to(device)
            conn_idx, old_val = sever_single_random_connection(model_dmg, layer_path, rng)
            damaged = _channel_summary(_run_hooked_activations(model_dmg, hook_path, eval_batch))
            frac_all, frac_nz = _fraction_affected(baseline, damaged, eps=eps)
            rows.append({
                "model": model_name,
                "hook_layer": hook_path,
                "damage_layer": layer_path,
                "friendly_damage_layer": _friendly_layer_name(model_spec, layer_path),
                "eval_images": eval_label,
                "perm_idx": perm,
                "weight_flat_index": conn_idx,
                "old_weight_value": old_val,
                "frac_penultimate_units_affected": frac_all,
                "frac_penultimate_units_affected_nonzero": frac_nz,
                "mean_abs_shift": float(np.mean(np.abs(damaged - baseline))),
                "mean_signed_shift": float(np.mean(damaged - baseline)),
            })
    df = pd.DataFrame(rows)
    df.to_csv(Path(output_dir) / f"{model_name}__single_connection_propagation.csv", index=False)
    _plot_single_connection_propagation(df, output_dir, model_name, hook_path)
    return df



def _plot_single_connection_propagation(df: pd.DataFrame, output_dir: str | Path, model_name: str, hook_path: str):
    out = _mkdir(Path(output_dir) / "plots_single_connection")
    agg = (
        df.groupby(["friendly_damage_layer", "damage_layer"], as_index=False)
          .agg(
              mean_frac=("frac_penultimate_units_affected", "mean"),
              sd_frac=("frac_penultimate_units_affected", "std"),
              mean_shift=("mean_abs_shift", "mean"),
              sd_shift=("mean_abs_shift", "std"),
          )
    )
    agg = agg.reset_index(drop=True)

    for ycol, yerr, ylabel, fname, expl in [
        (
            "mean_frac",
            "sd_frac",
            "Fraction of final-readout units affected",
            f"{model_name}__single_connection_frac_affected.png",
            "Each bar corresponds to severing one random weight in a given conv layer, repeating this many times. Higher values mean a single connection in that layer perturbs a larger fraction of units in the final readout layer.",
        ),
        (
            "mean_shift",
            "sd_shift",
            "Mean absolute shift in final-readout activations",
            f"{model_name}__single_connection_mean_abs_shift.png",
            "This shows the average absolute change induced in the final readout layer after severing one random connection in each earlier conv layer. Larger shifts imply broader downstream sensitivity to that layer's individual connections.",
        ),
    ]:
        fig = plt.figure(figsize=(10.0, 4.8))
        ax = plt.gca()
        x = np.arange(len(agg))
        ax.bar(x, agg[ycol], yerr=agg[yerr].fillna(0.0), capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(agg["friendly_damage_layer"], rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Layer containing the severed connection")
        ax.set_title(f"{model_name}: impact of severing one random connection on {hook_path}")
        fig.text(0.5, 0.01, expl, ha="center", va="bottom", wrap=True, fontsize=9)
        fig.tight_layout(rect=[0, 0.10, 1, 1])
        fig.savefig(out / fname, dpi=220)
        plt.close(fig)


def _resolve_activation_plot_layers(model_name: str, model_spec: Dict[str, Any], model: nn.Module) -> List[Tuple[str, str]]:
    """Return (source_weight_layer, readout_layer_for_plotting) pairs.

    The readout layer is always post-nonlinearity when possible. For CORnet-RT this means
    using .nonlin_input and .nonlin1 instead of .conv_input and .conv1, while optionally
    also retaining the block .output readout.
    """
    if model_name == "vgg16":
        candidates = [
            ("features.0", "features.1"),
            ("features.2", "features.3"),
            ("features.5", "features.6"),
            ("features.7", "features.8"),
            ("features.10", "features.11"),
            ("features.12", "features.13"),
            ("features.14", "features.15"),
            ("features.17", "features.18"),
            ("features.19", "features.20"),
            ("features.21", "features.22"),
            ("features.24", "features.25"),
            ("features.26", "features.27"),
            ("features.28", "features.30"),
        ]
    elif model_name == "alexnet":
        candidates = [
            ("features.0", "features.1"),
            ("features.3", "features.4"),
            ("features.6", "features.7"),
            ("features.8", "features.9"),
            ("features.10", "features.11"),
        ]
    elif model_name == "cornet_rt":
        candidates = [
            ("V1.conv_input", "V1.nonlin_input"),
            ("V1.conv1", "V1.nonlin1"),
            ("V1.conv1", "V1.output"),
            ("V2.conv_input", "V2.nonlin_input"),
            ("V2.conv1", "V2.nonlin1"),
            ("V2.conv1", "V2.output"),
            ("V4.conv_input", "V4.nonlin_input"),
            ("V4.conv1", "V4.nonlin1"),
            ("V4.conv1", "V4.output"),
            ("IT.conv_input", "IT.nonlin_input"),
            ("IT.conv1", "IT.nonlin1"),
            ("IT.conv1", "IT.output"),
        ]
    else:
        conv_layers = [normalize_module_name(p) for p in get_all_conv_layers(model, "")]
        candidates = [(c, c) for c in conv_layers]

    resolved: List[Tuple[str, str]] = []
    seen = set()
    for source_layer, plot_layer in candidates:
        source_norm = _canonicalize_weight_layer_for_model(source_layer, model_name=model_name)
        plot_norm = _canonicalize_measure_layer_for_model(plot_layer, model_name=model_name)
        key = (source_norm, plot_norm)
        if key in seen:
            continue
        if _layer_exists(model, source_norm) and _layer_exists(model, plot_norm):
            seen.add(key)
            resolved.append(key)
    return resolved



def _resolve_random_unit_sensitivity_pairs(model_name: str, model_spec: Dict[str, Any], model: nn.Module) -> List[Tuple[str, str]]:
    """Return (measure_layer, weight_layer) pairs for unit-local single-connection damage.

    The measure layer is the post-nonlinearity readout whose unit activation will be tracked.
    The weight layer is the immediately preceding weighted layer supplying that unit.
    """
    if model_name == "vgg16":
        candidates = [
            ("features.1", "features.0"),
            ("features.3", "features.2"),
            ("features.6", "features.5"),
            ("features.8", "features.7"),
            ("features.11", "features.10"),
            ("features.13", "features.12"),
            ("features.15", "features.14"),
            ("features.18", "features.17"),
            ("features.20", "features.19"),
            ("features.22", "features.21"),
            ("features.25", "features.24"),
            ("features.27", "features.26"),
            ("features.30", "features.28"),
        ]
    elif model_name == "alexnet":
        candidates = [
            ("features.1", "features.0"),
            ("features.4", "features.3"),
            ("features.7", "features.6"),
            ("features.9", "features.8"),
            ("features.11", "features.10"),
        ]
    elif model_name == "cornet_rt":
        candidates = [
            ("V1.nonlin_input", "V1.conv_input"),
            ("V1.nonlin1", "V1.conv1"),
            ("V2.nonlin_input", "V2.conv_input"),
            ("V2.nonlin1", "V2.conv1"),
            ("V4.nonlin_input", "V4.conv_input"),
            ("V4.nonlin1", "V4.conv1"),
            ("IT.nonlin_input", "IT.conv_input"),
            ("IT.nonlin1", "IT.conv1"),
        ]
    else:
        candidates = []

    resolved: List[Tuple[str, str]] = []
    seen = set()
    for measure_layer, weight_layer in candidates:
        measure_norm = _canonicalize_measure_layer_for_model(measure_layer, model_name=model_name)
        weight_norm = _canonicalize_weight_layer_for_model(weight_layer, model_name=model_name)
        key = (measure_norm, weight_norm)
        if key in seen:
            continue
        if _layer_exists(model, measure_norm) and _layer_exists(model, weight_norm):
            seen.add(key)
            resolved.append(key)
    return resolved


# -----------------------------------------------------------------------------
# Analysis 4: activation histograms / sparsity across conv layers
# -----------------------------------------------------------------------------

def analyze_activation_histograms(
    model_name: str,
    model_spec: Dict[str, Any],
    image_dir: str | Path,
    target_cat: str,
    max_layers: Optional[int],
    output_dir: str | Path,
) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = _load_model_with_fallbacks(model_spec["model_info"], hook_paths=[])
    model = model.to(device)
    batches, _, _ = get_stimuli_batches(image_dir, device)
    if target_cat not in batches:
        raise ValueError(f"Category '{target_cat}' not found in image dir.")
    x = batches[target_cat]

    plot_layers = _resolve_activation_plot_layers(model_name, model_spec, model)
    if max_layers is not None:
        plot_layers = plot_layers[: int(max_layers)]

    rows: List[Dict[str, Any]] = []
    plot_dir = _mkdir(Path(output_dir) / "plots_activation_histograms")

    for source_layer, plot_layer in plot_layers:
        acts = np.asarray(_run_hooked_activations(model, plot_layer, x))
        flat = acts.reshape(-1)
        rows.append({
            "model": model_name,
            "layer": plot_layer,
            "source_conv_layer": source_layer,
            "friendly_layer": _friendly_layer_name(model_spec, plot_layer),
            "category": target_cat,
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "median": float(np.median(flat)),
            "p95": float(np.percentile(flat, 95)),
            "p99": float(np.percentile(flat, 99)),
            "frac_exact_zero": float(np.mean(flat == 0)),
            "frac_near_zero_1e_6": float(np.mean(np.abs(flat) < 1e-6)),
            "n_values": int(flat.size),
        })

        fig = plt.figure(figsize=(8.0, 4.6))
        ax = plt.gca()
        if HAS_SNS:
            sns.histplot(flat, bins=120, stat="density", ax=ax)
        else:
            ax.hist(flat, bins=120, density=True)
        ax.set_xlabel("Activation value")
        ax.set_ylabel("Density")
        ax.set_title(f"{model_name}: activation distribution for {_friendly_layer_name(model_spec, plot_layer)}")
        expl = (
            f"Histogram of all activation values in {plot_layer} when the model sees {target_cat} images. "
            f"For VGG16/AlexNet these are post-ReLU readouts (and VGG block5 uses features.30). For CORnet-RT, the plots use block nonlinearity readouts such as .nonlin_input and .nonlin1 instead of raw conv_input / conv1 outputs, so the histogram reflects post-nonlinearity sparsity rather than pre-ReLU conv outputs."
        )
        fig.text(0.5, 0.01, expl, ha="center", va="bottom", wrap=True, fontsize=9)
        fig.tight_layout(rect=[0, 0.10, 1, 1])
        fig.savefig(plot_dir / f"{model_name}__{_sanitize(plot_layer)}__activation_hist.png", dpi=220)
        plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(Path(output_dir) / f"{model_name}__activation_histogram_summary.csv", index=False)
    _plot_activation_sparsity_summary(df, output_dir, model_name)
    return df



def _plot_activation_sparsity_summary(df: pd.DataFrame, output_dir: str | Path, model_name: str):
    out = _mkdir(Path(output_dir) / "plots_activation_histograms")
    fig = plt.figure(figsize=(10.0, 4.8))
    ax = plt.gca()
    x = np.arange(len(df))
    ax.plot(x, df["frac_near_zero_1e_6"], marker="o")
    ax.set_xticks(x)
    ax.set_xticklabels(df["friendly_layer"], rotation=35, ha="right")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fraction of activations near zero")
    ax.set_title(f"{model_name}: sparsity profile across conv layers")
    expl = "Higher values indicate that a larger share of the layer's activations stay at or extremely close to zero for the chosen category images, consistent with sparser activity."
    fig.text(0.5, 0.01, expl, ha="center", va="bottom", wrap=True, fontsize=9)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(out / f"{model_name}__activation_sparsity_profile.png", dpi=220)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Analysis 5: single-connection sensitivity of the directly damaged unit
# -----------------------------------------------------------------------------

def damage_specific_unit_incoming_connections(
    model: nn.Module,
    layer_path: str,
    unit_idx: int,
    rng: np.random.Generator,
    n_connections: int = 1,
    frac: Optional[float] = None,
) -> Dict[str, Any]:
    layer = get_layer_from_path(model, layer_path)
    if not hasattr(layer, "weight") or layer.weight is None:
        raise ValueError(f"Layer '{layer_path}' has no weight tensor.")
    with torch.no_grad():
        w = layer.weight.data
        if unit_idx < 0 or unit_idx >= w.shape[0]:
            raise IndexError(
                f"unit_idx {unit_idx} out of bounds for layer '{layer_path}' with shape {tuple(w.shape)}"
            )
        flat = w[unit_idx].reshape(-1)
        n_total = int(flat.numel())
        if frac is not None and frac > 0:
            k = max(1, int(round(float(frac) * n_total)))
        else:
            k = max(1, int(n_connections))
        k = min(k, n_total)
        idx = torch.as_tensor(rng.choice(n_total, size=k, replace=False), device=flat.device, dtype=torch.long)
        old_vals = flat[idx].detach().cpu().numpy().astype(float)
        flat[idx] = 0.0
    return {
        "weight_indices": idx.detach().cpu().numpy().astype(int).tolist(),
        "old_weight_values": old_vals.tolist(),
        "n_total_weights_for_unit": n_total,
        "n_deleted_weights": int(k),
        "deleted_weight_fraction": float(k / max(1, n_total)),
    }



def analyze_random_unit_input_sensitivity(
    model_name: str,
    model_spec: Dict[str, Any],
    image_dir: str | Path,
    n_units_per_layer: int,
    output_dir: str | Path,
    damage_fraction: float = 0.0,
    baseline_eps: float = 1e-8,
) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = _load_model_with_fallbacks(model_spec["model_info"], hook_paths=[])
    model = model.to(device)

    batches, _, _ = get_stimuli_batches(image_dir, device)
    eval_batch = _collect_all_images_batch(batches)

    layer_pairs = _resolve_random_unit_sensitivity_pairs(model_name, model_spec, model)
    if not layer_pairs:
        raise ValueError(f"No unit-sensitivity layer pairs were resolved for model '{model_name}'.")

    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(20260313)

    for measure_layer, weight_layer in layer_pairs:
        baseline_raw = np.asarray(_run_hooked_activations(model, measure_layer, eval_batch))
        baseline_summary = _channel_summary(baseline_raw)
        layer_sparsity = float(np.mean(np.abs(baseline_raw.reshape(-1)) < 1e-6))
        layer_exact_zero = float(np.mean(baseline_raw.reshape(-1) == 0))

        weight_obj = get_layer_from_path(model, weight_layer)
        if not hasattr(weight_obj, "weight") or weight_obj.weight is None:
            continue
        out_channels = int(weight_obj.weight.shape[0])
        max_units = min(int(baseline_summary.shape[1]), out_channels)
        if max_units <= 0:
            continue

        n_pick = min(int(n_units_per_layer), max_units)
        chosen_units = rng.choice(max_units, size=n_pick, replace=False)

        for unit_idx in chosen_units:
            baseline_act = baseline_summary[:, int(unit_idx)]
            model_dmg = copy.deepcopy(model).to(device)
            dmg_info = damage_specific_unit_incoming_connections(
                model_dmg,
                weight_layer,
                int(unit_idx),
                rng,
                n_connections=1,
                frac=(float(damage_fraction) if float(damage_fraction) > 0 else None),
            )
            damaged_raw = np.asarray(_run_hooked_activations(model_dmg, measure_layer, eval_batch))
            damaged_summary = _channel_summary(damaged_raw)
            damaged_act = damaged_summary[:, int(unit_idx)]

            mean_abs_baseline = float(np.mean(np.abs(baseline_act)))
            mean_abs_change = float(np.mean(np.abs(damaged_act - baseline_act)))
            mean_signed_change = float(np.mean(damaged_act - baseline_act))
            fractional_abs_change = float(mean_abs_change / (mean_abs_baseline + baseline_eps))
            per_image_fractional_abs_change = float(np.mean(np.abs(damaged_act - baseline_act) / (np.abs(baseline_act) + baseline_eps)))
            activation_ratio = float(np.mean(np.abs(damaged_act)) / (mean_abs_baseline + baseline_eps))

            rows.append({
                "model": model_name,
                "measure_layer": measure_layer,
                "weight_layer": weight_layer,
                "friendly_measure_layer": _friendly_layer_name(model_spec, measure_layer),
                "friendly_weight_layer": _friendly_layer_name(model_spec, weight_layer),
                "unit_index": int(unit_idx),
                "n_total_weights_for_unit": dmg_info["n_total_weights_for_unit"],
                "n_deleted_weights": dmg_info["n_deleted_weights"],
                "deleted_weight_fraction": dmg_info["deleted_weight_fraction"],
                "deleted_weight_indices": str(dmg_info["weight_indices"]),
                "baseline_mean_activation": float(np.mean(baseline_act)),
                "baseline_mean_abs_activation": mean_abs_baseline,
                "damaged_mean_activation": float(np.mean(damaged_act)),
                "damaged_mean_abs_activation": float(np.mean(np.abs(damaged_act))),
                "mean_abs_activation_change": mean_abs_change,
                "mean_signed_activation_change": mean_signed_change,
                "fractional_abs_change": fractional_abs_change,
                "per_image_fractional_abs_change": per_image_fractional_abs_change,
                "abs_activation_ratio_vs_baseline": activation_ratio,
                "layer_frac_near_zero_eval": layer_sparsity,
                "layer_frac_exact_zero_eval": layer_exact_zero,
            })

    df = pd.DataFrame(rows)
    df.to_csv(Path(output_dir) / f"{model_name}__random_unit_input_sensitivity.csv", index=False)
    _plot_random_unit_input_sensitivity(df, output_dir, model_name, damage_fraction=damage_fraction)
    return df



def _plot_random_unit_input_sensitivity(df: pd.DataFrame, output_dir: str | Path, model_name: str, damage_fraction: float = 0.0):
    out = _mkdir(Path(output_dir) / "plots_random_unit_sensitivity")
    if df.empty:
        return

    summary = (
        df.groupby(["friendly_measure_layer", "measure_layer"], as_index=False)
          .agg(
              mean_fractional_abs_change=("fractional_abs_change", "mean"),
              sd_fractional_abs_change=("fractional_abs_change", "std"),
              mean_abs_change=("mean_abs_activation_change", "mean"),
              sd_abs_change=("mean_abs_activation_change", "std"),
              layer_frac_near_zero_eval=("layer_frac_near_zero_eval", "first"),
          )
    )

    damage_desc = (
        "after deleting one randomly chosen incoming connection to that unit"
        if float(damage_fraction) <= 0
        else f"after deleting {float(damage_fraction):.4f} of that unit's incoming weights"
    )

    fig = plt.figure(figsize=(10.5, 4.8))
    ax = plt.gca()
    x = np.arange(len(summary))
    ax.bar(x, summary["mean_fractional_abs_change"], yerr=summary["sd_fractional_abs_change"].fillna(0.0), capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["friendly_measure_layer"], rotation=35, ha="right")
    ax.set_xlabel("Measured post-nonlinearity layer")
    ax.set_ylabel("Normalized activation change of the directly damaged unit")
    ax.set_title(f"{model_name}: sensitivity of a unit to damage within its own incoming weights")
    expl = (
        f"Each bar summarizes random units sampled from a layer. For each sampled unit, the script zeros a connection feeding that unit and measures how much that same unit's post-nonlinearity activation changes across all images, normalized by the unit's baseline mean absolute activation. {damage_desc}."
    )
    fig.text(0.5, 0.01, expl, ha="center", va="bottom", wrap=True, fontsize=9)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(out / f"{model_name}__random_unit_fractional_change_by_layer.png", dpi=220)
    plt.close(fig)

    fig = plt.figure(figsize=(8.8, 5.0))
    ax = plt.gca()
    ax.scatter(df["layer_frac_near_zero_eval"], df["fractional_abs_change"], alpha=0.75)
    ax.set_xlabel("Layer sparsity on all images (fraction of activations near zero)")
    ax.set_ylabel("Normalized activation change of damaged unit")
    ax.set_title(f"{model_name}: does unit sensitivity track layer sparsity?")
    xvals = df["layer_frac_near_zero_eval"].to_numpy(dtype=float)
    yvals = df["fractional_abs_change"].to_numpy(dtype=float)
    valid = np.isfinite(xvals) & np.isfinite(yvals)
    if np.sum(valid) >= 2 and np.unique(xvals[valid]).size >= 2:
        coeff = np.polyfit(xvals[valid], yvals[valid], deg=1)
        xs = np.linspace(np.min(xvals[valid]), np.max(xvals[valid]), 200)
        ys = coeff[0] * xs + coeff[1]
        ax.plot(xs, ys, linestyle="--")
        corr = float(np.corrcoef(xvals[valid], yvals[valid])[0, 1]) if np.sum(valid) >= 2 else float("nan")
        ax.text(0.03, 0.97, f"r = {corr:.3f}", transform=ax.transAxes, ha="left", va="top")
    expl = (
        "Each point is one randomly sampled unit. The x-axis is the overall sparsity of its layer across all images, and the y-axis is how much that unit's own post-nonlinearity activation changed after damaging one of its incoming connections. The dashed line is a simple least-squares fit over the raw unit-level points."
    )
    fig.text(0.5, 0.01, expl, ha="center", va="bottom", wrap=True, fontsize=9)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(out / f"{model_name}__random_unit_change_vs_sparsity.png", dpi=220)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Combined runner
# -----------------------------------------------------------------------------

def run_all(args: argparse.Namespace):
    root_out = _mkdir(args.output_dir)
    all_location_tables: List[pd.DataFrame] = []

    fractions = [float(x) for x in args.damage_fractions]

    for model_name in args.models:
        model_spec = _resolve_model_spec(model_name, cornet_time_steps=args.cornet_time_steps)
        model_out = _mkdir(root_out / model_name)
        print(f"\n=== Running diagnostics for {model_name} ===")
        print(f"  unit measure layer: {model_spec.get('unit_measure_layer', model_spec.get('unit_layer'))}")
        print(f"  unit weight layer:  {model_spec.get('unit_weight_layer', model_spec.get('unit_layer'))}")
        print(f"  propagation hook:   {model_spec['propagation_hook']}")

        top_df = analyze_top_unit_robustness(
            model_name=model_name,
            model_spec=model_spec,
            image_dir=args.image_dir,
            selectivity_root=args.selectivity_root,
            target_cat=args.target_cat,
            fractions=fractions,
            n_permutations=args.unit_permutations,
            output_dir=model_out,
        )
        print(f"  saved top-unit robustness: {len(top_df)} rows")

        loc_df = analyze_selective_unit_locations(
            model_name=model_name,
            model_spec=model_spec,
            selectivity_root=args.selectivity_root,
            target_cat=args.target_cat,
            top_global_frac=args.top_global_frac,
            output_dir=model_out,
        )
        all_location_tables.append(loc_df)
        print(f"  saved layerwise location summary: {len(loc_df)} rows")

        conn_df = analyze_single_connection_propagation(
            model_name=model_name,
            model_spec=model_spec,
            image_dir=args.image_dir,
            target_cat=args.connection_eval_category,
            n_permutations=args.connection_permutations,
            eps=args.affected_eps,
            output_dir=model_out,
        )
        print(f"  saved single-connection propagation: {len(conn_df)} rows")

        hist_df = analyze_activation_histograms(
            model_name=model_name,
            model_spec=model_spec,
            image_dir=args.image_dir,
            target_cat=args.target_cat,
            max_layers=args.max_hist_layers,
            output_dir=model_out,
        )
        print(f"  saved activation histogram summary: {len(hist_df)} rows")

        rand_unit_df = analyze_random_unit_input_sensitivity(
            model_name=model_name,
            model_spec=model_spec,
            image_dir=args.image_dir,
            n_units_per_layer=args.random_unit_samples_per_layer,
            output_dir=model_out,
            damage_fraction=args.random_unit_damage_fraction,
            baseline_eps=args.random_unit_baseline_eps,
        )
        print(f"  saved random-unit input sensitivity: {len(rand_unit_df)} rows")

    if all_location_tables:
        big = pd.concat(all_location_tables, ignore_index=True)
        big.to_csv(root_out / "all_models__selective_unit_locations.csv", index=False)
        plot_selective_unit_locations(big, root_out, args.target_cat)
        print(f"\nSaved cross-model selective-unit location comparison to: {root_out}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Selective unit / connection damage diagnostics")
    p.add_argument("--image_dir", type=str, default="stimuli", help="Directory containing category images")
    p.add_argument("--selectivity_root", type=str, default="unit_selectivity", help="Selectivity file or directory")
    p.add_argument("--output_dir", type=str, default="selective_damage_outputs", help="Output directory")
    p.add_argument("--models", nargs="+", default=["vgg16", "alexnet", "cornet_rt"], choices=sorted(MODEL_SPECS.keys()))
    p.add_argument("--target_cat", type=str, default="face", help="Target category for selectivity analyses")
    p.add_argument("--connection_eval_category", type=str, default="all", help="Category for single-connection propagation (or 'all')")
    p.add_argument("--damage_fractions", nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8], help="Fractions of incoming weights to delete for the top unit")
    p.add_argument("--unit_permutations", type=int, default=10, help="Permutations per damage fraction for top-unit analysis")
    p.add_argument("--connection_permutations", type=int, default=25, help="Random single-connection severings per conv layer")
    p.add_argument("--top_global_frac", type=float, default=0.05, help="Fraction of all rows in selectivity table treated as globally top-selective")
    p.add_argument("--affected_eps", type=float, default=1e-4, help="Relative threshold for calling a downstream unit 'affected'")
    p.add_argument("--cornet_time_steps", type=int, default=5, help="Time steps for CORnet-RT")
    p.add_argument("--max_hist_layers", type=int, default=None, help="Optional cap on number of activation-histogram layers to plot")
    p.add_argument("--random_unit_samples_per_layer", type=int, default=24, help="How many random units to sample per layer for direct unit-sensitivity analysis")
    p.add_argument("--random_unit_damage_fraction", type=float, default=0.0, help="If > 0, delete this fraction of a sampled unit's incoming weights instead of a single connection")
    p.add_argument("--random_unit_baseline_eps", type=float, default=1e-8, help="Stability constant when normalizing direct unit activation changes")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_all(args)
