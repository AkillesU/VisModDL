#!/usr/bin/env python3
"""
Streamlit app for interactive activation path tracing and damage analysis.

This is a practical single-file prototype aimed at the workflow described in the
user specification:
- models: VGG16, AlexNet, optional CORnet-RT if `cornet` is installed
- image directory selection with optional Tk folder picker
- selectivity file loading from CSV / pickle
- target-unit selection from top/bottom/random ranked units
- backward path tracing with greedy or beam search
- damage of one selected weighted edge and baseline vs damaged comparison
- optional CORnet-RT time selection by module occurrence

Run:
    streamlit run activation_path_streamlit_app.py

Recommended installs:
    pip install streamlit plotly pandas pillow
    pip install torch torchvision
    pip install git+https://github.com/dicarlolab/CORnet  # optional for CORnet-RT
"""

from __future__ import annotations

import copy
import io
import json
import math
import os
import pickle
import random
import subprocess
import sys
import traceback
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from plotly.subplots import make_subplots
import plotly.graph_objects as go


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
TRACEABLE_TYPES = (
    nn.Conv2d,
    nn.ReLU,
    nn.MaxPool2d,
    nn.AvgPool2d,
    nn.AdaptiveAvgPool2d,
    nn.Linear,
    nn.Dropout,
    nn.BatchNorm2d,
    nn.GroupNorm,
)


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass
class NodeRef:
    layer: str
    time: Optional[int]
    channel: int
    y: Optional[int]
    x: Optional[int]


@dataclass
class EdgeRef:
    layer: str
    time: Optional[int]
    out_channel: int
    out_y: Optional[int]
    out_x: Optional[int]
    in_channel: int
    in_y: Optional[int]
    in_x: Optional[int]
    ky: Optional[int]
    kx: Optional[int]
    linear_in_index: Optional[int] = None


@dataclass
class CandidateTransition:
    parent_node: NodeRef
    edge_ref: Optional[EdgeRef]
    edge_weight: Optional[float]
    local_contribution: Optional[float]
    rank_score: float
    note: str = ""


@dataclass
class TraceEvent:
    index: int
    name: str
    occurrence: int
    module_type: str
    module_ref: Optional[nn.Module]
    input_tensor: Optional[torch.Tensor]
    output_tensor: torch.Tensor
    synthetic: bool = False

    @property
    def time(self) -> Optional[int]:
        return self.occurrence if self.occurrence > 0 else 0

    @property
    def label(self) -> str:
        return f"{self.name} [t={self.occurrence}]" if self.occurrence is not None else self.name


@dataclass
class PathStep:
    step_index: int
    layer: str
    time: Optional[int]
    node: NodeRef
    pre_value: Optional[float]
    post_value: Optional[float]
    edge_weight: Optional[float]
    local_contribution: Optional[float]
    delta_value: Optional[float]
    cumulative_score: float
    event_index: int
    module_type: str
    edge_ref: Optional[EdgeRef] = None
    note: str = ""


@dataclass
class PathTrace:
    steps: List[PathStep]
    path_score: float
    path_mode: str
    path_rank: int
    search_mode: str


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


class AppError(RuntimeError):
    pass


class IdentityModule(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def is_leaf_module(module: nn.Module) -> bool:
    return len(list(module.children())) == 0


def ensure_streamlit():
    try:
        import streamlit as st  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise AppError(
            "Streamlit is not installed in this environment. Install it with: pip install streamlit"
        ) from exc
    return st


def try_import_torchvision():
    try:
        import torchvision  # type: ignore
        from torchvision import models, transforms  # type: ignore
    except Exception as exc:
        raise AppError(
            "torchvision could not be imported. This usually means either it is not installed "
            "or your torch/torchvision builds are mismatched. Reinstall compatible versions of "
            "torch and torchvision."
        ) from exc
    return torchvision, models, transforms


def try_import_cornet():
    try:
        import cornet  # type: ignore
    except Exception as exc:
        raise AppError(
            "CORnet is not installed. Install it with: pip install git+https://github.com/dicarlolab/CORnet"
        ) from exc
    return cornet


def safe_to_cpu(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    return x.detach().cpu()


def unpack_first_tensor(value: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            if torch.is_tensor(item):
                return item
    if isinstance(value, dict):
        for item in value.values():
            if torch.is_tensor(item):
                return item
    return None


def list_images(folder: str) -> List[Path]:
    root = Path(folder)
    if not root.exists() or not root.is_dir():
        return []
    return sorted([p for p in root.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def maybe_browse_directory() -> Optional[str]:
    if sys.platform == "darwin":
        try:
            script = 'POSIX path of (choose folder with prompt "Select image directory")'
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                selected = result.stdout.strip()
                return selected or None
            return None
        except Exception:
            return None

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory()
        root.destroy()
        return selected or None
    except Exception:
        return None


def load_image_pil(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_numpy_scalar(x: Optional[float]) -> Optional[float]:
    return None if x is None else float(x)


def score_key(value: float, mode: str) -> float:
    if mode == "strongest_signed_contribution":
        return float(value)
    if mode == "strongest_positive_contribution":
        return float(value) if value > 0 else -1e30
    if mode == "random":
        return abs(float(value))
    return abs(float(value))


def tensor_output_shape_no_batch(tensor: torch.Tensor) -> Tuple[int, ...]:
    return tuple(int(v) for v in tensor.shape[1:])


def tensor_rank_without_batch(tensor: torch.Tensor) -> int:
    return max(0, tensor.ndim - 1)


def scalar_from_tensor_node(tensor: torch.Tensor, node: NodeRef) -> float:
    t = tensor[0]
    if t.ndim == 1:
        return float(t[node.channel].item())
    if t.ndim == 3:
        yy = 0 if node.y is None else node.y
        xx = 0 if node.x is None else node.x
        return float(t[node.channel, yy, xx].item())
    if t.ndim == 2:
        yy = 0 if node.y is None else node.y
        return float(t[node.channel, yy].item())
    raise ValueError(f"Unsupported tensor rank for node lookup: {t.ndim}")


def get_event_node_value(event: TraceEvent, node: NodeRef, which: str = "output") -> Optional[float]:
    tensor = event.output_tensor if which == "output" else event.input_tensor
    if tensor is None:
        return None
    try:
        return scalar_from_tensor_node(tensor, node)
    except Exception:
        return None


def flatten_index_to_node(flat_index: int, prev_event: TraceEvent, layer_name: str, time: Optional[int]) -> NodeRef:
    shape = tensor_output_shape_no_batch(prev_event.output_tensor)
    if len(shape) == 1:
        return NodeRef(layer=layer_name, time=time, channel=int(flat_index), y=None, x=None)
    if len(shape) == 3:
        c, y, x = np.unravel_index(int(flat_index), shape)
        return NodeRef(layer=layer_name, time=time, channel=int(c), y=int(y), x=int(x))
    raise ValueError(f"Cannot unflatten index for shape {shape}")


def node_to_flat_index(node: NodeRef, event: TraceEvent) -> int:
    shape = tensor_output_shape_no_batch(event.output_tensor)
    if len(shape) == 1:
        return int(node.channel)
    if len(shape) == 3:
        return int(np.ravel_multi_index((node.channel, node.y or 0, node.x or 0), shape))
    raise ValueError(f"Cannot flatten node for shape {shape}")


def spatial_argmax_for_channel(event: TraceEvent, channel: int) -> Tuple[Optional[int], Optional[int]]:
    out = event.output_tensor[0]
    if out.ndim != 3:
        return None, None
    channel_map = out[channel]
    flat = int(torch.argmax(channel_map).item())
    h, w = channel_map.shape
    y, x = divmod(flat, w)
    return int(y), int(x)


def center_coords_for_channel(event: TraceEvent) -> Tuple[Optional[int], Optional[int]]:
    out = event.output_tensor[0]
    if out.ndim != 3:
        return None, None
    _, h, w = out.shape
    return int(h // 2), int(w // 2)


def describe_node(node: NodeRef) -> str:
    if node.y is None or node.x is None:
        return f"c={node.channel}"
    return f"c={node.channel}, y={node.y}, x={node.x}"


def pointwise_note(module_type: str) -> str:
    mapping = {
        "ReLU": "relu",
        "Dropout": "dropout",
        "BatchNorm2d": "batchnorm",
        "GroupNorm": "groupnorm",
        "Identity": "identity",
        "Flatten": "flatten",
        "FlattenSynthetic": "flatten",
        "Input": "input",
    }
    return mapping.get(module_type, module_type.lower())


@dataclass
class ResolvedUnit:
    node: NodeRef
    interpretation: str
    raw_unit: int


def resolve_unit_for_event(
    event: TraceEvent,
    raw_unit: int,
    spatial_choice: str = "argmax",
    y_file: Optional[int] = None,
    x_file: Optional[int] = None,
    manual_y: Optional[int] = None,
    manual_x: Optional[int] = None,
) -> ResolvedUnit:
    """Resolve a selectivity-file unit ID into an explicit NodeRef.

    Heuristic:
    - vector output: raw_unit is a direct feature index
    - spatial output with provided y/x and raw_unit < C: interpret as channel + supplied coords
    - spatial output with raw_unit >= C but < C*H*W: interpret as flattened CxHxW index
    - otherwise raw_unit < C means a channel-only identifier and spatial coords are chosen by rule
    """
    out = event.output_tensor
    if out.ndim == 2:
        n = int(out.shape[1])
        if raw_unit < 0 or raw_unit >= n:
            raise AppError(f"Unit index {raw_unit} is out of bounds for vector layer of size {n}.")
        return ResolvedUnit(
            node=NodeRef(layer=event.name, time=event.occurrence, channel=int(raw_unit), y=None, x=None),
            interpretation="vector_index",
            raw_unit=int(raw_unit),
        )

    if out.ndim != 4:
        raise AppError(f"Unsupported output rank for target selection: {tuple(out.shape)}")

    _, c, h, w = out.shape
    total = int(c * h * w)

    if y_file is not None and x_file is not None and 0 <= raw_unit < c:
        if not (0 <= y_file < h and 0 <= x_file < w):
            raise AppError(
                f"Provided y/x ({y_file}, {x_file}) are out of bounds for layer with spatial size ({h}, {w})."
            )
        return ResolvedUnit(
            node=NodeRef(layer=event.name, time=event.occurrence, channel=int(raw_unit), y=int(y_file), x=int(x_file)),
            interpretation="channel_plus_file_coords",
            raw_unit=int(raw_unit),
        )

    if 0 <= raw_unit < total and raw_unit >= c:
        ch, yy, xx = np.unravel_index(int(raw_unit), (c, h, w))
        return ResolvedUnit(
            node=NodeRef(layer=event.name, time=event.occurrence, channel=int(ch), y=int(yy), x=int(xx)),
            interpretation="flattened_spatial_index",
            raw_unit=int(raw_unit),
        )

    if raw_unit < 0 or raw_unit >= c:
        raise AppError(
            f"Unit index {raw_unit} is out of bounds for channel-only interpretation in a layer with {c} channels "
            f"and also not valid as a flattened index in total size {total}."
        )

    if spatial_choice == "from_file" and y_file is not None and x_file is not None:
        yy, xx = int(y_file), int(x_file)
        interpretation = "channel_plus_file_coords"
    elif spatial_choice == "argmax":
        yy, xx = spatial_argmax_for_channel(event, int(raw_unit))
        interpretation = "channel_plus_argmax_coords"
    elif spatial_choice == "center":
        yy, xx = center_coords_for_channel(event)
        interpretation = "channel_plus_center_coords"
    elif spatial_choice == "manual":
        if manual_y is None or manual_x is None:
            raise AppError("Manual spatial choice requires explicit manual y/x coordinates.")
        yy, xx = int(manual_y), int(manual_x)
        interpretation = "channel_plus_manual_coords"
    else:
        yy, xx = spatial_argmax_for_channel(event, int(raw_unit))
        interpretation = "channel_plus_argmax_coords"

    if yy is None or xx is None:
        raise AppError("Could not resolve spatial coordinates for the selected unit.")
    if not (0 <= yy < h and 0 <= xx < w):
        raise AppError(f"Resolved y/x ({yy}, {xx}) are out of bounds for layer with spatial size ({h}, {w}).")

    return ResolvedUnit(
        node=NodeRef(layer=event.name, time=event.occurrence, channel=int(raw_unit), y=int(yy), x=int(xx)),
        interpretation=interpretation,
        raw_unit=int(raw_unit),
    )


def render_full_width(callable_obj, *args, **kwargs):
    try:
        return callable_obj(*args, width="stretch", **kwargs)
    except TypeError:
        return callable_obj(*args, use_container_width=True, **kwargs)


# -----------------------------------------------------------------------------
# Selectivity loader
# -----------------------------------------------------------------------------


DEFAULT_LAYER_COLS = ["layer", "layer_name"]
DEFAULT_UNIT_COLS = ["unit", "unit_id", "channel"]
DEFAULT_Y_COLS = ["y", "row"]
DEFAULT_X_COLS = ["x", "col"]
DEFAULT_TIME_COLS = ["time", "t"]


@dataclass
class SelectivitySpec:
    df: pd.DataFrame
    layer_col: str
    unit_col: str
    y_col: Optional[str]
    x_col: Optional[str]
    time_col: Optional[str]
    score_cols: List[str]


def load_selectivity_table_from_any(uploaded_bytes: Optional[bytes], path_text: str) -> SelectivitySpec:
    if uploaded_bytes is not None:
        raw = io.BytesIO(uploaded_bytes)
        name = getattr(uploaded_bytes, "name", "uploaded") if not isinstance(uploaded_bytes, bytes) else "uploaded"
        if str(name).lower().endswith((".pkl", ".pickle")):
            raw.seek(0)
            obj = pickle.load(raw)
            df = obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
        else:
            raw.seek(0)
            df = pd.read_csv(raw)
    else:
        if not path_text:
            raise AppError("Please provide a selectivity file via upload or path.")
        path = Path(path_text)
        if not path.exists():
            raise AppError(f"Selectivity file does not exist: {path}")
        if path.suffix.lower() in {".pkl", ".pickle"}:
            obj = pd.read_pickle(path)
            df = obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
        else:
            df = pd.read_csv(path)
    return infer_selectivity_spec(df)


def infer_selectivity_spec(df: pd.DataFrame) -> SelectivitySpec:
    cols = list(df.columns)

    def first_match(candidates: Sequence[str]) -> Optional[str]:
        lookup = {c.lower(): c for c in cols}
        for cand in candidates:
            if cand.lower() in lookup:
                return lookup[cand.lower()]
        return None

    layer_col = first_match(DEFAULT_LAYER_COLS)
    unit_col = first_match(DEFAULT_UNIT_COLS)
    y_col = first_match(DEFAULT_Y_COLS)
    x_col = first_match(DEFAULT_X_COLS)
    time_col = first_match(DEFAULT_TIME_COLS)

    if layer_col is None or unit_col is None:
        raise AppError(
            "Could not infer required columns. The selectivity file needs at least a layer column and a unit column."
        )

    excluded = {layer_col, unit_col, y_col, x_col, time_col}
    score_cols: List[str] = []
    for c in cols:
        if c in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            score_cols.append(c)
    if not score_cols:
        raise AppError("No numeric selectivity columns were detected.")

    return SelectivitySpec(
        df=df.copy(),
        layer_col=layer_col,
        unit_col=unit_col,
        y_col=y_col,
        x_col=x_col,
        time_col=time_col,
        score_cols=score_cols,
    )


def parse_score_families(score_cols: Sequence[str]) -> Dict[str, List[str]]:
    families: Dict[str, List[str]] = defaultdict(list)
    for col in score_cols:
        if "_" in col:
            prefix, _ = col.split("_", 1)
        else:
            prefix = "all"
        families[prefix].append(col)
    return dict(families)


def candidate_units_from_selectivity(
    spec: SelectivitySpec,
    score_col: str,
    mode: str,
    n: int,
    layer_allowlist: Optional[Sequence[str]] = None,
    seed: int = 0,
) -> pd.DataFrame:
    df = spec.df.copy()
    if layer_allowlist:
        df = df[df[spec.layer_col].astype(str).isin(set(str(x) for x in layer_allowlist))]
    df = df.dropna(subset=[score_col])
    if df.empty:
        return df
    if mode == "top_n":
        df = df.sort_values(score_col, ascending=False).head(n)
    elif mode == "bottom_n":
        df = df.sort_values(score_col, ascending=True).head(n)
    elif mode == "random":
        rng = np.random.default_rng(seed)
        take = min(n, len(df))
        idx = rng.choice(len(df), size=take, replace=False)
        df = df.iloc[idx]
    else:
        raise ValueError(f"Unknown candidate unit mode: {mode}")
    return df.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Model loading and registry
# -----------------------------------------------------------------------------


@dataclass
class LoadedModel:
    name: str
    model: nn.Module
    device: torch.device
    preprocess: Any
    trace_aliases: Dict[str, str]
    traceable_leaf_names: List[str]


class SyntheticFlatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(x, 1)


def make_default_preprocess(transforms_module: Any):
    return transforms_module.Compose(
        [
            transforms_module.Resize(256),
            transforms_module.CenterCrop(224),
            transforms_module.ToTensor(),
            transforms_module.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_trace_registry(model: nn.Module) -> Tuple[Dict[str, str], List[str]]:
    aliases: Dict[str, str] = {}
    leaf_names: List[str] = []

    def is_traceable_leaf(name: str, module: nn.Module) -> bool:
        if not is_leaf_module(module):
            return False
        if isinstance(module, TRACEABLE_TYPES):
            return True
        if module.__class__.__name__ in {"Identity", "Flatten"}:
            return True
        return False

    for name, module in model.named_modules():
        if name == "":
            continue
        if is_traceable_leaf(name, module):
            leaf_names.append(name)
            aliases[name] = name

    # CORnet convenience aliases: if block has an output leaf, map block root -> block.output
    for name, module in model.named_modules():
        if name == "":
            continue
        child_names = [f"{name}.output", f"{name}.nonlin1"]
        for child in child_names:
            if child in aliases:
                aliases[name] = child
                break

    return aliases, sorted(set(leaf_names), key=leaf_names.index)


def load_model_bundle(model_name: str, pretrained: bool, device: torch.device, cornet_times: int) -> LoadedModel:
    seed_everything(0)
    if model_name in {"alexnet", "vgg16"}:
        _, models, transforms = try_import_torchvision()
        preprocess = make_default_preprocess(transforms)
        if model_name == "alexnet":
            model = _load_alexnet(models, pretrained)
        else:
            model = _load_vgg16(models, pretrained)
    elif model_name == "cornet-rt":
        cornet = try_import_cornet()
        _, _, transforms = try_import_torchvision()
        preprocess = make_default_preprocess(transforms)
        model = cornet.cornet_rt(pretrained=pretrained, map_location=device, times=int(cornet_times))
    else:
        raise AppError(f"Unsupported model: {model_name}")

    model = model.to(device)
    model.eval()
    aliases, leaf_names = build_trace_registry(model)
    return LoadedModel(
        name=model_name,
        model=model,
        device=device,
        preprocess=preprocess,
        trace_aliases=aliases,
        traceable_leaf_names=leaf_names,
    )


def _load_alexnet(models: Any, pretrained: bool) -> nn.Module:
    try:
        from torchvision.models import AlexNet_Weights  # type: ignore
        return models.alexnet(weights=AlexNet_Weights.DEFAULT if pretrained else None)
    except Exception:
        return models.alexnet(pretrained=pretrained)


def _load_vgg16(models: Any, pretrained: bool) -> nn.Module:
    try:
        from torchvision.models import VGG16_Weights  # type: ignore
        return models.vgg16(weights=VGG16_Weights.DEFAULT if pretrained else None)
    except Exception:
        return models.vgg16(pretrained=pretrained)


# -----------------------------------------------------------------------------
# Forward recorder
# -----------------------------------------------------------------------------


class ForwardRecorder:
    def __init__(self, model: nn.Module, model_name: str):
        self.model = model
        self.model_name = model_name
        self.handles: List[Any] = []
        self.events: List[TraceEvent] = []
        self.call_counts: Dict[str, int] = defaultdict(int)
        self.synthetic_flatten = SyntheticFlatten()
        self.leaf_name_set = set(build_trace_registry(model)[1])

    def clear(self) -> None:
        self.events = []
        self.call_counts = defaultdict(int)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _hook(self, name: str, module: nn.Module):
        def fn(module_: nn.Module, inputs: Tuple[Any, ...], output: Any):
            input_tensor = unpack_first_tensor(inputs)
            output_tensor = unpack_first_tensor(output)
            if output_tensor is None:
                return
            occurrence = self.call_counts[name]
            self.call_counts[name] += 1
            event = TraceEvent(
                index=-1,
                name=name,
                occurrence=occurrence,
                module_type=module.__class__.__name__,
                module_ref=module,
                input_tensor=safe_to_cpu(input_tensor),
                output_tensor=safe_to_cpu(output_tensor),
                synthetic=False,
            )
            self.events.append(event)

        return fn

    def register(self) -> None:
        self.close()
        self.clear()
        for name, module in self.model.named_modules():
            if name == "":
                continue
            if name not in self.leaf_name_set:
                continue
            self.handles.append(module.register_forward_hook(self._hook(name, module)))

    def run(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[TraceEvent]]:
        self.clear()
        with torch.no_grad():
            out = self.model(x)
        events = list(self.events)
        events = self._insert_synthetic_events(x, events)
        for i, ev in enumerate(events):
            ev.index = i
        return out, events

    def _insert_synthetic_events(self, x: torch.Tensor, events: List[TraceEvent]) -> List[TraceEvent]:
        synthetic_input = TraceEvent(
            index=-1,
            name="input",
            occurrence=0,
            module_type="Input",
            module_ref=None,
            input_tensor=None,
            output_tensor=safe_to_cpu(x),
            synthetic=True,
        )
        events = [synthetic_input] + events

        # torchvision AlexNet/VGG use functional torch.flatten between avgpool and classifier.
        if self.model_name in {"alexnet", "vgg16"}:
            avgpool_idx = None
            insert_idx = None
            for i, ev in enumerate(events):
                if ev.name == "avgpool":
                    avgpool_idx = i
                if avgpool_idx is not None and ev.name.startswith("classifier.") and ev.input_tensor is not None:
                    insert_idx = i
                    break
            if avgpool_idx is not None and insert_idx is not None:
                avg_ev = events[avgpool_idx]
                flat_out = torch.flatten(avg_ev.output_tensor, 1)
                flat_ev = TraceEvent(
                    index=-1,
                    name="__flatten__",
                    occurrence=0,
                    module_type="FlattenSynthetic",
                    module_ref=self.synthetic_flatten,
                    input_tensor=safe_to_cpu(avg_ev.output_tensor),
                    output_tensor=safe_to_cpu(flat_out),
                    synthetic=True,
                )
                events = events[: insert_idx] + [flat_ev] + events[insert_idx:]

        return events


# -----------------------------------------------------------------------------
# Backward tracing
# -----------------------------------------------------------------------------


POINTWISE_TYPES = {
    "ReLU",
    "Dropout",
    "BatchNorm2d",
    "GroupNorm",
    "Identity",
    "Flatten",
    "FlattenSynthetic",
    "Input",
}


class PathTracer:
    def __init__(
        self,
        events: List[TraceEvent],
        path_mode: str,
        search_mode: str = "greedy",
        beam_width: int = 5,
        random_paths_to_compute: int = 200,
        random_seed: int = 0,
    ):
        self.events = events
        self.path_mode = path_mode
        self.search_mode = search_mode
        self.beam_width = max(1, int(beam_width))
        self.random_paths_to_compute = max(1, int(random_paths_to_compute))
        self.random_seed = int(random_seed)

    def trace(self, start_event_index: int, start_node: NodeRef, top_k_paths: int = 8) -> List[PathTrace]:
        if self.path_mode == "random":
            return self._trace_random(start_event_index, start_node)

        beam_cap = 1 if self.search_mode == "greedy" else max(self.beam_width, top_k_paths)
        states = [
            {
                "event_index": int(start_event_index),
                "node": start_node,
                "score": 0.0,
                "steps_backwards": [],
            }
        ]

        while True:
            unfinished = [s for s in states if s["event_index"] > 0]
            if not unfinished:
                break
            new_states = []
            for state in states:
                event_index = state["event_index"]
                if event_index <= 0:
                    new_states.append(state)
                    continue
                current_event = self.events[event_index]
                prev_event = self.events[event_index - 1]
                node = state["node"]
                current_step_score = state["score"]
                transitions = self._backward_candidates(current_event, prev_event, node)
                if not transitions:
                    # force a stop at the current point if tracing cannot continue cleanly
                    input_step = self._make_step_for_current_node(current_event, node, None, current_step_score, note="stopped")
                    new_state = dict(state)
                    new_state["event_index"] = 0
                    new_state["steps_backwards"] = state["steps_backwards"] + [input_step]
                    new_states.append(new_state)
                    continue
                transitions = sorted(transitions, key=lambda t: t.rank_score, reverse=True)
                for transition in transitions[:beam_cap]:
                    next_score = current_step_score + float(transition.rank_score)
                    step = self._make_step_for_current_node(current_event, node, transition, next_score)
                    new_states.append(
                        {
                            "event_index": event_index - 1,
                            "node": transition.parent_node,
                            "score": next_score,
                            "steps_backwards": state["steps_backwards"] + [step],
                        }
                    )

            new_states = sorted(new_states, key=lambda s: s["score"], reverse=True)[:beam_cap]
            states = new_states

        traces: List[PathTrace] = []
        for rank, state in enumerate(sorted(states, key=lambda s: s["score"], reverse=True)[:top_k_paths], start=1):
            first_event = self.events[0]
            input_node = state["node"]
            input_step = PathStep(
                step_index=0,
                layer=first_event.name,
                time=0,
                node=input_node,
                pre_value=None,
                post_value=get_event_node_value(first_event, input_node, which="output"),
                edge_weight=None,
                local_contribution=None,
                delta_value=None,
                cumulative_score=float(state["score"]),
                event_index=0,
                module_type=first_event.module_type,
                edge_ref=None,
                note="input",
            )
            steps = [input_step] + list(reversed(state["steps_backwards"]))
            for i, step in enumerate(steps):
                step.step_index = i
            traces.append(
                PathTrace(
                    steps=steps,
                    path_score=float(state["score"]),
                    path_mode=self.path_mode,
                    path_rank=rank,
                    search_mode=self.search_mode,
                )
            )
        return traces

    def _path_signature_from_steps(self, steps: List[PathStep]) -> Tuple[Tuple[Any, ...], ...]:
        signature: List[Tuple[Any, ...]] = []
        for step in steps:
            signature.append((step.event_index, step.layer, step.node.channel, step.node.y, step.node.x, step.note))
        return tuple(signature)

    def _trace_random(self, start_event_index: int, start_node: NodeRef) -> List[PathTrace]:
        rng = random.Random(self.random_seed)
        sampled: List[PathTrace] = []
        seen: set = set()

        for _ in range(self.random_paths_to_compute):
            event_index = int(start_event_index)
            node = start_node
            cumulative_score = 0.0
            steps_backwards: List[PathStep] = []

            while event_index > 0:
                current_event = self.events[event_index]
                prev_event = self.events[event_index - 1]
                transitions = self._backward_candidates(current_event, prev_event, node)
                if not transitions:
                    input_step = self._make_step_for_current_node(current_event, node, None, cumulative_score, note="stopped")
                    steps_backwards.append(input_step)
                    event_index = 0
                    break
                transition = rng.choice(transitions)
                cumulative_score += float(transition.rank_score)
                step = self._make_step_for_current_node(current_event, node, transition, cumulative_score)
                steps_backwards.append(step)
                node = transition.parent_node
                event_index -= 1

            first_event = self.events[0]
            input_node = node
            input_step = PathStep(
                step_index=0,
                layer=first_event.name,
                time=0,
                node=input_node,
                pre_value=None,
                post_value=get_event_node_value(first_event, input_node, which="output"),
                edge_weight=None,
                local_contribution=None,
                delta_value=None,
                cumulative_score=float(cumulative_score),
                event_index=0,
                module_type=first_event.module_type,
                edge_ref=None,
                note="input",
            )
            steps = [input_step] + list(reversed(steps_backwards))
            for i, step in enumerate(steps):
                step.step_index = i
            signature = self._path_signature_from_steps(steps)
            if signature in seen:
                continue
            seen.add(signature)
            sampled.append(
                PathTrace(
                    steps=steps,
                    path_score=float(cumulative_score),
                    path_mode=self.path_mode,
                    path_rank=0,
                    search_mode="random_sampled",
                )
            )

        sampled = sorted(sampled, key=lambda p: p.path_score, reverse=True)
        for rank, path in enumerate(sampled, start=1):
            path.path_rank = rank
        return sampled

    def _make_step_for_current_node(
        self,
        event: TraceEvent,
        node: NodeRef,
        transition: Optional[CandidateTransition],
        cumulative_score: float,
        note: str = "",
    ) -> PathStep:
        pre_value = None
        if event.module_type in POINTWISE_TYPES and event.input_tensor is not None:
            try:
                if event.module_type in {"Flatten", "FlattenSynthetic"}:
                    flat_index = node_to_flat_index(node, event)
                    pre_value = float(event.input_tensor.flatten(start_dim=1)[0, flat_index].item())
                else:
                    pre_value = get_event_node_value(event, node, which="input")
            except Exception:
                pre_value = None

        post_value = get_event_node_value(event, node, which="output")
        local_contribution = to_numpy_scalar(transition.local_contribution) if transition is not None else None
        if local_contribution is None and event.module_type in {"ReLU", "BatchNorm2d", "GroupNorm", "Identity"}:
            local_contribution = post_value

        inferred_note = note or (transition.note if transition is not None else "")
        if not inferred_note and event.module_type in POINTWISE_TYPES:
            inferred_note = pointwise_note(event.module_type)

        step = PathStep(
            step_index=-1,
            layer=event.name,
            time=event.occurrence,
            node=node,
            pre_value=pre_value,
            post_value=post_value,
            edge_weight=to_numpy_scalar(transition.edge_weight) if transition is not None else None,
            local_contribution=local_contribution,
            delta_value=None,
            cumulative_score=float(cumulative_score),
            event_index=event.index,
            module_type=event.module_type,
            edge_ref=transition.edge_ref if transition is not None else None,
            note=inferred_note,
        )
        return step

    def _backward_candidates(self, current_event: TraceEvent, prev_event: TraceEvent, node: NodeRef) -> List[CandidateTransition]:
        module_type = current_event.module_type
        if current_event.input_tensor is None:
            return []

        if module_type == "Conv2d":
            return self._conv2d_candidates(current_event, prev_event, node)
        if module_type == "Linear":
            return self._linear_candidates(current_event, prev_event, node)
        if module_type == "MaxPool2d":
            return self._maxpool_candidates(current_event, prev_event, node)
        if module_type in {"AvgPool2d", "AdaptiveAvgPool2d"}:
            return self._avgpool_like_candidates(current_event, prev_event, node)
        if module_type in POINTWISE_TYPES:
            return self._passthrough_candidates(current_event, prev_event, node)
        return self._passthrough_candidates(current_event, prev_event, node)

    def _passthrough_candidates(self, current_event: TraceEvent, prev_event: TraceEvent, node: NodeRef) -> List[CandidateTransition]:
        # Same logical coordinate if shapes match. Flatten / FlattenSynthetic unflatten to prev_event.
        if current_event.module_type in {"Flatten", "FlattenSynthetic"}:
            flat_index = int(node.channel)
            parent = flatten_index_to_node(flat_index, prev_event, prev_event.name, prev_event.occurrence)
            return [
                CandidateTransition(
                    parent_node=parent,
                    edge_ref=None,
                    edge_weight=None,
                    local_contribution=None,
                    rank_score=0.0,
                    note="flatten",
                )
            ]

        if prev_event.output_tensor.ndim == current_event.input_tensor.ndim:
            parent = NodeRef(
                layer=prev_event.name,
                time=prev_event.occurrence,
                channel=node.channel,
                y=node.y,
                x=node.x,
            )
            return [
                CandidateTransition(
                    parent_node=parent,
                    edge_ref=None,
                    edge_weight=None,
                    local_contribution=None,
                    rank_score=0.0,
                    note=pointwise_note(current_event.module_type),
                )
            ]

        # Map flattened vector back to previous spatial event when shapes differ.
        if current_event.input_tensor.ndim == 2 and prev_event.output_tensor.ndim == 4:
            flat_index = int(node.channel)
            parent = flatten_index_to_node(flat_index, prev_event, prev_event.name, prev_event.occurrence)
            return [
                CandidateTransition(
                    parent_node=parent,
                    edge_ref=None,
                    edge_weight=None,
                    local_contribution=None,
                    rank_score=0.0,
                    note="reshape",
                )
            ]
        return []

    def _conv2d_candidates(self, current_event: TraceEvent, prev_event: TraceEvent, node: NodeRef) -> List[CandidateTransition]:
        module = current_event.module_ref
        assert isinstance(module, nn.Conv2d)
        inp = current_event.input_tensor[0]
        weight = module.weight.detach().cpu()

        out_c = int(node.channel)
        out_y = int(node.y or 0)
        out_x = int(node.x or 0)
        stride_y, stride_x = module.stride
        pad_y, pad_x = module.padding
        dil_y, dil_x = module.dilation
        k_h, k_w = module.kernel_size

        candidates: List[CandidateTransition] = []
        for in_c in range(inp.shape[0]):
            for ky in range(k_h):
                for kx in range(k_w):
                    iy = out_y * stride_y - pad_y + ky * dil_y
                    ix = out_x * stride_x - pad_x + kx * dil_x
                    if iy < 0 or ix < 0 or iy >= inp.shape[1] or ix >= inp.shape[2]:
                        continue
                    parent_act = float(inp[in_c, iy, ix].item())
                    edge_w = float(weight[out_c, in_c, ky, kx].item())
                    contrib = parent_act * edge_w
                    parent = NodeRef(layer=prev_event.name, time=prev_event.occurrence, channel=in_c, y=int(iy), x=int(ix))
                    edge = EdgeRef(
                        layer=current_event.name,
                        time=current_event.occurrence,
                        out_channel=out_c,
                        out_y=out_y,
                        out_x=out_x,
                        in_channel=in_c,
                        in_y=int(iy),
                        in_x=int(ix),
                        ky=int(ky),
                        kx=int(kx),
                    )
                    candidates.append(
                        CandidateTransition(
                            parent_node=parent,
                            edge_ref=edge,
                            edge_weight=edge_w,
                            local_contribution=contrib,
                            rank_score=score_key(contrib, self.path_mode),
                            note="conv",
                        )
                    )
        return candidates

    def _linear_candidates(self, current_event: TraceEvent, prev_event: TraceEvent, node: NodeRef) -> List[CandidateTransition]:
        module = current_event.module_ref
        assert isinstance(module, nn.Linear)
        inp = current_event.input_tensor[0]
        weight = module.weight.detach().cpu()
        out_idx = int(node.channel)

        candidates: List[CandidateTransition] = []
        for in_idx in range(inp.shape[0]):
            parent_act = float(inp[in_idx].item())
            edge_w = float(weight[out_idx, in_idx].item())
            contrib = parent_act * edge_w
            parent = flatten_index_to_node(in_idx, prev_event, prev_event.name, prev_event.occurrence)
            edge = EdgeRef(
                layer=current_event.name,
                time=current_event.occurrence,
                out_channel=out_idx,
                out_y=None,
                out_x=None,
                in_channel=parent.channel,
                in_y=parent.y,
                in_x=parent.x,
                ky=None,
                kx=None,
                linear_in_index=int(in_idx),
            )
            candidates.append(
                CandidateTransition(
                    parent_node=parent,
                    edge_ref=edge,
                    edge_weight=edge_w,
                    local_contribution=contrib,
                    rank_score=score_key(contrib, self.path_mode),
                    note="linear",
                )
            )
        return candidates

    def _maxpool_candidates(self, current_event: TraceEvent, prev_event: TraceEvent, node: NodeRef) -> List[CandidateTransition]:
        module = current_event.module_ref
        assert isinstance(module, nn.MaxPool2d)
        inp = current_event.input_tensor[0]
        c = int(node.channel)
        out_y = int(node.y or 0)
        out_x = int(node.x or 0)
        k = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        s = module.stride if module.stride is not None else k
        s = s if isinstance(s, tuple) else (s, s)
        p = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
        kh, kw = k
        sy, sx = s
        py, px = p
        y0 = out_y * sy - py
        x0 = out_x * sx - px

        candidates: List[CandidateTransition] = []
        best = None
        best_abs = -1.0
        for ky in range(kh):
            for kx in range(kw):
                iy = y0 + ky
                ix = x0 + kx
                if iy < 0 or ix < 0 or iy >= inp.shape[1] or ix >= inp.shape[2]:
                    continue
                val = float(inp[c, iy, ix].item())
                parent = NodeRef(layer=prev_event.name, time=prev_event.occurrence, channel=c, y=int(iy), x=int(ix))
                candidates.append(
                    CandidateTransition(
                        parent_node=parent,
                        edge_ref=None,
                        edge_weight=None,
                        local_contribution=float(val),
                        rank_score=score_key(float(val), self.path_mode),
                        note="maxpool",
                    )
                )
                if abs(val) > best_abs:
                    best_abs = abs(val)
                    best = candidates[-1]

        if self.path_mode == "random":
            return candidates
        if best is None:
            return []
        return [best]

    def _avgpool_like_candidates(self, current_event: TraceEvent, prev_event: TraceEvent, node: NodeRef) -> List[CandidateTransition]:
        inp = current_event.input_tensor[0]
        c = int(node.channel)
        out_y = int(node.y or 0)
        out_x = int(node.x or 0)

        if current_event.module_type == "AvgPool2d":
            module = current_event.module_ref
            assert isinstance(module, nn.AvgPool2d)
            k = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
            s = module.stride if module.stride is not None else k
            s = s if isinstance(s, tuple) else (s, s)
            p = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
            y0 = out_y * s[0] - p[0]
            x0 = out_x * s[1] - p[1]
            ys = range(y0, y0 + k[0])
            xs = range(x0, x0 + k[1])
        else:
            in_h, in_w = inp.shape[1:]
            out_h, out_w = current_event.output_tensor.shape[2:]
            y0 = math.floor(out_y * in_h / out_h)
            y1 = math.ceil((out_y + 1) * in_h / out_h)
            x0 = math.floor(out_x * in_w / out_w)
            x1 = math.ceil((out_x + 1) * in_w / out_w)
            ys = range(y0, y1)
            xs = range(x0, x1)

        candidates: List[CandidateTransition] = []
        for iy in ys:
            for ix in xs:
                if iy < 0 or ix < 0 or iy >= inp.shape[1] or ix >= inp.shape[2]:
                    continue
                val = float(inp[c, iy, ix].item())
                parent = NodeRef(layer=prev_event.name, time=prev_event.occurrence, channel=c, y=int(iy), x=int(ix))
                candidates.append(
                    CandidateTransition(
                        parent_node=parent,
                        edge_ref=None,
                        edge_weight=None,
                        local_contribution=val,
                        rank_score=score_key(val, self.path_mode),
                        note="avgpool",
                    )
                )
        return candidates


# -----------------------------------------------------------------------------
# Damage utilities
# -----------------------------------------------------------------------------


@dataclass
class DamageSpec:
    operation: str
    scalar: float = 0.0


@dataclass
class EdgeChoice:
    path_step_index: int
    layer: str
    time: Optional[int]
    edge_ref: EdgeRef


def apply_damage_to_model(model: nn.Module, edge: EdgeChoice, spec: DamageSpec) -> nn.Module:
    if spec.operation == "none":
        return copy.deepcopy(model)

    damaged = copy.deepcopy(model)
    modules = dict(damaged.named_modules())
    target = modules.get(edge.layer)
    if target is None:
        raise AppError(f"Could not find module to damage: {edge.layer}")

    if not hasattr(target, "weight"):
        raise AppError(f"Selected edge layer has no weight tensor: {edge.layer}")

    with torch.no_grad():
        if isinstance(target, nn.Conv2d):
            ref = edge.edge_ref
            value = target.weight[ref.out_channel, ref.in_channel, ref.ky, ref.kx]
            if spec.operation == "set_to_zero":
                value.zero_()
            elif spec.operation == "multiply_by_scalar":
                value.mul_(float(spec.scalar))
            elif spec.operation == "add_offset":
                value.add_(float(spec.scalar))
            else:
                raise ValueError(spec.operation)
        elif isinstance(target, nn.Linear):
            ref = edge.edge_ref
            idx = ref.linear_in_index
            if idx is None:
                raise AppError("Linear edge does not have a stored input index.")
            value = target.weight[ref.out_channel, idx]
            if spec.operation == "set_to_zero":
                value.zero_()
            elif spec.operation == "multiply_by_scalar":
                value.mul_(float(spec.scalar))
            elif spec.operation == "add_offset":
                value.add_(float(spec.scalar))
            else:
                raise ValueError(spec.operation)
        else:
            raise AppError(f"Damage currently supports Conv2d/Linear only, not {type(target).__name__}")
    return damaged



# -----------------------------------------------------------------------------
# Path recomputation and comparison helpers
# -----------------------------------------------------------------------------


def resolve_locked_path_against_events(baseline: PathTrace, damaged_events: List[TraceEvent]) -> PathTrace:
    new_steps: List[PathStep] = []
    events_by_index = {ev.index: ev for ev in damaged_events}

    for base_step in baseline.steps:
        event = events_by_index.get(base_step.event_index)
        if event is None:
            new_steps.append(copy.deepcopy(base_step))
            continue
        step = copy.deepcopy(base_step)
        step.post_value = get_event_node_value(event, step.node, which="output")
        if event.module_type in POINTWISE_TYPES and event.input_tensor is not None:
            if event.module_type in {"Flatten", "FlattenSynthetic"}:
                try:
                    flat_index = node_to_flat_index(step.node, event)
                    step.pre_value = float(event.input_tensor.flatten(start_dim=1)[0, flat_index].item())
                except Exception:
                    step.pre_value = None
            else:
                step.pre_value = get_event_node_value(event, step.node, which="input")
        else:
            step.pre_value = None

        if step.edge_ref is not None:
            step.edge_weight, step.local_contribution = recompute_edge_values(event, step.edge_ref)
        else:
            step.edge_weight = None
            if event.module_type == "MaxPool2d":
                step.local_contribution = step.post_value
            elif event.module_type in {"AvgPool2d", "AdaptiveAvgPool2d"}:
                step.local_contribution = step.post_value
            elif event.module_type in {"ReLU", "BatchNorm2d", "GroupNorm", "Identity"}:
                step.local_contribution = step.post_value
            else:
                step.local_contribution = None
        new_steps.append(step)

    return PathTrace(
        steps=new_steps,
        path_score=baseline.path_score,
        path_mode=baseline.path_mode,
        path_rank=baseline.path_rank,
        search_mode="locked",
    )


def recompute_edge_values(event: TraceEvent, edge_ref: EdgeRef) -> Tuple[Optional[float], Optional[float]]:
    module = event.module_ref
    if event.input_tensor is None or module is None:
        return None, None
    inp = event.input_tensor[0]
    if isinstance(module, nn.Conv2d):
        weight = float(module.weight.detach().cpu()[edge_ref.out_channel, edge_ref.in_channel, edge_ref.ky, edge_ref.kx].item())
        act = float(inp[edge_ref.in_channel, edge_ref.in_y, edge_ref.in_x].item())
        return weight, weight * act
    if isinstance(module, nn.Linear):
        idx = edge_ref.linear_in_index
        if idx is None:
            return None, None
        weight = float(module.weight.detach().cpu()[edge_ref.out_channel, idx].item())
        act = float(inp[idx].item())
        return weight, weight * act
    return None, None


def merge_delta_into_paths(baseline: PathTrace, damaged: PathTrace) -> Tuple[PathTrace, PathTrace]:
    base = copy.deepcopy(baseline)
    dam = copy.deepcopy(damaged)
    n = min(len(base.steps), len(dam.steps))
    for i in range(n):
        b = base.steps[i]
        d = dam.steps[i]
        if b.post_value is not None and d.post_value is not None:
            delta = float(d.post_value - b.post_value)
            b.delta_value = delta
            d.delta_value = delta
    return base, dam


# -----------------------------------------------------------------------------
# UI helper functions
# -----------------------------------------------------------------------------


def make_image_with_marker(img: Image.Image, node: NodeRef, input_size: Tuple[int, int]) -> Image.Image:
    marked = img.copy().convert("RGB")
    if node.y is None or node.x is None:
        return marked
    draw = ImageDraw.Draw(marked)
    w, h = marked.size
    in_h, in_w = input_size
    px = int(round((node.x / max(in_w - 1, 1)) * (w - 1)))
    py = int(round((node.y / max(in_h - 1, 1)) * (h - 1)))
    r = max(4, int(min(w, h) * 0.015))
    draw.ellipse((px - r, py - r, px + r, py + r), outline=(255, 40, 40), width=3)
    return marked


def path_to_dataframe(path: PathTrace) -> pd.DataFrame:
    rows = []
    for step in path.steps:
        row = {
            "step_index": step.step_index,
            "layer": step.layer,
            "time": step.time,
            "channel": step.node.channel,
            "y": step.node.y,
            "x": step.node.x,
            "pre_value": step.pre_value,
            "post_value": step.post_value,
            "edge_weight": step.edge_weight,
            "local_contribution": step.local_contribution,
            "delta_value": step.delta_value,
            "cumulative_score": step.cumulative_score,
            "module_type": step.module_type,
            "note": step.note,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def path_to_json(path: PathTrace) -> str:
    return json.dumps(
        {
            "path_score": path.path_score,
            "path_mode": path.path_mode,
            "path_rank": path.path_rank,
            "search_mode": path.search_mode,
            "steps": [
                {
                    **{k: v for k, v in asdict(step).items() if k != "edge_ref"},
                    "edge_ref": asdict(step.edge_ref) if step.edge_ref is not None else None,
                }
                for step in path.steps
            ],
        },
        indent=2,
    )


def make_path_figure(baseline: PathTrace, damaged: Optional[PathTrace] = None) -> go.Figure:
    y_labels = [f"{s.step_index}: {s.layer} | {describe_node(s.node)}" for s in baseline.steps]
    raw_base = [s.post_value for s in baseline.steps]
    raw_dam = [s.post_value for s in damaged.steps] if damaged is not None else None
    contrib_base = [s.local_contribution for s in baseline.steps]
    contrib_dam = [s.local_contribution for s in damaged.steps] if damaged is not None else None
    delta_vals = [0.0 if s.delta_value is None else s.delta_value for s in baseline.steps]
    hover_base = [[s.module_type, s.note, s.pre_value, s.post_value, s.edge_weight, s.local_contribution, s.delta_value] for s in baseline.steps]
    hover_dam = [[s.module_type, s.note, s.pre_value, s.post_value, s.edge_weight, s.local_contribution, s.delta_value] for s in damaged.steps] if damaged is not None else None

    fig = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        subplot_titles=("Raw activation trace", "Local contribution trace", "Delta trace"),
    )

    hovertemplate = (
        "<b>%{y}</b><br>module=%{customdata[0]}<br>note=%{customdata[1]}"
        "<br>pre=%{customdata[2]}<br>post=%{customdata[3]}"
        "<br>weight=%{customdata[4]}<br>local=%{customdata[5]}<br>delta=%{customdata[6]}<extra></extra>"
    )

    fig.add_trace(
        go.Scatter(x=raw_base, y=y_labels, mode="lines+markers", name="Baseline raw", customdata=hover_base, hovertemplate=hovertemplate),
        row=1,
        col=1,
    )
    if raw_dam is not None:
        fig.add_trace(
            go.Scatter(x=raw_dam, y=y_labels, mode="lines+markers", name="Damaged raw", customdata=hover_dam, hovertemplate=hovertemplate),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(x=contrib_base, y=y_labels, mode="lines+markers", name="Baseline contrib", customdata=hover_base, hovertemplate=hovertemplate),
        row=1,
        col=2,
    )
    if contrib_dam is not None:
        fig.add_trace(
            go.Scatter(x=contrib_dam, y=y_labels, mode="lines+markers", name="Damaged contrib", customdata=hover_dam, hovertemplate=hovertemplate),
            row=1,
            col=2,
        )

    fig.add_trace(
        go.Bar(x=delta_vals, y=y_labels, name="Delta", customdata=hover_base, hovertemplate=hovertemplate),
        row=1,
        col=3,
    )

    fig.update_xaxes(zeroline=True, zerolinewidth=1, row=1, col=2)
    fig.update_xaxes(zeroline=True, zerolinewidth=1, row=1, col=3)
    fig.update_layout(height=max(500, 100 + 42 * len(y_labels)), showlegend=True)
    return fig


def compute_all_image_summary(
    image_paths: Sequence[Path],
    bundle: LoadedModel,
    start_layer_alias: str,
    raw_unit: int,
    spatial_mode: str,
    y_file: Optional[int] = None,
    x_file: Optional[int] = None,
) -> pd.DataFrame:
    rows = []
    for path in image_paths:
        try:
            pil = load_image_pil(path)
            tensor = bundle.preprocess(pil).unsqueeze(0).to(bundle.device)
            recorder = ForwardRecorder(bundle.model, bundle.name)
            recorder.register()
            _, events = recorder.run(tensor)
            recorder.close()
            matching = [ev for ev in events if ev.name == start_layer_alias]
            if not matching:
                continue
            event = matching[-1]
            resolved = resolve_unit_for_event(
                event,
                raw_unit=int(raw_unit),
                spatial_choice=spatial_mode,
                y_file=y_file,
                x_file=x_file,
            )
            node = resolved.node
            value = get_event_node_value(event, node, which="output")
            rows.append({
                "image": path.name,
                "activation": value,
                "layer": event.name,
                "time": event.occurrence,
                "channel": node.channel,
                "y": node.y,
                "x": node.x,
                "interpretation": resolved.interpretation,
            })
        except Exception:
            rows.append({"image": path.name, "activation": np.nan, "layer": start_layer_alias, "time": np.nan})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Stage lesion helpers
# -----------------------------------------------------------------------------


def run_forward_events(model: nn.Module, model_name: str, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, List[TraceEvent]]:
    recorder = ForwardRecorder(model, model_name)
    recorder.register()
    try:
        out, events = recorder.run(input_tensor)
    finally:
        recorder.close()
    return out, events


def event_display_label(event: TraceEvent) -> str:
    return f"{event.index}: {event.label} ({event.module_type})"


def output_unit_count_for_event(event: TraceEvent) -> int:
    if event.output_tensor is None or event.output_tensor.ndim < 2:
        return 0
    return int(event.output_tensor.shape[1])


def sample_count_from_fraction(total: int, fraction: float) -> int:
    total = int(total)
    fraction = float(fraction)
    if total <= 0 or fraction <= 0:
        return 0
    k = int(round(total * fraction))
    if k == 0:
        k = 1
    return max(0, min(total, k))


def choose_random_indices(total: int, fraction: float, rng: np.random.Generator) -> np.ndarray:
    k = sample_count_from_fraction(total, fraction)
    if k <= 0:
        return np.zeros((0,), dtype=np.int64)
    return np.asarray(rng.choice(total, size=k, replace=False), dtype=np.int64)


def zero_selected_units_in_tensor(tensor: torch.Tensor, zero_indices: Sequence[int]) -> torch.Tensor:
    if tensor.ndim < 2 or len(zero_indices) == 0:
        return tensor
    mask = torch.ones(int(tensor.shape[1]), device=tensor.device, dtype=tensor.dtype)
    mask[torch.as_tensor(list(zero_indices), device=tensor.device, dtype=torch.long)] = 0
    view_shape = [1, int(tensor.shape[1])] + [1] * max(0, tensor.ndim - 2)
    return tensor * mask.view(*view_shape)


def zero_selected_units_in_output(output: Any, zero_indices: Sequence[int]) -> Any:
    if len(zero_indices) == 0:
        return output
    if torch.is_tensor(output):
        return zero_selected_units_in_tensor(output, zero_indices)
    if isinstance(output, tuple):
        replaced = False
        items = []
        for item in output:
            if not replaced and torch.is_tensor(item):
                items.append(zero_selected_units_in_tensor(item, zero_indices))
                replaced = True
            else:
                items.append(item)
        return tuple(items)
    if isinstance(output, list):
        replaced = False
        items = []
        for item in output:
            if not replaced and torch.is_tensor(item):
                items.append(zero_selected_units_in_tensor(item, zero_indices))
                replaced = True
            else:
                items.append(item)
        return items
    if isinstance(output, dict):
        replaced = False
        items = {}
        for key, value in output.items():
            if not replaced and torch.is_tensor(value):
                items[key] = zero_selected_units_in_tensor(value, zero_indices)
                replaced = True
            else:
                items[key] = value
        return items
    return output


def run_stage_lesion_once(
    bundle: LoadedModel,
    input_tensor: torch.Tensor,
    stage_event: TraceEvent,
    lesion_kind: str,
    fraction: float,
    rng_seed: int,
) -> Tuple[List[TraceEvent], Dict[str, Any]]:
    if lesion_kind == 'none' or fraction <= 0:
        _, events = run_forward_events(bundle.model, bundle.name, input_tensor)
        return events, {
            'lesion_kind': lesion_kind,
            'fraction_requested': float(fraction),
            'deleted_count': 0,
            'total_count': 0,
            'actual_fraction': 0.0,
            'target_event': event_display_label(stage_event),
        }

    damaged_model = copy.deepcopy(bundle.model).to(bundle.device)
    damaged_model.eval()
    modules = dict(damaged_model.named_modules())
    target = modules.get(stage_event.name)
    if target is None:
        raise AppError(f"Could not find stage module for lesion analysis: {stage_event.name}")

    rng = np.random.default_rng(int(rng_seed))
    hook_handles: List[Any] = []
    call_state = {'count': 0}
    lesion_meta: Dict[str, Any] = {
        'lesion_kind': lesion_kind,
        'fraction_requested': float(fraction),
        'deleted_count': 0,
        'total_count': 0,
        'actual_fraction': 0.0,
        'target_event': event_display_label(stage_event),
        'layer': stage_event.name,
        'occurrence': stage_event.occurrence,
    }

    try:
        if lesion_kind == 'units':
            total_units = output_unit_count_for_event(stage_event)
            if total_units <= 0:
                raise AppError('Selected stage does not expose a channel/unit dimension that can be lesioned.')
            zero_units = choose_random_indices(total_units, fraction, rng)
            lesion_meta['deleted_count'] = int(len(zero_units))
            lesion_meta['total_count'] = int(total_units)
            lesion_meta['actual_fraction'] = float(len(zero_units) / total_units) if total_units else 0.0

            def unit_hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any):
                occurrence = call_state['count']
                call_state['count'] += 1
                if occurrence != stage_event.occurrence:
                    return output
                return zero_selected_units_in_output(output, zero_units)

            hook_handles.append(target.register_forward_hook(unit_hook))

        elif lesion_kind == 'weights':
            if not hasattr(target, 'weight') or getattr(target, 'weight') is None:
                raise AppError('Selected stage has no weight tensor, so weight-level deletion is not available.')
            total_weights = int(target.weight.numel())
            zero_weights = choose_random_indices(total_weights, fraction, rng)
            lesion_meta['deleted_count'] = int(len(zero_weights))
            lesion_meta['total_count'] = int(total_weights)
            lesion_meta['actual_fraction'] = float(len(zero_weights) / total_weights) if total_weights else 0.0
            state: Dict[str, Any] = {'stored_weight': None, 'applied': False}

            def weight_pre_hook(module: nn.Module, inputs: Tuple[Any, ...]):
                occurrence = call_state['count']
                call_state['count'] += 1
                if occurrence != stage_event.occurrence or len(zero_weights) == 0:
                    return None
                state['stored_weight'] = module.weight.detach().clone()
                state['applied'] = True
                flat = module.weight.view(-1)
                index_tensor = torch.as_tensor(zero_weights, device=module.weight.device, dtype=torch.long)
                with torch.no_grad():
                    flat[index_tensor] = 0
                return None

            def weight_post_hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any):
                if state.get('applied') and state.get('stored_weight') is not None:
                    with torch.no_grad():
                        module.weight.copy_(state['stored_weight'])
                    state['stored_weight'] = None
                    state['applied'] = False
                return output

            hook_handles.append(target.register_forward_pre_hook(weight_pre_hook))
            hook_handles.append(target.register_forward_hook(weight_post_hook))
        else:
            raise AppError(f'Unsupported lesion kind: {lesion_kind}')

        _, events = run_forward_events(damaged_model, bundle.name, input_tensor)
        return events, lesion_meta
    finally:
        for handle in hook_handles:
            handle.remove()


def compute_stage_shift_rows(
    baseline_events: Sequence[TraceEvent],
    damaged_events: Sequence[TraceEvent],
    start_event_index: int,
    threshold: float,
    repeat_index: int,
    lesion_meta: Dict[str, Any],
    relative_eps: float = 1e-8,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    n_events = min(len(baseline_events), len(damaged_events))
    relative_eps = float(relative_eps)
    for event_index in range(int(start_event_index), n_events):
        base_event = baseline_events[event_index]
        dam_event = damaged_events[event_index]
        if tuple(base_event.output_tensor.shape) != tuple(dam_event.output_tensor.shape):
            continue
        base = base_event.output_tensor.float()
        dam = dam_event.output_tensor.float()
        delta = dam - base
        abs_delta = delta.abs()
        # Relative shift is computed per unit against that same unit's pre-damage activation.
        rel_abs_delta = abs_delta / base.abs().clamp_min(relative_eps)
        rows.append(
            {
                'repeat': int(repeat_index),
                'event_index': int(base_event.index),
                'event_label': base_event.label,
                'plot_label': f"{base_event.index}: {base_event.label}",
                'layer': base_event.name,
                'occurrence': int(base_event.occurrence),
                'module_type': base_event.module_type,
                'mean_abs_shift': float(abs_delta.mean().item()),
                'rms_shift': float(torch.sqrt((delta ** 2).mean()).item()),
                'mean_signed_shift': float(delta.mean().item()),
                'max_abs_shift': float(abs_delta.max().item()),
                'prop_shifted': float((abs_delta > float(threshold)).float().mean().item()),
                'mean_rel_abs_shift': float(rel_abs_delta.mean().item()),
                'rms_rel_shift': float(torch.sqrt((rel_abs_delta ** 2).mean()).item()),
                'max_rel_abs_shift': float(rel_abs_delta.max().item()),
                'prop_rel_shifted': float((rel_abs_delta > float(threshold)).float().mean().item()),
                'threshold': float(threshold),
                'relative_eps': float(relative_eps),
                'deleted_count': int(lesion_meta.get('deleted_count', 0)),
                'total_count': int(lesion_meta.get('total_count', 0)),
                'actual_fraction': float(lesion_meta.get('actual_fraction', 0.0)),
                'lesion_kind': lesion_meta.get('lesion_kind', ''),
                'target_event': lesion_meta.get('target_event', ''),
            }
        )
    return pd.DataFrame(rows)


def summarise_stage_shift_rows(rows_df: pd.DataFrame) -> pd.DataFrame:
    if rows_df.empty:
        return rows_df.copy()
    group_cols = ['event_index', 'event_label', 'plot_label', 'layer', 'occurrence', 'module_type']
    summary = (
        rows_df.groupby(group_cols, as_index=False)
        .agg(
            mean_abs_shift_mean=('mean_abs_shift', 'mean'),
            mean_abs_shift_std=('mean_abs_shift', 'std'),
            rms_shift_mean=('rms_shift', 'mean'),
            rms_shift_std=('rms_shift', 'std'),
            prop_shifted_mean=('prop_shifted', 'mean'),
            prop_shifted_std=('prop_shifted', 'std'),
            mean_rel_abs_shift_mean=('mean_rel_abs_shift', 'mean'),
            mean_rel_abs_shift_std=('mean_rel_abs_shift', 'std'),
            rms_rel_shift_mean=('rms_rel_shift', 'mean'),
            rms_rel_shift_std=('rms_rel_shift', 'std'),
            prop_rel_shifted_mean=('prop_rel_shifted', 'mean'),
            prop_rel_shifted_std=('prop_rel_shifted', 'std'),
            mean_signed_shift_mean=('mean_signed_shift', 'mean'),
            mean_signed_shift_std=('mean_signed_shift', 'std'),
            max_abs_shift_mean=('max_abs_shift', 'mean'),
            max_abs_shift_std=('max_abs_shift', 'std'),
            max_rel_abs_shift_mean=('max_rel_abs_shift', 'mean'),
            max_rel_abs_shift_std=('max_rel_abs_shift', 'std'),
            n_repeats=('repeat', 'nunique'),
            mean_actual_fraction=('actual_fraction', 'mean'),
            min_actual_fraction=('actual_fraction', 'min'),
            max_actual_fraction=('actual_fraction', 'max'),
            mean_relative_eps=('relative_eps', 'mean'),
        )
        .sort_values('event_index')
        .reset_index(drop=True)
    )
    for prefix in [
        'mean_abs_shift',
        'rms_shift',
        'prop_shifted',
        'mean_rel_abs_shift',
        'rms_rel_shift',
        'prop_rel_shifted',
        'mean_signed_shift',
        'max_abs_shift',
        'max_rel_abs_shift',
    ]:
        std_col = f'{prefix}_std'
        sem_col = f'{prefix}_sem'
        summary[std_col] = summary[std_col].fillna(0.0)
        summary[sem_col] = summary[std_col] / np.sqrt(summary['n_repeats'].clip(lower=1))
    return summary


def filter_stage_shift_summary_for_plot(summary_df: pd.DataFrame, flow_view_mode: str) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df.copy()
    if flow_view_mode == 'conv_to_conv_only':
        filtered = summary_df[summary_df['module_type'] == 'Conv2d'].copy()
        return filtered.reset_index(drop=True)
    return summary_df.copy().reset_index(drop=True)


def make_stage_shift_figure(
    summary_df: pd.DataFrame,
    show_error_bars: bool,
    error_bar_mode: str,
    shift_metric_mode: str,
    flow_view_mode: str,
) -> go.Figure:
    summary_df = filter_stage_shift_summary_for_plot(summary_df, flow_view_mode)
    relative_mode = shift_metric_mode == 'relative_to_each_unit_baseline'
    conv_only_mode = flow_view_mode == 'conv_to_conv_only'
    magnitude_prefix = 'mean_rel_abs_shift' if relative_mode else 'mean_abs_shift'
    prop_prefix = 'prop_rel_shifted' if relative_mode else 'prop_shifted'
    max_prefix = 'max_rel_abs_shift' if relative_mode else 'max_abs_shift'
    magnitude_title = (
        "Relative downstream shift |Δ| / max(|that unit's own pre-damage activation|, ε)"
        if relative_mode else
        'Magnitude of downstream shift'
    )
    prop_title = 'Proportion of activations relatively shifted' if relative_mode else 'Proportion of activations shifted'
    if conv_only_mode:
        magnitude_title += ' (Conv2d only)'
        prop_title += ' (Conv2d only)'
    magnitude_axis = 'Mean |Δ| / max(|unit baseline|, ε)' if relative_mode else 'Mean |Δ|'
    prop_axis = 'Proportion relatively shifted' if relative_mode else 'Proportion shifted'

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_xaxes=False,
        horizontal_spacing=0.08,
        subplot_titles=(magnitude_title, prop_title),
    )
    if summary_df.empty:
        empty_note = 'No downstream Conv2d events to plot.' if conv_only_mode else 'No downstream events to plot.'
        fig.add_annotation(text=empty_note, x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False)
        fig.update_layout(height=420)
        return fig

    x = list(range(len(summary_df)))
    ticktext = list(summary_df['plot_label'])
    abs_error = summary_df[f'{magnitude_prefix}_{error_bar_mode}'] if show_error_bars else None
    prop_error = summary_df[f'{prop_prefix}_{error_bar_mode}'] if show_error_bars else None
    customdata = np.stack(
        [
            summary_df['layer'].astype(str).to_numpy(),
            summary_df['occurrence'].astype(int).to_numpy(),
            summary_df['module_type'].astype(str).to_numpy(),
            summary_df['mean_actual_fraction'].astype(float).to_numpy(),
            summary_df[f'{max_prefix}_mean'].astype(float).to_numpy(),
            summary_df['mean_relative_eps'].astype(float).to_numpy(),
        ],
        axis=1,
    )
    max_label = 'max relative |Δ|' if relative_mode else 'max |Δ|'
    hovertemplate = (
        '<b>%{text}</b><br>layer=%{customdata[0]}<br>occurrence=%{customdata[1]}'
        '<br>module=%{customdata[2]}<br>mean lesion fraction=%{customdata[3]:.4f}'
        f'<br>{max_label}=%{{customdata[4]:.6f}}'
        '<br>relative ε=%{customdata[5]:.2e}<extra></extra>'
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=summary_df[f'{magnitude_prefix}_mean'],
            mode='lines+markers',
            name='Mean relative |Δ|' if relative_mode else 'Mean |Δ|',
            text=ticktext,
            customdata=customdata,
            hovertemplate=hovertemplate,
            error_y=dict(type='data', array=abs_error, visible=bool(show_error_bars)),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=summary_df[f'{prop_prefix}_mean'],
            mode='lines+markers',
            name='Proportion relatively shifted' if relative_mode else 'Proportion shifted',
            text=ticktext,
            customdata=customdata,
            hovertemplate=hovertemplate,
            error_y=dict(type='data', array=prop_error, visible=bool(show_error_bars)),
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(tickmode='array', tickvals=x, ticktext=ticktext, tickangle=45, row=1, col=1)
    fig.update_xaxes(tickmode='array', tickvals=x, ticktext=ticktext, tickangle=45, row=1, col=2)
    fig.update_yaxes(title_text=magnitude_axis, row=1, col=1)
    fig.update_yaxes(title_text=prop_axis, range=[0, 1.02], row=1, col=2)
    fig.update_layout(height=560, showlegend=False)
    return fig


# -----------------------------------------------------------------------------
# Main Streamlit app
# -----------------------------------------------------------------------------


def main() -> None:
    st = ensure_streamlit()
    st.set_page_config(page_title="Activation Path Tracing App", layout="wide")
    st.title("Activation Path Tracing and Damage Analysis")
    st.caption(
        "Single-file research prototype for VGG16, AlexNet, and optional CORnet-RT. "
        "It traces one chosen computational route rather than claiming a unique true causal path."
    )

    with st.sidebar:
        st.header("Model")
        model_name = st.selectbox("Model", ["alexnet", "vgg16", "cornet-rt"], index=0)
        pretrained = st.checkbox("Load pretrained weights", value=True)
        cornet_times = st.slider("CORnet-RT time steps", 1, 8, 5, disabled=model_name != "cornet-rt")
        device_pref = st.selectbox(
            "Device",
            ["auto", "cpu", "cuda"],
            index=0,
            help="Auto prefers CUDA when available.",
        )

        st.header("Images")
        if st.button("Browse image directory"):
            selected_dir = maybe_browse_directory()
            if selected_dir:
                st.session_state["image_dir"] = selected_dir
        image_dir = st.text_input("Image directory", value=st.session_state.get("image_dir", ""))
        image_mode = st.radio(
            "Image mode",
            ["single_image", "all_images_reference_plus_summary"],
            index=0,
            help="All-images mode still traces one active reference image but can also summarise target activations across all images.",
        )

        st.header("Selectivity file")
        uploaded = st.file_uploader("Upload selectivity CSV / pickle", type=["csv", "pkl", "pickle"])
        selectivity_path = st.text_input("or local selectivity file path", value="")

        st.header("Path tracing")
        path_mode = st.selectbox(
            "Path rule",
            [
                "strongest_signed_contribution",
                "strongest_absolute_contribution",
                "strongest_positive_contribution",
                "random",
            ],
            index=1,
            help=(
                "The three strongest_* modes keep the original ranked backward tracing. "
                "The random mode samples many candidate paths, ranks the sampled paths afterwards, "
                "and lets you pick a relative rank among them."
            ),
        )
        search_mode = st.selectbox("Search mode", ["greedy", "beam"], index=1, disabled=path_mode == "random")
        beam_width = st.slider("Beam width", 1, 12, 6, disabled=search_mode != "beam" or path_mode == "random")
        top_k_paths = st.slider("Number of candidate paths", 1, 12, 6, disabled=path_mode == "random")
        random_paths_to_compute = st.slider(
            "Random paths to compute",
            10,
            2000,
            300,
            10,
            disabled=path_mode != "random",
            help="Only used when Path rule = random. The app samples this many candidate backward paths, deduplicates them, and ranks the unique sampled paths by cumulative absolute contribution.",
        )
        random_relative_rank = st.slider(
            "Random relative rank (1 = strongest sampled, 100 = weakest sampled)",
            1,
            100,
            25,
            disabled=path_mode != "random",
            help="Chooses a path by relative position within the ranked set of randomly sampled unique paths.",
        )

    try:
        device = choose_device(device_pref)
        bundle = load_model_bundle(model_name, pretrained, device, cornet_times)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    images = list_images(image_dir) if image_dir else []
    if not images:
        st.info("Choose an image directory to continue.")
        st.stop()

    active_image_path = images[0]
    if image_mode == "single_image":
        active_image_path = st.selectbox("Active image", images, format_func=lambda p: p.name)
    else:
        active_image_path = st.selectbox(
            "Reference image used for actual path tracing",
            images,
            format_func=lambda p: p.name,
        )

    pil_img = load_image_pil(active_image_path)
    input_tensor = bundle.preprocess(pil_img).unsqueeze(0).to(bundle.device)

    try:
        _, baseline_events = run_forward_events(bundle.model, bundle.name, input_tensor)
    except Exception as exc:
        st.error(f"Forward pass failed: {exc}")
        st.stop()

    st.subheader("Reference image")
    render_full_width(st.image, pil_img, caption=active_image_path.name)
    st.write({"n_trace_events": len(baseline_events), "model": bundle.name})

    st.subheader("Stage lesion summary")
    model_module_names = set(dict(bundle.model.named_modules()).keys())
    available_stage_events = [ev for ev in baseline_events if ev.name in model_module_names]
    if not available_stage_events:
        st.info("No lesionable traced stages were found for the current model run.")
    else:
        stage_labels = [
            f"{event_display_label(ev)} | output={tuple(ev.output_tensor.shape)}"
            for ev in available_stage_events
        ]
        stage_col1, stage_col2, stage_col3, stage_col4 = st.columns([1.5, 1.0, 1.0, 1.0])
        with stage_col1:
            selected_stage_label = st.selectbox("Stage to lesion", stage_labels, key="stage_event_select")
        stage_event = available_stage_events[stage_labels.index(selected_stage_label)]
        supports_weights = bool(hasattr(stage_event.module_ref, "weight") and getattr(stage_event.module_ref, "weight", None) is not None)
        lesion_mode_options = ["none", "units"]
        if supports_weights:
            lesion_mode_options.append("weights")
        with stage_col2:
            lesion_kind = st.selectbox("Delete fraction of", lesion_mode_options, index=1 if "units" in lesion_mode_options else 0, key="stage_lesion_kind")
        with stage_col3:
            lesion_fraction = st.slider("Fraction deleted", 0.0, 1.0, 0.10, 0.01, key="stage_fraction")
        with stage_col4:
            lesion_repeats = st.slider("Random repeats", 1, 40, 8, key="stage_repeats")

        stage_col5, stage_col6, stage_col7 = st.columns([1.0, 1.2, 1.0])
        with stage_col5:
            lesion_seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=0, step=1, key="stage_seed")
        with stage_col6:
            shift_threshold = st.number_input(
                "Shift threshold",
                min_value=0.0,
                value=0.05,
                format="%.3g",
                key="stage_threshold",
                help="For absolute mode this is in activation units. For relative mode this is applied per unit to |Δ| / max(|that unit's own pre-damage activation|, ε).",
            )
        with stage_col7:
            error_bar_mode = st.selectbox("Error bars", ["sem", "std"], index=0, key="stage_error_mode")
        shift_metric_mode = st.radio(
            "Plot shift as",
            ["absolute", "relative_to_each_unit_baseline"],
            index=1,
            horizontal=True,
            key="stage_shift_metric_mode",
            help=(
                "Absolute uses |damaged - baseline|. Relative mode uses |damaged - baseline| / max(|that unit's own pre-damage activation|, ε), "
                "so each unit is normalised by its own baseline."
            ),
        )
        relative_eps = st.number_input(
            "Relative ε (per-unit baseline floor)",
            min_value=0.0,
            value=1e-8,
            format="%.1e",
            key="stage_relative_eps",
            help="Small floor used in the per-unit relative denominator: max(|that unit's own pre-damage activation|, ε).",
        )
        show_error_bars = st.checkbox("Show error bars", value=True, key="stage_show_error_bars")
        flow_view_mode = st.radio(
            "Plot event flow",
            ["all_downstream_events", "conv_to_conv_only"],
            index=0,
            horizontal=True,
            key="stage_flow_view_mode",
            help=(
                "All downstream events shows every traced module after the lesion stage. "
                "Conv-to-conv only filters the stage-lesion plots to Conv2d outputs so the lines connect one conv layer to the next."
            ),
        )

        if lesion_kind == "weights" and not supports_weights:
            st.warning("This stage does not have a weight tensor, so weight-level deletion is unavailable here.")
        else:
            try:
                repeat_tables: List[pd.DataFrame] = []
                lesion_meta_samples: List[Dict[str, Any]] = []
                for repeat_index in range(int(lesion_repeats)):
                    damaged_events, lesion_meta = run_stage_lesion_once(
                        bundle=bundle,
                        input_tensor=input_tensor,
                        stage_event=stage_event,
                        lesion_kind=lesion_kind,
                        fraction=float(lesion_fraction),
                        rng_seed=int(lesion_seed) + int(repeat_index),
                    )
                    lesion_meta_samples.append(lesion_meta)
                    repeat_rows = compute_stage_shift_rows(
                        baseline_events=baseline_events,
                        damaged_events=damaged_events,
                        start_event_index=stage_event.index,
                        threshold=float(shift_threshold),
                        repeat_index=repeat_index,
                        lesion_meta=lesion_meta,
                        relative_eps=float(relative_eps),
                    )
                    if not repeat_rows.empty:
                        repeat_tables.append(repeat_rows)

                stage_shift_rows = pd.concat(repeat_tables, ignore_index=True) if repeat_tables else pd.DataFrame()
                stage_shift_summary = summarise_stage_shift_rows(stage_shift_rows)
                mean_actual_fraction = float(np.mean([m.get("actual_fraction", 0.0) for m in lesion_meta_samples])) if lesion_meta_samples else 0.0
                deleted_count = int(np.mean([m.get("deleted_count", 0) for m in lesion_meta_samples])) if lesion_meta_samples else 0
                total_count = int(np.mean([m.get("total_count", 0) for m in lesion_meta_samples])) if lesion_meta_samples else 0

                st.write(
                    {
                        "selected_stage": event_display_label(stage_event),
                        "lesion_kind": lesion_kind,
                        "fraction_requested": float(lesion_fraction),
                        "shift_metric_mode": shift_metric_mode,
                        "relative_eps": float(relative_eps),
                        "mean_actual_fraction": mean_actual_fraction,
                        "mean_deleted_count": deleted_count,
                        "mean_total_count": total_count,
                        "downstream_events_summarised": int(len(stage_shift_summary)),
                        "plot_event_flow": flow_view_mode,
                        "events_plotted": int(len(filter_stage_shift_summary_for_plot(stage_shift_summary, flow_view_mode))),
                    }
                )
                lesion_fig = make_stage_shift_figure(
                    stage_shift_summary,
                    show_error_bars=show_error_bars,
                    error_bar_mode=error_bar_mode,
                    shift_metric_mode=shift_metric_mode,
                    flow_view_mode=flow_view_mode,
                )
                st.caption(
                    "Conv and ReLU stages often look smaller here because the summary averages shift across the full output tensor, "
                    "so sparse changes are diluted. MaxPool can magnify local winner-switches, and GroupNorm can spread one local perturbation "
                    "across many activations through centering and rescaling."
                )
                render_full_width(st.plotly_chart, lesion_fig)
                plotted_stage_shift_summary = filter_stage_shift_summary_for_plot(stage_shift_summary, flow_view_mode)
                with st.expander("Stage lesion summary table"):
                    render_full_width(st.dataframe, plotted_stage_shift_summary, height=320)
                with st.expander("Per-repeat stage lesion table"):
                    render_full_width(st.dataframe, stage_shift_rows, height=320)
            except Exception as exc:
                st.error(f"Stage lesion analysis failed: {exc}")
                st.code(traceback.format_exc())

    selectivity_is_provided = uploaded is not None or bool(selectivity_path)
    if not selectivity_is_provided:
        st.info("Upload a selectivity file if you also want the target-unit and path-tracing sections.")
    else:
        try:
            uploaded_bytes = uploaded.getvalue() if uploaded is not None else None
            spec = load_selectivity_table_from_any(uploaded_bytes, selectivity_path)
        except Exception as exc:
            st.error(str(exc))
            st.stop()

        layer_options = sorted(set(spec.df[spec.layer_col].astype(str)))
        valid_selectivity_layers = [x for x in layer_options if x in bundle.trace_aliases or x in bundle.traceable_leaf_names]
        if not valid_selectivity_layers:
            st.warning(
                "None of the layers in the selectivity file matched the model's traceable layers. "
                "You can still inspect the layer names below."
            )
            st.write("Traceable model layers:", bundle.traceable_leaf_names)
            st.stop()

        col_a, col_b, col_c = st.columns([1.2, 1.0, 1.0])
        with col_a:
            st.subheader("Selectivity and target unit")
            score_col = st.selectbox("Selectivity score column", spec.score_cols)
            layer_filter_mode = st.radio("Layer filter", ["matched_layers_only", "single_layer"], index=0, horizontal=True)
            if layer_filter_mode == "single_layer":
                layer_allow = [st.selectbox("Selectivity layer", valid_selectivity_layers)]
            else:
                layer_allow = valid_selectivity_layers
            unit_pick_mode = st.selectbox("Candidate unit mode", ["top_n", "bottom_n", "random"], index=0)
            candidate_n = st.slider("How many candidates to surface", 1, 50, 12)
            candidate_df = candidate_units_from_selectivity(
                spec=spec,
                score_col=score_col,
                mode=unit_pick_mode,
                n=candidate_n,
                layer_allowlist=layer_allow,
                seed=0,
            )
            if candidate_df.empty:
                st.error("No candidate units remained after filtering.")
                st.stop()
            candidate_labels = []
            for i, row in candidate_df.iterrows():
                y_txt = f", y={row[spec.y_col]}" if spec.y_col and pd.notna(row[spec.y_col]) else ""
                x_txt = f", x={row[spec.x_col]}" if spec.x_col and pd.notna(row[spec.x_col]) else ""
                candidate_labels.append(
                    f"{i}: {row[spec.layer_col]} | unit={row[spec.unit_col]} | {score_col}={row[score_col]:.4f}{y_txt}{x_txt}"
                )
            picked_label = st.selectbox("Pick candidate unit", candidate_labels)
            picked_idx = int(picked_label.split(":", 1)[0])
            picked_row = candidate_df.iloc[picked_idx]
            render_full_width(st.dataframe, candidate_df, height=250)

        selected_selectivity_layer = str(picked_row[spec.layer_col])
        start_layer_name = bundle.trace_aliases.get(selected_selectivity_layer, selected_selectivity_layer)
        matching_events = [ev for ev in baseline_events if ev.name == start_layer_name]
        if not matching_events:
            st.error(f"The selected layer {selected_selectivity_layer} did not resolve to a traced event.")
            st.stop()

        with col_b:
            st.subheader("Target node identity")
            if len(matching_events) > 1:
                occ = st.slider("Time / occurrence", 0, len(matching_events) - 1, len(matching_events) - 1)
                start_event = matching_events[occ]
            else:
                start_event = matching_events[0]
                st.write(f"Occurrence: {start_event.occurrence}")
            raw_unit = int(picked_row[spec.unit_col])
            y_file = int(picked_row[spec.y_col]) if spec.y_col and pd.notna(picked_row[spec.y_col]) else None
            x_file = int(picked_row[spec.x_col]) if spec.x_col and pd.notna(picked_row[spec.x_col]) else None
            spatial_choice = "from_file"
            manual_y = None
            manual_x = None
            if start_event.output_tensor.ndim == 4:
                spatial_choice = st.selectbox(
                    "Spatial coordinate rule",
                    ["argmax", "center", "manual", "from_file"],
                    index=0 if not (spec.y_col and spec.x_col) else 3,
                    help=(
                        "If the selectivity file unit is only a channel index, choose how y,x is set for the current image. "
                        "If the unit is actually a flattened CxHxW index, the app will detect that automatically."
                    ),
                )
                _, c, h, w = start_event.output_tensor.shape
                if spatial_choice == "manual":
                    manual_y = st.slider("Manual y", 0, h - 1, h // 2)
                    manual_x = st.slider("Manual x", 0, w - 1, w // 2)
            else:
                st.write("Selected layer is vector-like, so no spatial coordinate is needed.")

            try:
                resolved_unit = resolve_unit_for_event(
                    start_event,
                    raw_unit=raw_unit,
                    spatial_choice=spatial_choice,
                    y_file=y_file,
                    x_file=x_file,
                    manual_y=manual_y,
                    manual_x=manual_x,
                )
            except Exception as exc:
                st.error(f"Could not resolve the selected unit for this layer: {exc}")
                st.stop()

            start_node = resolved_unit.node
            st.json(
                {
                    "selectivity_layer": selected_selectivity_layer,
                    "resolved_trace_layer": start_event.name,
                    "raw_unit": raw_unit,
                    "interpretation": resolved_unit.interpretation,
                    "channel": start_node.channel,
                    "y": start_node.y,
                    "x": start_node.x,
                    "time": start_event.occurrence,
                    "start_value": get_event_node_value(start_event, start_node, "output"),
                }
            )

        with col_c:
            st.subheader("Reference image")
            marked = make_image_with_marker(pil_img, start_node, input_size=(input_tensor.shape[-2], input_tensor.shape[-1]))
            render_full_width(st.image, marked, caption=active_image_path.name)
            if image_mode != "single_image":
                if st.checkbox("Compute all-image target summary"):
                    summary_df = compute_all_image_summary(images, bundle, start_event.name, raw_unit, spatial_choice, y_file=y_file, x_file=x_file)
                    render_full_width(st.dataframe, summary_df, height=250)
                    if not summary_df.empty and "activation" in summary_df.columns:
                        st.write(
                            {
                                "mean_activation": float(summary_df["activation"].mean(skipna=True)),
                                "std_activation": float(summary_df["activation"].std(skipna=True)),
                                "n_images": int(len(summary_df)),
                            }
                        )

        tracer = PathTracer(baseline_events, path_mode=path_mode, search_mode=search_mode, beam_width=beam_width)
        traces = tracer.trace(start_event.index, start_node, top_k_paths=top_k_paths)
        if not traces:
            st.error("No path could be traced from the selected target node.")
            st.stop()

        chosen_rank = st.slider("Path rank", 1, len(traces), 1)
        baseline_trace = traces[chosen_rank - 1]

        st.subheader("Baseline path")
        st.write(
            {
                "path_score": baseline_trace.path_score,
                "path_mode": baseline_trace.path_mode,
                "path_rank": baseline_trace.path_rank,
                "search_mode": baseline_trace.search_mode,
            }
        )
        baseline_df = path_to_dataframe(baseline_trace)
        render_full_width(st.dataframe, baseline_df, height=350)

        weighted_steps = [s for s in baseline_trace.steps if s.edge_ref is not None and s.edge_weight is not None and s.module_type == "Conv2d"]

        st.subheader("Damage comparison")
        if not weighted_steps:
            st.info("The traced path contains no Conv2d edges that can currently be zeroed.")
            fig = make_path_figure(baseline_trace, None)
            render_full_width(st.plotly_chart, fig)
        else:
            edge_options = []
            for s in weighted_steps:
                edge = s.edge_ref
                assert edge is not None
                edge_options.append(
                    f"step {s.step_index} | {s.layer} | out=({edge.out_channel},{edge.out_y},{edge.out_x}) <- "
                    f"in=({edge.in_channel},{edge.in_y},{edge.in_x}) | w={s.edge_weight:.5f}"
                )
            dcol1, dcol2, dcol3, dcol4 = st.columns(4)
            with dcol1:
                edge_label = st.selectbox("Weighted edge to damage", edge_options)
            chosen_edge_idx = edge_options.index(edge_label)
            chosen_step = weighted_steps[chosen_edge_idx]
            assert chosen_step.edge_ref is not None
            edge_choice = EdgeChoice(
                path_step_index=chosen_step.step_index,
                layer=chosen_step.layer,
                time=chosen_step.time,
                edge_ref=chosen_step.edge_ref,
            )

            with dcol2:
                compare_mode = st.selectbox("Comparison mode", ["locked_path", "reselected_path"], index=0)
            with dcol3:
                damage_op = st.selectbox(
                    "Conv weight intervention",
                    ["none", "set_to_zero"],
                    index=0,
                    help="'none' keeps the model unchanged. 'set_to_zero' zeros one specific Conv2d weight from the displayed path and then reruns the full network.",
                )
            with dcol4:
                if damage_op == "none":
                    st.write("No scalar needed")
                else:
                    st.write("Single conv weight -> 0")
                damage_scalar = 0.0

            damage_spec = DamageSpec(operation=damage_op, scalar=float(damage_scalar))
            try:
                if damage_op == "none":
                    damaged_events = baseline_events
                    if compare_mode == "locked_path":
                        damaged_trace = resolve_locked_path_against_events(baseline_trace, damaged_events)
                    else:
                        damaged_trace = copy.deepcopy(baseline_trace)
                else:
                    damaged_model = apply_damage_to_model(bundle.model, edge_choice, damage_spec)
                    damaged_model = damaged_model.to(bundle.device)
                    damaged_model.eval()
                    _, damaged_events = run_forward_events(damaged_model, bundle.name, input_tensor)

                    if compare_mode == "locked_path":
                        damaged_trace = resolve_locked_path_against_events(baseline_trace, damaged_events)
                    else:
                        damaged_tracer = PathTracer(damaged_events, path_mode=path_mode, search_mode=search_mode, beam_width=beam_width)
                        damaged_candidates = damaged_tracer.trace(start_event.index, start_node, top_k_paths=max(chosen_rank, top_k_paths))
                        damaged_trace = damaged_candidates[chosen_rank - 1 if len(damaged_candidates) >= chosen_rank else 0]

                baseline_with_delta, damaged_with_delta = merge_delta_into_paths(baseline_trace, damaged_trace)
                fig = make_path_figure(baseline_with_delta, damaged_with_delta)
                render_full_width(st.plotly_chart, fig)

                e = chosen_step.edge_ref
                assert e is not None
                st.write(
                    {
                        "selected_edge_layer": chosen_step.layer,
                        "time": chosen_step.time,
                        "out_channel": e.out_channel,
                        "out_y": e.out_y,
                        "out_x": e.out_x,
                        "in_channel": e.in_channel,
                        "in_y": e.in_y,
                        "in_x": e.in_x,
                        "ky": e.ky,
                        "kx": e.kx,
                        "linear_in_index": e.linear_in_index,
                        "baseline_weight": chosen_step.edge_weight,
                        "damaged_weight": next((s.edge_weight for s in damaged_with_delta.steps if s.step_index == chosen_step.step_index), None),
                    }
                )

                st.markdown("**Baseline vs damaged path tables**")
                t1, t2 = st.columns(2)
                with t1:
                    st.write("Baseline")
                    render_full_width(st.dataframe, path_to_dataframe(baseline_with_delta), height=320)
                with t2:
                    st.write("Damaged")
                    render_full_width(st.dataframe, path_to_dataframe(damaged_with_delta), height=320)

                csv_bytes = path_to_dataframe(baseline_with_delta).to_csv(index=False).encode("utf-8")
                st.download_button("Download baseline path CSV", csv_bytes, file_name="baseline_path.csv", mime="text/csv")
                st.download_button(
                    "Download baseline path JSON",
                    path_to_json(baseline_with_delta).encode("utf-8"),
                    file_name="baseline_path.json",
                    mime="application/json",
                )
            except Exception as exc:
                st.error(f"Damage comparison failed: {exc}")
                st.code(traceback.format_exc())
                fig = make_path_figure(baseline_trace, None)
                render_full_width(st.plotly_chart, fig)

        with st.expander("Selectivity file details"):
            st.write(
                {
                    "layer_col": spec.layer_col,
                    "unit_col": spec.unit_col,
                    "y_col": spec.y_col,
                    "x_col": spec.x_col,
                    "time_col": spec.time_col,
                    "score_cols": spec.score_cols,
                }
            )

    with st.expander("Traceable model layers"):
        st.write(bundle.traceable_leaf_names)

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------


def choose_device(device_pref: str) -> torch.device:
    if device_pref == "cuda":
        if not torch.cuda.is_available():
            raise AppError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if device_pref == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    main()
