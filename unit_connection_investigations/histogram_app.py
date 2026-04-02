#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
from PIL import Image

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
WEIGHT_TYPES = (
    nn.Conv2d,
    nn.Linear,
    nn.BatchNorm2d,
    nn.GroupNorm,
)


class AppError(RuntimeError):
    pass


@dataclass
class ModelBundle:
    name: str
    model: nn.Module
    device: torch.device
    preprocess: Any
    activation_layers: List[str]
    weight_layers: List[str]
    layer_type_lookup: Dict[str, str]


@dataclass
class PanelSpec:
    panel_id: str
    row: int
    col: int
    analysis: str
    model_name: str
    layer_name: str
    bins: int
    use_global_source: bool
    image_dir: str
    source_mode: str
    selected_image_name: Optional[str]
    max_images: int
    title: str
    data: Optional[np.ndarray] = None
    subtitle: str = ""
    error: Optional[str] = None
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_max: Optional[float] = None


# -----------------------------------------------------------------------------
# Imports and small helpers
# -----------------------------------------------------------------------------


def ensure_streamlit():
    try:
        import streamlit as st  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise AppError("Streamlit is not installed. Install it with: pip install streamlit") from exc
    return st


def render_full_width(callable_obj, *args, **kwargs):
    try:
        return callable_obj(*args, width="stretch", **kwargs)
    except TypeError:
        return callable_obj(*args, use_container_width=True, **kwargs)


def unpack_first_tensor(value: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if torch.is_tensor(item):
                return item
    if isinstance(value, dict):
        for item in value.values():
            if torch.is_tensor(item):
                return item
    return None


def safe_array(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().numpy()


def choose_device(device_pref: str) -> torch.device:
    if device_pref == "mps":
        if not torch.backends.mps.is_available():
            raise AppError("MPS was requested but is not available.")
        return torch.device("mps")
    if device_pref == "cuda":
        if not torch.cuda.is_available():
            raise AppError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if device_pref == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@contextmanager
def default_device_context(device: torch.device):
    if hasattr(torch, "set_default_device"):
        old_device = None
        if hasattr(torch, "get_default_device"):
            try:
                old_device = torch.get_default_device()
            except Exception:
                old_device = None
        try:
            torch.set_default_device(str(device))
            yield
        finally:
            if old_device is not None:
                try:
                    torch.set_default_device(old_device)
                except Exception:
                    pass
    else:
        yield


def forward_model(bundle: ModelBundle, x: torch.Tensor) -> Any:
    with default_device_context(bundle.device):
        return bundle.model(x)


def list_images(folder: str) -> List[Path]:
    root = Path(folder)
    if not folder or not root.exists() or not root.is_dir():
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
                return result.stdout.strip() or None
        except Exception:
            return None
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
    return Image.open(path).convert("RGB")


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------


def try_import_torchvision():
    try:
        import torchvision  # type: ignore
        from torchvision import models, transforms  # type: ignore
    except Exception as exc:
        raise AppError(
            "torchvision could not be imported. Reinstall compatible torch and torchvision versions."
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


def is_leaf_module(module: nn.Module) -> bool:
    return len(list(module.children())) == 0


def make_default_preprocess(transforms_module: Any):
    return transforms_module.Compose(
        [
            transforms_module.Resize(256),
            transforms_module.CenterCrop(224),
            transforms_module.ToTensor(),
            transforms_module.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
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


def build_layer_registry(model: nn.Module) -> Tuple[List[str], List[str], Dict[str, str]]:
    activation_layers: List[str] = []
    weight_layers: List[str] = []
    layer_type_lookup: Dict[str, str] = {}

    for name, module in model.named_modules():
        if name == "":
            continue
        if is_leaf_module(module) and isinstance(module, TRACEABLE_TYPES):
            activation_layers.append(name)
            layer_type_lookup[name] = module.__class__.__name__
        if isinstance(module, WEIGHT_TYPES):
            weight_layers.append(name)
            layer_type_lookup[name] = module.__class__.__name__

    return activation_layers, weight_layers, layer_type_lookup


@torch.no_grad()
def _get_model_bundle_uncached(model_name: str, pretrained: bool, device_str: str, cornet_times: int) -> ModelBundle:
    device = torch.device(device_str)
    if model_name in {"alexnet", "vgg16"}:
        _, models, transforms = try_import_torchvision()
        preprocess = make_default_preprocess(transforms)
        model = _load_alexnet(models, pretrained) if model_name == "alexnet" else _load_vgg16(models, pretrained)
    elif model_name == "cornet-rt":
        cornet = try_import_cornet()
        _, _, transforms = try_import_torchvision()
        preprocess = make_default_preprocess(transforms)
        with default_device_context(device):
            model = cornet.cornet_rt(pretrained=pretrained, map_location=device, times=int(cornet_times))
    else:
        raise AppError(f"Unsupported model: {model_name}")

    model = model.to(device)
    model.eval()
    activation_layers, weight_layers, layer_type_lookup = build_layer_registry(model)
    return ModelBundle(
        name=model_name,
        model=model,
        device=device,
        preprocess=preprocess,
        activation_layers=activation_layers,
        weight_layers=weight_layers,
        layer_type_lookup=layer_type_lookup,
    )


# -----------------------------------------------------------------------------
# Histogram data extraction
# -----------------------------------------------------------------------------


def get_layer_weight_values(bundle: ModelBundle, layer_name: str) -> np.ndarray:
    modules = dict(bundle.model.named_modules())
    module = modules.get(layer_name)
    if module is None:
        raise AppError(f"Layer not found in model: {layer_name}")
    if not hasattr(module, "weight") or getattr(module, "weight") is None:
        raise AppError(f"Layer does not expose a weight tensor: {layer_name}")
    weight = getattr(module, "weight")
    return safe_array(weight).ravel()


@torch.no_grad()
def get_single_image_activation_values(bundle: ModelBundle, layer_name: str, image_path: Path) -> np.ndarray:
    modules = dict(bundle.model.named_modules())
    module = modules.get(layer_name)
    if module is None:
        raise AppError(f"Layer not found in model: {layer_name}")

    holder: Dict[str, np.ndarray] = {}

    def hook_fn(_module: nn.Module, _inputs: Tuple[Any, ...], output: Any):
        tensor = unpack_first_tensor(output)
        if tensor is not None:
            holder["values"] = safe_array(tensor).ravel()

    handle = module.register_forward_hook(hook_fn)
    try:
        pil = load_image_pil(image_path)
        x = bundle.preprocess(pil).unsqueeze(0).to(bundle.device)
        forward_model(bundle, x)
    finally:
        handle.remove()

    if "values" not in holder:
        raise AppError(f"Could not capture activations from layer: {layer_name}")
    return holder["values"]


@torch.no_grad()
def get_multi_image_activation_values(
    bundle: ModelBundle,
    layer_name: str,
    image_paths: Sequence[Path],
) -> np.ndarray:
    values: List[np.ndarray] = []
    for image_path in image_paths:
        values.append(get_single_image_activation_values(bundle, layer_name, image_path))
    if not values:
        raise AppError("No images were available for activation extraction.")
    return np.concatenate(values, axis=0)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def compute_histogram_stats(values: np.ndarray, bins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise AppError("All selected values were non-finite, so the histogram could not be created.")
    counts, edges = np.histogram(finite, bins=int(bins))
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    return counts, edges, centers, float(np.min(edges)), float(np.max(edges)), float(np.max(counts))


def make_histogram_figure(
    panel: PanelSpec,
    sync_x_range: Optional[Tuple[float, float]],
    sync_y_max: Optional[float],
) -> go.Figure:
    assert panel.data is not None
    counts, edges, centers, auto_x_min, auto_x_max, auto_y_max = compute_histogram_stats(panel.data, panel.bins)
    fig = go.Figure(
        data=[
            go.Bar(
                x=centers,
                y=counts,
                width=np.diff(edges),
                hovertemplate="bin center=%{x:.5g}<br>count=%{y}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=panel.title,
        xaxis_title="Value",
        yaxis_title="Count",
        bargap=0.0,
        height=350,
        margin=dict(l=25, r=15, t=50, b=45),
        annotations=[
            dict(
                text=panel.subtitle,
                x=0,
                xref="paper",
                y=1.12,
                yref="paper",
                xanchor="left",
                showarrow=False,
                font=dict(size=11),
            )
        ],
    )
    x_range = sync_x_range if sync_x_range is not None else (auto_x_min, auto_x_max)
    fig.update_xaxes(range=list(x_range))
    y_upper = sync_y_max if sync_y_max is not None else auto_y_max
    if y_upper <= 0:
        y_upper = 1.0
    fig.update_yaxes(range=[0, y_upper * 1.05])
    panel.x_min = auto_x_min
    panel.x_max = auto_x_max
    panel.y_max = auto_y_max
    return fig


# -----------------------------------------------------------------------------
# UI state helpers
# -----------------------------------------------------------------------------


def ensure_panel_defaults(st, rows: int, cols: int) -> None:
    for r in range(rows):
        for c in range(cols):
            prefix = f"panel_{r}_{c}"
            st.session_state.setdefault(f"{prefix}_analysis", "activations")
            st.session_state.setdefault(f"{prefix}_model", "alexnet")
            st.session_state.setdefault(f"{prefix}_layer", "")
            st.session_state.setdefault(f"{prefix}_local_dir", "")
            st.session_state.setdefault(f"{prefix}_source_mode", "single image")
            st.session_state.setdefault(f"{prefix}_selected_image", "")
            st.session_state.setdefault(f"{prefix}_bins", 60)
            st.session_state.setdefault(f"{prefix}_use_global", True)
            st.session_state.setdefault(f"{prefix}_title", f"Panel {r + 1},{c + 1}")


def get_layer_options(bundle: ModelBundle, analysis: str) -> List[str]:
    if analysis == "weights":
        return bundle.weight_layers
    return bundle.activation_layers


# -----------------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------------


def main() -> None:
    st = ensure_streamlit()
    st.set_page_config(page_title="Model Histogram Grid", layout="wide")
    st.title("Model histogram grid")
    st.caption(
        "Grid-based Streamlit app for layer-wise weight and activation histograms across AlexNet, VGG16, and CORnet-RT. "
        "The design is inspired by your uploaded activation-path prototype, but simplified for multi-panel histogram comparison."
    )

    if not hasattr(st, "cache_resource") or not hasattr(st, "cache_data"):
        st.error("This app expects a recent Streamlit version with cache_resource and cache_data.")
        st.stop()

    @st.cache_resource(show_spinner=False)
    def load_bundle_cached(model_name: str, pretrained: bool, device_str: str, cornet_times: int) -> ModelBundle:
        return _get_model_bundle_uncached(model_name, pretrained, device_str, cornet_times)

    @st.cache_data(show_spinner=False)
    def list_images_cached(folder: str) -> List[str]:
        return [str(p) for p in list_images(folder)]

    with st.sidebar:
        st.header("Grid")
        rows = st.slider("Rows", 1, 4, 2)
        cols = st.slider("Columns", 1, 4, 2)
        ensure_panel_defaults(st, rows, cols)

        st.header("Global histogram settings")
        default_bins = st.slider("Default bins", 10, 200, 60)
        sync_x = st.checkbox("Share x-axis across all panels", value=False)
        sync_y = st.checkbox("Share y-axis across all panels", value=False)

        st.header("Model loading")
        pretrained = st.checkbox("Load pretrained weights", value=True)
        device_pref = st.selectbox("Device", ["auto", "cpu", "mps", "cuda"], index=0)
        cornet_times = st.slider("CORnet-RT time steps", 1, 8, 5)

        st.header("Global image source for activation panels")
        use_global_activation_source = st.checkbox("Use one global image source", value=True)
        if st.button("Browse global image directory"):
            selected = maybe_browse_directory()
            if selected:
                st.session_state["global_image_dir"] = selected
        global_image_dir = st.text_input("Global image directory", value=st.session_state.get("global_image_dir", ""))
        global_source_mode = st.radio("Global activation source", ["single image", "all images"], index=0)
        global_max_images = st.slider(
            "Max images when using 'all images'",
            1,
            256,
            24,
            help="Keeps 'all images' panels responsive. Increase it if you want a larger pooled activation histogram.",
        )

    try:
        device = choose_device(device_pref)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    st.caption(
        f"Active device: {device} | mps_built={torch.backends.mps.is_built()} | "
        f"mps_available={torch.backends.mps.is_available()} | cuda_available={torch.cuda.is_available()}"
    )

    global_image_paths: List[Path] = []
    global_selected_image_name: Optional[str] = None
    if global_image_dir:
        global_image_paths = [Path(p) for p in list_images_cached(global_image_dir)]
        if global_image_paths:
            global_selected_image_name = st.selectbox(
                "Global image",
                [p.name for p in global_image_paths],
                key="global_selected_image",
            )
        elif use_global_activation_source:
            st.warning("The global image directory currently contains no supported images.")

    panel_specs: List[PanelSpec] = []
    panel_placeholders: List[Tuple[Any, Any, Any]] = []
    used_models: Dict[str, ModelBundle] = {}

    st.subheader("Panels")
    for r in range(rows):
        row_cols = st.columns(cols)
        for c in range(cols):
            panel_id = f"panel_{r}_{c}"
            with row_cols[c]:
                box = st.container(border=True)
                with box:
                    st.markdown(f"**Panel {r + 1},{c + 1}**")
                    title = st.text_input("Title", key=f"{panel_id}_title")
                    analysis = st.selectbox(
                        "Histogram type",
                        ["activations", "weights"],
                        key=f"{panel_id}_analysis",
                    )
                    model_name = st.selectbox(
                        "Model",
                        ["alexnet", "vgg16", "cornet-rt"],
                        key=f"{panel_id}_model",
                    )

                    try:
                        if model_name not in used_models:
                            used_models[model_name] = load_bundle_cached(model_name, pretrained, str(device), cornet_times)
                        bundle = used_models[model_name]
                        layer_options = get_layer_options(bundle, analysis)
                    except Exception as exc:
                        layer_options = []
                        st.error(str(exc))

                    layer_display_options = [f"{name} ({bundle.layer_type_lookup.get(name, 'module')})" for name in layer_options] if layer_options else []
                    selected_layer_display = st.selectbox(
                        "Layer",
                        layer_display_options if layer_display_options else [""],
                        key=f"{panel_id}_layer_display",
                    )
                    layer_name = selected_layer_display.split(" (")[0] if selected_layer_display else ""

                    panel_use_global = use_global_activation_source
                    image_dir = global_image_dir
                    source_mode = global_source_mode
                    selected_image_name = global_selected_image_name
                    max_images = int(global_max_images)

                    if analysis == "activations":
                        if not use_global_activation_source:
                            panel_use_global = st.checkbox(
                                "Use global image source",
                                value=False,
                                key=f"{panel_id}_use_global",
                            )
                        if analysis == "activations" and not panel_use_global:
                            if st.button("Browse panel image directory", key=f"{panel_id}_browse"):
                                selected = maybe_browse_directory()
                                if selected:
                                    st.session_state[f"{panel_id}_local_dir"] = selected
                            image_dir = st.text_input("Panel image directory", key=f"{panel_id}_local_dir")
                            source_mode = st.radio(
                                "Panel activation source",
                                ["single image", "all images"],
                                key=f"{panel_id}_source_mode",
                                horizontal=True,
                            )
                            max_images = st.slider(
                                "Max images",
                                1,
                                256,
                                int(st.session_state.get(f"{panel_id}_max_images", 24)),
                                key=f"{panel_id}_max_images",
                            )
                            local_image_paths = [Path(p) for p in list_images_cached(image_dir)] if image_dir else []
                            if local_image_paths:
                                selected_image_name = st.selectbox(
                                    "Panel image",
                                    [p.name for p in local_image_paths],
                                    key=f"{panel_id}_selected_image",
                                )
                            else:
                                selected_image_name = None

                    bins = st.number_input(
                        "Bins",
                        min_value=5,
                        max_value=500,
                        value=int(st.session_state.get(f"{panel_id}_bins", default_bins)),
                        step=1,
                        key=f"{panel_id}_bins",
                    )

                    plot_placeholder = st.empty()
                    info_placeholder = st.empty()
                    error_placeholder = st.empty()

                    panel_specs.append(
                        PanelSpec(
                            panel_id=panel_id,
                            row=r,
                            col=c,
                            analysis=analysis,
                            model_name=model_name,
                            layer_name=layer_name,
                            bins=int(bins),
                            use_global_source=bool(panel_use_global),
                            image_dir=image_dir or "",
                            source_mode=source_mode,
                            selected_image_name=selected_image_name,
                            max_images=max_images,
                            title=title or f"Panel {r + 1},{c + 1}",
                        )
                    )
                    panel_placeholders.append((plot_placeholder, info_placeholder, error_placeholder))

    # ------------------------------------------------------------------
    # Resolve panel data first, then render with optional shared axes.
    # ------------------------------------------------------------------
    for panel in panel_specs:
        try:
            if panel.model_name not in used_models:
                used_models[panel.model_name] = load_bundle_cached(panel.model_name, pretrained, str(device), cornet_times)
            bundle = used_models[panel.model_name]

            if not panel.layer_name:
                raise AppError("Please choose a layer for this panel.")

            if panel.analysis == "weights":
                panel.data = get_layer_weight_values(bundle, panel.layer_name)
                panel.subtitle = f"{panel.model_name} · {panel.layer_name} · weights · n={panel.data.size:,}"
            else:
                image_paths = [Path(p) for p in list_images_cached(panel.image_dir)] if panel.image_dir else []
                if not image_paths:
                    raise AppError("No supported images were found for this activation panel.")

                if panel.source_mode == "single image":
                    if not panel.selected_image_name:
                        raise AppError("Please choose an image for this activation panel.")
                    chosen_matches = [p for p in image_paths if p.name == panel.selected_image_name]
                    if not chosen_matches:
                        raise AppError("The selected image was not found in the current directory.")
                    chosen = chosen_matches[0]
                    panel.data = get_single_image_activation_values(bundle, panel.layer_name, chosen)
                    panel.subtitle = (
                        f"{panel.model_name} · {panel.layer_name} · activations · image={chosen.name} · n={panel.data.size:,}"
                    )
                else:
                    chosen_paths = image_paths[: max(1, int(panel.max_images))]
                    panel.data = get_multi_image_activation_values(bundle, panel.layer_name, chosen_paths)
                    panel.subtitle = (
                        f"{panel.model_name} · {panel.layer_name} · pooled activations over {len(chosen_paths)} images · n={panel.data.size:,}"
                    )

            # precompute stats for shared ranges
            _, _, _, x_min, x_max, y_max = compute_histogram_stats(panel.data, panel.bins)
            panel.x_min = x_min
            panel.x_max = x_max
            panel.y_max = y_max
        except Exception as exc:
            panel.error = str(exc)

    sync_x_range: Optional[Tuple[float, float]] = None
    sync_y_max: Optional[float] = None
    valid_panels = [p for p in panel_specs if p.error is None and p.data is not None]
    if valid_panels and sync_x:
        sync_x_range = (
            min(float(p.x_min) for p in valid_panels if p.x_min is not None),
            max(float(p.x_max) for p in valid_panels if p.x_max is not None),
        )
    if valid_panels and sync_y:
        sync_y_max = max(float(p.y_max) for p in valid_panels if p.y_max is not None)

    for panel, placeholders in zip(panel_specs, panel_placeholders):
        plot_placeholder, info_placeholder, error_placeholder = placeholders
        if panel.error is not None:
            error_placeholder.error(panel.error)
            continue
        assert panel.data is not None
        fig = make_histogram_figure(panel, sync_x_range=sync_x_range, sync_y_max=sync_y_max)
        render_full_width(plot_placeholder.plotly_chart, fig)
        info_placeholder.caption(
            f"min={np.nanmin(panel.data):.5g} · max={np.nanmax(panel.data):.5g} · mean={np.nanmean(panel.data):.5g} · std={np.nanstd(panel.data):.5g}"
        )

    with st.expander("Notes"):
        st.markdown(
            "- Streamlit is sufficient here, so there was no need to switch packages.\n"
            "- Activation histograms flatten the selected layer output tensor.\n"
            "- 'All images' pools flattened activations across multiple images into one histogram.\n"
            "- Shared axes apply across all valid visible panels, including mixed weight and activation panels.\n"
            "- CORnet-RT requires the optional CORnet package to be installed."
        )


if __name__ == "__main__":
    main()
