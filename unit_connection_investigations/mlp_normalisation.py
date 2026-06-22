import io
import math
import time
import copy
import json
import uuid
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------
# Utility helpers
# -----------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


# -----------------------------
# Data generation
# -----------------------------

def make_cluster_dataset(
    n_samples: int,
    n_features: int,
    n_classes: int,
    class_spread: float,
    center_scale: float,
    anisotropy: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates clustered, noisy n-dimensional stimuli.
    Returns X, y, class_centers.
    """
    rng = np.random.default_rng(seed)

    # class centers: meaningful structure from clustered category manifolds
    centers = rng.normal(0.0, center_scale, size=(n_classes, n_features))

    # add a low-rank shared latent structure so categories are not trivially orthogonal
    latent_rank = max(2, min(8, n_features // 2))
    shared_basis = rng.normal(0.0, 1.0, size=(latent_rank, n_features))
    shared_basis /= np.linalg.norm(shared_basis, axis=1, keepdims=True) + 1e-8

    X_list = []
    y_list = []
    per_class = [n_samples // n_classes] * n_classes
    for i in range(n_samples % n_classes):
        per_class[i] += 1

    for c in range(n_classes):
        n_c = per_class[c]
        latent = rng.normal(0.0, 1.0, size=(n_c, latent_rank))
        latent_component = latent @ shared_basis

        feature_scales = rng.uniform(1.0 - anisotropy, 1.0 + anisotropy, size=(n_features,))
        noise = rng.normal(0.0, class_spread, size=(n_c, n_features)) * feature_scales[None, :]

        X_c = centers[c][None, :] + 0.45 * latent_component + noise
        X_list.append(X_c)
        y_list.append(np.full(n_c, c, dtype=np.int64))

    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list)

    # standardize features globally for more stable training
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-6)
    return X, y, centers.astype(np.float32)


# -----------------------------
# Block and model definitions
# -----------------------------

class IdentityLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class VectorMaxPool(nn.Module):
    def __init__(self, kernel_size: int = 2):
        super().__init__()
        self.kernel_size = kernel_size
        self.pool = nn.MaxPool1d(kernel_size=kernel_size, stride=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D] -> [B, 1, D]
        if x.shape[-1] < self.kernel_size:
            return x
        return self.pool(x.unsqueeze(1)).squeeze(1)


class VectorGroupNorm(nn.Module):
    def __init__(self, num_features: int, num_groups: int):
        super().__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_features, affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.unsqueeze(-1)).squeeze(-1)


class VectorLayerNorm(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.norm = nn.LayerNorm(num_features, elementwise_affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class TopKActivationSparsifier(nn.Module):
    def __init__(self, keep_frac: float = 1.0, apply_during_eval: bool = True):
        super().__init__()
        self.keep_frac = float(keep_frac)
        self.apply_during_eval = bool(apply_during_eval)

    def _should_apply(self) -> bool:
        return self.keep_frac < 0.999999 and (self.training or self.apply_during_eval)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self._should_apply()) or (x.shape[-1] <= 1):
            return x
        d = int(x.shape[-1])
        k = max(1, int(math.ceil(self.keep_frac * d)))
        if k >= d:
            return x
        topk_idx = torch.topk(x, k=k, dim=-1).indices
        mask = torch.zeros_like(x)
        src = torch.ones_like(topk_idx, dtype=x.dtype, device=x.device)
        mask.scatter_(dim=-1, index=topk_idx, src=src)
        return x * mask


class MLPBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        block_index: int,
        post_op: str = "layer_norm",
        pool_kernel: int = 2,
        num_groups: int = 2,
        activation: str = "relu",
        dropout: float = 0.0,
        activation_keep_frac: float = 1.0,
        apply_sparsity_eval: bool = True,
    ):
        super().__init__()
        self.block_index = block_index
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.post_op_name = post_op
        self.pool_kernel = pool_kernel
        self.num_groups = num_groups
        self.dropout_p = dropout
        self.activation_name = activation
        self.activation_keep_frac = float(activation_keep_frac)
        self.apply_sparsity_eval = bool(apply_sparsity_eval)

        self.linear = nn.Linear(in_dim, hidden_dim)
        self.activation = self._make_activation(activation)
        self.sparsifier = TopKActivationSparsifier(
            keep_frac=self.activation_keep_frac,
            apply_during_eval=self.apply_sparsity_eval,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else IdentityLayer()
        self.post = self._make_post(post_op, hidden_dim, pool_kernel, num_groups)
        self.out_dim = self._infer_out_dim(hidden_dim, post_op, pool_kernel)

    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        if name == "tanh":
            return nn.Tanh()
        raise ValueError(f"Unsupported activation: {name}")

    @staticmethod
    def _infer_out_dim(hidden_dim: int, post_op: str, pool_kernel: int) -> int:
        if post_op == "maxpool":
            return max(1, hidden_dim // pool_kernel)
        return hidden_dim

    @staticmethod
    def _make_post(post_op: str, hidden_dim: int, pool_kernel: int, num_groups: int) -> nn.Module:
        if post_op == "none":
            return IdentityLayer()
        if post_op == "layer_norm":
            return VectorLayerNorm(hidden_dim)
        if post_op == "maxpool":
            return VectorMaxPool(pool_kernel)
        if post_op == "group_norm":
            if hidden_dim % num_groups != 0:
                raise ValueError(
                    f"GroupNorm requires hidden_dim ({hidden_dim}) divisible by num_groups ({num_groups})."
                )
            return VectorGroupNorm(hidden_dim, num_groups)
        raise ValueError(f"Unsupported post op: {post_op}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.activation(x)
        x = self.sparsifier(x)
        x = self.dropout(x)
        x = self.post(x)
        return x


class ConfigurableMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, blocks_config: List[Dict[str, Any]]):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.blocks_config = copy.deepcopy(blocks_config)

        blocks = []
        current_dim = input_dim
        self.block_output_dims = []
        for i, cfg in enumerate(blocks_config):
            block = MLPBlock(
                in_dim=current_dim,
                hidden_dim=int(cfg["hidden_dim"]),
                block_index=i,
                post_op=str(cfg["post_op"]),
                pool_kernel=int(cfg.get("pool_kernel", 2)),
                num_groups=int(cfg.get("num_groups", 2)),
                activation=str(cfg.get("activation", "relu")),
                dropout=float(cfg.get("dropout", 0.0)),
                activation_keep_frac=float(cfg.get("activation_keep_frac", 1.0)),
                apply_sparsity_eval=bool(cfg.get("apply_sparsity_eval", True)),
            )
            blocks.append(block)
            current_dim = block.out_dim
            self.block_output_dims.append(current_dim)

        self.blocks = nn.ModuleList(blocks)
        self.classifier = nn.Linear(current_dim, output_dim)
        self.penultimate_dim = current_dim

    def forward(self, x: torch.Tensor, return_penultimate: bool = False):
        for block in self.blocks:
            x = block(x)
        penultimate = x
        logits = self.classifier(penultimate)
        if return_penultimate:
            return logits, penultimate
        return logits

    def forward_with_intermediates(self, x: torch.Tensor):
        block_outputs = []
        for block in self.blocks:
            x = block(x)
            block_outputs.append(x)
        penultimate = x
        logits = self.classifier(penultimate)
        return logits, penultimate, block_outputs

    def get_named_ablatable_layers(self) -> List[str]:
        names = []
        for i, _ in enumerate(self.blocks):
            names.append(f"block_{i}.linear")
        return names

    def get_layer_by_name(self, name: str) -> nn.Module:
        if name.startswith("block_") and name.endswith(".linear"):
            idx = int(name.split(".")[0].split("_")[1])
            return self.blocks[idx].linear
        raise KeyError(f"Unknown layer name: {name}")


# -----------------------------
# Representation metrics
# -----------------------------

def compute_pairwise_representation_correlations(reps: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """
    reps: [N, D]
    Computes sample-by-sample Pearson correlations in representation space.
    """
    reps = reps - reps.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(reps, axis=1, keepdims=True) + 1e-8
    reps = reps / denom
    corr = reps @ reps.T

    n = reps.shape[0]
    iu = np.triu_indices(n, k=1)
    corr_vals = corr[iu]
    same = labels[iu[0]] == labels[iu[1]]
    diff = ~same

    within = corr_vals[same]
    between = corr_vals[diff]
    separation = float(np.mean(within) - np.mean(between)) if len(within) and len(between) else np.nan

    return {
        "corr_matrix": corr,
        "within_values": within,
        "between_values": between,
        "within_mean": float(np.mean(within)) if len(within) else np.nan,
        "between_mean": float(np.mean(between)) if len(between) else np.nan,
        "separation": separation,
    }


def compute_activation_sparsity_diagnostics(
    model: ConfigurableMLP,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset = TensorDataset(torch.from_numpy(X).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    per_sample_rows = []
    per_unit_sum = [None for _ in model.blocks]
    per_unit_count = [0 for _ in model.blocks]

    with torch.no_grad():
        sample_offset = 0
        for (xb,) in loader:
            xb = xb.to(device)
            _, _, block_outputs = model.forward_with_intermediates(xb)
            batch_n = xb.shape[0]
            for block_idx, out in enumerate(block_outputs):
                zero_mask = (out == 0)
                sample_zero_frac = zero_mask.float().mean(dim=1)
                for i in range(batch_n):
                    per_sample_rows.append(
                        {
                            "block": f"block_{block_idx}",
                            "block_index": block_idx,
                            "sample_index": sample_offset + i,
                            "zero_fraction": float(sample_zero_frac[i].item()),
                            "active_fraction": float(1.0 - sample_zero_frac[i].item()),
                            "n_units": int(out.shape[1]),
                        }
                    )
                unit_zero_prob = zero_mask.float().sum(dim=0).detach().cpu().numpy()
                if per_unit_sum[block_idx] is None:
                    per_unit_sum[block_idx] = unit_zero_prob.astype(np.float64)
                else:
                    per_unit_sum[block_idx] += unit_zero_prob.astype(np.float64)
                per_unit_count[block_idx] += batch_n
            sample_offset += batch_n

    per_sample_df = pd.DataFrame(per_sample_rows)

    unit_rows = []
    summary_rows = []
    for block_idx, block in enumerate(model.blocks):
        if per_unit_sum[block_idx] is None or per_unit_count[block_idx] == 0:
            continue
        zero_prob = per_unit_sum[block_idx] / max(per_unit_count[block_idx], 1)
        keep_frac_cfg = float(getattr(block, "activation_keep_frac", 1.0))
        sparsity_pct_cfg = float((1.0 - keep_frac_cfg) * 100.0)
        apply_eval_cfg = bool(getattr(block, "apply_sparsity_eval", True))
        block_name = f"block_{block_idx}"
        for unit_idx, zp in enumerate(zero_prob):
            unit_rows.append(
                {
                    "block": block_name,
                    "block_index": block_idx,
                    "unit_index": int(unit_idx),
                    "unit_zero_probability": float(zp),
                    "unit_active_probability": float(1.0 - zp),
                    "configured_keep_fraction": keep_frac_cfg,
                    "configured_sparsity_pct": sparsity_pct_cfg,
                    "apply_sparsity_eval": apply_eval_cfg,
                }
            )
        block_samples = per_sample_df[per_sample_df["block_index"] == block_idx]
        summary_rows.append(
            {
                "block": block_name,
                "block_index": block_idx,
                "n_units": int(block.out_dim),
                "configured_keep_fraction": keep_frac_cfg,
                "configured_sparsity_pct": sparsity_pct_cfg,
                "apply_sparsity_eval": apply_eval_cfg,
                "mean_zero_fraction": float(block_samples["zero_fraction"].mean()),
                "mean_active_fraction": float(block_samples["active_fraction"].mean()),
                "std_zero_fraction": float(block_samples["zero_fraction"].std(ddof=0)),
                "mean_unit_zero_probability": float(np.mean(zero_prob)),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    unit_df = pd.DataFrame(unit_rows)
    return summary_df, per_sample_df, unit_df


# -----------------------------
# Ablation helpers
# -----------------------------

def get_penultimate_representations(
    model: ConfigurableMLP,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
    hook_fn=None,
) -> np.ndarray:
    dataset = TensorDataset(torch.from_numpy(X).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    reps = []
    handle = None
    if hook_fn is not None:
        handle = hook_fn()
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            _, pen = model(xb, return_penultimate=True)
            reps.append(to_numpy(pen))
    if handle is not None:
        handle.remove()
    return np.concatenate(reps, axis=0)


class ActivationUnitAblator:
    def __init__(self, layer: nn.Module, fraction: float, seed: int):
        self.layer = layer
        self.fraction = fraction
        self.seed = seed
        self.handle = None
        self.mask = None

    def __call__(self):
        out_dim = self.layer.out_features
        rng = np.random.default_rng(self.seed)
        n_ablate = int(round(self.fraction * out_dim))
        idx = rng.choice(out_dim, size=n_ablate, replace=False) if n_ablate > 0 else np.array([], dtype=int)
        mask = torch.ones(out_dim, dtype=torch.float32)
        if len(idx) > 0:
            mask[idx] = 0.0
        self.mask = mask

        def hook(_module, _inp, out):
            local_mask = self.mask.to(out.device)
            return out * local_mask

        self.handle = self.layer.register_forward_hook(hook)
        return self.handle


class WeightAblator:
    def __init__(self, layer: nn.Linear, fraction: float, seed: int):
        self.layer = layer
        self.fraction = fraction
        self.seed = seed
        self.original_weight = None

    def __call__(self):
        rng = np.random.default_rng(self.seed)
        w = self.layer.weight.data
        self.original_weight = w.clone()
        numel = w.numel()
        n_ablate = int(round(self.fraction * numel))
        if n_ablate > 0:
            flat_idx = rng.choice(numel, size=n_ablate, replace=False)
            flat = w.view(-1)
            flat[torch.from_numpy(flat_idx).to(flat.device)] = 0.0
        return self

    def remove(self):
        if self.original_weight is not None:
            self.layer.weight.data.copy_(self.original_weight)


# -----------------------------
# Training helpers
# -----------------------------

def evaluate_model(
    model: ConfigurableMLP,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.shape[0]
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total += xb.shape[0]
    return {
        "loss": total_loss / max(total, 1),
        "acc": total_correct / max(total, 1),
    }


def train_model_live(
    model: ConfigurableMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    min_delta: float,
    update_every: int,
    chart_placeholder,
    metric_placeholder,
) -> Tuple[ConfigurableMLP, pd.DataFrame, Dict[str, Any]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history = []
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    wait = 0

    start = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.shape[0]
            train_correct += (logits.argmax(dim=1) == yb).sum().item()
            train_total += xb.shape[0]

        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        val_metrics = evaluate_model(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
        }
        history.append(row)
        hist_df = pd.DataFrame(history)

        if (epoch == 1) or (epoch % update_every == 0) or (epoch == max_epochs):
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Accuracy"))
            fig.add_trace(
                go.Scatter(x=hist_df["epoch"], y=hist_df["train_loss"], mode="lines", name="Training loss", legendgroup="Training loss"),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=hist_df["epoch"], y=hist_df["val_loss"], mode="lines", name="Validation loss", legendgroup="Validation loss"),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=hist_df["epoch"], y=hist_df["train_acc"], mode="lines", name="Training accuracy", legendgroup="Training accuracy"),
                row=1,
                col=2,
            )
            fig.add_trace(
                go.Scatter(x=hist_df["epoch"], y=hist_df["val_acc"], mode="lines", name="Validation accuracy", legendgroup="Validation accuracy"),
                row=1,
                col=2,
            )
            fig.update_xaxes(title_text="Epoch", row=1, col=1)
            fig.update_xaxes(title_text="Epoch", row=1, col=2)
            fig.update_yaxes(title_text="Cross-entropy loss", row=1, col=1)
            fig.update_yaxes(title_text="Classification accuracy", row=1, col=2)
            fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20), legend_title_text="Training curve")
            chart_placeholder.plotly_chart(fig, use_container_width=True)
            metric_placeholder.markdown(
                f"**Epoch {epoch}**  \\n                Train loss: `{train_loss:.4f}` · Train acc: `{train_acc:.3f}`  \\n                Val loss: `{val_metrics['loss']:.4f}` · Val acc: `{val_metrics['acc']:.3f}`"
            )

        if val_metrics["loss"] < best_val_loss - min_delta:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    elapsed = time.time() - start
    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "elapsed_sec": elapsed,
        "stopped_epoch": history[-1]["epoch"],
    }
    return model, pd.DataFrame(history), summary


# -----------------------------
# Serialization
# -----------------------------

def build_model_bundle(
    name: str,
    model: ConfigurableMLP,
    data_config: Dict[str, Any],
    blocks_config: List[Dict[str, Any]],
    training_config: Dict[str, Any],
    history_df: pd.DataFrame,
    split_data: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "name": name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "state_dict": copy.deepcopy(model.state_dict()),
        "input_dim": model.input_dim,
        "output_dim": model.output_dim,
        "blocks_config": copy.deepcopy(blocks_config),
        "data_config": copy.deepcopy(data_config),
        "training_config": copy.deepcopy(training_config),
        "history": history_df.to_dict(orient="records"),
        "split_data": {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in split_data.items()},
    }


def bundle_to_bytes(bundle: Dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    torch.save(bundle, buffer)
    return buffer.getvalue()


def bytes_to_bundle(raw: bytes) -> Dict[str, Any]:
    buffer = io.BytesIO(raw)
    return torch.load(buffer, map_location="cpu")


def instantiate_model_from_bundle(bundle: Dict[str, Any], device: torch.device) -> ConfigurableMLP:
    model = ConfigurableMLP(
        input_dim=int(bundle["input_dim"]),
        output_dim=int(bundle["output_dim"]),
        blocks_config=bundle["blocks_config"],
    )
    model.load_state_dict(bundle["state_dict"])
    model.to(device)
    model.eval()
    return model


# -----------------------------
# Plotting helpers
# -----------------------------

def render_architecture_graph(input_dim: int, blocks_config: List[Dict[str, Any]], output_dim: int):
    lines = ["digraph G {", "rankdir=LR;", 'node [shape=record, style="rounded,filled", fillcolor="lightgray"];']
    lines.append(f'input [label="Input|dim={input_dim}"];')
    prev = "input"
    curr_dim = input_dim
    for i, cfg in enumerate(blocks_config):
        hidden = int(cfg["hidden_dim"])
        post = cfg["post_op"]
        pool = int(cfg.get("pool_kernel", 2))
        out_dim = hidden if post != "maxpool" else max(1, hidden // pool)
        label = (
            f"Block {i}|Linear {curr_dim}->{hidden}|Act={cfg.get('activation','relu')}|"
            f"Post={post}|Out={out_dim}"
        )
        name = f"b{i}"
        lines.append(f'{name} [label="{label}"];')
        lines.append(f"{prev} -> {name};")
        prev = name
        curr_dim = out_dim
    lines.append(f'out [label="Classifier|{curr_dim}->{output_dim}", fillcolor="lightblue"];')
    lines.append(f"{prev} -> out;")
    lines.append("}")
    dot = "\n".join(lines)
    try:
        st.graphviz_chart(dot, use_container_width=True)
    except Exception:
        st.code(dot, language="dot")


def plot_dataset_projection(X: np.ndarray, y: np.ndarray):
    pca = PCA(n_components=min(3, X.shape[1]))
    emb = pca.fit_transform(X)
    if emb.shape[1] == 1:
        df = pd.DataFrame({"pc1": emb[:, 0], "class": y.astype(str)})
        fig = px.histogram(
            df,
            x="pc1",
            color="class",
            barmode="overlay",
            labels={"pc1": "PC 1 score", "class": "Stimulus class"},
        )
    elif emb.shape[1] == 2:
        df = pd.DataFrame({"pc1": emb[:, 0], "pc2": emb[:, 1], "class": y.astype(str)})
        fig = px.scatter(
            df,
            x="pc1",
            y="pc2",
            color="class",
            opacity=0.75,
            labels={"pc1": "PC 1 score", "pc2": "PC 2 score", "class": "Stimulus class"},
        )
    else:
        df = pd.DataFrame({"pc1": emb[:, 0], "pc2": emb[:, 1], "pc3": emb[:, 2], "class": y.astype(str)})
        fig = px.scatter_3d(
            df,
            x="pc1",
            y="pc2",
            z="pc3",
            color="class",
            opacity=0.75,
            labels={"pc1": "PC 1 score", "pc2": "PC 2 score", "pc3": "PC 3 score", "class": "Stimulus class"},
        )
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=30, b=20), legend_title_text="Stimulus class")
    st.plotly_chart(fig, use_container_width=True)


def plot_correlation_matrix(corr: np.ndarray, labels: np.ndarray, title: str):
    order = np.argsort(labels)
    corr_sorted = corr[order][:, order]
    fig = px.imshow(corr_sorted, aspect="auto", origin="lower", title=title)
    fig.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)


def plot_within_between_distributions(within: np.ndarray, between: np.ndarray, title: str):
    max_n = 4000
    rng = np.random.default_rng(0)
    within_show = within if len(within) <= max_n else rng.choice(within, size=max_n, replace=False)
    between_show = between if len(between) <= max_n else rng.choice(between, size=max_n, replace=False)
    df = pd.DataFrame(
        {
            "correlation": np.concatenate([within_show, between_show]),
            "pair_type": ["within"] * len(within_show) + ["between"] * len(between_show),
        }
    )
    fig = px.histogram(
        df,
        x="correlation",
        color="pair_type",
        barmode="overlay",
        nbins=50,
        title=title,
        labels={"correlation": "Pairwise representation correlation", "pair_type": "Pair type"},
    )
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20), legend_title_text="Pair type")
    st.plotly_chart(fig, use_container_width=True)


def plot_activation_sparsity_summary(summary_df: pd.DataFrame):
    if summary_df.empty:
        st.info("No activation sparsity summary available.")
        return
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Mean zero fraction by block", "Configured sparsity by block"))
    ordered = summary_df.sort_values("block_index")
    fig.add_trace(
        go.Bar(x=ordered["block"], y=ordered["mean_zero_fraction"], name="Observed activation zero fraction"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=ordered["block"], y=ordered["configured_sparsity_pct"], name="Configured top-k sparsity target (%)"),
        row=1,
        col=2,
    )
    fig.update_yaxes(title_text="Fraction", row=1, col=1)
    fig.update_yaxes(title_text="Percent", row=1, col=2)
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20), legend_title_text="Summary metric")
    st.plotly_chart(fig, use_container_width=True)


def plot_activation_sparsity_distributions(per_sample_df: pd.DataFrame):
    if per_sample_df.empty:
        st.info("No per-sample activation sparsity data available.")
        return
    fig = px.box(
        per_sample_df,
        x="block",
        y="zero_fraction",
        color="block",
        points=False,
        title="Per-sample activation zero fraction by block",
        labels={"block": "Block", "zero_fraction": "Per-sample zero fraction"},
    )
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20), legend_title_text="Block")
    st.plotly_chart(fig, use_container_width=True)


def plot_unit_zero_probabilities(unit_df: pd.DataFrame, block_name: str):
    sub = unit_df[unit_df["block"] == block_name].sort_values("unit_index")
    if sub.empty:
        st.info("No per-unit sparsity data available for this block.")
        return
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Unit zero probability", "Distribution across units"))
    fig.add_trace(
        go.Scatter(
            x=sub["unit_index"],
            y=sub["unit_zero_probability"],
            mode="lines+markers",
            name=f"{block_name} · unit zero probability",
            legendgroup="unit_zero_probability",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(
            x=sub["unit_zero_probability"],
            nbinsx=30,
            name=f"{block_name} · zero-probability histogram",
            legendgroup="unit_zero_probability",
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Unit index", row=1, col=1)
    fig.update_xaxes(title_text="Unit zero probability", row=1, col=2)
    fig.update_yaxes(title_text="Zero probability", row=1, col=1)
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20), legend_title_text="Trace")
    st.plotly_chart(fig, use_container_width=True)


def plot_ablation_results(df: pd.DataFrame):
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Separation", "Within-class corr", "Between-class corr"))
    for metric, col in [("separation", 1), ("within_mean", 2), ("between_mean", 3)]:
        for name in sorted(df["condition"].unique()):
            sub = df[df["condition"] == name].sort_values("fraction")
            fig.add_trace(
                go.Scatter(
                    x=sub["fraction"],
                    y=sub[metric],
                    mode="lines+markers",
                    name=name,
                    legendgroup=name,
                    showlegend=(col == 1),
                    hovertemplate=(
                        "Model/layer: %{fullData.name}<br>"
                        "Ablation fraction: %{x:.3f}<br>"
                        f"Metric value ({metric}): %{{y:.4f}}<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )
    fig.update_xaxes(title_text="Ablation fraction", row=1, col=1)
    fig.update_xaxes(title_text="Ablation fraction", row=1, col=2)
    fig.update_xaxes(title_text="Ablation fraction", row=1, col=3)
    fig.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20), legend_title_text="Model · ablated layer")
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# App state init
# -----------------------------

def init_session_state():
    defaults = {
        "saved_models": [],
        "latest_bundle": None,
        "latest_history": None,
        "latest_train_summary": None,
        "latest_data_preview": None,
        "latest_blocks_config": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# -----------------------------
# Main app
# -----------------------------

def main():
    st.set_page_config(page_title="MLP Representation & Ablation Lab", layout="wide")
    init_session_state()

    st.title("MLP Representation & Ablation Lab")
    st.markdown(
        "Generate clustered n-dimensional stimuli, build a configurable MLP with per-block post-operations, "
        "train to asymptotic performance, inspect penultimate-layer geometry and layer sparsity, and test random unit or weight ablations."
    )

    with st.sidebar:
        st.header("Run settings")
        seed = st.number_input("Random seed", min_value=0, max_value=100000, value=7, step=1)
        prefer_gpu = st.checkbox("Prefer GPU / MPS if available", value=True)
        device = get_device(prefer_gpu=prefer_gpu)
        st.info(f"Device: {device}")
        set_seed(int(seed))

    tabs = st.tabs([
        "1) Data + model builder",
        "2) Train + save",
        "3) Saved models",
        "4) Representation analysis",
        "5) Ablation study",
    ])

    # ------------------------------------------------------------------
    # Tab 1: Data + model builder
    # ------------------------------------------------------------------
    with tabs[0]:
        st.subheader("Dataset generation")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            n_samples = st.number_input("Total stimuli", min_value=60, max_value=20000, value=900, step=60)
            n_features = st.number_input("Stimulus dimensionality", min_value=2, max_value=2048, value=32, step=1)
        with c2:
            n_classes = st.number_input("Number of categories / output units", min_value=3, max_value=20, value=3, step=1)
            class_spread = st.slider("Within-class noise", min_value=0.05, max_value=3.0, value=0.85, step=0.05)
        with c3:
            center_scale = st.slider("Class center separation", min_value=0.25, max_value=6.0, value=2.0, step=0.05)
            anisotropy = st.slider("Cluster anisotropy", min_value=0.0, max_value=1.0, value=0.30, step=0.05)
        with c4:
            test_size = st.slider("Validation fraction", min_value=0.05, max_value=0.5, value=0.25, step=0.05)
            batch_size = st.select_slider("Batch size", options=[16, 32, 64, 128, 256, 512], value=64)

        X, y, centers = make_cluster_dataset(
            n_samples=int(n_samples),
            n_features=int(n_features),
            n_classes=int(n_classes),
            class_spread=float(class_spread),
            center_scale=float(center_scale),
            anisotropy=float(anisotropy),
            seed=int(seed),
        )
        st.session_state["latest_data_preview"] = {"X": X, "y": y}

        st.markdown("**Stimulus projection (PCA)**")
        plot_dataset_projection(X, y)

        st.subheader("Model builder")
        n_blocks = st.number_input("Number of blocks", min_value=1, max_value=8, value=3, step=1)
        blocks_config = []
        curr_dim = int(n_features)
        for i in range(int(n_blocks)):
            st.markdown(f"**Block {i}**")
            b1, b2, b3, b4, b5, b6 = st.columns(6)
            with b1:
                hidden_dim = st.number_input(
                    f"Hidden dim · block {i}",
                    min_value=2,
                    max_value=4096,
                    value=max(8, curr_dim if i == 0 else min(curr_dim * 2, 128)),
                    step=1,
                    key=f"hidden_dim_{i}",
                )
            with b2:
                activation = st.selectbox(
                    f"Activation · block {i}",
                    options=["relu", "gelu", "tanh"],
                    index=0,
                    key=f"activation_{i}",
                )
            with b3:
                post_op = st.selectbox(
                    f"Post-op · block {i}",
                    options=["layer_norm", "maxpool", "group_norm", "none"],
                    index=0 if i < int(n_blocks) - 1 else 1,
                    key=f"post_op_{i}",
                )
            with b4:
                pool_kernel = st.number_input(
                    f"Pool kernel · block {i}",
                    min_value=2,
                    max_value=8,
                    value=2,
                    step=1,
                    key=f"pool_kernel_{i}",
                    disabled=(post_op != "maxpool"),
                )
            with b5:
                num_groups = st.number_input(
                    f"Groups · block {i}",
                    min_value=1,
                    max_value=64,
                    value=2,
                    step=1,
                    key=f"num_groups_{i}",
                    disabled=(post_op != "group_norm"),
                )
            with b6:
                activation_sparsity_pct = st.slider(
                    f"Activation sparsity % · block {i}",
                    min_value=0,
                    max_value=99,
                    value=0,
                    step=1,
                    key=f"activation_sparsity_pct_{i}",
                    help="Bottom n% of post-activation units are zeroed per sample. Top activations are kept.",
                )
            s1, s2 = st.columns(2)
            with s1:
                dropout = st.slider(
                    f"Dropout · block {i}",
                    min_value=0.0,
                    max_value=0.9,
                    value=0.0,
                    step=0.05,
                    key=f"dropout_{i}",
                )
            with s2:
                apply_sparsity_eval = st.checkbox(
                    f"Apply sparsity at eval / analysis · block {i}",
                    value=True,
                    key=f"apply_sparsity_eval_{i}",
                    help="When enabled, the same top-k activation sparsity rule is also used during evaluation and representation analysis.",
                )
            activation_keep_frac = max(0.01, 1.0 - (float(activation_sparsity_pct) / 100.0))
            blocks_config.append(
                {
                    "hidden_dim": int(hidden_dim),
                    "activation": activation,
                    "post_op": post_op,
                    "pool_kernel": int(pool_kernel),
                    "num_groups": int(num_groups),
                    "dropout": float(dropout),
                    "activation_sparsity_pct": int(activation_sparsity_pct),
                    "activation_keep_frac": float(activation_keep_frac),
                    "apply_sparsity_eval": bool(apply_sparsity_eval),
                }
            )
            curr_dim = int(hidden_dim) if post_op != "maxpool" else max(1, int(hidden_dim) // int(pool_kernel))

        st.session_state["latest_blocks_config"] = blocks_config

        valid = True
        validation_messages = []
        temp_dim = int(n_features)
        for i, cfg in enumerate(blocks_config):
            h = int(cfg["hidden_dim"])
            if cfg["post_op"] == "group_norm" and h % int(cfg["num_groups"]) != 0:
                valid = False
                validation_messages.append(f"Block {i}: hidden_dim must be divisible by num_groups for GroupNorm.")
            if cfg["post_op"] == "maxpool" and h < int(cfg["pool_kernel"]):
                valid = False
                validation_messages.append(f"Block {i}: hidden_dim must be >= pool kernel for MaxPool.")
            temp_dim = h if cfg["post_op"] != "maxpool" else max(1, h // int(cfg["pool_kernel"]))

        if valid:
            st.success(f"Model is valid. Penultimate dimension = {temp_dim}")
        else:
            for msg in validation_messages:
                st.error(msg)

        st.markdown("**Architecture preview**")
        render_architecture_graph(int(n_features), blocks_config, int(n_classes))

        summary_rows = []
        prev_dim = int(n_features)
        for i, cfg in enumerate(blocks_config):
            out_dim = int(cfg["hidden_dim"]) if cfg["post_op"] != "maxpool" else max(1, int(cfg["hidden_dim"]) // int(cfg["pool_kernel"]))
            summary_rows.append(
                {
                    "block": i,
                    "input_dim": prev_dim,
                    "hidden_dim": int(cfg["hidden_dim"]),
                    "activation": cfg["activation"],
                    "post_op": cfg["post_op"],
                    "post_op_arg": int(cfg["pool_kernel"]) if cfg["post_op"] == "maxpool" else int(cfg["num_groups"]),
                    "dropout": float(cfg["dropout"]),
                    "activation_sparsity_pct": int(cfg.get("activation_sparsity_pct", round((1.0 - float(cfg.get("activation_keep_frac", 1.0))) * 100))),
                    "apply_sparsity_eval": bool(cfg.get("apply_sparsity_eval", True)),
                    "output_dim": out_dim,
                }
            )
            prev_dim = out_dim
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # Tab 2: Train + save
    # ------------------------------------------------------------------
    with tabs[1]:
        st.subheader("Training")
        if st.session_state["latest_blocks_config"] is None:
            st.warning("Build a model in tab 1 first.")
        else:
            t1, t2, t3, t4, t5 = st.columns(5)
            with t1:
                learning_rate = st.number_input("Learning rate", min_value=1e-5, max_value=1.0, value=1e-3, step=1e-4, format="%.5f")
            with t2:
                weight_decay = st.number_input("Weight decay", min_value=0.0, max_value=1.0, value=1e-4, step=1e-4, format="%.5f")
            with t3:
                max_epochs = st.number_input("Max epochs", min_value=5, max_value=5000, value=300, step=5)
            with t4:
                patience = st.number_input("Early-stop patience", min_value=2, max_value=500, value=35, step=1)
            with t5:
                min_delta = st.number_input("Min val-loss improvement", min_value=0.0, max_value=1.0, value=1e-4, step=1e-4, format="%.5f")

            update_every = st.select_slider("Live chart update every N epochs", options=[1, 2, 5, 10, 20, 25, 50], value=5)
            if "draft_model_name" not in st.session_state:
                st.session_state["draft_model_name"] = f"mlp_run_{time.strftime('%Y%m%d_%H%M%S')}"
            st.text_input("Model name", key="draft_model_name")

            if st.button("Train model", type="primary", disabled=not valid):
                model_name = str(st.session_state.get("draft_model_name", "")).strip()
                if model_name == "":
                    model_name = f"mlp_run_{time.strftime('%Y%m%d_%H%M%S')}"
                    st.session_state["draft_model_name"] = model_name
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=float(test_size), random_state=int(seed), stratify=y
                )

                train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
                val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
                train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True)
                val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False)

                model = ConfigurableMLP(int(n_features), int(n_classes), blocks_config).to(device)
                chart_placeholder = st.empty()
                metric_placeholder = st.empty()

                trained_model, history_df, train_summary = train_model_live(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    learning_rate=float(learning_rate),
                    weight_decay=float(weight_decay),
                    max_epochs=int(max_epochs),
                    patience=int(patience),
                    min_delta=float(min_delta),
                    update_every=int(update_every),
                    chart_placeholder=chart_placeholder,
                    metric_placeholder=metric_placeholder,
                )

                criterion = nn.CrossEntropyLoss()
                final_train = evaluate_model(trained_model, train_loader, criterion, device)
                final_val = evaluate_model(trained_model, val_loader, criterion, device)

                data_config = {
                    "n_samples": int(n_samples),
                    "n_features": int(n_features),
                    "n_classes": int(n_classes),
                    "class_spread": float(class_spread),
                    "center_scale": float(center_scale),
                    "anisotropy": float(anisotropy),
                    "seed": int(seed),
                    "test_size": float(test_size),
                    "batch_size": int(batch_size),
                }
                training_config = {
                    "learning_rate": float(learning_rate),
                    "weight_decay": float(weight_decay),
                    "max_epochs": int(max_epochs),
                    "patience": int(patience),
                    "min_delta": float(min_delta),
                }
                split_data = {
                    "X_train": X_train,
                    "y_train": y_train,
                    "X_val": X_val,
                    "y_val": y_val,
                }
                bundle = build_model_bundle(
                    name=model_name,
                    model=trained_model,
                    data_config=data_config,
                    blocks_config=blocks_config,
                    training_config=training_config,
                    history_df=history_df,
                    split_data=split_data,
                )
                st.session_state["latest_bundle"] = bundle
                st.session_state["latest_history"] = history_df
                st.session_state["latest_train_summary"] = {
                    **train_summary,
                    "final_train_acc": final_train["acc"],
                    "final_val_acc": final_val["acc"],
                    "final_train_loss": final_train["loss"],
                    "final_val_loss": final_val["loss"],
                }
                st.session_state["saved_models"].append(bundle)

                st.success("Training complete and model saved to the current session.")
                st.json(st.session_state["latest_train_summary"])

            if st.session_state["latest_bundle"] is not None:
                raw = bundle_to_bytes(st.session_state["latest_bundle"])
                st.download_button(
                    "Download latest model bundle",
                    data=raw,
                    file_name=f"{st.session_state['latest_bundle']['name']}.pt",
                    mime="application/octet-stream",
                )

    # ------------------------------------------------------------------
    # Tab 3: Saved models
    # ------------------------------------------------------------------
    with tabs[2]:
        st.subheader("Saved model instances")
        uploaded = st.file_uploader("Upload a saved model bundle (.pt)", type=["pt", "pth"])
        if uploaded is not None:
            try:
                bundle = bytes_to_bundle(uploaded.read())
                st.session_state["saved_models"].append(bundle)
                st.success(f"Loaded bundle: {bundle['name']}")
            except Exception as e:
                st.error(f"Could not load bundle: {e}")

        if len(st.session_state["saved_models"]) == 0:
            st.info("No saved models yet. Train one in tab 2 or upload a bundle here.")
        else:
            saved_rows = []
            for b in st.session_state["saved_models"]:
                hist = pd.DataFrame(b.get("history", []))
                final_val_acc = float(hist["val_acc"].iloc[-1]) if not hist.empty else np.nan
                saved_rows.append(
                    {
                        "name": b["name"],
                        "timestamp": b["timestamp"],
                        "input_dim": b["input_dim"],
                        "output_dim": b["output_dim"],
                        "n_blocks": len(b["blocks_config"]),
                        "final_val_acc": final_val_acc,
                        "id": b["id"],
                    }
                )
            st.dataframe(pd.DataFrame(saved_rows), use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # Tab 4: Representation analysis
    # ------------------------------------------------------------------
    with tabs[3]:
        st.subheader("Penultimate-layer / final-normalisation-layer representation analysis")
        if len(st.session_state["saved_models"]) == 0:
            st.info("Train or load a model first.")
        else:
            options = [f"{i}: {b['name']} ({b['timestamp']})" for i, b in enumerate(st.session_state["saved_models"])]
            selected_text = st.selectbox("Select model bundle", options=options, key="repr_bundle_select")
            selected_idx = int(selected_text.split(":", 1)[0])
            bundle = st.session_state["saved_models"][selected_idx]
            split_name = st.selectbox("Which stimuli set?", options=["validation", "train"], index=0)
            use_key = "X_val" if split_name == "validation" else "X_train"
            label_key = "y_val" if split_name == "validation" else "y_train"

            model = instantiate_model_from_bundle(bundle, device=device)
            X_use = bundle["split_data"][use_key].astype(np.float32)
            y_use = bundle["split_data"][label_key].astype(np.int64)
            reps = get_penultimate_representations(model, X_use, batch_size=int(bundle["data_config"]["batch_size"]), device=device)
            metrics = compute_pairwise_representation_correlations(reps, y_use)

            a, b, c = st.columns(3)
            a.metric("Within-class mean corr", f"{metrics['within_mean']:.4f}")
            b.metric("Between-class mean corr", f"{metrics['between_mean']:.4f}")
            c.metric("Within - between", f"{metrics['separation']:.4f}")

            plot_correlation_matrix(metrics["corr_matrix"], y_use, title="Sample-by-sample penultimate-layer correlation matrix")
            plot_within_between_distributions(
                metrics["within_values"],
                metrics["between_values"],
                title="Within- vs between-class representation correlations",
            )

            pca = PCA(n_components=min(3, reps.shape[1]))
            emb = pca.fit_transform(reps)
            if emb.shape[1] >= 3:
                df = pd.DataFrame({"pc1": emb[:, 0], "pc2": emb[:, 1], "pc3": emb[:, 2], "class": y_use.astype(str)})
                fig = px.scatter_3d(df, x="pc1", y="pc2", z="pc3", color="class", title="Penultimate-layer PCA")
            else:
                cols = {"pc1": emb[:, 0], "pc2": emb[:, 1] if emb.shape[1] > 1 else np.zeros_like(emb[:, 0]), "class": y_use.astype(str)}
                df = pd.DataFrame(cols)
                fig = px.scatter(df, x="pc1", y="pc2", color="class", title="Penultimate-layer PCA")
            fig.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20), legend_title_text="Model · ablated layer")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Activation sparsity diagnostics**")
            sparsity_summary_df, sparsity_per_sample_df, sparsity_unit_df = compute_activation_sparsity_diagnostics(
                model,
                X_use,
                batch_size=int(bundle["data_config"]["batch_size"]),
                device=device,
            )
            if sparsity_summary_df.empty:
                st.info("No activation sparsity diagnostics available for this model.")
            else:
                st.dataframe(sparsity_summary_df, use_container_width=True, hide_index=True)
                plot_activation_sparsity_summary(sparsity_summary_df)
                plot_activation_sparsity_distributions(sparsity_per_sample_df)
                block_options = list(sparsity_summary_df.sort_values("block_index")["block"])
                selected_block_name = st.selectbox(
                    "Block for per-unit sparsity view",
                    options=block_options,
                    key=f"unit_sparsity_block_{bundle['id']}_{split_name}",
                )
                plot_unit_zero_probabilities(sparsity_unit_df, selected_block_name)

    # ------------------------------------------------------------------
    # Tab 5: Ablation study
    # ------------------------------------------------------------------
    with tabs[4]:
        st.subheader("Random unit / weight ablation and representational change")
        if len(st.session_state["saved_models"]) == 0:
            st.info("Train or load a model first.")
        else:
            options = [f"{i}: {b['name']} ({b['timestamp']})" for i, b in enumerate(st.session_state["saved_models"])]
            default_selection = [options[-1]] if options else []
            selected_texts = st.multiselect(
                "Select model bundle(s)",
                options=options,
                default=default_selection,
                key="ablation_bundle_select_multi",
            )

            if len(selected_texts) == 0:
                st.info("Select at least one saved model bundle to run the ablation study.")
            else:
                selected_indices = [int(txt.split(":", 1)[0]) for txt in selected_texts]
                selected_bundles = [st.session_state["saved_models"][idx] for idx in selected_indices]

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    ablation_mode = st.selectbox("Ablation mode", options=["unit", "weight"])
                with c2:
                    split_name = st.selectbox("Stimuli set", options=["validation", "train"], index=0, key="ablation_split")
                with c3:
                    repeats = st.number_input("Random repeats per fraction", min_value=1, max_value=50, value=5, step=1)
                with c4:
                    st.markdown(f"**Selected models:** {len(selected_bundles)}")

                frac_text = st.text_input("Fractions (comma separated)", value="0.0,0.1,0.2,0.4,0.6,0.8")
                fractions = []
                fraction_parse_error = None
                for item in frac_text.split(","):
                    item = item.strip()
                    if item:
                        try:
                            fractions.append(float(item))
                        except ValueError:
                            fraction_parse_error = item
                            break
                if fraction_parse_error is not None:
                    st.error(f"Could not parse ablation fraction: '{fraction_parse_error}'")
                    fractions = []
                fractions = [min(max(f, 0.0), 1.0) for f in fractions]
                fractions = sorted(set(fractions))

                st.markdown("**Per-model layer selection**")
                layer_selection_rows = []
                layer_name_by_bundle_id = {}
                for bundle in selected_bundles:
                    preview_model = instantiate_model_from_bundle(bundle, device=device)
                    ablation_layers = preview_model.get_named_ablatable_layers()
                    layer_name = st.selectbox(
                        f"Hidden layer to ablate · {bundle['name']}",
                        options=ablation_layers,
                        key=f"ablation_layer_{bundle['id']}",
                    )
                    layer_name_by_bundle_id[bundle["id"]] = layer_name
                    layer_selection_rows.append(
                        {
                            "model_name": bundle["name"],
                            "layer_name": layer_name,
                            "n_blocks": len(bundle["blocks_config"]),
                            "input_dim": bundle["input_dim"],
                            "output_dim": bundle["output_dim"],
                        }
                    )
                st.dataframe(pd.DataFrame(layer_selection_rows), use_container_width=True, hide_index=True)

                use_key = "X_val" if split_name == "validation" else "X_train"
                label_key = "y_val" if split_name == "validation" else "y_train"
                baseline_rows = []
                for bundle in selected_bundles:
                    preview_model = instantiate_model_from_bundle(bundle, device=device)
                    X_use = bundle["split_data"][use_key].astype(np.float32)
                    y_use = bundle["split_data"][label_key].astype(np.int64)
                    base_batch_size = int(bundle["data_config"]["batch_size"])
                    baseline_reps = get_penultimate_representations(
                        preview_model,
                        X_use,
                        batch_size=base_batch_size,
                        device=device,
                    )
                    baseline_metrics = compute_pairwise_representation_correlations(baseline_reps, y_use)
                    baseline_rows.append(
                        {
                            "model_name": bundle["name"],
                            "layer_name": layer_name_by_bundle_id[bundle["id"]],
                            "n_stimuli": int(X_use.shape[0]),
                            "within_mean": baseline_metrics["within_mean"],
                            "between_mean": baseline_metrics["between_mean"],
                            "separation": baseline_metrics["separation"],
                        }
                    )

                st.markdown("**Baselines before ablation**")
                baseline_df = pd.DataFrame(baseline_rows)
                st.dataframe(baseline_df, use_container_width=True, hide_index=True)

                can_run = len(fractions) > 0
                if st.button("Run ablation study", type="primary", disabled=not can_run):
                    rows = []
                    progress = st.progress(0)
                    status = st.empty()
                    total_steps = max(1, len(selected_bundles) * len(fractions) * int(repeats))
                    step_i = 0

                    for bundle in selected_bundles:
                        model = instantiate_model_from_bundle(bundle, device=device)
                        target_layer_name = layer_name_by_bundle_id[bundle["id"]]
                        target_layer = model.get_layer_by_name(target_layer_name)
                        X_use = bundle["split_data"][use_key].astype(np.float32)
                        y_use = bundle["split_data"][label_key].astype(np.int64)
                        base_batch_size = int(bundle["data_config"]["batch_size"])

                        baseline_reps = get_penultimate_representations(
                            model,
                            X_use,
                            batch_size=base_batch_size,
                            device=device,
                        )
                        baseline_metrics = compute_pairwise_representation_correlations(baseline_reps, y_use)
                        condition_label = f"{bundle['name']} · {target_layer_name}"

                        for frac in fractions:
                            for rep in range(int(repeats)):
                                rep_seed = int(seed) + rep + int(frac * 10000)

                                if ablation_mode == "unit":
                                    hook_factory = lambda layer=target_layer, fraction=frac, s=rep_seed: ActivationUnitAblator(layer, fraction, s)()
                                    reps_after = get_penultimate_representations(
                                        model,
                                        X_use,
                                        batch_size=base_batch_size,
                                        device=device,
                                        hook_fn=hook_factory,
                                    )
                                else:
                                    ablator = WeightAblator(target_layer, frac, rep_seed)
                                    ablator()
                                    reps_after = get_penultimate_representations(
                                        model,
                                        X_use,
                                        batch_size=base_batch_size,
                                        device=device,
                                        hook_fn=None,
                                    )
                                    ablator.remove()

                                met = compute_pairwise_representation_correlations(reps_after, y_use)
                                rows.append(
                                    {
                                        "model_name": bundle["name"],
                                        "model_id": bundle["id"],
                                        "layer_name": target_layer_name,
                                        "fraction": float(frac),
                                        "repeat": int(rep),
                                        "condition": condition_label,
                                        "ablation_mode": ablation_mode,
                                        "split_name": split_name,
                                        "within_mean": met["within_mean"],
                                        "between_mean": met["between_mean"],
                                        "separation": met["separation"],
                                        "baseline_within": baseline_metrics["within_mean"],
                                        "baseline_between": baseline_metrics["between_mean"],
                                        "baseline_separation": baseline_metrics["separation"],
                                        "delta_within": met["within_mean"] - baseline_metrics["within_mean"],
                                        "delta_between": met["between_mean"] - baseline_metrics["between_mean"],
                                        "delta_separation": met["separation"] - baseline_metrics["separation"],
                                    }
                                )
                                step_i += 1
                                progress.progress(step_i / total_steps)
                                status.write(f"Completed {step_i}/{total_steps} ablation runs")

                    raw_df = pd.DataFrame(rows)
                    summary_df = (
                        raw_df.groupby(["model_name", "model_id", "layer_name", "condition", "fraction"], as_index=False)
                        .agg(
                            within_mean=("within_mean", "mean"),
                            between_mean=("between_mean", "mean"),
                            separation=("separation", "mean"),
                            delta_within=("delta_within", "mean"),
                            delta_between=("delta_between", "mean"),
                            delta_separation=("delta_separation", "mean"),
                        )
                    )

                    st.markdown("**Mean results across repeats**")
                    plot_ablation_results(summary_df)
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)

                    csv = raw_df.to_csv(index=False).encode("utf-8")
                    file_stub = "combined" if len(selected_bundles) > 1 else selected_bundles[0]["name"]
                    st.download_button(
                        "Download raw ablation results (CSV)",
                        data=csv,
                        file_name=f"ablation_{file_stub}_{ablation_mode}_{split_name}.csv",
                        mime="text/csv",
                    )

                    change_fig = make_subplots(rows=1, cols=3, subplot_titles=("Δ separation", "Δ within", "Δ between"))
                    metric_label_map = {
                        "delta_separation": "Change in within-minus-between correlation",
                        "delta_within": "Change in within-class correlation",
                        "delta_between": "Change in between-class correlation",
                    }
                    for metric, col in [("delta_separation", 1), ("delta_within", 2), ("delta_between", 3)]:
                        for name in sorted(summary_df["condition"].unique()):
                            sub = summary_df[summary_df["condition"] == name].sort_values("fraction")
                            change_fig.add_trace(
                                go.Scatter(
                                    x=sub["fraction"],
                                    y=sub[metric],
                                    mode="lines+markers",
                                    name=name,
                                    legendgroup=name,
                                    showlegend=(col == 1),
                                    hovertemplate=(
                                        "Model/layer: %{fullData.name}<br>"
                                        "Ablation fraction: %{x:.3f}<br>"
                                        f"{metric_label_map[metric]}: %{{y:.4f}}<extra></extra>"
                                    ),
                                ),
                                row=1,
                                col=col,
                            )
                    change_fig.update_xaxes(title_text="Ablation fraction", row=1, col=1)
                    change_fig.update_xaxes(title_text="Ablation fraction", row=1, col=2)
                    change_fig.update_xaxes(title_text="Ablation fraction", row=1, col=3)
                    change_fig.update_yaxes(title_text="Δ within - between", row=1, col=1)
                    change_fig.update_yaxes(title_text="Δ within-class corr", row=1, col=2)
                    change_fig.update_yaxes(title_text="Δ between-class corr", row=1, col=3)
                    change_fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20), legend_title_text="Model · ablated layer")
                    st.plotly_chart(change_fig, use_container_width=True)

    st.caption(
        "Notes: LayerNorm and GroupNorm are configured as non-learnable (affine=False). "
        "GroupNorm is applied to feature vectors by treating the feature dimension as channels with spatial length 1. "
        "MaxPool is applied over the feature dimension after the block activation. "
        "Activation sparsity zeroes the bottom n% of post-activation units per sample and keeps the top activations."
    )


if __name__ == "__main__":
    main()
