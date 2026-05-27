#!/usr/bin/env python3
"""
Plot category-selective IT unit responses from saved damage activation files.

Expected activation layout:

    <damaged_model_dir>/<damage_type>/<damage_layer>/activations/<activation_layer>/
        damaged_<level>/
            <permutation>__activ_<n_features>.zarr

Pickle and CSV activation matrices are also accepted when they contain one
image-by-unit matrix per file.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

try:
    import zarr
except ImportError:  # zarr is only needed when reading .zarr activation stores.
    zarr = None


CATEGORY_ALIASES: Dict[str, List[str]] = {
    "animal": ["animal", "animals"],
    "face": ["face", "faces"],
    "object": ["object", "objects"],
    "place": ["place", "places", "scene", "scenes"],
}

CATEGORY_COLORS = {
    "animal": "#2f9e44",
    "face": "#d9480f",
    "object": "#1c7ed6",
    "place": "#7048e8",
    "joint": "#212529",
}


@dataclass(frozen=True)
class ActivationBatch:
    df: pd.DataFrame
    perm: int
    source: Path


def natural_key(value: Any) -> Tuple[Any, ...]:
    parts = re.split(r"(\d+(?:\.\d+)?)", str(value))
    key: List[Any] = []
    for part in parts:
        if not part:
            continue
        try:
            key.append(float(part))
        except ValueError:
            key.append(part.lower())
    return tuple(key)


def safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_damage_level(path_or_name: Any) -> Optional[float]:
    match = re.search(r"damaged_([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", str(path_or_name))
    return safe_float(match.group(1)) if match else None


def parse_perm_index(path: Path, fallback: int = 0) -> int:
    match = re.match(r"(\d+)(?:__|\.|_|$)", path.name)
    if match:
        return int(match.group(1))
    digits = re.findall(r"\d+", path.stem)
    return int(digits[0]) if digits else int(fallback)


def sanitize_filename(text: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-")
    return cleaned or "plot"


def canonical_category(category: str) -> str:
    lower = str(category).lower()
    for key, aliases in CATEGORY_ALIASES.items():
        if lower == key or lower in aliases:
            return key
    return lower


def infer_image_category(image_name: Any, category_aliases: Mapping[str, Sequence[str]]) -> str:
    text = Path(str(image_name)).stem.lower()
    tokens = [tok for tok in re.split(r"[^a-z0-9]+", text) if tok]
    for category, aliases in category_aliases.items():
        canonical = canonical_category(category)
        for token in tokens:
            for alias in aliases:
                alias = str(alias).lower()
                suffix = token[len(alias):]
                if token == alias or (token.startswith(alias) and (not suffix or suffix.isdigit())):
                    return canonical
    return "unknown"


def selectivity_model_tags(model_name: str) -> List[str]:
    tags = [str(model_name)]
    base = re.sub(r"(\+b)?$", "", str(model_name))
    base = re.sub(r"_(c|all|ut)$", "", base)
    tags.append(base)
    if re.match(r"cornet_rt\d*", base):
        tags.append("cornet_rt")
    if base.startswith("vgg16"):
        tags.append("vgg16")
    if base.startswith("alexnet"):
        tags.append("alexnet")
    return list(dict.fromkeys(tag for tag in tags if tag))


def resolve_selectivity_path(path_or_dir: str | Path, model_name: str) -> Path:
    path = Path(path_or_dir)
    if path.is_file():
        return path
    if path.is_dir():
        candidates = []
        for tag in selectivity_model_tags(model_name):
            candidates.extend([
                path / f"{tag}_all_layers_units_mannwhitneyu.pkl",
                path / f"{tag}_all_layers_units_mannwhitneyu.csv",
            ])
        candidates.extend([
            path / "all_layers_units_mannwhitneyu.pkl",
            path / "all_layers_units_mannwhitneyu.csv",
        ])
        for candidate in candidates:
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Could not resolve selectivity table from: {path_or_dir}")


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def infer_column(df: pd.DataFrame, candidates: Sequence[str], label: str) -> str:
    lookup = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise ValueError(f"Could not infer {label} column. Tried: {', '.join(candidates)}")


def score_column_for_category(
    df: pd.DataFrame,
    category: str,
    score_prefix: str,
    score_columns: Mapping[str, str],
) -> str:
    score_columns = {canonical_category(key): value for key, value in score_columns.items()}
    if category in score_columns:
        return score_columns[category]
    aliases = CATEGORY_ALIASES.get(category, [category])
    candidates: List[str] = []
    for alias in aliases:
        candidates.extend([
            f"{score_prefix}_{alias}",
            f"{score_prefix}{alias}",
            alias,
        ])
    lookup = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise ValueError(f"No score column found for category '{category}' with prefix '{score_prefix}'.")


def parse_unit_index(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        text = str(value).replace(",", ":")
        parts = [part.strip() for part in text.split(":") if part.strip()]
        if len(parts) >= 3:
            return int(float(parts[-3]))
        if parts:
            return int(float(parts[-1]))
    raise ValueError(f"Could not parse unit index from {value!r}")


def select_top_units(config: Mapping[str, Any]) -> pd.DataFrame:
    model_name = str(config.get("model_name", "cornet_rt5_c"))
    categories = [canonical_category(c) for c in config.get("categories", list(CATEGORY_ALIASES))]
    top_n = int(config.get("top_n_units", 1))
    ranking_score_prefix = str(config.get("ranking_score_prefix", "hg"))
    score_columns = config.get("score_columns", {}) or {}
    unit_layer = config.get("unit_layer", "module.IT")

    selectivity_path = resolve_selectivity_path(config.get("selectivity_path", "unit_selectivity"), model_name)
    df = read_table(selectivity_path)
    layer_col = config.get("selectivity_layer_col") or infer_column(df, ["layer", "layer_name"], "layer")
    unit_col = config.get("unit_col") or infer_column(df, ["unit", "unit_id", "channel"], "unit")

    layer_df = df[df[layer_col].astype(str) == str(unit_layer)].copy()
    if layer_df.empty:
        aliases = {str(unit_layer), str(unit_layer).replace("module.", ""), f"module.{unit_layer}"}
        layer_df = df[df[layer_col].astype(str).isin(aliases)].copy()
    if layer_df.empty:
        available = sorted(df[layer_col].astype(str).unique(), key=natural_key)
        raise ValueError(f"No rows found for unit_layer={unit_layer!r}. Available layers include: {available[:12]}")

    selected: List[Dict[str, Any]] = []
    for category in categories:
        score_col = score_column_for_category(layer_df, category, ranking_score_prefix, score_columns)
        ranked = layer_df.dropna(subset=[score_col]).sort_values(score_col, ascending=False).head(top_n)
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            selected.append(
                {
                    "target_category": category,
                    "rank": int(rank),
                    "unit": parse_unit_index(row[unit_col]),
                    "unit_source_value": row[unit_col],
                    "score": float(row[score_col]),
                    "score_col": score_col,
                    "selectivity_layer": str(row[layer_col]),
                    "selectivity_path": str(selectivity_path),
                }
            )

    if not selected:
        raise ValueError("No units were selected from the selectivity table.")
    return pd.DataFrame(selected)


def activation_root(config: Mapping[str, Any], damage_type: str, damage_layer: str, activation_layer: str) -> Path:
    model_name = str(config.get("model_name", "cornet_rt5_c"))
    damaged_model_dir = Path(config.get("damaged_model_dir") or f"data/haupt_stim_activ/damaged/{model_name}")
    return damaged_model_dir / damage_type / damage_layer / "activations" / activation_layer


def discover_damage_dirs(root: Path, requested_levels: Optional[Sequence[Any]] = None) -> List[Tuple[float, Path]]:
    if not root.exists():
        raise FileNotFoundError(f"Activation directory does not exist: {root}")
    found: List[Tuple[float, Path]] = []
    for child in sorted(root.iterdir(), key=lambda p: natural_key(p.name)):
        if not child.is_dir() or not child.name.startswith("damaged_"):
            continue
        level = parse_damage_level(child.name)
        if level is not None:
            found.append((level, child))
    if requested_levels is None:
        return found
    wanted = {round(float(level), 10) for level in requested_levels}
    return [(level, path) for level, path in found if round(float(level), 10) in wanted]


def discover_activation_files(damage_dir: Path) -> List[Path]:
    files: List[Path] = []
    files.extend([p for p in damage_dir.rglob("*.zarr") if p.is_dir()])
    files.extend([p for p in damage_dir.rglob("*.pkl") if p.is_file()])
    files.extend([p for p in damage_dir.rglob("*.pickle") if p.is_file()])
    files.extend([p for p in damage_dir.rglob("*.csv") if p.is_file()])
    return sorted(files, key=lambda p: natural_key(p.relative_to(damage_dir)))


def dataframe_from_array(arr: np.ndarray, image_names: Optional[Sequence[Any]], column_names: Optional[Sequence[Any]]) -> pd.DataFrame:
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D image-by-unit matrix, got shape {arr.shape}.")
    index = list(image_names) if image_names is not None and len(image_names) == arr.shape[0] else list(range(arr.shape[0]))
    columns = list(column_names) if column_names is not None and len(column_names) == arr.shape[1] else list(range(arr.shape[1]))
    return pd.DataFrame(arr, index=index, columns=columns)


def load_zarr_batches(path: Path) -> List[ActivationBatch]:
    if zarr is None:
        raise ImportError("zarr is required to read .zarr activation stores.")
    root = zarr.open(path, mode="r")
    if "activ" not in root:
        raise ValueError(f"{path} has no 'activ' dataset.")
    arr = np.asarray(root["activ"])
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported zarr activation shape in {path}: {arr.shape}")

    image_names = list(root.attrs.get("image_names", [])) or None
    column_names = list(root.attrs.get("column_names", [])) or None
    perm_indices = list(root.attrs.get("perm_indices", []))
    if not perm_indices:
        perm_indices = [parse_perm_index(path, fallback=i) for i in range(arr.shape[0])]

    batches = []
    for i in range(arr.shape[0]):
        perm = int(perm_indices[i]) if i < len(perm_indices) else parse_perm_index(path, fallback=i)
        batches.append(ActivationBatch(dataframe_from_array(arr[i], image_names, column_names), perm, path))
    return batches


def load_pickle_batches(path: Path) -> List[ActivationBatch]:
    with open(path, "rb") as f:
        obj = pickle.load(f)

    perm = parse_perm_index(path)
    if isinstance(obj, pd.DataFrame):
        return [ActivationBatch(obj, perm, path)]
    if isinstance(obj, dict):
        for key in ["df", "dataframe", "activations_df"]:
            if isinstance(obj.get(key), pd.DataFrame):
                return [ActivationBatch(obj[key], int(obj.get("perm", perm)), path)]
        arr = obj.get("activations", obj.get("activ", obj.get("data")))
        if arr is None:
            raise ValueError(f"Unsupported activation pickle keys in {path}: {list(obj.keys())[:20]}")
        arr = np.asarray(arr)
        image_names = obj.get("image_names", obj.get("index"))
        column_names = obj.get("column_names", obj.get("columns"))
        if arr.ndim == 2:
            return [ActivationBatch(dataframe_from_array(arr, image_names, column_names), int(obj.get("perm", perm)), path)]
        if arr.ndim == 3:
            perm_indices = obj.get("perm_indices", list(range(arr.shape[0])))
            return [
                ActivationBatch(dataframe_from_array(arr[i], image_names, column_names), int(perm_indices[i]), path)
                for i in range(arr.shape[0])
            ]
    arr = np.asarray(obj)
    return [ActivationBatch(dataframe_from_array(arr, None, None), perm, path)]


def load_activation_batches(path: Path) -> List[ActivationBatch]:
    suffix = path.suffix.lower()
    if suffix == ".zarr":
        return load_zarr_batches(path)
    if suffix in {".pkl", ".pickle"}:
        return load_pickle_batches(path)
    if suffix == ".csv":
        return [ActivationBatch(pd.read_csv(path, index_col=0), parse_perm_index(path), path)]
    raise ValueError(f"Unsupported activation file: {path}")


def unit_series(df: pd.DataFrame, unit: int) -> pd.Series:
    for key in [unit, str(unit), float(unit)]:
        if key in df.columns:
            return pd.to_numeric(df[key], errors="coerce")
    if 0 <= int(unit) < df.shape[1]:
        return pd.to_numeric(df.iloc[:, int(unit)], errors="coerce")
    raise IndexError(f"Unit index {unit} is out of bounds for activation matrix with {df.shape[1]} columns.")


def rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def mannwhitney_u(target_values: np.ndarray, other_values: np.ndarray, normalize: bool = False) -> float:
    target_values = np.asarray(target_values, dtype=float)
    other_values = np.asarray(other_values, dtype=float)
    target_values = target_values[np.isfinite(target_values)]
    other_values = other_values[np.isfinite(other_values)]
    n_target = len(target_values)
    n_other = len(other_values)
    if n_target == 0 or n_other == 0:
        return float("nan")
    combined = np.concatenate([target_values, other_values])
    ranks = rankdata_average(combined)
    u_value = float(ranks[:n_target].sum() - n_target * (n_target + 1) / 2.0)
    return u_value / float(n_target * n_other) if normalize else u_value


def sd(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return 0.0
    return float(np.std(arr, ddof=1))


def stable_seed(key: Any, base_seed: int) -> int:
    digest = hashlib.md5(repr(key).encode("utf-8")).hexdigest()
    return int((int(digest[:8], 16) + int(base_seed)) % (2**32 - 1))


def bootstrap_mean_ci(
    values: Sequence[float],
    weights: Optional[Sequence[float]] = None,
    n_boot: int = 2000,
    ci: float = 95.0,
    seed: int = 0,
) -> Tuple[float, float]:
    values_arr = np.asarray(values, dtype=float)
    if weights is None:
        weights_arr = np.ones_like(values_arr, dtype=float)
    else:
        weights_arr = np.asarray(weights, dtype=float)

    valid = np.isfinite(values_arr) & np.isfinite(weights_arr) & (weights_arr > 0)
    values_arr = values_arr[valid]
    weights_arr = weights_arr[valid]
    if len(values_arr) == 0:
        return float("nan"), float("nan")
    if len(values_arr) == 1 or n_boot <= 0:
        mean = float(np.average(values_arr, weights=weights_arr))
        return mean, mean

    rng = np.random.default_rng(int(seed))
    boot_means = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        idx = rng.integers(0, len(values_arr), size=len(values_arr))
        boot_means[i] = float(np.average(values_arr[idx], weights=weights_arr[idx]))

    alpha = (100.0 - float(ci)) / 2.0
    low, high = np.percentile(boot_means, [alpha, 100.0 - alpha])
    return float(low), float(high)


def collect_iteration_summaries(
    config: Mapping[str, Any],
    selected_units: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    categories = [canonical_category(c) for c in config.get("categories", list(CATEGORY_ALIASES))]
    category_aliases = {
        canonical_category(cat): aliases
        for cat, aliases in {**CATEGORY_ALIASES, **(config.get("category_aliases", {}) or {})}.items()
    }
    damage_types = config.get("damage_types", [config.get("damage_type", "unit_activations")])
    damage_layers = config.get("damage_layers", ["V1", "V2", "V4", "IT"])
    activation_layers = config.get("activation_layers", ["IT"])
    requested_levels = config.get("damage_levels")
    normalize_u = bool(config.get("mannwhitney_normalize", False))

    activation_rows: List[Dict[str, Any]] = []
    mw_rows: List[Dict[str, Any]] = []

    for damage_type in damage_types:
        for damage_layer in damage_layers:
            for activation_layer in activation_layers:
                root = activation_root(config, damage_type, damage_layer, activation_layer)
                for damage_level, damage_dir in discover_damage_dirs(root, requested_levels):
                    activation_files = discover_activation_files(damage_dir)
                    if not activation_files:
                        print(f"Warning: no activation files under {damage_dir}")
                        continue
                    for activation_file in activation_files:
                        for batch in load_activation_batches(activation_file):
                            image_categories = np.asarray(
                                [infer_image_category(idx, category_aliases) for idx in batch.df.index],
                                dtype=object,
                            )
                            for _, unit_row in selected_units.iterrows():
                                target_category = canonical_category(unit_row["target_category"])
                                if target_category not in categories:
                                    continue
                                values = unit_series(batch.df, int(unit_row["unit"])).to_numpy(dtype=float)
                                target_mask = image_categories == target_category
                                other_mask = np.isin(image_categories, categories) & ~target_mask
                                target_values = values[target_mask]
                                other_values = values[other_mask]

                                for status, status_values in [("target", target_values), ("non_target", other_values)]:
                                    finite = status_values[np.isfinite(status_values)]
                                    activation_rows.append(
                                        {
                                            "damage_type": damage_type,
                                            "damage_layer": damage_layer,
                                            "activation_layer": activation_layer,
                                            "damage_level": float(damage_level),
                                            "perm": int(batch.perm),
                                            "target_category": target_category,
                                            "rank": int(unit_row["rank"]),
                                            "unit": int(unit_row["unit"]),
                                            "score": float(unit_row["score"]),
                                            "target_status": status,
                                            "mean_activation": float(np.mean(finite)) if len(finite) else np.nan,
                                            "sd_activation": sd(finite),
                                            "n_images": int(len(finite)),
                                            "source": str(batch.source),
                                        }
                                    )

                                mw_rows.append(
                                    {
                                        "damage_type": damage_type,
                                        "damage_layer": damage_layer,
                                        "activation_layer": activation_layer,
                                        "damage_level": float(damage_level),
                                        "perm": int(batch.perm),
                                        "target_category": target_category,
                                        "rank": int(unit_row["rank"]),
                                        "unit": int(unit_row["unit"]),
                                        "score": float(unit_row["score"]),
                                        "mannwhitney_u": mannwhitney_u(target_values, other_values, normalize=normalize_u),
                                        "n_target": int(np.isfinite(target_values).sum()),
                                        "n_other": int(np.isfinite(other_values).sum()),
                                        "source": str(batch.source),
                                    }
                                )

    return pd.DataFrame(activation_rows), pd.DataFrame(mw_rows)


def pooled_activation_summary(
    rows: pd.DataFrame,
    group_cols: Sequence[str],
    bootstrap_iterations: int,
    bootstrap_ci: float,
    bootstrap_seed: int,
) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()

    def pool(group: pd.DataFrame) -> pd.Series:
        n = group["n_images"].astype(float).to_numpy()
        means = group["mean_activation"].astype(float).to_numpy()
        sds = group["sd_activation"].fillna(0).astype(float).to_numpy()
        valid = np.isfinite(means) & (n > 0)
        if not valid.any():
            return pd.Series({"mean": np.nan, "sd": np.nan, "n": 0, "n_permutations": 0})
        n = n[valid]
        means = means[valid]
        sds = sds[valid]
        total_n = float(n.sum())
        mean = float((n * means).sum() / total_n)
        if total_n <= 1:
            pooled_sd = 0.0
        else:
            ss = ((n - 1) * (sds ** 2) + n * ((means - mean) ** 2)).sum()
            pooled_sd = float(math.sqrt(max(ss / (total_n - 1), 0.0)))
        group_key = tuple(group[col].iloc[0] for col in group_cols if col in group.columns)
        ci_low, ci_high = bootstrap_mean_ci(
            means,
            weights=n,
            n_boot=int(bootstrap_iterations),
            ci=float(bootstrap_ci),
            seed=stable_seed(group_key, bootstrap_seed),
        )
        return pd.Series(
            {
                "mean": mean,
                "sd": pooled_sd,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n": int(total_n),
                "n_permutations": int(group.loc[valid, "perm"].nunique()),
            }
        )

    return rows.groupby(list(group_cols)).apply(pool).reset_index()


def mw_summary(
    rows: pd.DataFrame,
    group_cols: Sequence[str],
    bootstrap_iterations: int,
    bootstrap_ci: float,
    bootstrap_seed: int,
) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()

    def summarise(group: pd.DataFrame) -> pd.Series:
        values = pd.to_numeric(group["mannwhitney_u"], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        group_key = tuple(group[col].iloc[0] for col in group_cols if col in group.columns)
        ci_low, ci_high = bootstrap_mean_ci(
            values,
            n_boot=int(bootstrap_iterations),
            ci=float(bootstrap_ci),
            seed=stable_seed(group_key, bootstrap_seed),
        )
        return pd.Series(
            {
                "mean": float(np.mean(values)) if len(values) else np.nan,
                "sd": sd(values),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n": int(len(values)),
                "n_permutations": int(group["perm"].nunique()),
            }
        )

    return rows.groupby(list(group_cols)).apply(summarise).reset_index().sort_values(list(group_cols)).reset_index(drop=True)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 400,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, out_base: Path, dpi: int) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def errorbar_values(summary: pd.DataFrame, mode: str) -> Any:
    if mode == "sd":
        return summary["sd"].to_numpy(dtype=float)
    if mode == "bootstrap_ci":
        lower = summary["mean"].to_numpy(dtype=float) - summary["ci_low"].to_numpy(dtype=float)
        upper = summary["ci_high"].to_numpy(dtype=float) - summary["mean"].to_numpy(dtype=float)
        return np.vstack([np.clip(lower, 0, None), np.clip(upper, 0, None)])
    raise ValueError(f"Unsupported error bar mode: {mode}")


def errorbar_label(mode: str, ci: float) -> str:
    if mode == "sd":
        return "SD"
    if mode == "bootstrap_ci":
        return f"{ci:g}% bootstrap CI"
    return str(mode)


def plot_activation_summary(
    summary: pd.DataFrame,
    title: str,
    color: str,
    output_base: Path,
    dpi: int,
    figure_size: Sequence[float],
    error_bar_mode: str,
    bootstrap_ci: float,
) -> None:
    fig, ax = plt.subplots(figsize=tuple(figure_size))
    for status, linestyle, label in [("target", "-", "Target images"), ("non_target", "--", "Non-target images")]:
        sub = summary[summary["target_status"] == status].sort_values("damage_level")
        if sub.empty:
            continue
        line_color = color if status == "target" else "#495057"
        ax.errorbar(
            sub["damage_level"],
            sub["mean"],
            yerr=errorbar_values(sub, error_bar_mode),
            marker="o",
            linestyle=linestyle,
            linewidth=2.0,
            markersize=4.5,
            capsize=3,
            color=line_color,
            label=label,
        )
    ax.set_title(f"{title} | {errorbar_label(error_bar_mode, bootstrap_ci)}")
    ax.set_xlabel("Damage level")
    ax.set_ylabel("Activation")
    ax.grid(True, axis="y", color="#dee2e6", linewidth=0.8)
    ax.set_box_aspect(1)
    ax.legend(loc="best")
    save_figure(fig, output_base, dpi)


def plot_mw_summary(
    summary: pd.DataFrame,
    title: str,
    color: str,
    output_base: Path,
    dpi: int,
    figure_size: Sequence[float],
    normalize: bool,
    error_bar_mode: str,
    bootstrap_ci: float,
) -> None:
    sub = summary.sort_values("damage_level")
    fig, ax = plt.subplots(figsize=tuple(figure_size))
    ax.errorbar(
        sub["damage_level"],
        sub["mean"],
        yerr=errorbar_values(sub, error_bar_mode),
        marker="o",
        linestyle="-",
        linewidth=2.0,
        markersize=4.5,
        capsize=3,
        color=color,
    )
    ax.set_title(f"{title} | {errorbar_label(error_bar_mode, bootstrap_ci)}")
    ax.set_xlabel("Damage level")
    ax.set_ylabel("Normalized Mann-Whitney U" if normalize else "Mann-Whitney U")
    ax.grid(True, axis="y", color="#dee2e6", linewidth=0.8)
    ax.set_box_aspect(1)
    save_figure(fig, output_base, dpi)


def plot_all(config: Mapping[str, Any], activation_rows: pd.DataFrame, mw_rows: pd.DataFrame) -> None:
    output_dir = Path(config.get("output_dir", "plots/damage_unit_activations"))
    figure_size = config.get("figure_size", [6, 6])
    dpi = int(config.get("dpi", 400))
    normalize_u = bool(config.get("mannwhitney_normalize", False))
    top_n = int(config.get("top_n_units", 1))
    plot_individual = bool(config.get("plot_individual_units", True))
    plot_category_average = bool(config.get("plot_category_averages", True))
    plot_joint_average = bool(config.get("plot_joint_average", True))
    error_bar_modes = list(config.get("error_bars", ["sd", "bootstrap_ci"]))
    bootstrap_iterations = int(config.get("bootstrap_iterations", 2000))
    bootstrap_ci = float(config.get("bootstrap_ci", 95.0))
    bootstrap_seed = int(config.get("bootstrap_seed", 1234))

    combo_cols = ["damage_type", "damage_layer", "activation_layer"]
    combos = activation_rows[combo_cols].drop_duplicates().sort_values(combo_cols)

    base_activation_group = combo_cols + ["target_category", "rank", "unit", "target_status", "damage_level"]
    base_mw_group = combo_cols + ["target_category", "rank", "unit", "damage_level"]
    activation_unit_summary = pooled_activation_summary(
        activation_rows,
        base_activation_group,
        bootstrap_iterations,
        bootstrap_ci,
        bootstrap_seed,
    )
    mw_unit_summary = mw_summary(
        mw_rows,
        base_mw_group,
        bootstrap_iterations,
        bootstrap_ci,
        bootstrap_seed,
    )

    for _, combo in combos.iterrows():
        combo_filter = np.ones(len(activation_rows), dtype=bool)
        mw_filter = np.ones(len(mw_rows), dtype=bool)
        combo_name_parts = []
        for col in combo_cols:
            combo_filter &= activation_rows[col].eq(combo[col]).to_numpy()
            mw_filter &= mw_rows[col].eq(combo[col]).to_numpy()
            combo_name_parts.append(sanitize_filename(combo[col]))
        combo_name = "_".join(combo_name_parts)

        if plot_individual:
            for _, unit_meta in activation_rows.loc[combo_filter, ["target_category", "rank", "unit"]].drop_duplicates().iterrows():
                category = unit_meta["target_category"]
                rank = int(unit_meta["rank"])
                unit = int(unit_meta["unit"])
                color = CATEGORY_COLORS.get(category, "#1f77b4")

                act_sub = activation_unit_summary[
                    (activation_unit_summary["damage_type"] == combo["damage_type"])
                    & (activation_unit_summary["damage_layer"] == combo["damage_layer"])
                    & (activation_unit_summary["activation_layer"] == combo["activation_layer"])
                    & (activation_unit_summary["target_category"] == category)
                    & (activation_unit_summary["rank"] == rank)
                    & (activation_unit_summary["unit"] == unit)
                ]
                mw_sub = mw_unit_summary[
                    (mw_unit_summary["damage_type"] == combo["damage_type"])
                    & (mw_unit_summary["damage_layer"] == combo["damage_layer"])
                    & (mw_unit_summary["activation_layer"] == combo["activation_layer"])
                    & (mw_unit_summary["target_category"] == category)
                    & (mw_unit_summary["rank"] == rank)
                    & (mw_unit_summary["unit"] == unit)
                ]
                label = f"{category} rank {rank}, unit {unit}"
                file_stub = f"{combo_name}_{sanitize_filename(category)}_rank{rank}_unit{unit}"
                for error_bar_mode in error_bar_modes:
                    plot_activation_summary(
                        act_sub,
                        f"Activation | {label} | damage {combo['damage_layer']}",
                        color,
                        output_dir / "activation" / error_bar_mode / "individual_units" / file_stub,
                        dpi,
                        figure_size,
                        error_bar_mode,
                        bootstrap_ci,
                    )
                    plot_mw_summary(
                        mw_sub,
                        f"Selectivity | {label} | damage {combo['damage_layer']}",
                        color,
                        output_dir / "mannwhitney_u" / error_bar_mode / "individual_units" / file_stub,
                        dpi,
                        figure_size,
                        normalize_u,
                        error_bar_mode,
                        bootstrap_ci,
                    )

        if plot_category_average:
            act_cat_summary = pooled_activation_summary(
                activation_rows.loc[combo_filter],
                combo_cols + ["target_category", "target_status", "damage_level"],
                bootstrap_iterations,
                bootstrap_ci,
                bootstrap_seed,
            )
            mw_cat_summary = mw_summary(
                mw_rows.loc[mw_filter],
                combo_cols + ["target_category", "damage_level"],
                bootstrap_iterations,
                bootstrap_ci,
                bootstrap_seed,
            )
            for category in sorted(act_cat_summary["target_category"].unique(), key=natural_key):
                color = CATEGORY_COLORS.get(category, "#1f77b4")
                file_stub = f"{combo_name}_{sanitize_filename(category)}_top{top_n}_average"
                for error_bar_mode in error_bar_modes:
                    plot_activation_summary(
                        act_cat_summary[act_cat_summary["target_category"] == category],
                        f"Activation | {category} top {top_n} average | damage {combo['damage_layer']}",
                        color,
                        output_dir / "activation" / error_bar_mode / "category_average" / file_stub,
                        dpi,
                        figure_size,
                        error_bar_mode,
                        bootstrap_ci,
                    )
                    plot_mw_summary(
                        mw_cat_summary[mw_cat_summary["target_category"] == category],
                        f"Selectivity | {category} top {top_n} average | damage {combo['damage_layer']}",
                        color,
                        output_dir / "mannwhitney_u" / error_bar_mode / "category_average" / file_stub,
                        dpi,
                        figure_size,
                        normalize_u,
                        error_bar_mode,
                        bootstrap_ci,
                    )

        if plot_joint_average:
            act_joint = pooled_activation_summary(
                activation_rows.loc[combo_filter],
                combo_cols + ["target_status", "damage_level"],
                bootstrap_iterations,
                bootstrap_ci,
                bootstrap_seed,
            )
            mw_joint = mw_summary(
                mw_rows.loc[mw_filter],
                combo_cols + ["damage_level"],
                bootstrap_iterations,
                bootstrap_ci,
                bootstrap_seed,
            )
            file_stub = f"{combo_name}_joint_top{top_n}_average"
            for error_bar_mode in error_bar_modes:
                plot_activation_summary(
                    act_joint,
                    f"Activation | joint top {top_n} average | damage {combo['damage_layer']}",
                    CATEGORY_COLORS["joint"],
                    output_dir / "activation" / error_bar_mode / "joint_average" / file_stub,
                    dpi,
                    figure_size,
                    error_bar_mode,
                    bootstrap_ci,
                )
                plot_mw_summary(
                    mw_joint,
                    f"Selectivity | joint top {top_n} average | damage {combo['damage_layer']}",
                    CATEGORY_COLORS["joint"],
                    output_dir / "mannwhitney_u" / error_bar_mode / "joint_average" / file_stub,
                    dpi,
                    figure_size,
                    normalize_u,
                    error_bar_mode,
                    bootstrap_ci,
                )

    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    activation_rows.to_csv(table_dir / "activation_iteration_summary.csv", index=False)
    mw_rows.to_csv(table_dir / "mannwhitney_iteration_summary.csv", index=False)
    activation_unit_summary.to_csv(table_dir / "activation_unit_summary.csv", index=False)
    mw_unit_summary.to_csv(table_dir / "mannwhitney_unit_summary.csv", index=False)


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        config = yaml.safe_load(f) or {}
    if "damage_type" in config and "damage_types" not in config:
        config["damage_types"] = [config["damage_type"]]
    config.setdefault("categories", ["animal", "face", "object", "place"])
    config["categories"] = [canonical_category(cat) for cat in config["categories"]]
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot saved damage activations for category-selective IT units.")
    parser.add_argument("config", help="YAML config file.")
    parser.add_argument("--top-n", type=int, default=None, help="Override top_n_units from the config.")
    parser.add_argument("--output-dir", default=None, help="Override output_dir from the config.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.top_n is not None:
        config["top_n_units"] = int(args.top_n)
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    configure_plot_style()
    selected_units = select_top_units(config)
    output_dir = Path(config.get("output_dir", "plots/damage_unit_activations"))
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    selected_units.to_csv(output_dir / "tables" / "selected_units.csv", index=False)

    activation_rows, mw_rows = collect_iteration_summaries(config, selected_units)
    if activation_rows.empty or mw_rows.empty:
        raise RuntimeError("No activation data were loaded. Check damaged_model_dir, damage_type, layers, and damage levels.")
    plot_all(config, activation_rows, mw_rows)
    print(f"Saved plots and tables under: {output_dir}")


if __name__ == "__main__":
    main()
