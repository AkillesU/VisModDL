"""Compare category differentiation values across matched damage iterations.

The selective RDMs are used instead of the average-selectivity cache because
their filenames retain the iteration identity needed for paired tests.

Example
-------
python category_comparison_stats.py \
    --main-dir data/haupt_stim_activ/damaged/cornet_rt5_c \
    --damage-type connections \
    --damage-layer IT \
    --activation-layer IT \
    --damage-level 0.75 \
    --selectivity-fraction 0.05
"""

from __future__ import annotations

import argparse
import itertools
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_CATEGORIES = ("animal", "face", "object", "place")
_CATEGORY_ALIASES = {
    "animal": "animal",
    "animals": "animal",
    "face": "face",
    "faces": "face",
    "object": "object",
    "objects": "object",
    "place": "place",
    "places": "place",
    "scene": "place",
    "scenes": "place",
}
_DAMAGE_LEVEL_RE = re.compile(
    r"^damaged_([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def normalize_category(category: str) -> str:
    key = str(category).strip().lower()
    if key not in _CATEGORY_ALIASES:
        choices = ", ".join(DEFAULT_CATEGORIES)
        raise ValueError(f"Unknown category {category!r}; choose from {choices}.")
    return _CATEGORY_ALIASES[key]


def categories_from_image_names(image_names) -> np.ndarray:
    """Match the category parsing used by utils.assign_categories."""
    labels = []
    for image_name in image_names:
        stem = Path(str(image_name)).stem
        label = re.sub(r"\d+", "", stem).rstrip("_-. ").lower()
        labels.append(normalize_category(label))
    return np.asarray(labels, dtype=object)


def category_differentiation(
    correlation_matrix: np.ndarray,
    image_categories: np.ndarray,
    category: str,
    metric: str = "observed_difference",
) -> float:
    """Calculate the same category statistic used by categ_corr_lineplot."""
    category = normalize_category(category)
    matrix = np.asarray(correlation_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"RDM must be square; received shape {matrix.shape}.")
    if matrix.shape[0] != len(image_categories):
        raise ValueError(
            f"RDM size ({matrix.shape[0]}) does not match image_names "
            f"({len(image_categories)})."
        )

    category_indices = np.flatnonzero(image_categories == category)
    other_indices = np.flatnonzero(image_categories != category)
    if len(category_indices) < 2 or len(other_indices) == 0:
        raise ValueError(f"Not enough images to calculate category {category!r}.")

    within_matrix = matrix[np.ix_(category_indices, category_indices)]
    within = within_matrix[~np.eye(len(category_indices), dtype=bool)]
    between = matrix[np.ix_(category_indices, other_indices)].ravel()
    avg_within = float(np.mean(within))
    avg_between = float(np.mean(between))
    metrics = {
        "avg_within": avg_within,
        "avg_between": avg_between,
        "observed_difference": avg_within - avg_between,
    }
    if metric not in metrics:
        raise ValueError(
            "metric must be observed_difference, avg_within, or avg_between."
        )
    return metrics[metric]


def resolve_damage_directory(
    category_root: Path,
    requested_level: float,
    tolerance: float,
) -> tuple[Path, float]:
    exact = category_root / f"damaged_{requested_level}"
    if exact.is_dir():
        return exact, float(requested_level)

    candidates = []
    if category_root.is_dir():
        for path in category_root.iterdir():
            if not path.is_dir():
                continue
            match = _DAMAGE_LEVEL_RE.match(path.name)
            if match:
                candidates.append((abs(float(match.group(1)) - requested_level), path))
    if not candidates:
        raise FileNotFoundError(f"No damaged_* directories found under {category_root}.")

    difference, selected = min(candidates, key=lambda item: item[0])
    selected_level = float(_DAMAGE_LEVEL_RE.match(selected.name).group(1))
    if difference > tolerance:
        available = sorted(
            float(_DAMAGE_LEVEL_RE.match(path.name).group(1))
            for _, path in candidates
        )
        raise FileNotFoundError(
            f"No damage level matching {requested_level:g} within tolerance "
            f"{tolerance:g} under {category_root}. Available levels: {available}"
        )
    return selected, selected_level


def load_category_values(
    category_root: Path,
    damage_level: float,
    category: str,
    metric: str,
    tolerance: float,
) -> tuple[dict[str, float], float]:
    damage_dir, selected_level = resolve_damage_directory(
        category_root, damage_level, tolerance
    )
    values = {}
    for path in sorted(damage_dir.glob("*.pkl")):
        with path.open("rb") as handle:
            record = pickle.load(handle)
        if not isinstance(record, dict) or "RDM" not in record:
            raise ValueError(f"{path} does not contain an RDM record.")
        image_names = record.get("image_names")
        if image_names is None or len(image_names) == 0:
            raise ValueError(f"{path} does not contain image_names.")
        image_categories = categories_from_image_names(image_names)
        values[path.stem] = category_differentiation(
            record["RDM"], image_categories, category, metric=metric
        )
    if not values:
        raise FileNotFoundError(f"No RDM pickle files found in {damage_dir}.")
    return values, selected_level


def collect_matched_values(
    main_dir: str | Path,
    damage_type: str,
    damage_layer: str,
    activation_layer: str,
    damage_level: float,
    selectivity_fraction: float,
    selection_mode: str,
    categories,
    metric: str,
    value_mode: str,
    baseline_level: float,
    tolerance: float,
) -> tuple[pd.DataFrame, dict]:
    categories = tuple(normalize_category(cat) for cat in categories)
    rdm_root = (
        Path(main_dir)
        / damage_type
        / damage_layer
        / f"RDM_{selectivity_fraction:.2f}_{selection_mode}"
        / activation_layer
    )
    if not rdm_root.is_dir():
        raise FileNotFoundError(f"Selective RDM directory not found: {rdm_root}")

    category_values = {}
    levels_used = {}
    for category in categories:
        category_root = rdm_root / f"{category}_selective"
        if category == "place" and not category_root.is_dir():
            category_root = rdm_root / "scene_selective"
        damaged, damaged_level_used = load_category_values(
            category_root, damage_level, category, metric, tolerance
        )
        levels_used[f"{category}_damaged"] = damaged_level_used

        if value_mode == "differentiation":
            category_values[category] = {
                replicate_id: value
                for replicate_id, value in damaged.items()
                if np.isfinite(value)
            }
            continue

        baseline, baseline_level_used = load_category_values(
            category_root, baseline_level, category, metric, tolerance
        )
        levels_used[f"{category}_baseline"] = baseline_level_used
        matched_ids = sorted(set(damaged) & set(baseline))
        transformed = {}
        for replicate_id in matched_ids:
            baseline_value = baseline[replicate_id]
            damaged_value = damaged[replicate_id]
            if (
                not np.isfinite(baseline_value)
                or not np.isfinite(damaged_value)
                or np.isclose(baseline_value, 0.0)
            ):
                continue
            ratio = damaged_value / baseline_value
            if value_mode == "scaled_percent":
                transformed[replicate_id] = 100.0 * ratio
            elif value_mode == "relative_drop":
                transformed[replicate_id] = 1.0 - ratio
            elif value_mode == "relative_drop_percent":
                transformed[replicate_id] = 100.0 * (1.0 - ratio)
            else:
                raise ValueError(f"Unknown value_mode: {value_mode}")
        category_values[category] = transformed

    common_ids = set.intersection(
        *(set(values) for values in category_values.values())
    )
    if len(common_ids) < 2:
        counts = {cat: len(vals) for cat, vals in category_values.items()}
        raise ValueError(
            "Fewer than two complete matched iterations remain across categories. "
            f"Per-category counts: {counts}"
        )

    rows = []
    for replicate_id in sorted(common_ids):
        row = {"replicate": replicate_id}
        row.update({
            category: float(category_values[category][replicate_id])
            for category in categories
        })
        rows.append(row)
    matched = pd.DataFrame(rows)
    metadata = {
        "rdm_root": str(rdm_root),
        "levels_used": levels_used,
        "per_category_counts": {
            category: len(values) for category, values in category_values.items()
        },
        "complete_n": len(matched),
    }
    return matched, metadata


def holm_adjust(p_values) -> np.ndarray:
    """Holm family-wise error correction without a statsmodels dependency."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(p_values))
    if len(finite_indices) == 0:
        return adjusted

    order = finite_indices[np.argsort(p_values[finite_indices])]
    running_max = 0.0
    number_tests = len(order)
    for rank, original_index in enumerate(order):
        candidate = (number_tests - rank) * p_values[original_index]
        running_max = max(running_max, candidate)
        adjusted[original_index] = min(running_max, 1.0)
    return adjusted


def paired_sign_flip_test(
    differences: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> float:
    differences = np.asarray(differences, dtype=float)
    observed = abs(float(np.mean(differences)))
    if np.allclose(differences, 0.0):
        return 1.0

    exceedances = 0
    completed = 0
    batch_size = min(2000, n_permutations)
    while completed < n_permutations:
        current = min(batch_size, n_permutations - completed)
        signs = rng.choice((-1.0, 1.0), size=(current, len(differences)))
        null_means = np.mean(signs * differences, axis=1)
        exceedances += int(np.sum(np.abs(null_means) >= observed - 1e-15))
        completed += current
    return float((exceedances + 1) / (n_permutations + 1))


def summarize_categories(matched: pd.DataFrame, categories) -> pd.DataFrame:
    rows = []
    for category in categories:
        values = matched[category].to_numpy(dtype=float)
        n_values = len(values)
        std = float(np.std(values, ddof=1))
        sem = std / np.sqrt(n_values)
        critical = float(stats.t.ppf(0.975, df=n_values - 1))
        mean = float(np.mean(values))
        rows.append({
            "category": category,
            "n": n_values,
            "mean": mean,
            "sd": std,
            "sem": sem,
            "ci95_lower": mean - critical * sem,
            "ci95_upper": mean + critical * sem,
        })
    return pd.DataFrame(rows)


def pairwise_tests(
    matched: pd.DataFrame,
    categories,
    alpha: float,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for category_a, category_b in itertools.combinations(categories, 2):
        a = matched[category_a].to_numpy(dtype=float)
        b = matched[category_b].to_numpy(dtype=float)
        differences = a - b
        n_values = len(differences)
        mean_difference = float(np.mean(differences))
        sd_difference = float(np.std(differences, ddof=1))

        if np.allclose(differences, 0.0):
            t_statistic, t_p = 0.0, 1.0
        else:
            t_result = stats.ttest_rel(a, b)
            t_statistic, t_p = float(t_result.statistic), float(t_result.pvalue)

        sem_difference = sd_difference / np.sqrt(n_values)
        critical = float(stats.t.ppf(0.975, df=n_values - 1))
        permutation_p = paired_sign_flip_test(
            differences, n_permutations=n_permutations, rng=rng
        )
        rows.append({
            "category_a": category_a,
            "category_b": category_b,
            "n_pairs": n_values,
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
            "mean_difference_a_minus_b": mean_difference,
            "difference_ci95_lower": mean_difference - critical * sem_difference,
            "difference_ci95_upper": mean_difference + critical * sem_difference,
            "paired_t": t_statistic,
            "df": n_values - 1,
            "paired_t_p": t_p,
            "cohens_dz": (
                mean_difference / sd_difference
                if not np.isclose(sd_difference, 0.0)
                else np.nan
            ),
            "sign_flip_p": permutation_p,
        })

    results = pd.DataFrame(rows)
    results["paired_t_p_holm"] = holm_adjust(results["paired_t_p"])
    results["paired_t_significant"] = results["paired_t_p_holm"] < alpha
    results["sign_flip_p_holm"] = holm_adjust(results["sign_flip_p"])
    results["sign_flip_significant"] = results["sign_flip_p_holm"] < alpha
    return results


def omnibus_friedman(matched: pd.DataFrame, categories) -> dict:
    arrays = [matched[category].to_numpy(dtype=float) for category in categories]
    if all(np.allclose(arrays[0], values) for values in arrays[1:]):
        statistic, p_value = 0.0, 1.0
    else:
        result = stats.friedmanchisquare(*arrays)
        statistic, p_value = float(result.statistic), float(result.pvalue)
    return {
        "test": "Friedman repeated-measures test",
        "statistic": statistic,
        "df": len(categories) - 1,
        "p_value": p_value,
        "n_complete_iterations": len(matched),
    }


def safe_condition_tag(value) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")


def run_analysis(args) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    categories = tuple(normalize_category(cat) for cat in args.categories)
    if len(set(categories)) != len(categories):
        raise ValueError("Categories must be unique after normalizing aliases.")
    if len(categories) < 2:
        raise ValueError("At least two categories are required.")
    if args.n_permutations < 1:
        raise ValueError("--n-permutations must be at least 1.")

    matched, metadata = collect_matched_values(
        main_dir=args.main_dir,
        damage_type=args.damage_type,
        damage_layer=args.damage_layer,
        activation_layer=args.activation_layer,
        damage_level=args.damage_level,
        selectivity_fraction=args.selectivity_fraction,
        selection_mode=args.selection_mode,
        categories=categories,
        metric=args.metric,
        value_mode=args.value_mode,
        baseline_level=args.baseline_level,
        tolerance=args.tolerance,
    )
    summary = summarize_categories(matched, categories)
    pairwise = pairwise_tests(
        matched,
        categories,
        alpha=args.alpha,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )
    omnibus = omnibus_friedman(matched, categories)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = "_".join([
        safe_condition_tag(args.damage_type),
        safe_condition_tag(args.damage_layer),
        safe_condition_tag(args.activation_layer),
        f"dmg{args.damage_level:g}",
        f"top{args.selectivity_fraction:.2f}",
        safe_condition_tag(args.value_mode),
    ])
    matched.to_csv(output_dir / f"{tag}_matched-values.csv", index=False)
    summary.to_csv(output_dir / f"{tag}_summary.csv", index=False)
    pairwise.to_csv(output_dir / f"{tag}_pairwise.csv", index=False)
    pd.DataFrame([omnibus]).to_csv(
        output_dir / f"{tag}_omnibus.csv", index=False
    )

    print(f"RDM root: {metadata['rdm_root']}")
    print(f"Complete matched iterations: {metadata['complete_n']}")
    print(f"Value mode: {args.value_mode}")
    print()
    print("Category summaries")
    print(summary.to_string(index=False))
    print()
    print("Omnibus comparison")
    print(pd.DataFrame([omnibus]).to_string(index=False))
    print()
    print("Pairwise comparisons (two-sided, paired; Holm-corrected)")
    print(pairwise.to_string(index=False))
    print()
    print(f"Saved CSV files to: {output_dir}")
    return summary, pairwise, omnibus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare category differentiation across matched selective-RDM "
            "damage iterations."
        )
    )
    parser.add_argument("--main-dir", required=True)
    parser.add_argument("--damage-type", required=True)
    parser.add_argument("--damage-layer", required=True)
    parser.add_argument("--activation-layer", required=True)
    parser.add_argument("--damage-level", required=True, type=float)
    parser.add_argument("--selectivity-fraction", required=True, type=float)
    parser.add_argument("--selection-mode", default="percentage")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(DEFAULT_CATEGORIES),
    )
    parser.add_argument(
        "--metric",
        choices=("observed_difference", "avg_within", "avg_between"),
        default="observed_difference",
    )
    parser.add_argument(
        "--value-mode",
        choices=(
            "scaled_percent",
            "differentiation",
            "relative_drop",
            "relative_drop_percent",
        ),
        default="scaled_percent",
        help=(
            "Use scaled_percent to match plot_category_relative_drop_bar with "
            "value_mode='scaled_percent'."
        ),
    )
    parser.add_argument("--baseline-level", type=float, default=0.0)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--output-dir",
        default="stats/category_comparisons",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
