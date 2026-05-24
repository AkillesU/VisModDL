import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from utils import (
    _category_scaled_percent_from_df,
    _ci95_mean,
    _collect_categ_corr_lineplot_data,
    _normalize_category_name,
)


try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - fallback for stripped-down environments
    scipy_stats = None


def _as_list(value):
    if value is None:
        return []
    return value if isinstance(value, (list, tuple)) else [value]


def _finite(values):
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _integrate_trapezoid(y, x):
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    return float(integrate(y, x))


def _relative_drop_auc_per_replicate(x_to_values, baseline_x=0.0, tolerance=1e-8):
    levels = sorted(float(x) for x in x_to_values.keys())
    if len(levels) < 2:
        return []
    baseline_candidates = [x for x in levels if abs(x - baseline_x) <= tolerance]
    if not baseline_candidates:
        raise ValueError(f"No baseline level {baseline_x:g} found for AUC calculation.")
    baseline_level = baseline_candidates[0]

    min_n = min(len(x_to_values[x]) for x in levels)
    if min_n == 0:
        return []

    xs = np.asarray(levels, dtype=float)
    damage_range = float(xs[-1] - xs[0])
    if damage_range <= 0:
        return []
    baseline_idx = levels.index(baseline_level)

    aucs = []
    for idx in range(min_n):
        ys = np.asarray([x_to_values[x][idx] for x in levels], dtype=float)
        baseline_y = float(ys[baseline_idx])
        if not np.isfinite(baseline_y) or np.isclose(baseline_y, 0.0):
            aucs.append(np.nan)
            continue
        drops = 1.0 - (ys / baseline_y)
        aucs.append(_integrate_trapezoid(drops, xs) / damage_range)
    return _finite(aucs).tolist()


def _ttest(a_values, b_values):
    a = _finite(a_values)
    b = _finite(b_values)
    if len(a) < 2 or len(b) < 2:
        return {"method": "insufficient_n", "stat": np.nan, "p": np.nan, "diff": np.nan}

    paired = len(a) == len(b)
    diff = float(np.mean(a) - np.mean(b))
    if paired and np.allclose(a, b, equal_nan=False):
        return {"method": "paired t-test", "stat": 0.0, "p": 1.0, "diff": 0.0}
    if not paired and np.isclose(diff, 0.0) and np.isclose(np.var(a), 0.0) and np.isclose(np.var(b), 0.0):
        return {"method": "Welch t-test", "stat": 0.0, "p": 1.0, "diff": 0.0}

    if scipy_stats is not None:
        if paired:
            stat, p = scipy_stats.ttest_rel(a, b, nan_policy="omit")
            method = "paired t-test"
        else:
            stat, p = scipy_stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
            method = "Welch t-test"
        return {"method": method, "stat": float(stat), "p": float(p), "diff": diff}

    if paired:
        d = a - b
        se = np.std(d, ddof=1) / math.sqrt(len(d))
        stat = float(np.mean(d) / se) if se > 0 else np.nan
        p = math.erfc(abs(stat) / math.sqrt(2)) if np.isfinite(stat) else np.nan
        return {"method": "paired z approx", "stat": stat, "p": p, "diff": diff}

    se = math.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
    stat = diff / se if se > 0 else np.nan
    p = math.erfc(abs(stat) / math.sqrt(2)) if np.isfinite(stat) else np.nan
    return {"method": "Welch z approx", "stat": float(stat), "p": p, "diff": diff}


def _print_group_header(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def _print_series_summary(series_values):
    print("Series summaries:")
    for label, values in series_values.items():
        arr = _finite(values)
        mean = float(np.mean(arr)) if len(arr) else np.nan
        ci95 = _ci95_mean(arr) if len(arr) else np.nan
        print(f"  {label}: n={len(arr)}, mean={mean:.6g}, 95% CI half-width={ci95:.6g}")


def _print_pairwise(series_values):
    print("Pairwise comparisons:")
    labels = list(series_values.keys())
    for a_label, b_label in itertools.combinations(labels, 2):
        result = _ttest(series_values[a_label], series_values[b_label])
        print(
            f"  {a_label} vs {b_label}: "
            f"diff={result['diff']:.6g}, "
            f"{result['method']}, stat={result['stat']:.6g}, p={result['p']:.6g}"
        )


def _line_label(layer, act, cat, categories, activations_layers):
    if len(categories) == 1 and len(activations_layers) == 1:
        return str(layer)
    return f"{layer}-{act}-{cat}"


def _lineplot_auc_stats(task_index, params):
    categories = params.get("categories", ("overall",))
    activations_layers = params.get("activations_layers", ())
    data, raw_points = _collect_categ_corr_lineplot_data(
        damage_layers=params.get("damage_layers", ()),
        activations_layers=activations_layers,
        damage_type=params.get("damage_type"),
        main_dir=params.get("main_dir", "data/haupt_stim_activ/damaged/cornet_rt5_all/"),
        categories=categories,
        metric=params.get("metric", "observed_difference"),
        subdir_regex=params.get("subdir_regex", r"damaged_([\d\.]+)(?:_|/|$)"),
        data_type=params.get("data_type", "selectivity"),
        verbose=params.get("verbose", 0),
        percentage=params.get("percentage", False),
        selectivity_fraction=params.get("selectivity_fraction"),
        selection_mode=params.get("selection_mode", "percentage"),
        selectivity_file=params.get("selectivity_file", "unit_selectivity/all_layers_units_mannwhitneyu.pkl"),
        model_tag=params.get("model_tag"),
    )

    series_values = {}
    for key, frac_dict in data.items():
        layer, act, cat = key
        if cat not in categories or not frac_dict:
            continue
        x_to_values = {
            float(x): list(map(float, raw_points.get(key, {}).get(x, [])))
            for x in sorted(frac_dict.keys())
        }
        label = _line_label(layer, act, cat, categories, activations_layers)
        try:
            series_values[label] = _relative_drop_auc_per_replicate(x_to_values)
        except ValueError as exc:
            print(f"  [WARN] {label}: {exc}")

    if len(series_values) < 2:
        print("  [SKIP] Fewer than two line series with replicate-level AUC values.")
        return

    _print_group_header(
        f"Lineplot AUC stats | task={task_index} | damage_type={params.get('damage_type')}"
    )
    _print_series_summary(series_values)
    _print_pairwise(series_values)


def _collect_total_bar_values(params):
    damage_layer = _as_list(params.get("damage_layers", ("IT",)))[0]
    activation_layer = _as_list(params.get("activations_layers", ("IT",)))[0]
    category = _as_list(params.get("categories", ("total",)))[0]
    main_dirs = _as_list(params.get("main_dir"))
    if not main_dirs:
        main_dirs = [params.get("main_dir")]

    series_values = {}
    for damage_type, levels in params.get("damage_pairs", {}).items():
        for level in _as_list(levels):
            label = f"{damage_type}, dmg={level}"
            for candidate_main_dir in main_dirs:
                try:
                    data, raw_points = _collect_categ_corr_lineplot_data(
                        damage_layers=[damage_layer],
                        activations_layers=[activation_layer],
                        damage_type=damage_type,
                        main_dir=candidate_main_dir,
                        categories=[category],
                        metric=params.get("metric", "observed_difference"),
                        subdir_regex=params.get("subdir_regex", r"damaged_([\d\.]+)(?:_|/|$)"),
                        data_type=params.get("data_type", "selectivity"),
                        verbose=params.get("verbose", 0),
                        percentage=params.get("percentage", False),
                        selectivity_fraction=params.get("selectivity_fraction"),
                        selection_mode=params.get("selection_mode", "percentage"),
                        selectivity_file=params.get("selectivity_file", "unit_selectivity/all_layers_units_mannwhitneyu.pkl"),
                        model_tag=params.get("model_tag"),
                    )
                except Exception as exc:
                    print(f"  [WARN] Could not collect {label} from {candidate_main_dir}: {exc}")
                    continue
                key = (damage_layer, activation_layer, category)
                frac_dict = data.get(key, {})
                level_float = float(level)
                matched_level = None
                for available in frac_dict:
                    if abs(float(available) - level_float) <= params.get("tolerance", 1e-6):
                        matched_level = available
                        break
                if matched_level is None:
                    continue
                series_values[label] = list(map(float, raw_points.get(key, {}).get(matched_level, [])))
                break
            if label not in series_values:
                print(f"  [WARN] No data found for total bar {damage_type}, dmg={level}")
    return series_values


def _total_bar_stats(task_index, params):
    series_values = _collect_total_bar_values(params)
    if len(series_values) < 2:
        print("  [SKIP] Fewer than two total bar series.")
        return
    _print_group_header(f"Total bar stats | task={task_index}")
    _print_series_summary(series_values)
    _print_pairwise(series_values)


def _collect_category_scaled_values(params):
    collector_categories = tuple(_normalize_category_name(c) for c in params.get("categories", ()))
    layer = params.get("layer", "IT")
    activation_layer = params.get("activation_layer", "IT")
    data, raw_points = _collect_categ_corr_lineplot_data(
        damage_layers=[layer],
        activations_layers=[activation_layer],
        damage_type=params.get("damage_type"),
        main_dir=params.get("main_dir"),
        categories=collector_categories,
        metric=params.get("metric", "observed_difference"),
        subdir_regex=params.get("subdir_regex", r"damaged_([\d\.]+)(?:_|/|$)"),
        data_type=params.get("data_type", "selectivity"),
        verbose=params.get("verbose", 0),
        percentage=False,
        selectivity_fraction=params.get("selectivity_fraction"),
        selection_mode=params.get("selection_mode", "percentage"),
        selectivity_file=params.get("selectivity_file", "unit_selectivity/all_layers_units_mannwhitneyu.pkl"),
        model_tag=params.get("model_tag"),
    )

    rows = []
    for cat in collector_categories:
        key = (layer, activation_layer, cat)
        for level, rec in data.get(key, {}).items():
            vals = raw_points.get(key, {}).get(level, [])
            if not vals:
                vals = [rec[0]]
            for replicate, value in enumerate(vals):
                rows.append({
                    "damage_type": params.get("damage_type"),
                    "damage_layer": layer,
                    "activation_layer": activation_layer,
                    "damage_level": level,
                    "category": cat,
                    "replicate": replicate,
                    "differentiation": value,
                })

    if not rows:
        return {}

    summary, _matched = _category_scaled_percent_from_df(
        pd.DataFrame(rows),
        damage_type=params.get("damage_type"),
        damage_level=params.get("damage_level"),
        layer=layer,
        category_col="category",
        y_col="differentiation",
        damage_type_col="damage_type",
        damage_level_col="damage_level",
        layer_col="damage_layer",
        activation_layer=activation_layer,
        activation_layer_col="activation_layer",
        tolerance=params.get("tolerance", 1e-6),
        category_order=params.get("categories", ()),
        verbose=params.get("verbose", 0),
    )
    return {
        row["category_label"]: row["raw_values"]
        for _, row in summary.iterrows()
    }


def _category_bar_stats(task_index, params):
    series_values = _collect_category_scaled_values(params)
    if len(series_values) < 2:
        print("  [SKIP] Fewer than two category bar series.")
        return
    _print_group_header(
        f"Category scaled bar stats | task={task_index} | "
        f"damage_type={params.get('damage_type')} | level={params.get('damage_level')}"
    )
    _print_series_summary(series_values)
    _print_pairwise(series_values)


def run_stats(config_path):
    config_path = Path(config_path)
    with config_path.open("r") as handle:
        config = yaml.safe_load(handle)

    tasks = config.get("tasks", [])
    print(f"Loaded {len(tasks)} tasks from {config_path}")

    for idx, task in enumerate(tasks):
        function = task.get("function")
        params = dict(task.get("parameters", {}))

        if function == "categ_corr_lineplot":
            if params.get("side_summary_metric") not in (None, "auc_loss"):
                continue
            _lineplot_auc_stats(idx, params)
        elif function == "plot_total_differentiation_bar":
            _total_bar_stats(idx, params)
        elif function == "plot_category_relative_drop_bar":
            _category_bar_stats(idx, params)


def main():
    parser = argparse.ArgumentParser(description="Print paper figure statistics to stdout.")
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/plots/plot_paper_figs.yaml",
        help="Plot YAML config to mirror for statistical tests.",
    )
    args = parser.parse_args()
    run_stats(args.config)


if __name__ == "__main__":
    main()
