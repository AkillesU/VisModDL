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


def _permutation_test(a_values, b_values, n_permutations=10000, rng=None):
    """
    Two-sided permutation p-value for the mean difference.

    Equal-length arrays are treated as paired replicate-level measurements and
    tested with sign flips of the within-replicate differences. Unequal-length
    arrays use an unpaired label-shuffle test. P-values use the +1 correction to
    avoid returning zero from finite Monte Carlo samples.
    """
    a = _finite(a_values)
    b = _finite(b_values)
    if len(a) < 2 or len(b) < 2:
        return {
            "method": "insufficient_n",
            "stat": np.nan,
            "p": np.nan,
            "diff": np.nan,
            "iterations": 0,
        }

    if rng is None:
        rng = np.random.default_rng()

    paired = len(a) == len(b)
    observed = float(np.mean(a) - np.mean(b))
    if paired and np.allclose(a, b, equal_nan=False):
        return {
            "method": "paired sign-flip permutation",
            "stat": 0.0,
            "p": 1.0,
            "diff": 0.0,
            "iterations": int(n_permutations),
        }

    if paired:
        diffs = a - b
        diffs = diffs[np.isfinite(diffs)]
        if len(diffs) < 2:
            return {
                "method": "insufficient_n",
                "stat": np.nan,
                "p": np.nan,
                "diff": observed,
                "iterations": 0,
            }
        count = 0
        done = 0
        batch_size = min(1000, int(n_permutations))
        while done < int(n_permutations):
            current = min(batch_size, int(n_permutations) - done)
            signs = rng.choice((-1.0, 1.0), size=(current, len(diffs)))
            null_stats = np.mean(signs * diffs, axis=1)
            count += int(np.sum(np.abs(null_stats) >= (abs(observed) - 1e-12)))
            done += current
        p_value = (count + 1.0) / (int(n_permutations) + 1.0)
        return {
            "method": "paired sign-flip permutation",
            "stat": observed,
            "p": float(p_value),
            "diff": observed,
            "iterations": int(n_permutations),
        }

    combined = np.concatenate([a, b])
    n_a = len(a)
    if np.isclose(observed, 0.0) and np.isclose(np.var(a), 0.0) and np.isclose(np.var(b), 0.0):
        return {
            "method": "unpaired label permutation",
            "stat": 0.0,
            "p": 1.0,
            "diff": 0.0,
            "iterations": int(n_permutations),
        }

    null_stats = np.empty(int(n_permutations), dtype=float)
    for idx in range(int(n_permutations)):
        shuffled = rng.permutation(combined)
        null_stats[idx] = float(np.mean(shuffled[:n_a]) - np.mean(shuffled[n_a:]))

    count = int(np.sum(np.abs(null_stats) >= (abs(observed) - 1e-12)))
    p_value = (count + 1.0) / (int(n_permutations) + 1.0)
    return {
        "method": "unpaired label permutation",
        "stat": observed,
        "p": float(p_value),
        "diff": observed,
        "iterations": int(n_permutations),
    }


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


def _print_pairwise(series_values, n_permutations=10000, rng=None):
    print("Pairwise comparisons:")
    labels = list(series_values.keys())
    for a_label, b_label in itertools.combinations(labels, 2):
        t_result = _ttest(series_values[a_label], series_values[b_label])
        perm_result = _permutation_test(
            series_values[a_label],
            series_values[b_label],
            n_permutations=n_permutations,
            rng=rng,
        )
        print(
            f"  {a_label} vs {b_label}: "
            f"diff={t_result['diff']:.6g}; "
            f"parametric={t_result['method']}, stat={t_result['stat']:.6g}, p={t_result['p']:.6g}; "
            f"permutation={perm_result['method']}, "
            f"iterations={perm_result['iterations']}, "
            f"stat={perm_result['stat']:.6g}, p={perm_result['p']:.6g}"
        )


def _line_label(layer, act, cat, categories, activations_layers):
    if len(categories) == 1 and len(activations_layers) == 1:
        return str(layer)
    return f"{layer}-{act}-{cat}"


def _lineplot_auc_stats(task_index, params, n_permutations=10000, rng=None):
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
    _print_pairwise(series_values, n_permutations=n_permutations, rng=rng)


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


def _total_bar_stats(task_index, params, n_permutations=10000, rng=None):
    series_values = _collect_total_bar_values(params)
    if len(series_values) < 2:
        print("  [SKIP] Fewer than two total bar series.")
        return
    _print_group_header(f"Total bar stats | task={task_index}")
    _print_series_summary(series_values)
    _print_pairwise(series_values, n_permutations=n_permutations, rng=rng)


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


def _category_bar_stats(task_index, params, n_permutations=10000, rng=None):
    series_values = _collect_category_scaled_values(params)
    if len(series_values) < 2:
        print("  [SKIP] Fewer than two category bar series.")
        return
    _print_group_header(
        f"Category scaled bar stats | task={task_index} | "
        f"damage_type={params.get('damage_type')} | level={params.get('damage_level')}"
    )
    _print_series_summary(series_values)
    _print_pairwise(series_values, n_permutations=n_permutations, rng=rng)


def run_stats(config_path, n_permutations=10000, seed=12345):
    config_path = Path(config_path)
    with config_path.open("r") as handle:
        config = yaml.safe_load(handle)

    rng = np.random.default_rng(seed)
    tasks = config.get("tasks", [])
    print(f"Loaded {len(tasks)} tasks from {config_path}")
    print(f"Permutation tests: n_permutations={int(n_permutations)}, seed={seed}")

    for idx, task in enumerate(tasks):
        function = task.get("function")
        params = dict(task.get("parameters", {}))

        if function == "categ_corr_lineplot":
            if params.get("side_summary_metric") not in (None, "auc_loss"):
                continue
            _lineplot_auc_stats(idx, params, n_permutations=n_permutations, rng=rng)
        elif function == "plot_total_differentiation_bar":
            _total_bar_stats(idx, params, n_permutations=n_permutations, rng=rng)
        elif function == "plot_category_relative_drop_bar":
            _category_bar_stats(idx, params, n_permutations=n_permutations, rng=rng)


def main():
    parser = argparse.ArgumentParser(description="Print paper figure statistics to stdout.")
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/plots/plot_paper_figs.yaml",
        help="Plot YAML config to mirror for statistical tests.",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=10000,
        help="Number of Monte Carlo permutations/sign flips per pairwise test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for permutation tests.",
    )
    args = parser.parse_args()
    run_stats(args.config, n_permutations=args.n_permutations, seed=args.seed)


if __name__ == "__main__":
    main()
