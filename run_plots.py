import argparse
from pathlib import Path

import yaml

from utils import (
    categ_corr_lineplot,
    grouped_categ_corr_lineplot,
    plot_categ_differences,
    plot_avg_corr_mat,
    pair_corr_scatter_subplots,
    damage_type_lineplot,
    plot_category_relative_drop_bar,
    plot_total_differentiation_bar,
)


PLOT_FUNCTIONS = {
    "categ_corr_lineplot": categ_corr_lineplot,
    "grouped_categ_corr_lineplot": grouped_categ_corr_lineplot,
    "plot_categ_differences": plot_categ_differences,
    "plot_avg_corr_mat": plot_avg_corr_mat,
    "pair_corr_scatter_subplots": pair_corr_scatter_subplots,
    "damage_type_lineplot": damage_type_lineplot,
    "plot_category_relative_drop_bar": plot_category_relative_drop_bar,
    "plot_total_differentiation_bar": plot_total_differentiation_bar,
}


def run_function(function_name, **kwargs):
    try:
        plot_function = PLOT_FUNCTIONS[function_name]
    except KeyError:
        raise ValueError(f"Function '{function_name}' is not defined.") from None
    plot_function(**kwargs)


def main(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file) or {}

    subdir = config.get("subdir") or ""
    plot_dir = str(Path("plots") / subdir)

    for task in config.get("tasks", []):
        function_name = task.get("function")
        parameters = dict(task.get("parameters", {}))
        parameters["plot_dir"] = plot_dir

        if function_name:
            print(f"Running {function_name} with parameters: {parameters}")
            run_function(function_name, **parameters)
        else:
            print("No function specified for a task.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run functions based on a YAML config file.")
    parser.add_argument("config", help="Path to the YAML configuration file.")
    args = parser.parse_args()
    main(args.config)
