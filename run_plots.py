import yaml
from utils import categ_corr_lineplot, plot_categ_differences, plot_avg_corr_mat, pair_corr_scatter_subplots, damage_type_lineplot

def run_function(function_name, **kwargs):
    """Executes the specified function with the given parameters."""
    functions = {
        "categ_corr_lineplot": categ_corr_lineplot,
        "plot_categ_differences": plot_categ_differences,
        "plot_avg_corr_mat": plot_avg_corr_mat,
        "pair_corr_scatter_subplots": pair_corr_scatter_subplots,
        "damage_type_lineplot": damage_type_lineplot
    }

    if function_name in functions:
        functions[function_name](**kwargs)
    else:
        raise ValueError(f"Function '{function_name}' is not defined.")

def main(config_path):
    """Main function to execute functions based on YAML config."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    if config["subdir"] is not None:
        print("subdir found")
        subdir = config["subdir"]
    else:
        subdir = ""
    for task in config.get("tasks", []):
        function_name = task.get("function")
        parameters = task.get("parameters", {})
        # Add plot directory to params
        parameters["plot_dir"] = "plots/" + subdir
        print(parameters)

        if function_name:
            print(f"Running {function_name} with parameters: {parameters}")
            run_function(function_name, **parameters)
        else:
            print("No function specified for a task.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run functions based on a YAML config file.")
    parser.add_argument("config", help="Path to the YAML configuration file.")

    args = parser.parse_args()
    main(args.config)
