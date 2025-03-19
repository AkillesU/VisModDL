from utils import svm_process_directory
import yaml

def main(config_path):
    """Main function to execute functions based on YAML config."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    parent_dir = config["parent_dir"]
    training_samples = config["training_samples"]
    activation_layers = config["activation_layers"]

    svm_process_directory(parent_dir, training_samples=training_samples, allowed_subdirs=activation_layers)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run functions based on a YAML config file.")
    parser.add_argument("config", help="Path to the YAML configuration file.")

    args = parser.parse_args()
    main(args.config)
