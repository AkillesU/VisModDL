import os
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from utils import svm_process_file

def run_svm_on_directory(directory, training_samples=15, max_permutations=None):
    """
    Finds all .pkl files in 'directory', processes each with svm_process_file,
    concatenates them into one DataFrame, and returns it.
    """
    all_results = []

    pkl_files = sorted(glob.glob(os.path.join(directory, "*.pkl")))
    for pkl_file in tqdm(pkl_files, desc=f"Processing {directory}"):
        df_res = svm_process_file(
            pkl_file,
            training_samples=training_samples,
            max_permutations=max_permutations
        )
        if df_res is not None and len(df_res) > 0:
            # Optionally tag with 'pkl_file' so we know which file the rows came from
            df_res["source_file"] = os.path.basename(pkl_file)
            all_results.append(df_res)

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()


def main():
    directory = "test_activations/IT_connections_0.2_10/"
    training_samples = 15  # typically means 256 permutations possible

    # 1) Full permutations (256) => no limit
    print("Running full permutations (256) ...")
    df_full = run_svm_on_directory(directory, training_samples=training_samples, max_permutations=None)
    df_full.to_pickle("svm_results_full.pkl")
    #df_full = pd.read_pickle("svm_results_full.pkl")

    # 2) Limited permutations (50)
    print("Running limited permutations (50) ...")
    df_limited = run_svm_on_directory(directory, training_samples=training_samples, max_permutations=50)
    df_limited.to_pickle("svm_results_50.pkl")
    #df_limited = pd.read_pickle("svm_results_50.pkl")

    # 3) Compare distributions
    cat_cols = [c for c in df_full.columns if "_vs_" in c]
    # For each row, compute average across category pairs
    df_full["avg_acc"] = df_full[cat_cols].mean(axis=1)
    df_limited["avg_acc"] = df_limited[cat_cols].mean(axis=1)

    data_full = df_full["avg_acc"].values
    data_limited = df_limited["avg_acc"].values

    mean_full = np.mean(data_full)
    std_full  = np.std(data_full)
    mean_lim  = np.mean(data_limited)
    std_lim   = np.std(data_limited)

    # t-test
    tstat, pval = ttest_ind(data_full, data_limited, equal_var=False)

    print("\n=== Comparison of Full (256) vs. Limited (50) permutations ===")
    print(f"Full permutations: mean={mean_full:.4f}, std={std_full:.4f}, n={len(data_full)}")
    print(f"Limited (50)    : mean={mean_lim:.4f}, std={std_lim:.4f}, n={len(data_limited)}")
    print(f"T-test: t={tstat:.4f}, p={pval:.5g}")

if __name__ == "__main__":
    main()

