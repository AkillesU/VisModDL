import os
import glob
import random
import pickle
import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from collections import defaultdict

def run_svm(image_indices, labels_dict, n_test_per_image, all_data, seed=None):
    """
    image_indices:  list of int indices, e.g., [2,1]
    labels_dict:    dict mapping each index -> label, e.g., {2:0,1:1}
    n_test_per_image: int number of test samples per index
    all_data:       big list of (feature_vector, idx)
    seed:           random seed (optional) to allow reproducible splits
    
    Returns:
        accuracy_test (float)
    """
    # Optionally set a random seed so each run can be different if desired
    if seed is not None:
        random.seed(seed)

    # Group relevant data
    grouped = defaultdict(list)
    for features, idx in all_data:
        if idx in image_indices:
            grouped[idx].append(features)

    train_data = []
    test_data = []
    
    # For each idx in the chosen set, do the train/test split
    for idx in image_indices:
        feature_list = grouped[idx]
        random.shuffle(feature_list)

        if len(feature_list) < n_test_per_image:
            raise ValueError(
                f"Not enough samples ({len(feature_list)}) for image {idx} "
                f"to allocate {n_test_per_image} test samples."
            )

        test_feats = feature_list[:n_test_per_image]
        train_feats = feature_list[n_test_per_image:]

        for f in test_feats:
            test_data.append((f, idx))
        for f in train_feats:
            train_data.append((f, idx))

    # Convert to NumPy
    X_train, y_train = [], []
    for features, idx in train_data:
        X_train.append(features)
        y_train.append(labels_dict[idx])
    X_train = np.array(X_train, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.float64)

    X_test, y_test = [], []
    for features, idx in test_data:
        X_test.append(features)
        y_test.append(labels_dict[idx])
    X_test = np.array(X_test, dtype=np.float64)
    y_test = np.array(y_test, dtype=np.float64)

    # (Optional) clamp any infinities/NaNs in raw data
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e308, neginf=-1e308)
    X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=1e308, neginf=-1e308)

    # Print distributions
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    #print("Training set distribution (label: count):")
    #for lbl, cnt in zip(unique_train, counts_train):
        #print(f"  Label {lbl}: {cnt}")

    unique_test, counts_test = np.unique(y_test, return_counts=True)
    #print("Test set distribution (label: count):")
    #for lbl, cnt in zip(unique_test, counts_test):
        #print(f"  Label {lbl}: {cnt}")

    # Scale with RobustScaler
    scaler = RobustScaler()
    scaler.fit(X_train)

    # Ensure minimal scale so we avoid division by tiny numbers
    min_epsilon = 1e-6
    scaler.scale_[scaler.scale_ < min_epsilon] = min_epsilon

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Final clip if needed
    X_train_scaled = np.clip(X_train_scaled, -1e308, 1e308)
    X_test_scaled  = np.clip(X_test_scaled,  -1e308, 1e308)

    # Train SVM
    clf = SVC(kernel="linear", random_state=42)
    clf.fit(X_train_scaled, y_train)

    # Evaluate
    preds_test = clf.predict(X_test_scaled)
    accuracy_test = np.mean(preds_test == y_test)

    #print("Test predictions:", preds_test)
    #print("Test labels:     ", y_test)
    print(f"Test Accuracy:   {accuracy_test:.2f}")
    print("-" * 50)
    return accuracy_test


if __name__ == "__main__":
    # Directories
    pkl_dirs = [
        "data/haupt_stim_activ/damaged/cornet_rt5_c/connections/V1/activations/IT/damaged_0.1",
        "data/haupt_stim_activ/damaged/cornet_rt5_c/connections/V1/activations/IT/damaged_0.85",
        "data/haupt_stim_activ/damaged/cornet_rt5_c/connections/V1/activations/V1/damaged_0.1",
        "data/haupt_stim_activ/damaged/cornet_rt5_c/connections/V1/activations/V1/damaged_0.85",
        "data/haupt_stim_activ/damaged/cornet_rt5_c/noise/V1/activations/IT/damaged_0.5",
        "data/haupt_stim_activ/damaged/cornet_rt5_c/noise/V1/activations/IT/damaged_2.5",
        "data/haupt_stim_activ/damaged/cornet_rt5_c/noise/V1/activations/V1/damaged_0.5",
        "data/haupt_stim_activ/damaged/cornet_rt5_c/noise/V1/activations/V1/damaged_2.5"
    ]

    # Index groups and label dicts
    image_indices = [

        [60, 59]
    ]
    labels_dicts = [

        {60: 0, 59: 1}
    ]

    # Test set sizes
    n_test_per_image_list = [198,199]

    # We'll store accuracies in a dictionary keyed by (pkl_dir, tuple_of_idxs, n_test)
    results_dict = defaultdict(list)

    for pkl_dir in pkl_dirs:
        print(f"\n=== Loading data from directory: {pkl_dir} ===")
        pkl_files = sorted(glob.glob(os.path.join(pkl_dir, "*.pkl")))

        all_data = []
        # Collect all interesting indices
        all_interesting_indices = set()
        for idx_list in image_indices:
            all_interesting_indices.update(idx_list)

        # Load
        for pkl_file in pkl_files:
            #print(f"  Loading {pkl_file} ...")
            df = pd.read_pickle(pkl_file)
            df = df.drop("numeric_index", axis=1, errors="ignore")
            df.columns = df.columns.astype(str)

            for idx in all_interesting_indices:
                row_np = df.iloc[idx].to_numpy()
                all_data.append((row_np, idx))

        print(f"  Total loaded samples: {len(all_data)}\n")

        # Now run SVM multiple times for each combination
        for i, idx_list in enumerate(image_indices):
            label_dict = labels_dicts[i]
            for n_test in n_test_per_image_list:
                # Repeat 5 times
                for run_i in range(10):
                    print(f"DIR: {pkl_dir} | Image indices: {idx_list} | "
                          f"N_test: {n_test} | Run #{run_i+1}")
                    accuracy = run_svm(
                        image_indices=idx_list,
                        labels_dict=label_dict,
                        n_test_per_image=n_test,
                        all_data=all_data,
                        seed=42 + run_i  # optional for distinct splits
                    )
                    # Store result
                    key = (pkl_dir, tuple(idx_list), n_test)
                    results_dict[key].append(accuracy)

    print("\n=== FINAL AVERAGE ACCURACIES ===")
    for (pkl_dir, idxs, n_test), acc_list in results_dict.items():
        avg_acc = np.mean(acc_list)
        print(f"{pkl_dir}, indices={list(idxs)}, N_test={n_test}: "
              f"{avg_acc:.2f} (averaged over {len(acc_list)} runs)")
