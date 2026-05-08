import argparse
import os
import numpy as np


def split_and_save(features_path, labels_path, output_dir,
                   split=(0.70, 0.15, 0.15), seed=42):
    X = np.load(features_path)
    y = np.load(labels_path)
    assert X.shape[0] == y.shape[0], "features and labels must have the same number of samples"

    N = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)

    n_train = int(split[0] * N)
    n_val   = int(split[1] * N)

    splits = {
        "train": idx[:n_train],
        "val":   idx[n_train:n_train + n_val],
        "test":  idx[n_train + n_val:],
    }

    os.makedirs(output_dir, exist_ok=True)
    for name, ids in splits.items():
        np.save(os.path.join(output_dir, f"{name}_features.npy"), X[ids])
        np.save(os.path.join(output_dir, f"{name}_labels.npy"),   y[ids])
        print(f"{name:5s}: features {X[ids].shape}  labels {y[ids].shape}")

    print(f"Saved splits to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split numpy arrays into train/val/test.")
    parser.add_argument("--features", required=True, help="Path to features .npy file")
    parser.add_argument("--labels",   required=True, help="Path to labels .npy file")
    parser.add_argument("--output",   required=True, help="Output directory")
    parser.add_argument("--split",    nargs=3, type=float, default=[0.70, 0.15, 0.15],
                        metavar=("TRAIN", "VAL", "TEST"),
                        help="Train/val/test fractions (default: 0.70 0.15 0.15)")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    total = sum(args.split)
    assert abs(total - 1.0) < 1e-6, f"Split fractions must sum to 1.0 (got {total})"

    split_and_save(args.features, args.labels, args.output, tuple(args.split), args.seed)
