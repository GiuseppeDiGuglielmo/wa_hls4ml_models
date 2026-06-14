#!/usr/bin/env python3
"""Build depth-extrapolation splits from per-layer-count numpy groups.

Inputs (produced by catapult_asic_to_numpy.py, one --prefix per group):
    <raw>/L1_features.npy, L1_labels.npy   (1-layer designs)
    <raw>/L2_*.npy, L3_*.npy, L4_*.npy

Outputs (each a base-dir for run_sumdecomp.py):
    <out>/split_12/   train+val = shuffle(L1 u L2)  85/15 ; test = L3 u L4
    <out>/split_123/  train+val = shuffle(L1 u L2 u L3) 85/15 ; test = L4

Groups may have different max-layer padding; we re-pad every array to the global
max with -1 before concatenating, matching the converter's padding convention.
Labels keep all 3 columns [latency, area, thruput] (driver selects [0,1] +
throughput-LUT col).
"""
import argparse
import os
import numpy as np


def load_group(raw, name):
    f = np.load(os.path.join(raw, f"{name}_features.npy"))
    l = np.load(os.path.join(raw, f"{name}_labels.npy"))
    return f, l


def repad(arr, L):
    """(N, Li, F) -> (N, L, F), padding extra layer rows with -1."""
    if arr.shape[1] == L:
        return arr
    out = np.full((arr.shape[0], L, arr.shape[2]), -1.0, dtype=arr.dtype)
    out[:, :arr.shape[1], :] = arr
    return out


def dense_count_hist(feats):
    """Distribution of Dense-layer counts (feature col 9 == 1) across designs."""
    counts = {}
    for d in feats:
        n = sum(1 for r in d if not np.all(r == -1) and int(r[9]) == 1)
        counts[n] = counts.get(n, 0) + 1
    return dict(sorted(counts.items()))


def write_split(outdir, parts):
    os.makedirs(outdir, exist_ok=True)
    for nm, (X, y) in parts.items():
        np.save(os.path.join(outdir, f"{nm}_features.npy"), X.astype(np.float32))
        np.save(os.path.join(outdir, f"{nm}_labels.npy"), y.astype(np.float32))
    comp = {nm: dense_count_hist(X) for nm, (X, y) in parts.items()}
    print(f"  wrote {outdir}")
    for nm in ("train", "val", "test"):
        X, y = parts[nm]
        print(f"    {nm:5s}: N={len(X):>7d}  layer-counts={comp[nm]}")


def pool_trainval(groups_X, groups_y, rng, val_frac):
    X = np.concatenate(groups_X, axis=0)
    y = np.concatenate(groups_y, axis=0)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    nval = int(val_frac * len(X))
    return (X[nval:], y[nval:]), (X[:nval], y[:nval])  # (train), (val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="dir with L{1,2,3,4}_{features,labels}.npy")
    ap.add_argument("--out", required=True, help="output root (creates split_12/ and split_123/)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    G = {n: load_group(args.raw, f"L{n}") for n in (1, 2, 3, 4)}
    for n, (f, l) in G.items():
        print(f"L{n}: features {f.shape}  labels {l.shape}  layer-counts={dense_count_hist(f)}")

    L = max(f.shape[1] for f, _ in G.values())
    print(f"\nglobal max_layers (padding axis) = {L}")
    G = {n: (repad(f, L), l) for n, (f, l) in G.items()}

    rng = np.random.default_rng(args.seed)

    # ---- split_12: train/val from {1,2}; test = {3,4} ----
    print("\n=== split_12  (train 1+2 -> test 3 [Run A] and 4 [Run B]) ===")
    (Xtr, ytr), (Xv, yv) = pool_trainval([G[1][0], G[2][0]], [G[1][1], G[2][1]], rng, args.val_frac)
    Xte = np.concatenate([G[3][0], G[4][0]], axis=0)
    yte = np.concatenate([G[3][1], G[4][1]], axis=0)
    write_split(os.path.join(args.out, "split_12"),
                {"train": (Xtr, ytr), "val": (Xv, yv), "test": (Xte, yte)})

    # ---- split_123: train/val from {1,2,3}; test = {4} ----
    print("\n=== split_123  (train 1+2+3 -> test 4 [Run C]) ===")
    rng2 = np.random.default_rng(args.seed)
    (Xtr, ytr), (Xv, yv) = pool_trainval([G[1][0], G[2][0], G[3][0]],
                                         [G[1][1], G[2][1], G[3][1]], rng2, args.val_frac)
    Xte = G[4][0]; yte = G[4][1]
    write_split(os.path.join(args.out, "split_123"),
                {"train": (Xtr, ytr), "val": (Xv, yv), "test": (Xte, yte)})

    print("\nDONE.")


if __name__ == "__main__":
    main()
