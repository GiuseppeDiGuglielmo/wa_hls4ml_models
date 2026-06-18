#!/usr/bin/env python3
"""Build a random 70/15/15 train/val/test split of all 22nm {L1+L2+L3} designs.
Saves to $SCRATCH/catapult_22nm/rand123/.  No layer-depth held out — pure in-distribution.
Used for the 'scratch vs FT-from-45nm-{1,2,3}' comparison study."""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_layer_splits import load_group, repad, write_split

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True, help="dir with L1_features.npy, L2_features.npy, L3_features.npy")
ap.add_argument("--out", required=True, help="output dir (rand123/ will be created here)")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

G = {n: load_group(a.raw, f"L{n}") for n in (1, 2, 3)}
L = max(f.shape[1] for f, _ in G.values())
G = {n: (repad(f, L), lbl) for n, (f, lbl) in G.items()}
n1, n2, n3 = len(G[1][0]), len(G[2][0]), len(G[3][0])
print(f"Loaded: L1={n1}  L2={n2}  L3={n3}  total={n1+n2+n3}  max_layers={L}")

X_all = np.concatenate([G[n][0] for n in (1, 2, 3)], axis=0)
y_all = np.concatenate([G[n][1] for n in (1, 2, 3)], axis=0)
N = len(X_all)
rng = np.random.default_rng(a.seed)
perm = rng.permutation(N)
X_all, y_all = X_all[perm], y_all[perm]

n_val  = int(0.15 * N)
n_test = int(0.15 * N)
n_train = N - n_val - n_test

Xtr, ytr = X_all[:n_train],          y_all[:n_train]
Xv,  yv  = X_all[n_train:n_train+n_val], y_all[n_train:n_train+n_val]
Xte, yte = X_all[n_train+n_val:],    y_all[n_train+n_val:]

out = os.path.join(a.out, "rand123")
write_split(out, {"train": (Xtr, ytr), "val": (Xv, yv), "test": (Xte, yte)})
print(f"Wrote {out}: train={n_train}  val={n_val}  test={n_test}")
