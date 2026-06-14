#!/usr/bin/env python3
"""Generic depth-split builder: arbitrary train-group / test-group layer sets.
Reuses build_layer_splits helpers. Example:
  python build_split_generic.py --raw <raw> --out <out>/split_1234 \
      --train-groups 1 2 3 4 --test-groups 5
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_layer_splits import load_group, repad, write_split, pool_trainval

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--train-groups", type=int, nargs="+", required=True)
ap.add_argument("--test-groups", type=int, nargs="+", required=True)
ap.add_argument("--val-frac", type=float, default=0.15)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

groups = sorted(set(a.train_groups + a.test_groups))
G = {n: load_group(a.raw, f"L{n}") for n in groups}
L = max(f.shape[1] for f, _ in G.values())
G = {n: (repad(f, L), l) for n, (f, l) in G.items()}
print(f"groups={groups} global max_layers={L}")
rng = np.random.default_rng(a.seed)
(Xtr, ytr), (Xv, yv) = pool_trainval([G[n][0] for n in a.train_groups], [G[n][1] for n in a.train_groups], rng, a.val_frac)
Xte = np.concatenate([G[n][0] for n in a.test_groups]); yte = np.concatenate([G[n][1] for n in a.test_groups])
write_split(a.out, {"train": (Xtr, ytr), "val": (Xv, yv), "test": (Xte, yte)})
