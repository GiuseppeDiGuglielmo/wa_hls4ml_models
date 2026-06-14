#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the 22nm (gf22fdx) transfer-study splits. All runs share ONE fixed held-out 3-layer
test set so numbers are directly comparable. Emits:
  ft22_{100,500,2000,all}/ : train = N {1,2}-layer designs, val = fixed {1,2} val, test = fixed 3L
  ceil22/                  : train = {1,2} pool + (3L minus test), val = fixed {1,2} val, test = fixed 3L
The ft splits feed both from-scratch and fine-tuning (same data; the difference is --finetune-from)."""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_layer_splits import load_group, repad, write_split

ap = argparse.ArgumentParser()
ap.add_argument("--raw", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n-test", type=int, default=8000, help="held-out 3-layer test designs")
ap.add_argument("--n-val", type=int, default=1500, help="fixed {1,2}-layer val designs")
ap.add_argument("--sizes", type=int, nargs="+", default=[100, 500, 2000])
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

G = {n: load_group(a.raw, f"L{n}") for n in (1, 2, 3)}
L = max(f.shape[1] for f, _ in G.values())
G = {n: (repad(f, L), l) for n, (f, l) in G.items()}
print(f"22nm groups loaded; global max_layers={L}  | n1={len(G[1][0])} n2={len(G[2][0])} n3={len(G[3][0])}")
rng = np.random.default_rng(a.seed)

# ---- fixed held-out 3-layer TEST (shared by every run) ----
X3, y3 = G[3]
idx3 = rng.permutation(len(X3))
te, pool3 = idx3[:a.n_test], idx3[a.n_test:]
Xte, yte = X3[te], y3[te]
X3pool, y3pool = X3[pool3], y3[pool3]
print(f"fixed 3L test: {len(Xte)}   | 3L ceiling-train pool: {len(X3pool)}")

# ---- {1,2}-layer pool: fixed val + train pool ----
X12 = np.concatenate([G[1][0], G[2][0]]); y12 = np.concatenate([G[1][1], G[2][1]])
p = rng.permutation(len(X12)); X12, y12 = X12[p], y12[p]
Xv, yv = X12[:a.n_val], y12[:a.n_val]
Xtr, ytr = X12[a.n_val:], y12[a.n_val:]
print(f"fixed {{1,2}} val: {len(Xv)}   | {{1,2}} train pool: {len(Xtr)}")

sizes = [s for s in a.sizes if s <= len(Xtr)] + [len(Xtr)]
for N in sorted(set(sizes)):
    name = "all" if N == len(Xtr) else str(N)
    write_split(os.path.join(a.out, f"ft22_{name}"), {
        "train": (Xtr[:N], ytr[:N]), "val": (Xv, yv), "test": (Xte, yte)})
    print(f"  wrote ft22_{name}: train={N} val={len(Xv)} test={len(Xte)}")

# ---- in-distribution ceiling: {1,2} train pool + 3L pool ----
Xc = np.concatenate([Xtr, X3pool]); yc = np.concatenate([ytr, y3pool])
write_split(os.path.join(a.out, "ceil22"), {"train": (Xc, yc), "val": (Xv, yv), "test": (Xte, yte)})
print(f"  wrote ceil22: train={len(Xc)} (incl 3L) val={len(Xv)} test={len(Xte)}")
