#!/usr/bin/env python3
"""Active-learning replay on archived Catapult data (experiment A: depth).

Hide the pool's labels, start from a small random seed set, let a selection rule
"synthesize" a batch by looking up the stored result, retrain (warm start), and
record test accuracy against cumulative Catapult hours. No synthesis is run.
Design note: wa-hls4ml-paper/HANDOFF_ACTIVE_LEARNING.md.

    python replay.py --raw $SCRATCH/al_replay/raw --cost $SCRATCH/al_replay/cost_L12.tsv \
        --rules random coverage mcdropout --seeds 0 --out $SCRATCH/al_replay/A

Pool = L1+L2 (minus --pool-exclude runs); test = a fixed sample of L3 plus L4..L10.
Each round appends one JSON line per (seed, rule, round) to <out>/curves.jsonl.

Rules
  random     uniform over the unlabeled pool (R0).
  coverage   greedy: designs that add the most per-layer contexts not yet labeled
             (layer type, in/out width, precision, RF, activation, strategy, io); R2.
  mcdropout  highest MC-dropout variance of log predictions (T passes); a cheap
             stand-in for R3's ensemble disagreement.
  mcdropout_hour  mcdropout score divided by estimated Catapult hours (mean measured
             cost of labeled designs of the same depth); R4.

Experiment A' (depth choice): --pool-groups L1 L2 L3 --pool-subset L3=<ids.tsv>
--test-groups L4 L5 L6 L8 L10, with the L3 designs' cost table added to --cost.
"""
import argparse
import heapq
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "GNN"))
from Models import FPGA_GNN_SumDecomp  # noqa: E402

# Feature layout of dataset/catapult_asic_to_numpy.py (same as transformer/GNN/Dataset2.py)
NUM_COLS = [0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 15]
LAYER_TYPE, ACT, PAD, STRAT, IO = 9, 10, 14, 8, 17
N_LAYER_TYPES, N_ACT, N_PAD = 12, 6, 3
CONTEXT_COLS = [9, 0, 3, 6, 7, 10, 8, 17]
TARGETS = ["LATENCY", "AREA"]
LOG_SHIFT = 1e-6


# ---------------------------------------------------------------- data
def load_group(raw, name):
    X = np.load(os.path.join(raw, f"{name}_features.npy"))
    y = np.load(os.path.join(raw, f"{name}_labels.npy"))[:, :2]   # latency, area
    with open(os.path.join(raw, f"{name}_ids.tsv")) as f:
        next(f)
        ids = [tuple(line.rstrip("\n").split("\t")) for line in f]
    return X, y, ids


def feature_stats(X):
    """Mean/std of numerical columns over valid layer rows, -1 entries ignored (Dataset2 rule)."""
    rows = X[~np.all(X == -1, axis=2)]
    means, stds = [], []
    for c in NUM_COLS:
        v = rows[:, c]
        v = v[v != -1]
        means.append(v.mean() if len(v) else 0.0)
        s = v.std() if len(v) else 1.0
        stds.append(s if s > 1e-5 else 1.0)
    return np.array(means, np.float32), np.array(stds, np.float32)


def one_hot(v, n):
    out = np.zeros((len(v), n), np.float32)
    ok = (v >= 0) & (v < n)
    out[np.nonzero(ok)[0], v[ok].astype(int)] = 1.0
    return out


def to_graphs(X, fmean, fstd):
    """Same tensors as FPGAGraphDataset.get (Dataset2.py), without labels."""
    graphs = []
    for d in X:
        rows = d[~np.all(d == -1, axis=1)]
        num = rows[:, NUM_COLS].copy()
        num[num == -1] = 0.0
        num = (num - fmean) / fstd
        x = np.concatenate([num, one_hot(rows[:, LAYER_TYPE], N_LAYER_TYPES),
                            one_hot(rows[:, ACT], N_ACT), one_hot(rows[:, PAD], N_PAD)], axis=1)
        n = len(rows)
        ei = (torch.stack([torch.arange(n - 1), torch.arange(1, n)]) if n > 1
              else torch.empty((2, 0), dtype=torch.long))
        g = Data(x=torch.from_numpy(x), edge_index=ei)
        g.strategy = torch.from_numpy(one_hot(rows[:1, STRAT], 2))
        g.io_type = torch.from_numpy(one_hot(rows[:1, IO], 2))
        graphs.append(g)
    return graphs


def contexts(X):
    """Per design, the set of per-layer context tuples (valid rows only)."""
    out = []
    for d in X:
        rows = d[~np.all(d == -1, axis=1)]
        out.append({tuple(r[CONTEXT_COLS].tolist()) for r in rows})
    return out


def load_cost(paths, ids):
    """Catapult hours per pool design from design_cost.py tables. Uses all_s (every
    attempt, retries included) when the table has it, else catapult_s."""
    table = {}
    for path in paths:
        with open(path) as f:
            hdr = next(f).rstrip("\n").split("\t")
            col = "all_s" if "all_s" in hdr else "catapult_s"
            ir, is_, ic, ist = hdr.index("run"), hdr.index("stem"), hdr.index(col), hdr.index("status")
            for line in f:
                p = line.rstrip("\n").split("\t")
                if p[ist] == "ok":
                    table[(p[ir], p[is_])] = float(p[ic])
        print(f"cost: {path} ({col})")
    cost = np.array([table.get(k, np.nan) for k in ids])
    miss = np.isnan(cost)
    if miss.any():
        print(f"cost: {miss.sum()} of {len(cost)} pool designs have no cost; using the median")
        cost[miss] = np.nanmedian(cost)
    return cost / 3600.0


# ---------------------------------------------------------------- model
class Trainer:
    def __init__(self, node_dim, device, args):
        self.device, self.args = device, args
        self.model = FPGA_GNN_SumDecomp(node_feature_dim=node_dim, num_targets=2, hidden_dim=256,
                                        num_gnn_layers=4, num_attention_heads=4, mlp_hidden_dim=128,
                                        dropout_rate=0.2).to(device)
        self.lmean = self.lstd = None

    def set_label_stats(self, y):
        ly = np.log(y + LOG_SHIFT)
        self.lmean = torch.tensor(ly.mean(0), device=self.device)
        self.lstd = torch.tensor(np.where(ly.std(0) > 1e-5, ly.std(0), 1.0), device=self.device)

    def norm_log(self, total):
        return (torch.log(total.clamp_min(0.0) + LOG_SHIFT) - self.lmean) / self.lstd

    def _loader(self, graphs, y=None, shuffle=False):
        if y is not None:   # labeled copies: the shared pool graphs must stay label-free
            yn = (np.log(y + LOG_SHIFT) - self.lmean.cpu().numpy()) / self.lstd.cpu().numpy()
            graphs = [Data(x=g.x, edge_index=g.edge_index, strategy=g.strategy, io_type=g.io_type,
                           y=torch.tensor(t, dtype=torch.float32).unsqueeze(0))
                      for g, t in zip(graphs, yn)]
        return DataLoader(graphs, batch_size=self.args.batch_size, shuffle=shuffle)

    def fit(self, tr_g, tr_y, va_g, va_y, epochs, min_steps=0):
        a, m = self.args, self.model
        opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=5e-6)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=8, min_lr=1e-7)
        mse = nn.MSELoss()
        tl, vl = self._loader(tr_g, tr_y, True), self._loader(va_g, va_y)
        # Small labeled sets give few batches per epoch: guarantee a number of gradient
        # steps (min_steps) and count patience in steps too, so round 0 is not undertrained.
        bpe = max(1, len(tl))
        epochs = max(epochs, -(-min_steps // bpe))
        patience = max(a.patience, -(-a.patience_steps // bpe))
        best, bad, best_state, ep = float("inf"), 0, None, 0
        for ep in range(1, epochs + 1):
            m.train()
            for b in tl:
                b = b.to(self.device)
                opt.zero_grad()
                loss = mse(self.norm_log(m(b)), b.y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
            m.eval()
            s, n = 0.0, 0
            with torch.no_grad():
                for b in vl:
                    b = b.to(self.device)
                    s += mse(self.norm_log(m(b)), b.y).item() * b.num_graphs
                    n += b.num_graphs
            v = s / max(n, 1)
            sched.step(v)
            if v < best:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in m.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        m.load_state_dict(best_state)
        return ep, best

    @torch.no_grad()
    def predict(self, graphs, dropout=False):
        self.model.train(dropout)
        out = []
        for b in DataLoader(graphs, batch_size=self.args.eval_batch_size):
            out.append(self.model(b.to(self.device)).cpu().numpy())
        self.model.eval()
        return np.concatenate(out)


def metrics(y, p):
    out = {}
    for i, t in enumerate(TARGETS):
        yt, yp = y[:, i], p[:, i]
        r2 = 1.0 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2)
        smape = 100.0 * np.mean(2 * np.abs(yp - yt) / (np.abs(yt) + np.abs(yp) + 1e-8))
        out[t] = {"r2": float(r2), "smape": float(smape)}
    return out


# ---------------------------------------------------------------- rules
def pick_random(rng, unl, k, **_):
    return rng.choice(unl, size=min(k, len(unl)), replace=False)


def pick_coverage(rng, unl, k, ctx, covered, **_):
    # lazy greedy: gains only shrink as coverage grows, so a popped entry whose
    # recomputed gain is unchanged is the true maximum. Random tie-break.
    tie = rng.permutation(len(unl))
    heap = [(-len(ctx[i] - covered), t, i) for t, i in zip(tie, unl)]
    heapq.heapify(heap)
    chosen, cov = [], set(covered)
    while heap and len(chosen) < k:
        g, t, i = heapq.heappop(heap)
        now = len(ctx[i] - cov)
        if now < -g:
            heapq.heappush(heap, (-now, t, i))
            continue
        chosen.append(i)
        cov |= ctx[i]
    return np.array(chosen)


def pick_mcdropout(rng, unl, k, trainer, pool_g, T, **_):
    g = [pool_g[i] for i in unl]
    preds = np.stack([np.log(trainer.predict(g, dropout=True) + LOG_SHIFT) for _ in range(T)])
    score = (preds.var(0) / preds.var(0).mean(0)).sum(1)   # per target, scaled so both count
    return unl[np.argsort(-score)[:k]]


def pick_mcdropout_hour(rng, unl, k, trainer, pool_g, T, grp, cost_h, labeled, **_):
    """R4: MC-dropout score per estimated hour. The estimate is the mean measured cost
    of the labeled designs in the same depth group (known only after synthesis)."""
    g = [pool_g[i] for i in unl]
    preds = np.stack([np.log(trainer.predict(g, dropout=True) + LOG_SHIFT) for _ in range(T)])
    score = (preds.var(0) / preds.var(0).mean(0)).sum(1)
    known = {d: cost_h[labeled & (grp == d)].mean() for d in np.unique(grp) if np.any(labeled & (grp == d))}
    fallback = cost_h[labeled].mean()
    est = np.array([known.get(d, fallback) for d in grp[unl]])
    return unl[np.argsort(-(score / est))[:k]]


RULES = {"random": pick_random, "coverage": pick_coverage, "mcdropout": pick_mcdropout,
         "mcdropout_hour": pick_mcdropout_hour}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", required=True)
    ap.add_argument("--cost", required=True, nargs="+", help="design_cost.py TSV(s)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rules", nargs="+", default=["random", "coverage", "mcdropout"], choices=list(RULES))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--pool-groups", nargs="+", default=["L1", "L2"])
    ap.add_argument("--pool-exclude", nargs="*", default=["run_20260613"],
                    help="run-name prefixes left out of the pool (default: 1-layer sz128 runs "
                         "made after the paper's splits)")
    ap.add_argument("--pool-subset", nargs="*", default=[], metavar="GROUP=IDS_TSV",
                    help="keep only these (run, stem) of a pool group, e.g. L3=L3_pool_ids.tsv")
    ap.add_argument("--test-groups", nargs="+", default=["L3", "L4", "L5", "L6", "L8", "L10"])
    ap.add_argument("--l3-sample", type=int, default=65000, help="fixed random L3 subset (0 = all)")
    ap.add_argument("--seed-frac", type=float, default=0.01)
    ap.add_argument("--batch-frac", type=float, default=0.01)
    ap.add_argument("--max-frac", type=float, default=0.30, help="stop when this share of the pool is labeled")
    ap.add_argument("--max-rounds", type=int, default=0, help="stop after this many rounds (0 = no limit)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--epochs-first", type=int, default=200)
    ap.add_argument("--epochs-round", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--min-steps-first", type=int, default=4000, help="gradient steps at least, round 0 and cold retrain")
    ap.add_argument("--min-steps-round", type=int, default=1500, help="gradient steps at least, warm-started rounds")
    ap.add_argument("--patience-steps", type=int, default=1000, help="early-stopping patience in steps (or --patience epochs, the larger)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--eval-batch-size", type=int, default=4096)
    ap.add_argument("--mc-passes", type=int, default=10)
    ap.add_argument("--final-retrain", action="store_true", help="cold retrain at the end of each curve")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=1)
    t0 = time.time()

    # pool
    parts = [load_group(a.raw, g) for g in a.pool_groups]
    X = np.concatenate([p[0] for p in parts]) if len({p[0].shape[1] for p in parts}) == 1 else None
    if X is None:
        L = max(p[0].shape[1] for p in parts)
        X = np.concatenate([np.pad(p[0], ((0, 0), (0, L - p[0].shape[1]), (0, 0)), constant_values=-1)
                            for p in parts])
    y = np.concatenate([p[1] for p in parts])
    ids = [i for p in parts for i in p[2]]
    grp = np.array([g for g, p in zip(a.pool_groups, parts) for _ in p[2]])
    subset = {}
    for spec in a.pool_subset:
        g, path = spec.split("=", 1)
        with open(path) as f:
            next(f)
            subset[g] = {tuple(line.rstrip("\n").split("\t")[:2]) for line in f}
    keep = np.array([not any(r.startswith(e) for e in a.pool_exclude) and
                     (g not in subset or (r, s) in subset[g]) for (r, s), g in zip(ids, grp)])
    X, y, grp, ids = X[keep], y[keep], grp[keep], [i for i, k in zip(ids, keep) if k]
    cost_h = load_cost(a.cost, ids)
    N = len(ids)
    print(f"pool: {N} designs, {cost_h.sum():.1f} Catapult hours in all; " +
          ", ".join(f"{g} {np.sum(grp == g)} ({cost_h[grp == g].sum():.0f} h)" for g in a.pool_groups))

    fmean, fstd = feature_stats(X)          # inputs are known for every candidate: no label leak
    pool_g = to_graphs(X, fmean, fstd)
    ctx = contexts(X)
    n_dense = ((X[:, :, LAYER_TYPE] == 1) & ~np.all(X == -1, axis=2)).sum(1)   # as run_sumdecomp.py

    # tests
    tests = {}
    for g in a.test_groups:
        Xt, yt, _ = load_group(a.raw, g)
        if g == "L3" and a.l3_sample and len(Xt) > a.l3_sample:
            sel = np.sort(np.random.default_rng(42).choice(len(Xt), a.l3_sample, replace=False))
            Xt, yt = Xt[sel], yt[sel]
        tests[g] = (to_graphs(Xt, fmean, fstd), yt)
        print(f"test {g}: {len(yt)} designs")
    print(f"data ready in {time.time() - t0:.0f}s on {device}")

    log = open(os.path.join(a.out, "curves.jsonl"), "a")
    k_seed, k_batch = max(1, int(a.seed_frac * N)), max(1, int(a.batch_frac * N))
    for seed in a.seeds:
        rng0 = np.random.default_rng(seed)
        seed_set = rng0.choice(N, k_seed, replace=False)
        is_val = rng0.random(N) < a.val_frac           # fixed per design, so val grows with labels
        for rule in a.rules:
            torch.manual_seed(seed)
            rng = np.random.default_rng([seed, list(RULES).index(rule)])
            trainer = Trainer(pool_g[0].x.shape[1], device, a)
            labeled = np.zeros(N, bool)
            labeled[seed_set] = True
            trainer.set_label_stats(y[seed_set])       # fixed for the whole curve
            trainer.model.init_head_bias(y[seed_set].mean(0), n_dense[seed_set].mean())
            rnd = 0
            while True:
                lab = np.nonzero(labeled)[0]
                tr, va = lab[~is_val[lab]], lab[is_val[lab]]
                if len(va) == 0:
                    tr, va = lab[:-1], lab[-1:]
                ts = time.time()
                ep, best = trainer.fit([pool_g[i] for i in tr], y[tr], [pool_g[i] for i in va], y[va],
                                       a.epochs_first if rnd == 0 else a.epochs_round,
                                       a.min_steps_first if rnd == 0 else a.min_steps_round)
                t_train = time.time() - ts
                ts = time.time()
                res = {g: metrics(yt, trainer.predict(gs)) for g, (gs, yt) in tests.items()}
                t_eval = time.time() - ts
                row = dict(seed=seed, rule=rule, round=rnd, n_labeled=int(labeled.sum()),
                           frac=float(labeled.mean()), hours=float(cost_h[labeled].sum()),
                           by_group={g: int(np.sum(labeled & (grp == g))) for g in a.pool_groups},
                           epochs=ep, val_loss=best, t_train=t_train, t_eval=t_eval, test=res)
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(f"seed {seed} {rule:9s} r{rnd:02d} n={row['n_labeled']:6d} h={row['hours']:7.1f} "
                      f"[{' '.join(f'{g}:{c}' for g, c in row['by_group'].items())}] "
                      f"ep={ep:3d} train {t_train:5.0f}s eval {t_eval:4.0f}s | " +
                      " ".join(f"{g}:{res[g]['LATENCY']['r2']:.3f}/{res[g]['AREA']['r2']:.3f}" for g in res),
                      flush=True)
                rnd += 1
                if labeled.mean() >= a.max_frac or (a.max_rounds and rnd >= a.max_rounds):
                    break
                unl = np.nonzero(~labeled)[0]
                covered = set().union(*(ctx[i] for i in lab))
                ts = time.time()
                pick = RULES[rule](rng, unl, k_batch, ctx=ctx, covered=covered, trainer=trainer,
                                   pool_g=pool_g, T=a.mc_passes, grp=grp, cost_h=cost_h, labeled=labeled)
                print(f"  picked {len(pick)} by {rule} in {time.time() - ts:.0f}s", flush=True)
                labeled[pick] = True
            np.save(os.path.join(a.out, f"labeled_s{seed}_{rule}.npy"), np.nonzero(labeled)[0])
            if a.final_retrain:
                cold = Trainer(pool_g[0].x.shape[1], device, a)
                cold.lmean, cold.lstd = trainer.lmean, trainer.lstd
                cold.model.init_head_bias(y[seed_set].mean(0), n_dense[seed_set].mean())
                lab = np.nonzero(labeled)[0]
                tr, va = lab[~is_val[lab]], lab[is_val[lab]]
                ep, best = cold.fit([pool_g[i] for i in tr], y[tr], [pool_g[i] for i in va], y[va], a.epochs_first, a.min_steps_first)
                res = {g: metrics(yt, cold.predict(gs)) for g, (gs, yt) in tests.items()}
                log.write(json.dumps(dict(seed=seed, rule=rule, round="final_cold", n_labeled=int(labeled.sum()),
                                          frac=float(labeled.mean()), hours=float(cost_h[labeled].sum()),
                                          epochs=ep, val_loss=best, test=res)) + "\n")
                log.flush()
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
