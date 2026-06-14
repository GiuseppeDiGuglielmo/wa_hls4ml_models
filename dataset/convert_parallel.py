#!/usr/bin/env python3
"""Parallel, memory-light Catapult report -> numpy converter.

Reuses CatapultASICProcessor.process_json (the exact, tested per-file logic) but
runs it across a process pool and stacks numpy arrays directly instead of holding
hundreds of thousands of pandas DataFrames. Output is byte-identical in content to
catapult_asic_to_numpy.py --archive, just faster and far lighter on memory.

  python convert_parallel.py --archive <mlp-Nlayer> --exclude <prefixes> \
      --output <dir> --prefix L3 --workers 32
"""
import argparse
import glob
import os
import sys
import numpy as np
from multiprocessing import Pool

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)
from catapult_asic_to_numpy import CatapultASICProcessor  # noqa: E402

_PROC = None


def _init():
    global _PROC
    _PROC = CatapultASICProcessor()


def _work(fp):
    try:
        df, labels = _PROC.process_json(fp)
    except Exception:
        return None
    if df is None or labels is None:
        return None
    feat = df.values.astype(np.float32)                                   # (n_layers, F)
    lab = np.array([labels[c] for c in _PROC.label_columns], dtype=np.float32)
    return feat, lab


def collect_files(archive, exclude):
    files, skipped = [], []
    for rd in sorted(glob.glob(os.path.join(archive, "run_*"))):
        if any(os.path.basename(rd).startswith(ex) for ex in exclude):
            skipped.append(os.path.basename(rd))
            continue
        files.extend(sorted(glob.glob(os.path.join(rd, "reports", "*.json"))))
    if skipped:
        print(f"  excluded {len(skipped)} run(s): {', '.join(skipped)}")
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--output", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("WA_NUM_WORKERS", "16")))
    args = ap.parse_args()

    files = collect_files(args.archive, args.exclude)
    print(f"[{args.prefix}] {len(files)} reports from {args.archive} (workers={args.workers})", flush=True)
    if not files:
        print(f"[{args.prefix}] NO FILES — abort"); sys.exit(2)

    feats, labs, bad = [], [], 0
    with Pool(args.workers, initializer=_init) as pool:
        for i, res in enumerate(pool.imap_unordered(_work, files, chunksize=64)):
            if res is None:
                bad += 1
            else:
                feats.append(res[0]); labs.append(res[1])
            if (i + 1) % 50000 == 0:
                print(f"  {i + 1}/{len(files)} processed", flush=True)

    if not feats:
        print(f"[{args.prefix}] No valid data."); sys.exit(3)
    L = max(f.shape[0] for f in feats)
    F = feats[0].shape[1]
    N = len(feats)
    X = np.full((N, L, F), -1.0, dtype=np.float32)
    y = np.zeros((N, len(labs[0])), dtype=np.float32)
    for i, (f, l) in enumerate(zip(feats, labs)):
        X[i, :f.shape[0], :] = f
        y[i] = l
    os.makedirs(args.output, exist_ok=True)
    np.save(os.path.join(args.output, f"{args.prefix}_features.npy"), X)
    np.save(os.path.join(args.output, f"{args.prefix}_labels.npy"), y)
    print(f"[{args.prefix}] saved features {X.shape} labels {y.shape}  (skipped {bad})", flush=True)


if __name__ == "__main__":
    main()
