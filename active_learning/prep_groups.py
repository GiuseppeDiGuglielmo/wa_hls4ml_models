#!/usr/bin/env python3
"""Convert archived Catapult reports to numpy, one file set per depth group, with ids.

Same per-report logic as dataset/convert_parallel.py (CatapultASICProcessor.process_json),
but the order is kept and each design's (run, stem) is written next to it, so the
active-learning replay can join synthesis cost and name the designs a rule picks.

    python prep_groups.py --archive <nangate45 dir> --groups 1 2 3 4 5 6 8 10 \
        --out $SCRATCH/al_replay/raw --workers 32

Writes <out>/L<N>_features.npy, L<N>_labels.npy, L<N>_ids.tsv (run, stem), one row
per design that the processor accepts. --exclude defaults to the runs the paper's
depth splits left out (slurm/prep_v2_arghya.sh).
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "dataset"))
from catapult_asic_to_numpy import CatapultASICProcessor  # noqa: E402

PAPER_EXCLUDE = ["run_20260504", "run_20260505", "run_20260506", "run_20260507_143427"]
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
    return (df.values.astype(np.float32),
            np.array([labels[c] for c in _PROC.label_columns], dtype=np.float32))


def collect(group_dir, exclude):
    files = []
    for rd in sorted(glob.glob(os.path.join(group_dir, "run_*"))):
        name = os.path.basename(rd)
        if any(name.startswith(ex) for ex in exclude):
            print(f"  excluded {name}")
            continue
        files.extend(sorted(glob.glob(os.path.join(rd, "reports", "*.json"))))
    return files


def convert(group_dir, prefix, out, exclude, workers):
    files = collect(group_dir, exclude)
    print(f"[{prefix}] {len(files)} reports from {group_dir}", flush=True)
    feats, labs, ids, bad = [], [], [], 0
    with Pool(workers, initializer=_init) as pool:
        for i, (fp, res) in enumerate(zip(files, pool.imap(_work, files, chunksize=64))):
            if res is None:
                bad += 1
                continue
            feats.append(res[0])
            labs.append(res[1])
            ids.append((os.path.basename(os.path.dirname(os.path.dirname(fp))),
                        os.path.basename(fp)[:-len(".json")]))
            if (i + 1) % 50000 == 0:
                print(f"  {i + 1}/{len(files)}", flush=True)
    L = max(f.shape[0] for f in feats)
    X = np.full((len(feats), L, feats[0].shape[1]), -1.0, dtype=np.float32)
    for i, f in enumerate(feats):
        X[i, :f.shape[0]] = f
    y = np.stack(labs)
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, f"{prefix}_features.npy"), X)
    np.save(os.path.join(out, f"{prefix}_labels.npy"), y)
    with open(os.path.join(out, f"{prefix}_ids.tsv"), "w") as f:
        f.write("run\tstem\n")
        f.writelines(f"{r}\t{s}\n" for r, s in ids)
    print(f"[{prefix}] features {X.shape} labels {y.shape} skipped {bad}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--archive", required=True, help="tech dir holding mlp-<N>layer/ folders")
    ap.add_argument("--groups", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", nargs="*", default=PAPER_EXCLUDE)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    for n in a.groups:
        convert(os.path.join(a.archive, f"mlp-{n}layer"), f"L{n}", a.out, a.exclude, a.workers)


if __name__ == "__main__":
    main()
