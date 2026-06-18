# GNN/eval_ckpt.py
#
# Eval-only: load a trained sum-decomp checkpoint (+ its saved normalization stats) and evaluate
# on an arbitrary features/labels .npy pair (e.g. the held-out 5-layer set). No training.
#
#   python eval_ckpt.py --ckpt-dir <run_dir>/best_model --variant plain \
#       --features $SCRATCH/catapult_v2/raw/L5_features.npy \
#       --labels   $SCRATCH/catapult_v2/raw/L5_labels.npy \
#       --tag D_seed1 [--json-out results.json]
import os
import sys
import json
import argparse
import pickle

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(_REPO, "transformer"))
sys.path.insert(0, os.path.join(_REPO, "transformer", "GNN"))

from Models import (FPGA_GNN_SumDecomp, FPGA_GNN_SumDecompCorr,         # noqa: E402
                    FPGA_GNN_GATv2_SumDecomp, FPGA_GNN_GATv2_SumDecompCorr,
                    FPGA_GNN, FPGA_GNN_GATv2)  # pooling baselines (output NORMALIZED space)
from Dataset2 import create_dataloaders_from_split_data                 # noqa: E402


def r2(a, p):
    ss = np.sum((a - p) ** 2); tot = np.sum((a - a.mean()) ** 2)
    return 1.0 - ss / tot if tot > 0 else 1.0


def smape(a, p):
    return float(100.0 * np.mean(2.0 * np.abs(p - a) / (np.abs(a) + np.abs(p) + 1e-12)))


def main():
    ap = argparse.ArgumentParser(description="Eval-only for sum-decomp checkpoints.")
    ap.add_argument("--ckpt-dir", required=True, help="run's best_model dir (model.pt + lognormalization_stats.npy)")
    ap.add_argument("--variant", choices=["plain", "corr", "gatv2", "gatv2corr",
                                           "poolsage", "poolgat"], default="plain")
    ap.add_argument("--features", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--thruput-lookup", default=os.path.join(_REPO, "dataset", "thruput_lookup.pkl"))
    ap.add_argument("--json-out", default="")
    ap.add_argument("--save-preds", default="", help="npz path to dump actual+pred arrays for plotting")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats_path = os.path.join(args.ckpt_dir, "lognormalization_stats.npy")
    ckpt_path = os.path.join(args.ckpt_dir, "model.pt")
    assert os.path.exists(stats_path) and os.path.exists(ckpt_path), f"missing artifacts in {args.ckpt_dir}"

    # all three loaders point at the SAME eval file; stats are LOADED from the checkpoint's
    # training stats (never recomputed), so features/labels are normalized exactly as in training.
    _, _, test_loader, node_feature_dim, num_targets = create_dataloaders_from_split_data(
        train_features_path=args.features, train_labels_path=args.labels,
        val_features_path=args.features, val_labels_path=args.labels,
        test_features_path=args.features, test_labels_path=args.labels,
        stats_load_path=stats_path, stats_save_path=None,
        batch_size=args.batch_size, num_workers=4, pin_memory=(device.type == "cuda"),
        mode="gnn", use_log_transform=True, log_epsilon=1e-6, label_cols=[0, 1],
    )
    test_loader.dataset.mode = "gnn"

    # Pooling baselines (FPGA_GNN / FPGA_GNN_GATv2) predict in NORMALIZED log-space and need
    # denormalize_labels() on the prediction; the SumDecomp models already emit LINEAR totals.
    POOLING = ("poolsage", "poolgat")
    if args.variant == "poolgat":   # run_gnn.py build_model('gatv2') config
        mkw = dict(hidden_dim=512, num_gnn_layers=5, num_attention_heads=5, mlp_hidden_dim=512,
                   dropout_rate=0.2, edge_dim=None, concat_heads=True, residual_connections=True)
    elif args.variant == "poolsage":  # run_gnn.py build_model('gnn') config
        mkw = dict(hidden_dim=256, num_gnn_layers=5, mlp_hidden_dim=128, dropout_rate=0.3)
    elif args.variant in ("gatv2", "gatv2corr"):
        mkw = dict(hidden_dim=512, num_gnn_layers=5, num_attention_heads=5, mlp_hidden_dim=128, dropout_rate=0.2)
    else:
        mkw = dict(hidden_dim=256, num_gnn_layers=4, num_attention_heads=4, mlp_hidden_dim=128, dropout_rate=0.2)
    ModelCls = {"plain": FPGA_GNN_SumDecomp, "corr": FPGA_GNN_SumDecompCorr,
                "gatv2": FPGA_GNN_GATv2_SumDecomp, "gatv2corr": FPGA_GNN_GATv2_SumDecompCorr,
                "poolsage": FPGA_GNN, "poolgat": FPGA_GNN_GATv2}[args.variant]
    model = ModelCls(node_feature_dim=node_feature_dim, num_targets=num_targets, **mkw).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    yp, yt = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            yp.append(model(batch).cpu().numpy())                      # SumDecomp: LINEAR; pooling: NORMALIZED
            y = batch.y
            yt.append((y.squeeze(1) if y.dim() == 3 else y).cpu().numpy())
    yp = np.concatenate(yp, axis=0)
    if args.variant in POOLING:  # invert z-score+log on the learned columns, same as the truth
        yp = test_loader.dataset.denormalize_labels(torch.tensor(yp)).numpy()
    yt = test_loader.dataset.denormalize_labels(torch.tensor(np.concatenate(yt, axis=0))).numpy()

    # throughput via exact LUT (sanity + completeness)
    feats = np.load(args.features)
    with open(args.thruput_lookup, "rb") as f:
        lut = pickle.load(f)
    miss = 0; der = []
    for d in feats:
        rows = [r for r in d if not np.all(r == -1)]
        keys = [(int(r[0]), int(r[3]), int(r[6]), int(r[7])) for r in rows if int(r[9]) == 1]
        vals = [lut[k] for k in keys if k in lut]
        miss += sum(1 for k in keys if k not in lut)
        der.append(float(max(vals)) if vals else 0.0)
    raw_labels = np.load(args.labels)
    thr_true = raw_labels[:, 2]
    thr_r2 = r2(thr_true, np.array(der))

    if args.save_preds:
        np.savez(args.save_preds,
                 lat_true=yt[:, 0], lat_pred=yp[:, 0], area_true=yt[:, 1], area_pred=yp[:, 1],
                 thr_true=thr_true, thr_pred=np.array(der))
        print(f"saved predictions -> {args.save_preds}")

    out = {"tag": args.tag, "ckpt": args.ckpt_dir, "variant": args.variant,
           "n": int(len(yp)), "lut_miss": int(miss), "THROUGHPUT_r2": round(thr_r2, 4)}
    for j, tg in enumerate(["LATENCY", "AREA"]):
        a, p = yt[:, j].astype(np.float64), yp[:, j].astype(np.float64)
        out[tg] = {"r2": round(float(r2(a, p)), 4), "smape_pct": round(float(smape(a, p)), 2),
                   "mae": round(float(np.mean(np.abs(p - a))), 2),
                   "median_ratio": round(float(np.median(p / np.clip(a, 1e-9, None))), 3)}
    print(json.dumps(out, indent=1))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
