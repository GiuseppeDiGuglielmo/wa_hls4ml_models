# GNN/run_sumdecomp.py
#
# Sum-decomposition (deep-sets) GNN surrogate for DEPTH EXTRAPOLATION.
# The model predicts a non-negative per-node (per-layer) contribution and SUMS over nodes to get
# the design total. Trained only on graph-level totals, but the additive inductive bias lets a
# model trained on shallow (1-2 layer) designs extend to deeper (3-layer) designs.
#
# Differs from run_gnn.py: the model returns the LINEAR total, so we apply log+z-score inside the
# loss (using the dataset's saved label stats) to keep the objective in the same normalized-log
# space as the other models (comparable metrics).
#
# Example (extrapolation split):
#   cd GNN && python run_sumdecomp.py --epochs 200 --lr 1e-3 --batch-size 256 \
#       --base-dir /pscratch/sd/a/arghyara/catapult_asic_data/split_layerextrap \
#       --drop-throughput --thruput-lookup ../dataset/thruput_lookup.pkl

import os
import sys
import argparse
import pickle
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(_REPO, "transformer"))
sys.path.insert(0, os.path.join(_REPO, "transformer", "GNN"))

from Models import (FPGA_GNN_SumDecomp, FPGA_GNN_SumDecompCorr,         # noqa: E402
                    FPGA_GNN_GATv2_SumDecomp, FPGA_GNN_GATv2_SumDecompCorr)
from Dataset2 import create_dataloaders_from_split_data                 # noqa: E402
from train import (calculate_metrics, calculate_metrics_per_feature,    # noqa: E402
                   calculate_metrics_by_layer_count)
from plot import plot_loss, plot_box_plots_symlog, plot_results_simplified  # noqa: E402
from run_gnn import get_model_types                                     # noqa: E402


def _t(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def main():
    ap = argparse.ArgumentParser(description="Sum-decomposition GNN for depth extrapolation.")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--base-dir", default="/pscratch/sd/a/arghyara/catapult_asic_data/split_layerextrap")
    ap.add_argument("--drop-throughput", action="store_true")
    ap.add_argument("--thruput-lookup", default=os.path.join(_REPO, "dataset", "thruput_lookup.pkl"))
    ap.add_argument("--num-workers", type=int, default=int(os.environ.get("WA_NUM_WORKERS", "16")))
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--variant", choices=["plain", "corr", "gatv2", "gatv2corr"], default="plain",
                    help="plain = small GATv2 + Σcᵢ; corr = + bounded correction; "
                         "gatv2 = lui-gnn GATv2 (512/5/5, residual) + Σcᵢ; gatv2corr = gatv2 + bounded correction")
    ap.add_argument("--optimizer", choices=["adamw", "nadam"], default="adamw")
    ap.add_argument("--run-tag", default="", help="extra label appended to the output dir (avoids collisions)")
    ap.add_argument("--finetune-from", default="", help="pretrained best_model dir or model.pt to init weights from (transfer learning)")
    ap.add_argument("--freeze-encoder", action="store_true", help="freeze the GNN encoder; fine-tune only node_head")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | arch: sumdecomp")
    timestamp = datetime.now().strftime("%m_%d_%H_%M")
    vsuffix = {"plain": "", "corr": "corr", "gatv2": "gatv2", "gatv2corr": "gatv2corr"}[args.variant]
    rtag = f"_{args.run_tag}" if args.run_tag else ""
    tag = f"sumdecomp{vsuffix}_{args.optimizer}_{timestamp}_{args.epochs}epochs_{args.lr}lr_{args.batch_size}bs{rtag}"
    outdir = os.path.join(_THIS_DIR, "gnn_results_plots", tag)
    best_model_dir = os.path.join(outdir, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)
    ckpt = os.path.join(best_model_dir, "model.pt")

    base_dir = args.base_dir
    _label_cols = [0, 1] if args.drop_throughput else None
    _stats_save = os.path.join(best_model_dir, "lognormalization_stats.npy")
    _stats_load = os.path.join(base_dir, "lognormalization_stats.npy")

    train_loader, val_loader, test_loader, node_feature_dim, num_targets = create_dataloaders_from_split_data(
        train_features_path=os.path.join(base_dir, "train_features.npy"),
        train_labels_path=os.path.join(base_dir, "train_labels.npy"),
        val_features_path=os.path.join(base_dir, "val_features.npy"),
        val_labels_path=os.path.join(base_dir, "val_labels.npy"),
        test_features_path=os.path.join(base_dir, "test_features.npy"),
        test_labels_path=os.path.join(base_dir, "test_labels.npy"),
        stats_load_path=_stats_load, stats_save_path=_stats_save,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), mode="gnn",
        use_log_transform=True, log_epsilon=1e-6, label_cols=_label_cols,
    )
    for loader in (train_loader, val_loader, test_loader):
        loader.dataset.mode = "gnn"

    output_features = ["LATENCY", "AREA"] if args.drop_throughput else ["LATENCY", "AREA", "THROUGHPUT"][:num_targets]
    print(f"node_feature_dim={node_feature_dim} | num_targets={num_targets} | output_features={output_features}")

    # label-normalization stats (for the log-space loss on the summed LINEAR total)
    ds = train_loader.dataset
    lmean = _t(ds.label_means, device)
    lstd = _t(ds.label_stds, device)
    lshift = float(ds.log_shift)

    def norm_log(total):
        return (torch.log(total.clamp_min(0.0) + lshift) - lmean) / lstd

    if args.variant == "gatv2":
        ModelCls = FPGA_GNN_GATv2_SumDecomp
        mkw = dict(hidden_dim=512, num_gnn_layers=5, num_attention_heads=5, mlp_hidden_dim=128, dropout_rate=0.2)
    elif args.variant == "gatv2corr":
        ModelCls = FPGA_GNN_GATv2_SumDecompCorr
        mkw = dict(hidden_dim=512, num_gnn_layers=5, num_attention_heads=5, mlp_hidden_dim=128, dropout_rate=0.2)
    elif args.variant == "corr":
        ModelCls = FPGA_GNN_SumDecompCorr
        mkw = dict(hidden_dim=256, num_gnn_layers=4, num_attention_heads=4, mlp_hidden_dim=128, dropout_rate=0.2)
    else:
        ModelCls = FPGA_GNN_SumDecomp
        mkw = dict(hidden_dim=256, num_gnn_layers=4, num_attention_heads=4, mlp_hidden_dim=128, dropout_rate=0.2)
    model = ModelCls(node_feature_dim=node_feature_dim, num_targets=num_targets, **mkw).to(device)
    print(f"variant: {args.variant} ({ModelCls.__name__})")

    # transfer learning: initialize from a pretrained (e.g. 45nm) checkpoint
    if args.finetune_from:
        src = args.finetune_from
        if os.path.isdir(src):
            src = os.path.join(src, "model.pt")
        model.load_state_dict(torch.load(src, map_location=device), strict=True)
        print(f"FINE-TUNE: loaded pretrained weights from {src}")

    # init the per-node head bias near the data magnitude so the summed total starts sensibly
    # (for fine-tuning this resets only the head's output bias to the NEW node's scale — the
    #  ~log(area_ratio) shift — while keeping the pretrained encoder + head transform)
    raw_train = np.load(os.path.join(base_dir, "train_labels.npy"))
    raw_train = raw_train[:, _label_cols] if _label_cols is not None else raw_train
    feats_train = np.load(os.path.join(base_dir, "train_features.npy"))
    mean_layers = np.mean([sum(1 for r in d if not np.all(r == -1) and int(r[9]) == 1) for d in feats_train])
    model.init_head_bias(raw_train.mean(axis=0), mean_layers)
    print(f"init head bias from mean_total={raw_train.mean(axis=0)} mean_layers={mean_layers:.2f}")

    if args.freeze_encoder:
        for name, p in model.named_parameters():
            if not name.startswith("node_head"):
                p.requires_grad_(False)
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        print(f"FREEZE-ENCODER: training node_head only — {n_tr:,}/{n_all:,} params")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model sumdecomp: {n_params:,} parameters")
    trainable = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "nadam":
        optimizer = torch.optim.NAdam(trainable, lr=args.lr, weight_decay=5e-6)
    else:
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=5e-6)
    print(f"optimizer: {args.optimizer} | lr {args.lr}")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8, min_lr=1e-7)
    mse = nn.MSELoss()

    best_val, bad = float("inf"), 0
    train_losses, val_losses = [], []
    for epoch in range(1, args.epochs + 1):
        model.train()
        tl, n = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred_total = model(batch)                 # LINEAR total
            loss = mse(norm_log(pred_total), batch.y)  # compare in normalized-log space
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item() * batch.num_graphs
            n += batch.num_graphs
        tl /= max(n, 1)

        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                vl += mse(norm_log(model(batch)), batch.y).item() * batch.num_graphs
                vn += batch.num_graphs
        vl /= max(vn, 1)
        scheduler.step(vl)
        train_losses.append(tl)
        val_losses.append(vl)
        print(f"Epoch {epoch:3d} — Train Loss: {tl:.4f} | Validation Loss: {vl:.4f}")
        if vl < best_val:
            best_val, bad = vl, 0
            torch.save(model.state_dict(), ckpt)
            print(f"  ↳ New best model (val={vl:.4f}), saved to {ckpt}")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping at epoch {epoch} (no val improvement for {args.patience} epochs)")
                break
    model.load_state_dict(torch.load(ckpt, map_location=device))

    # ---- Test eval ----  model outputs LINEAR totals directly
    model.eval()
    yt_norm, yp_lin = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            yp_lin.append(model(batch).cpu().numpy())
            y = batch.y
            yt_norm.append((y.squeeze(1) if y.dim() == 3 else y).cpu().numpy())
    yp_denorm = np.concatenate(yp_lin, axis=0)                                   # already LINEAR
    yt_denorm = test_loader.dataset.denormalize_labels(torch.tensor(np.concatenate(yt_norm, axis=0))).numpy()

    test_features_np = np.load(os.path.join(base_dir, "test_features.npy"))
    if args.drop_throughput:
        with open(args.thruput_lookup, "rb") as f:
            lut = pickle.load(f)
        der = []
        for d in test_features_np:
            rows = [r for r in d if not np.all(r == -1)]
            lt = [lut[(int(r[0]), int(r[3]), int(r[6]), int(r[7]))]
                  for r in rows if int(r[9]) == 1 and (int(r[0]), int(r[3]), int(r[6]), int(r[7])) in lut]
            der.append(float(max(lt)) if lt else 0.0)
        der = np.array(der).reshape(-1, 1)
        yp_denorm = np.hstack([yp_denorm, der])
        raw = np.load(os.path.join(base_dir, "test_labels.npy"))
        yt_denorm = np.hstack([yt_denorm, raw[:, 2:3]])
        output_features = ["LATENCY", "AREA", "THROUGHPUT"]

    print("\n===== OVERALL TEST METRICS (sum-decomposition) =====")
    calculate_metrics(yt_denorm, yp_denorm)
    calculate_metrics_per_feature(yt_denorm, yp_denorm, output_features)
    print("\n\n--- Metrics by layer count ---")
    calculate_metrics_by_layer_count(yt_denorm, yp_denorm, test_features_np, output_features)

    plots_dir = os.path.join(outdir, "plots")
    plot_loss(train_losses, val_losses, outdir=plots_dir)
    plot_box_plots_symlog(yp_denorm, yt_denorm, folder_name=outdir, output_features=output_features)
    plot_results_simplified(name="run1", mpl_plots=True, y_test=yt_denorm, y_pred=yp_denorm,
                            output_features=output_features, folder_name=outdir, model_types=get_model_types(test_features_np))
    print(f"\nBest checkpoint: {ckpt}\nPlots under: {outdir}")


if __name__ == "__main__":
    main()
