# GNN/run_gnn.py
#
# Catapult-ASIC GNN surrogate driver — the GNN analogue of transformer/run.py.
# Trains a graph neural network (GraphSAGE baseline / GATv2 "lui-gnn" / enhanced GATv2)
# on the Catapult ASIC dataset, predicting [LATENCY, AREA] (+ THROUGHPUT derived
# analytically from the lookup table, matching the transformer's --drop-throughput v2 mode).
#
# Reuses, unchanged:
#   - transformer/GNN/Dataset2.py  (Catapult-ready FPGAGraphDataset, mode='gnn' -> PyG graphs)
#   - transformer/train.py         (calculate_metrics* — true-first arg order)
#   - dataset/thruput_lookup.pkl   (exact throughput LUT)
#   - GNN/Models.py                (the 3 GNN classes — NO edits; num_targets is a ctor arg)
#
# Example:
#   module load pytorch/2.8.0 && source $SCRATCH/venv_wa_hls4ml_models/bin/activate
#   cd GNN && python run_gnn.py --arch gatv2 --epochs 200 --lr 1e-3 --batch-size 256 \
#       --base-dir /pscratch/sd/a/arghyara/catapult_asic_data/split \
#       --drop-throughput --thruput-lookup ../dataset/thruput_lookup.pkl

import os
import sys
import argparse
import pickle
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

# --- path setup: reuse the transformer's Catapult-ready loader + metrics ---------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))            # <repo>/GNN
_REPO = os.path.dirname(_THIS_DIR)                                # <repo>
sys.path.insert(0, _THIS_DIR)                                    # Models.py
sys.path.insert(0, os.path.join(_REPO, "transformer"))          # train.py (metrics)
sys.path.insert(0, os.path.join(_REPO, "transformer", "GNN"))   # Dataset2.py (Catapult loader)

from Models import FPGA_GNN, FPGA_GNN_GATv2, FPGA_GNN_GATv2_Enhanced  # noqa: E402
from Dataset2 import create_dataloaders_from_split_data               # noqa: E402
from train import (                                                   # noqa: E402
    calculate_metrics,
    calculate_metrics_per_feature,
    calculate_metrics_by_layer_count,
)
from plot import plot_loss, plot_box_plots_symlog, plot_results_simplified  # noqa: E402


def build_model(arch, node_feature_dim, num_targets):
    """Construct the requested GNN. node_feature_dim/num_targets come from the loader."""
    if arch == "gnn":  # GraphSAGE baseline (FPGA_GNN)
        return FPGA_GNN(
            node_feature_dim=node_feature_dim, num_targets=num_targets,
            hidden_dim=256, num_gnn_layers=5, mlp_hidden_dim=128, dropout_rate=0.3,
        )
    if arch == "gatv2":  # the published "lui-gnn" (FPGA_GNN_GATv2)
        return FPGA_GNN_GATv2(
            node_feature_dim=node_feature_dim, num_targets=num_targets,
            hidden_dim=512, num_gnn_layers=5, num_attention_heads=5, mlp_hidden_dim=512,
            dropout_rate=0.2, edge_dim=None, concat_heads=True, residual_connections=True,
        )
    if arch == "gatv2_enhanced":  # experimental GATv2 + edge features (disabled: no edge_attr in Catapult data)
        return FPGA_GNN_GATv2_Enhanced(
            node_feature_dim=node_feature_dim, num_targets=num_targets,
            hidden_dim=128, num_gnn_layers=4, num_attention_heads=4, mlp_hidden_dim=160,
            dropout_rate=0.2, use_edge_features=False,
        )
    raise ValueError(f"unknown --arch {arch}")


def _targets(batch):
    """batch.y is [B, num_targets] after PyG batching; squeeze a stray middle dim if present."""
    y = batch.y
    return y.squeeze(1) if y.dim() == 3 else y


def get_model_types(features_np):
    """Label each design Dense/Conv1D/Conv2D from layer_type (feature index 9) for plot coloring."""
    model_types = []
    for design in features_np:
        valid_layers = [layer for layer in design if not np.all(layer == -1)]
        layer_types = [int(layer[9]) for layer in valid_layers]
        if any(lt == 2 for lt in layer_types):
            model_types.append("Conv1D")
        elif any(lt == 3 for lt in layer_types):
            model_types.append("Conv2D")
        else:
            model_types.append("Dense")
    return model_types


def main():
    parser = argparse.ArgumentParser(description="Catapult-ASIC GNN surrogate (mirrors transformer/run.py).")
    parser.add_argument("--arch", choices=["gnn", "gatv2", "gatv2_enhanced"], default="gatv2",
                        help="gnn=GraphSAGE baseline, gatv2=published lui-gnn, gatv2_enhanced=GATv2+edge feats")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--base-dir", type=str, default="/pscratch/sd/a/arghyara/catapult_asic_data/split")
    parser.add_argument("--drop-throughput", action="store_true",
                        help="learn LATENCY+AREA only; derive THROUGHPUT via the 1-layer lookup at inference")
    parser.add_argument("--thruput-lookup", type=str, default=os.path.join(_REPO, "dataset", "thruput_lookup.pkl"))
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("WA_NUM_WORKERS", "8")))
    parser.add_argument("--patience", type=int, default=30, help="early-stopping patience (epochs)")
    args = parser.parse_args()

    if args.drop_throughput and not args.thruput_lookup:
        raise ValueError("--thruput-lookup is required when --drop-throughput is set")

    timestamp = datetime.now().strftime("%m_%d_%H_%M")
    USE_LOG_TRANSFORM = True
    LOG_EPSILON = 1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | arch: {args.arch}")

    tag = f"{args.arch}_{timestamp}_{args.epochs}epochs_{args.lr}lr_{args.batch_size}bs"
    outdir = os.path.join(_THIS_DIR, "gnn_results_plots", ("testing_only_" if args.eval_only else "") + tag)
    best_model_dir = os.path.join(outdir, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)

    base_dir = args.base_dir

    # Stats live next to the checkpoint so they always match (same convention as run.py).
    if args.eval_only:
        _model_path = args.model_path or os.path.join(best_model_dir, "model.pt")
        _stats_load = os.path.join(os.path.dirname(_model_path), "lognormalization_stats.npy")
        _stats_save = None
    else:
        _stats_load = os.path.join(base_dir, "lognormalization_stats.npy")
        _stats_save = os.path.join(best_model_dir, "lognormalization_stats.npy")

    _label_cols = [0, 1] if args.drop_throughput else None

    train_loader, val_loader, test_loader, node_feature_dim, num_targets = create_dataloaders_from_split_data(
        train_features_path=os.path.join(base_dir, "train_features.npy"),
        train_labels_path=os.path.join(base_dir, "train_labels.npy"),
        val_features_path=os.path.join(base_dir, "val_features.npy"),
        val_labels_path=os.path.join(base_dir, "val_labels.npy"),
        test_features_path=os.path.join(base_dir, "test_features.npy"),
        test_labels_path=os.path.join(base_dir, "test_labels.npy"),
        stats_load_path=_stats_load,
        stats_save_path=_stats_save,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        mode="gnn",
        use_log_transform=USE_LOG_TRANSFORM,
        log_epsilon=LOG_EPSILON,
        label_cols=_label_cols,
    )
    # belt-and-suspenders: ensure graph mode on every loader (mirrors run.py:116-118)
    for loader in [train_loader, val_loader, test_loader]:
        loader.dataset.mode = "gnn"

    if args.drop_throughput:
        output_features = ["LATENCY", "AREA"]
    elif num_targets == 3:
        output_features = ["LATENCY", "AREA", "THROUGHPUT"]
    else:
        output_features = [f"TARGET_{i}" for i in range(num_targets)]

    print(f"node_feature_dim={node_feature_dim} | num_targets={num_targets} | output_features={output_features}")

    model = build_model(args.arch, node_feature_dim, num_targets).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model {args.arch}: {n_params:,} parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-6)
    loss_fn = nn.MSELoss()
    ckpt = os.path.join(best_model_dir, "model.pt")
    train_losses, val_losses = [], []

    if args.eval_only:
        model_path = args.model_path or ckpt
        print(f"Loading model weights from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        if args.resume:
            print(f"Warm-starting from checkpoint: {args.resume}")
            model.load_state_dict(torch.load(args.resume, map_location=device))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8, min_lr=1e-7)
        best_val = float("inf")
        bad_epochs = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            tl, n = 0.0, 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                preds = model(batch)
                loss = loss_fn(preds, _targets(batch))
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
                    vl += loss_fn(model(batch), _targets(batch)).item() * batch.num_graphs
                    vn += batch.num_graphs
            vl /= max(vn, 1)
            scheduler.step(vl)
            train_losses.append(tl)
            val_losses.append(vl)
            print(f"Epoch {epoch:3d} — Train Loss: {tl:.4f} | Validation Loss: {vl:.4f}")

            if vl < best_val:
                best_val = vl
                bad_epochs = 0
                torch.save(model.state_dict(), ckpt)
                print(f"  ↳ New best model (val={vl:.4f}), saved to {ckpt}")
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    print(f"Early stopping at epoch {epoch} (no val improvement for {args.patience} epochs)")
                    break
        # reload best checkpoint for evaluation
        model.load_state_dict(torch.load(ckpt, map_location=device))

    # ---- Test evaluation ----
    model.eval()
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            y_pred_list.append(model(batch).cpu().numpy())
            y_true_list.append(_targets(batch).cpu().numpy())
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)

    # Denormalize (inverts z-score + log on the learned columns only)
    y_true_denorm = test_loader.dataset.denormalize_labels(torch.tensor(y_true)).numpy()
    y_pred_denorm = test_loader.dataset.denormalize_labels(torch.tensor(y_pred)).numpy()

    test_features_np = np.load(os.path.join(base_dir, "test_features.npy"))  # (N, max_layers, 18)

    # Derive THROUGHPUT analytically (verbatim port of transformer/run.py:166-191)
    if args.drop_throughput:
        with open(args.thruput_lookup, "rb") as f:
            lut = pickle.load(f)
        thruput_derived = []
        for design in test_features_np:
            rows = [row for row in design if not np.all(row == -1)]
            layer_thruputs = []
            for row in rows:
                if int(row[9]) != 1:  # feature index 9 = layer_type; 1 = Dense
                    continue
                key = (int(row[0]), int(row[3]), int(row[6]), int(row[7]))
                t = lut.get(key)
                if t is not None:
                    layer_thruputs.append(t)
            thruput_derived.append(float(max(layer_thruputs)) if layer_thruputs else 0.0)
        thruput_derived = np.array(thruput_derived).reshape(-1, 1)
        y_pred_denorm = np.hstack([y_pred_denorm, thruput_derived])
        all_labels_raw = np.load(os.path.join(base_dir, "test_labels.npy"))
        y_true_denorm = np.hstack([y_true_denorm, all_labels_raw[:, 2:3]])
        output_features = ["LATENCY", "AREA", "THROUGHPUT"]

    print("\n===== OVERALL TEST METRICS =====")
    calculate_metrics(y_true_denorm, y_pred_denorm)
    calculate_metrics_per_feature(y_true_denorm, y_pred_denorm, output_features)
    print("\n\n--- Metrics by layer count ---")
    calculate_metrics_by_layer_count(y_true_denorm, y_pred_denorm, test_features_np, output_features)

    # ---- Plots (reuse the transformer's plot.py) ----
    plots_dir = os.path.join(outdir, "plots")
    if not args.eval_only and len(train_losses) > 0:
        plot_loss(train_losses, val_losses, outdir=plots_dir)
    test_model_types = get_model_types(test_features_np)
    plot_box_plots_symlog(y_pred_denorm, y_true_denorm, folder_name=outdir, output_features=output_features)
    plot_results_simplified(
        name="run1", mpl_plots=True, y_test=y_true_denorm, y_pred=y_pred_denorm,
        output_features=output_features, folder_name=outdir, model_types=test_model_types,
    )
    print(f"\nBest checkpoint: {ckpt}")
    print(f"Plots written under: {outdir}")


if __name__ == "__main__":
    main()
