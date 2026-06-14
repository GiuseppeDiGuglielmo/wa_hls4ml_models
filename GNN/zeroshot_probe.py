# GNN/zeroshot_probe.py
#
# Zero-shot transfer probe: apply a 45nm-trained sum-decomp checkpoint to 22nm data with NO
# fine-tuning, then fit a single per-target scalar on the 22nm {1,2}-layer set and re-measure.
# Tells us how much of the cross-node gap is "just a scale shift" vs structural.
#
#   python zeroshot_probe.py --ckpt-dir <45nm>/best_model --variant plain \
#     --fit-features 22nm/ft22_all/train_features.npy --fit-labels .../train_labels.npy \
#     --test-features 22nm/ft22_all/test_features.npy  --test-labels .../test_labels.npy
import os, sys, json, argparse
import numpy as np, torch

_THIS = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.dirname(_THIS)
sys.path.insert(0, _THIS); sys.path.insert(0, os.path.join(_REPO, "transformer"))
sys.path.insert(0, os.path.join(_REPO, "transformer", "GNN"))
from Models import (FPGA_GNN_SumDecomp, FPGA_GNN_SumDecompCorr,            # noqa: E402
                    FPGA_GNN_GATv2_SumDecomp, FPGA_GNN_GATv2_SumDecompCorr)
from Dataset2 import create_dataloaders_from_split_data                    # noqa: E402


def r2(a, p):
    ss = np.sum((a - p) ** 2); tot = np.sum((a - a.mean()) ** 2)
    return 1.0 - ss / tot if tot > 0 else 1.0


def smape(a, p):
    return float(100.0 * np.mean(2.0 * np.abs(p - a) / (np.abs(a) + np.abs(p) + 1e-12)))


def spearman(a, p):
    ra = np.argsort(np.argsort(a)); rp = np.argsort(np.argsort(p))
    ra = ra - ra.mean(); rp = rp - rp.mean()
    d = np.sqrt((ra ** 2).sum() * (rp ** 2).sum())
    return float((ra * rp).sum() / d) if d > 0 else 0.0


def predict(ckpt_dir, variant, feats, labs, device):
    stats = os.path.join(ckpt_dir, "lognormalization_stats.npy")
    _, _, loader, ndim, ntgt = create_dataloaders_from_split_data(
        train_features_path=feats, train_labels_path=labs, val_features_path=feats, val_labels_path=labs,
        test_features_path=feats, test_labels_path=labs, stats_load_path=stats, stats_save_path=None,
        batch_size=512, num_workers=4, pin_memory=(device.type == "cuda"),
        mode="gnn", use_log_transform=True, log_epsilon=1e-6, label_cols=[0, 1])
    loader.dataset.mode = "gnn"
    mkw = (dict(hidden_dim=512, num_gnn_layers=5, num_attention_heads=5, mlp_hidden_dim=128, dropout_rate=0.2)
           if variant in ("gatv2", "gatv2corr") else
           dict(hidden_dim=256, num_gnn_layers=4, num_attention_heads=4, mlp_hidden_dim=128, dropout_rate=0.2))
    Cls = {"plain": FPGA_GNN_SumDecomp, "corr": FPGA_GNN_SumDecompCorr,
           "gatv2": FPGA_GNN_GATv2_SumDecomp, "gatv2corr": FPGA_GNN_GATv2_SumDecompCorr}[variant]
    model = Cls(node_feature_dim=ndim, num_targets=ntgt, **mkw).to(device)
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "model.pt"), map_location=device))
    model.eval()
    yp, yt = [], []
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            yp.append(model(b).cpu().numpy())
            y = b.y
            yt.append((y.squeeze(1) if y.dim() == 3 else y).cpu().numpy())
    yp = np.concatenate(yp, 0)
    yt = loader.dataset.denormalize_labels(torch.tensor(np.concatenate(yt, 0))).numpy()
    return yt.astype(np.float64), yp.astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True); ap.add_argument("--variant", default="plain")
    ap.add_argument("--fit-features", required=True); ap.add_argument("--fit-labels", required=True)
    ap.add_argument("--test-features", required=True); ap.add_argument("--test-labels", required=True)
    ap.add_argument("--label", default="probe"); ap.add_argument("--json-out", default="")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yt_fit, yp_fit = predict(a.ckpt_dir, a.variant, a.fit_features, a.fit_labels, dev)
    yt_te, yp_te = predict(a.ckpt_dir, a.variant, a.test_features, a.test_labels, dev)

    out = {"label": a.label, "ckpt": a.ckpt_dir, "variant": a.variant, "n_test": int(len(yt_te))}
    for j, tg in enumerate(["LATENCY", "AREA"]):
        af, pf = yt_fit[:, j], yp_fit[:, j]
        at, pt = yt_te[:, j], yp_te[:, j]
        scalar = float(np.median(af / np.clip(pf, 1e-9, None)))     # fit on 22nm {1,2}
        out[tg] = {
            "zeroshot_r2": round(r2(at, pt), 4), "zeroshot_smape": round(smape(at, pt), 2),
            "spearman": round(spearman(at, pt), 4),                  # rank transfer (scale-free)
            "fit_scalar": round(scalar, 4),
            "calibrated_r2": round(r2(at, pt * scalar), 4), "calibrated_smape": round(smape(at, pt * scalar), 2),
        }
    print(json.dumps(out, indent=1))
    if a.json_out:
        json.dump(out, open(a.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
