# GNN/render_plots.py
# Regenerate plots (loss curve from the training log + predicted-vs-true grid + symlog
# box plots from the checkpoint) for a finished GNN run, into <run-dir>/plots/.
# Reuses transformer/plot.py and the throughput LUT, exactly like run_gnn.py.

import os
import sys
import argparse
import pickle
import re

import numpy as np
import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_THIS)
sys.path.insert(0, _THIS)
sys.path.insert(0, os.path.join(_REPO, "transformer"))
sys.path.insert(0, os.path.join(_REPO, "transformer", "GNN"))

from Dataset2 import create_dataloaders_from_split_data  # noqa: E402
from run_gnn import build_model, get_model_types          # noqa: E402
from plot import plot_loss, plot_box_plots_symlog, plot_results_simplified  # noqa: E402


def parse_loss(logpath):
    tr, va = [], []
    pat = re.compile(r"Train Loss:\s*([0-9.eE+-]+)\s*\|\s*Validation Loss:\s*([0-9.eE+-]+)")
    with open(logpath) as f:
        for line in f:
            m = pat.search(line)
            if m:
                tr.append(float(m.group(1)))
                va.append(float(m.group(2)))
    return tr, va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="training run dir containing best_model/")
    ap.add_argument("--arch", required=True, choices=["gnn", "gatv2", "gatv2_enhanced"])
    ap.add_argument("--base-dir", default="/pscratch/sd/a/arghyara/catapult_asic_data/split")
    ap.add_argument("--thruput-lookup", default=os.path.join(_REPO, "dataset", "thruput_lookup.pkl"))
    ap.add_argument("--log", default=None, help="slurm .out log to parse the loss curve from")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best = os.path.join(args.run_dir, "best_model")
    stats = os.path.join(best, "lognormalization_stats.npy")

    tr, va, te, nfd, nt = create_dataloaders_from_split_data(
        train_features_path=f"{args.base_dir}/train_features.npy",
        train_labels_path=f"{args.base_dir}/train_labels.npy",
        val_features_path=f"{args.base_dir}/val_features.npy",
        val_labels_path=f"{args.base_dir}/val_labels.npy",
        test_features_path=f"{args.base_dir}/test_features.npy",
        test_labels_path=f"{args.base_dir}/test_labels.npy",
        stats_load_path=stats, stats_save_path=None, batch_size=512, num_workers=8,
        pin_memory=(device.type == "cuda"), mode="gnn",
        use_log_transform=True, log_epsilon=1e-6, label_cols=[0, 1],
    )
    for L in (tr, va, te):
        L.dataset.mode = "gnn"

    model = build_model(args.arch, nfd, nt).to(device)
    model.load_state_dict(torch.load(os.path.join(best, "model.pt"), map_location=device))
    model.eval()

    yt, yp = [], []
    with torch.no_grad():
        for b in te:
            b = b.to(device)
            yp.append(model(b).cpu().numpy())
            y = b.y
            yt.append((y.squeeze(1) if y.dim() == 3 else y).cpu().numpy())
    yt = np.concatenate(yt)
    yp = np.concatenate(yp)
    ytd = te.dataset.denormalize_labels(torch.tensor(yt)).numpy()
    ypd = te.dataset.denormalize_labels(torch.tensor(yp)).numpy()

    tf = np.load(f"{args.base_dir}/test_features.npy")
    with open(args.thruput_lookup, "rb") as f:
        lut = pickle.load(f)
    der = []
    for d in tf:
        rows = [r for r in d if not np.all(r == -1)]
        lt = []
        for r in rows:
            if int(r[9]) != 1:
                continue
            k = (int(r[0]), int(r[3]), int(r[6]), int(r[7]))
            t = lut.get(k)
            if t is not None:
                lt.append(t)
        der.append(float(max(lt)) if lt else 0.0)
    der = np.array(der).reshape(-1, 1)
    ypd = np.hstack([ypd, der])
    raw = np.load(f"{args.base_dir}/test_labels.npy")
    ytd = np.hstack([ytd, raw[:, 2:3]])
    of = ["LATENCY", "AREA", "THROUGHPUT"]

    pdir = os.path.join(args.run_dir, "plots")
    if args.log and os.path.exists(args.log):
        t, v = parse_loss(args.log)
        if t:
            plot_loss(t, v, outdir=pdir)
            print(f"loss curve: {len(t)} epochs")
    plot_box_plots_symlog(ypd, ytd, folder_name=args.run_dir, output_features=of)
    plot_results_simplified(name="run1", mpl_plots=True, y_test=ytd, y_pred=ypd,
                            output_features=of, folder_name=args.run_dir, model_types=get_model_types(tf))
    print("PLOTS_DONE", os.path.join(args.run_dir, "plots"))


if __name__ == "__main__":
    main()
