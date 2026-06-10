# GNN/replot_sumdecomp.py — regenerate plots for a finished FPGA_GNN_SumDecomp run from its checkpoint.
import os, sys, argparse, pickle, re
import numpy as np, torch
_THIS = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.dirname(_THIS)
sys.path.insert(0, _THIS); sys.path.insert(0, os.path.join(_REPO, "transformer")); sys.path.insert(0, os.path.join(_REPO, "transformer", "GNN"))
from Models import FPGA_GNN_SumDecomp
from Dataset2 import create_dataloaders_from_split_data
from plot import plot_box_plots_symlog, plot_results_simplified, plot_loss
from run_gnn import get_model_types

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--base-dir", default="/pscratch/sd/a/arghyara/catapult_asic_data/split_layerextrap")
ap.add_argument("--thruput-lookup", default=os.path.join(_REPO, "dataset", "thruput_lookup.pkl"))
ap.add_argument("--log", default=None)
a = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best = os.path.join(a.run_dir, "best_model"); stats = os.path.join(best, "lognormalization_stats.npy")
tr, va, te, nfd, nt = create_dataloaders_from_split_data(
    f"{a.base_dir}/train_features.npy", f"{a.base_dir}/train_labels.npy",
    f"{a.base_dir}/val_features.npy", f"{a.base_dir}/val_labels.npy",
    f"{a.base_dir}/test_features.npy", f"{a.base_dir}/test_labels.npy",
    stats_load_path=stats, stats_save_path=None, batch_size=512, num_workers=8,
    pin_memory=(dev.type == "cuda"), mode="gnn", use_log_transform=True, log_epsilon=1e-6, label_cols=[0, 1])
for L in (tr, va, te): L.dataset.mode = "gnn"
m = FPGA_GNN_SumDecomp(node_feature_dim=nfd, num_targets=nt, hidden_dim=256, num_gnn_layers=4,
                       num_attention_heads=4, mlp_hidden_dim=128).to(dev)
m.load_state_dict(torch.load(os.path.join(best, "model.pt"), map_location=dev)); m.eval()
yp, yt = [], []
with torch.no_grad():
    for b in te:
        b = b.to(dev); yp.append(m(b).cpu().numpy())
        y = b.y; yt.append((y.squeeze(1) if y.dim() == 3 else y).cpu().numpy())
yp = np.concatenate(yp)
ytd = te.dataset.denormalize_labels(torch.tensor(np.concatenate(yt))).numpy()
tf = np.load(f"{a.base_dir}/test_features.npy")
with open(a.thruput_lookup, "rb") as f: lut = pickle.load(f)
der = []
for d in tf:
    rows = [r for r in d if not np.all(r == -1)]
    lt = [lut[(int(r[0]), int(r[3]), int(r[6]), int(r[7]))] for r in rows
          if int(r[9]) == 1 and (int(r[0]), int(r[3]), int(r[6]), int(r[7])) in lut]
    der.append(float(max(lt)) if lt else 0.0)
yp = np.hstack([yp, np.array(der).reshape(-1, 1)])
ytd = np.hstack([ytd, np.load(f"{a.base_dir}/test_labels.npy")[:, 2:3]])
of = ["LATENCY", "AREA", "THROUGHPUT"]
if a.log and os.path.exists(a.log):
    trn, vln, pat = [], [], re.compile(r"Train Loss:\s*([0-9.eE+-]+)\s*\|\s*Validation Loss:\s*([0-9.eE+-]+)")
    for line in open(a.log):
        mm = pat.search(line)
        if mm: trn.append(float(mm.group(1))); vln.append(float(mm.group(2)))
    if trn: plot_loss(trn, vln, outdir=os.path.join(a.run_dir, "plots"))
plot_box_plots_symlog(yp, ytd, folder_name=a.run_dir, output_features=of)
plot_results_simplified(name="run1", mpl_plots=True, y_test=ytd, y_pred=yp, output_features=of,
                        folder_name=a.run_dir, model_types=get_model_types(tf))
print("REPLOT_DONE", os.path.join(a.run_dir, "plots"))
