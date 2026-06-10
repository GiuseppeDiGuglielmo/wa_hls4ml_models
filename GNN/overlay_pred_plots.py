# Overlaid predicted-vs-true scatter (3 targets) — Standard GATv2 vs Sum-Decomp, depth-extrapolation test.
import os, sys, pickle
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.dirname(_THIS)
sys.path.insert(0, _THIS); sys.path.insert(0, os.path.join(_REPO, "transformer")); sys.path.insert(0, os.path.join(_REPO, "transformer", "GNN"))
from Dataset2 import create_dataloaders_from_split_data
from run_gnn import build_model
from Models import FPGA_GNN_SumDecomp

SPLIT = "/pscratch/sd/a/arghyara/catapult_asic_data/split_layerextrap"
STD = f"{_REPO}/GNN/gnn_results_plots/gatv2_06_02_19_19_200epochs_0.001lr_256bs"
SD = f"{_REPO}/GNN/gnn_results_plots/sumdecomp_06_02_19_50_200epochs_0.001lr_256bs"
OUT = f"{_REPO}/GNN/gnn_results_plots/comparison_extrap"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tr, va, te, nfd, nt = create_dataloaders_from_split_data(
    f"{SPLIT}/train_features.npy", f"{SPLIT}/train_labels.npy",
    f"{SPLIT}/val_features.npy", f"{SPLIT}/val_labels.npy",
    f"{SPLIT}/test_features.npy", f"{SPLIT}/test_labels.npy",
    stats_load_path=f"{STD}/best_model/lognormalization_stats.npy", stats_save_path=None,
    batch_size=1024, num_workers=8, pin_memory=(dev.type == "cuda"),
    mode="gnn", use_log_transform=True, log_epsilon=1e-6, label_cols=[0, 1])
for L in (tr, va, te):
    L.dataset.mode = "gnn"

std = build_model("gatv2", nfd, nt).to(dev); std.load_state_dict(torch.load(f"{STD}/best_model/model.pt", map_location=dev)); std.eval()
sd = FPGA_GNN_SumDecomp(nfd, nt, hidden_dim=256, num_gnn_layers=4, num_attention_heads=4, mlp_hidden_dim=128).to(dev)
sd.load_state_dict(torch.load(f"{SD}/best_model/model.pt", map_location=dev)); sd.eval()

yt_n, p_std_n, p_sd_lin = [], [], []
with torch.no_grad():
    for b in te:
        b = b.to(dev)
        p_std_n.append(std(b).cpu().numpy())
        p_sd_lin.append(sd(b).cpu().numpy())
        y = b.y; yt_n.append((y.squeeze(1) if y.dim() == 3 else y).cpu().numpy())
yt_n = np.concatenate(yt_n); p_std_n = np.concatenate(p_std_n); p_sd = np.concatenate(p_sd_lin)
ds = te.dataset
yt = ds.denormalize_labels(torch.tensor(yt_n)).numpy()      # [N,2] linear true
p_std = ds.denormalize_labels(torch.tensor(p_std_n)).numpy()  # [N,2] linear (standard model)

# throughput via the exact LUT (identical for both models)
tf = np.load(f"{SPLIT}/test_features.npy")
with open(f"{_REPO}/dataset/thruput_lookup.pkl", "rb") as f:
    lut = pickle.load(f)
der = []
for d in tf:
    rows = [r for r in d if not np.all(r == -1)]
    lt = [lut[(int(r[0]), int(r[3]), int(r[6]), int(r[7]))] for r in rows
          if int(r[9]) == 1 and (int(r[0]), int(r[3]), int(r[6]), int(r[7])) in lut]
    der.append(float(max(lt)) if lt else 0.0)
der = np.array(der).reshape(-1, 1)
raw = np.load(f"{SPLIT}/test_labels.npy")
yt3 = np.hstack([yt, raw[:, 2:3]])
p_std3 = np.hstack([p_std, der])
p_sd3 = np.hstack([p_sd, der])

rng = np.random.default_rng(0)
idx = rng.choice(len(yt3), size=min(4000, len(yt3)), replace=False)
feats = ["LATENCY", "AREA", "THROUGHPUT"]
fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
for j, f in enumerate(feats):
    a = ax[j]; t = yt3[idx, j]
    if j < 2:
        a.scatter(t, p_std3[idx, j], s=7, alpha=0.25, color="C0", label="Standard GATv2", edgecolors="none")
        a.scatter(t, p_sd3[idx, j], s=7, alpha=0.25, color="C1", label="Sum-Decomp", edgecolors="none")
    else:
        a.scatter(t, p_sd3[idx, j], s=7, alpha=0.3, color="C2", label="Both (exact LUT)", edgecolors="none")
    lo = max(min(t.min(), p_std3[idx, j].min(), p_sd3[idx, j].min()), 1e-1)
    hi = max(t.max(), p_std3[idx, j].max(), p_sd3[idx, j].max())
    a.plot([lo, hi], [lo, hi], "r-", lw=1.2, zorder=5)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel(f"Actual {f}"); a.set_ylabel(f"Predicted {f}"); a.set_title(f)
    a.legend(markerscale=3, fontsize=9)
fig.suptitle("Predicted vs. True on unseen 3-layer designs — Standard GATv2 (below the line) vs Sum-Decomp (on the line)", y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT}/comparison_pred_vs_true.png", dpi=120, bbox_inches="tight")
print("DONE", f"{OUT}/comparison_pred_vs_true.png")
