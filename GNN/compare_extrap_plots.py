# Comparison plots: Standard GATv2 vs Sum-Decomposition on the depth-extrapolation split.
# Loss overlay (from the two training logs) + grouped R²/SMAPE bars (test metrics).
import os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = "/global/u2/a/arghyara/work/wa_hls4ml_models"
LOGS = f"{REPO}/slurm/logs"
OUT = f"{REPO}/GNN/gnn_results_plots/comparison_extrap"
os.makedirs(OUT, exist_ok=True)

def parse(p):
    tr, va = [], []
    pat = re.compile(r"Train Loss:\s*([0-9.eE+-]+)\s*\|\s*Validation Loss:\s*([0-9.eE+-]+)")
    for line in open(p):
        m = pat.search(line)
        if m:
            tr.append(float(m.group(1))); va.append(float(m.group(2)))
    return tr, va

std_tr, std_va = parse(f"{LOGS}/wa_train_gnn_53824320.out")        # standard GATv2 extrapolation
sd_tr, sd_va = parse(f"{LOGS}/wa_train_sumdecomp_53825701.out")    # sum-decomp extrapolation

# ---- loss overlay ----
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(std_va) + 1), std_va, color="C0", label="Standard GATv2 — val")
plt.plot(range(1, len(std_tr) + 1), std_tr, color="C0", ls="--", alpha=0.45, label="Standard GATv2 — train")
plt.plot(range(1, len(sd_va) + 1), sd_va, color="C1", label="Sum-Decomp — val")
plt.plot(range(1, len(sd_tr) + 1), sd_tr, color="C1", ls="--", alpha=0.45, label="Sum-Decomp — train")
plt.yscale("log"); plt.xlabel("Epoch"); plt.ylabel("MSE loss (normalized-log space)")
plt.title("Loss — Standard GATv2 vs Sum-Decomp (extrapolation split)")
plt.legend(fontsize=9); plt.tight_layout()
plt.savefig(f"{OUT}/comparison_loss.png", dpi=120); plt.close()

# ---- metric bars (verified test metrics from jobs 53824320 / 53825701) ----
targets = ["Overall", "LATENCY", "AREA"]
r2_std = [0.8083, 0.6935, 0.7314];  r2_sd = [0.9920, 0.9922, 0.9837]
sm_std = [75.97, 37.82, 38.15];     sm_sd = [15.86, 5.76, 10.10]
x = np.arange(len(targets)); w = 0.36
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].bar(x - w/2, r2_std, w, color="C0", label="Standard GATv2")
ax[0].bar(x + w/2, r2_sd, w, color="C1", label="Sum-Decomp")
ax[0].set_xticks(x); ax[0].set_xticklabels(targets); ax[0].set_ylim(0, 1.08)
ax[0].set_ylabel("R²  (higher = better)"); ax[0].set_title("Depth-extrapolation test R²"); ax[0].legend()
for i, (a, b) in enumerate(zip(r2_std, r2_sd)):
    ax[0].text(i - w/2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
    ax[0].text(i + w/2, b + 0.01, f"{b:.2f}", ha="center", fontsize=8)
ax[1].bar(x - w/2, sm_std, w, color="C0", label="Standard GATv2")
ax[1].bar(x + w/2, sm_sd, w, color="C1", label="Sum-Decomp")
ax[1].set_xticks(x); ax[1].set_xticklabels(targets)
ax[1].set_ylabel("SMAPE %  (lower = better)"); ax[1].set_title("Depth-extrapolation test SMAPE"); ax[1].legend()
for i, (a, b) in enumerate(zip(sm_std, sm_sd)):
    ax[1].text(i - w/2, a + 0.6, f"{a:.1f}", ha="center", fontsize=8)
    ax[1].text(i + w/2, b + 0.6, f"{b:.1f}", ha="center", fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/comparison_metrics.png", dpi=120); plt.close()
print("DONE", OUT, "| std epochs", len(std_va), "| sumdecomp epochs", len(sd_va))
