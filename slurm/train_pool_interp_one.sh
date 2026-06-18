#!/bin/bash
# INTERPOLATION refresh: train a pooling model (FPGA_GNN GraphSAGE OR FPGA_GNN_GATv2) on the random
# {1,2,3} 45nm split (split_rand123, 303k train) and eval on its HELD-OUT {1,2,3} test (65k) — the
# in-distribution regime. Refreshes the stale §1.1 numbers (old smaller dataset) with catapult_v2 + 4 seeds.
# Env: ARCH(gnn|gatv2) VARIANT(poolsage|poolgat) SEED LR EP PAT
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q express_amsc
#SBATCH --gpus-per-node=1
#SBATCH -c 32
#SBATCH -A amsc011
#SBATCH -t 03:00:00
#SBATCH -J wa_poolinterp
#SBATCH -o /global/u2/a/arghyara/work/wa_hls4ml_models/slurm/logs/%x_%j.out
#SBATCH -e /global/u2/a/arghyara/work/wa_hls4ml_models/slurm/logs/%x_%j.err

set -euo pipefail
REPO=/global/u2/a/arghyara/work/wa_hls4ml_models
CKROOT=${REPO}/GNN/gnn_results_plots
SPLIT=${SCRATCH}/catapult_v2/split_rand123
RES=${SCRATCH}/catapult_v2/eval_results_interp
mkdir -p "${RES}"

ARCH=${ARCH:?}; VARIANT=${VARIANT:?}; SEED=${SEED:?}
LR=${LR:-1e-3}; EP=${EP:-120}; PAT=${PAT:-25}

module load pytorch/2.8.0
source "${SCRATCH}/venv_wa_hls4ml_models/bin/activate"
export WA_NUM_WORKERS=16
cd "${REPO}/GNN"

tag="interp_${ARCH}_${SEED}"
echo "===== TRAIN ${ARCH} ${tag}  (split=rand123 interp, lr=${LR} ep=${EP}) ====="
python run_gnn.py --arch "${ARCH}" --base-dir "${SPLIT}" \
  --lr "${LR}" --epochs "${EP}" --patience "${PAT}" --batch-size 256 --drop-throughput \
  --run-tag "${tag}"

ck=$(ls -dt ${CKROOT}/${ARCH}_*_${tag}/best_model 2>/dev/null | head -1)
if [ -z "${ck}" ]; then echo "NO CKPT for ${tag}"; exit 1; fi
echo "checkpoint: ${ck}"

echo "===== EVAL ${tag} on held-out {1,2,3} test (interpolation) ====="
python eval_ckpt.py --ckpt-dir "${ck}" --variant "${VARIANT}" \
  --features "${SPLIT}/test_features.npy" --labels "${SPLIT}/test_labels.npy" \
  --tag "${VARIANT}__interp__${SEED}" \
  --json-out "${RES}/${VARIANT}__interp__${SEED}.json" \
  --save-preds "${RES}/${VARIANT}__interp__${SEED}.npz"
echo "DONE_POOLINTERP ${ARCH} ${SEED}"
