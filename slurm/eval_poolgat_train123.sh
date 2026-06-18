#!/bin/bash
# Eval-only rescue: the train123 pooling-GATv2 jobs hit the 2h wall during TRAINING (no early-stop;
# val kept creeping down past epoch 100) and were killed before the eval step — but the best_model
# checkpoints were saved. This just runs the fast eval on those checkpoints (L6/L8/L10), 4 seeds.
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q express_amsc
#SBATCH --gpus-per-node=1
#SBATCH -c 32
#SBATCH -A amsc011
#SBATCH -t 00:30:00
#SBATCH -J wa_poolgat_eval
#SBATCH -o /global/u2/a/arghyara/work/wa_hls4ml_models/slurm/logs/%x_%j.out
#SBATCH -e /global/u2/a/arghyara/work/wa_hls4ml_models/slurm/logs/%x_%j.err

set -euo pipefail
REPO=/global/u2/a/arghyara/work/wa_hls4ml_models
CKROOT=${REPO}/GNN/gnn_results_plots
RAW=${SCRATCH}/catapult_v2/raw
RES=${SCRATCH}/catapult_v2/eval_results
mkdir -p "${RES}"

module load pytorch/2.8.0
source "${SCRATCH}/venv_wa_hls4ml_models/bin/activate"
cd "${REPO}/GNN"

for seed in a b c d; do
  ck=$(ls -d ${CKROOT}/gatv2_*_poolgat_train123_${seed}/best_model 2>/dev/null | head -1)
  if [ -z "${ck}" ]; then echo "NO CKPT train123 ${seed}"; continue; fi
  for ts in L6 L8 L10; do
    echo "===== EVAL poolgat train123 ${seed} on ${ts}  <- ${ck} ====="
    python eval_ckpt.py --ckpt-dir "${ck}" --variant poolgat \
      --features "${RAW}/${ts}_features.npy" --labels "${RAW}/${ts}_labels.npy" \
      --tag "poolgat__train123__${seed}__${ts}" \
      --json-out "${RES}/poolgat__train123__${seed}__${ts}.json" \
      --save-preds "${RES}/poolgat__train123__${seed}__${ts}.npz" || echo "EVAL FAILED ${seed} ${ts}"
  done
done
echo "DONE_POOLGAT_EVAL train123"
