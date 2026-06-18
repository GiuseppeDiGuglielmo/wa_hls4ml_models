#!/bin/bash
# Train the PLAIN POOLING GATv2 (FPGA_GNN_GATv2, 54.47M, multi-pool->MLP — the original Table 1/2
# architecture, NOT SumDecomp) for ONE (config, seed), then eval on each held-out depth test set.
# This refreshes the stale "GNN GATv2 extrap" row with the new catapult_v2 data + 4 seeds.
# Env: SPLIT CONFIG SEED LR EP PAT TESTSETS
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q express_amsc
#SBATCH --gpus-per-node=1
#SBATCH -c 32
#SBATCH -A amsc011
#SBATCH -t 02:00:00
#SBATCH -J wa_poolgat
#SBATCH -o /global/u2/a/arghyara/work/wa_hls4ml_models/slurm/logs/%x_%j.out
#SBATCH -e /global/u2/a/arghyara/work/wa_hls4ml_models/slurm/logs/%x_%j.err

set -euo pipefail
REPO=/global/u2/a/arghyara/work/wa_hls4ml_models
CKROOT=${REPO}/GNN/gnn_results_plots
RAW=${SCRATCH}/catapult_v2/raw
RES=${SCRATCH}/catapult_v2/eval_results
mkdir -p "${RES}"

SPLIT=${SPLIT:?}; CONFIG=${CONFIG:?}; SEED=${SEED:?}
LR=${LR:-1e-3}; EP=${EP:-200}; PAT=${PAT:-30}
TESTSETS=${TESTSETS:-"L3 L6 L8 L10"}

module load pytorch/2.8.0
source "${SCRATCH}/venv_wa_hls4ml_models/bin/activate"
export WA_NUM_WORKERS=16
cd "${REPO}/GNN"

tag="poolgat_${CONFIG}_${SEED}"
echo "===== TRAIN poolgat ${tag}  (split=$(basename ${SPLIT}) lr=${LR} ep=${EP}) ====="
python run_gnn.py --arch gatv2 --base-dir "${SPLIT}" \
  --lr "${LR}" --epochs "${EP}" --patience "${PAT}" --batch-size 256 --drop-throughput \
  --run-tag "${tag}"

ck=$(ls -dt ${CKROOT}/gatv2_*_${tag}/best_model 2>/dev/null | head -1)
if [ -z "${ck}" ]; then echo "NO CKPT for ${tag}"; exit 1; fi
echo "checkpoint: ${ck}"

for ts in ${TESTSETS}; do
  echo "===== EVAL ${tag} on ${ts} ====="
  python eval_ckpt.py --ckpt-dir "${ck}" --variant poolgat \
    --features "${RAW}/${ts}_features.npy" --labels "${RAW}/${ts}_labels.npy" \
    --tag "poolgat__${CONFIG}__${SEED}__${ts}" \
    --json-out "${RES}/poolgat__${CONFIG}__${SEED}__${ts}.json" \
    --save-preds "${RES}/poolgat__${CONFIG}__${SEED}__${ts}.npz" || echo "EVAL FAILED ${tag} ${ts}"
done
echo "DONE_POOLGAT ${CONFIG} ${SEED}"
