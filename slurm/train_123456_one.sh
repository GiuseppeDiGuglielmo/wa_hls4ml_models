#!/bin/bash
# Train ONE seed of the deepest config: train {1,2,3,4,5,6} -> internal test {8}.
# Parametrized by env: VARIANT, OPTIM, LR, EPOCHS, PATIENCE, RUN_TAG.
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q express_amsc
#SBATCH --gpus-per-node=1
#SBATCH -c 32
#SBATCH -A amsc011
#SBATCH -t 06:00:00
#SBATCH -J wa_train_123456
#SBATCH -o /global/u2/a/arghyara/work/wa_hls4ml_models/slurm/logs/%x_%j.out
#SBATCH -e /global/u2/a/arghyara/work/wa_hls4ml_models/slurm/logs/%x_%j.err

set -euo pipefail
REPO=/global/u2/a/arghyara/work/wa_hls4ml_models
SPLIT=${SCRATCH}/catapult_v2/split_123456
VARIANT=${VARIANT:-plain}; OPTIM=${OPTIM:-adamw}; LR=${LR:-1e-3}
EPOCHS=${EPOCHS:-200}; BATCH=${BATCH:-256}; PATIENCE=${PATIENCE:-30}; RUN_TAG=${RUN_TAG:-x123456}

module load pytorch/2.8.0
source "${SCRATCH}/venv_wa_hls4ml_models/bin/activate"
export WA_NUM_WORKERS=16
cd "${REPO}/GNN"
python run_sumdecomp.py --epochs "${EPOCHS}" --lr "${LR}" --batch-size "${BATCH}" \
  --variant "${VARIANT}" --optimizer "${OPTIM}" --patience "${PATIENCE}" \
  --base-dir "${SPLIT}" --drop-throughput --run-tag "${RUN_TAG}" \
  --thruput-lookup "${REPO}/dataset/thruput_lookup.pkl"
