#!/bin/bash
# Sum-decomposition (deep-sets) GNN for depth extrapolation. GPU job.
# Default split = layer-extrapolation (train 1+2-layer, test unseen 3-layer).
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q express_amsc
#SBATCH --gpus-per-node=1
#SBATCH -c 32
#SBATCH -A amsc011
#SBATCH -t 06:00:00
#SBATCH -J wa_train_sumdecomp
#SBATCH -o /global/u2/a/arghyara/work/wa_hls4ml_models/slurm/logs/%x_%j.out
#SBATCH -e /global/u2/a/arghyara/work/wa_hls4ml_models/slurm/logs/%x_%j.err

set -euo pipefail
REPO=/global/u2/a/arghyara/work/wa_hls4ml_models
SPLIT=${SPLIT:-/pscratch/sd/a/arghyara/catapult_asic_data/split_layerextrap}
EPOCHS=${EPOCHS:-200}; LR=${LR:-1e-3}; BATCH=${BATCH:-256}; VARIANT=${VARIANT:-plain}; OPTIM=${OPTIM:-adamw}

module load pytorch/2.8.0
source "${SCRATCH}/venv_wa_hls4ml_models/bin/activate"
export WA_NUM_WORKERS=16

cd "${REPO}/GNN"
python run_sumdecomp.py --epochs "${EPOCHS}" --lr "${LR}" --batch-size "${BATCH}" --variant "${VARIANT}" \
  --optimizer "${OPTIM}" --base-dir "${SPLIT}" --drop-throughput --thruput-lookup "${REPO}/dataset/thruput_lookup.pkl"
