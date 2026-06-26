#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q express_amsc
#SBATCH --gpus-per-node=1
#SBATCH -A amsc011
#SBATCH -t 06:00:00
#SBATCH -J train_surrogate
#SBATCH -o /global/homes/g/gdg/research/projects/genesis/wa-hls4ml-paper/wa-hls4ml-models/slurm/logs/%x_%j.out
#SBATCH -e /global/homes/g/gdg/research/projects/genesis/wa-hls4ml-paper/wa-hls4ml-models/slurm/logs/%x_%j.err

set -euo pipefail

ARCHIVE=/global/cfs/cdirs/amsc011/shared/wa-hls4ml-catapult
REPO=/global/homes/g/gdg/research/projects/genesis/wa-hls4ml-paper/wa-hls4ml-models

module load pytorch/2.8.0

cd "${REPO}"

EPOCHS=${EPOCHS:-200}
BATCH=${BATCH:-256}
RESUME=${RESUME:-""}
CONTINUE_EPOCHS=${CONTINUE_EPOCHS:-0}

TRAIN_ARGS=""
if [ -n "${RESUME}" ]; then
    # run.py executes from inside transformer/, so the path must be absolute
    RESUME=$(realpath "${RESUME}")
    TRAIN_ARGS="--resume ${RESUME}"
fi

make train EPOCHS="${EPOCHS}" BATCH="${BATCH}" TRAIN_ARGS="${TRAIN_ARGS}"

# Auto-submit a resume job if requested (keeps each job within 6h wall-time)
if [ "${CONTINUE_EPOCHS}" -gt 0 ]; then
    LATEST_CKPT=$(ls -t "${REPO}/transformer/new_results_plots"/*/best_model/model.pt 2>/dev/null | head -1)
    if [ -n "${LATEST_CKPT}" ]; then
        echo "Submitting continuation: ${CONTINUE_EPOCHS} more epochs from ${LATEST_CKPT}"
        RESUME="${LATEST_CKPT}" EPOCHS="${CONTINUE_EPOCHS}" \
            sbatch "${REPO}/slurm/train_surrogate.sh"
    else
        echo "WARNING: no checkpoint found, skipping continuation"
    fi
fi
