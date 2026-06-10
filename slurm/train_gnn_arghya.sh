#!/bin/bash
# Catapult-ASIC surrogate: train a GNN (GraphSAGE baseline / GATv2 lui-gnn) on the
# prepared split. GPU job. Arghya's paths. Mirrors train_surrogate_arghya.sh.
#
# Pick the model with ARCH below: gnn | gatv2 | gatv2_enhanced  (default: gatv2 = published lui-gnn)
# Run prepare_data_arghya.sh (or `make data split`) first so the split exists.
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q express_amsc
#SBATCH --gpus-per-node=1
#SBATCH -c 32
#SBATCH -A amsc011
#SBATCH -t 06:00:00
#SBATCH -J wa_train_gnn
#SBATCH -o /global/u2/a/arghyara/work/wa_hls4ml_models/slurm/logs/%x_%j.out
#SBATCH -e /global/u2/a/arghyara/work/wa_hls4ml_models/slurm/logs/%x_%j.err

set -euo pipefail

REPO=/global/u2/a/arghyara/work/wa_hls4ml_models
SPLIT=${SPLIT:-/pscratch/sd/a/arghyara/catapult_asic_data/split}

# ── knobs (override on submit: `sbatch --export=ALL,ARCH=gnn,EPOCHS=300 ...`) ──
ARCH=${ARCH:-gatv2}        # gnn | gatv2 | gatv2_enhanced
EPOCHS=${EPOCHS:-200}
LR=${LR:-1e-3}
BATCH=${BATCH:-256}

module load pytorch/2.8.0
source "${SCRATCH}/venv_wa_hls4ml_models/bin/activate"
export WA_NUM_WORKERS=16   # data loaders use the 32 cores requested above

cd "${REPO}/GNN"
python run_gnn.py \
  --arch "${ARCH}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --batch-size "${BATCH}" \
  --base-dir "${SPLIT}" \
  --drop-throughput \
  --thruput-lookup "${REPO}/dataset/thruput_lookup.pkl"
