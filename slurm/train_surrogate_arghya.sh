#!/bin/bash
# Catapult-ASIC surrogate: train the transformer on the prepared split. GPU job.
# Arghya's paths. Run prepare_data_arghya.sh (or `make data split`) first so the
# split exists at $SCRATCH/catapult_asic_data/split/.
#
# Account/QoS note: Giuseppe's original uses `-A amsc011 -q express_amsc` and it
# runs. CLAUDE.md prefers `-A amsc011_g` for GPU jobs. Both accounts carry the
# express_amsc / express_amsc_g / gpu_* QoS for this user, so either works;
# confirm the intended billing account with Giuseppe.
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q express_amsc
#SBATCH --gpus-per-node=1
#SBATCH -c 32
#SBATCH -A amsc011
#SBATCH -t 06:00:00
#SBATCH -J wa_train_surrogate
#SBATCH -o /global/u2/a/arghyara/work/wa_hls4ml_models/slurm/logs/%x_%j.out
#SBATCH -e /global/u2/a/arghyara/work/wa_hls4ml_models/slurm/logs/%x_%j.err

set -euo pipefail

REPO=/global/u2/a/arghyara/work/wa_hls4ml_models

module load pytorch/2.8.0
# Activate venv after the module so the venv's python (with plotly) wins for any bare `python`.
# 'make train' uses the explicit $(VENV)/bin/python regardless, but this makes manual reruns safe too.
source "${SCRATCH}/venv_wa_hls4ml_models/bin/activate"

# Data loaders are the bottleneck; use the 32 cores requested above (run.py defaults to 16, override here).
export WA_NUM_WORKERS=16

cd "${REPO}"
# 'make train' = v2: learn LATENCY + AREA, derive THROUGHPUT analytically
#                (run.py --drop-throughput --thruput-lookup dataset/thruput_lookup.pkl)
# Use 'make train-legacy' to learn all 3 (LATENCY/AREA/THROUGHPUT).
make train EPOCHS=200 BATCH=256
