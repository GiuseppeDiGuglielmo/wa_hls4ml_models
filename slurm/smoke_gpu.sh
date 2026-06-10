#!/bin/bash
# Short GPU smoke test: proves job submission + GPU allocation + training runs.
# Trains 2 epochs on the toy split already staged at $SCRATCH/catapult_asic_smoke.
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q express_amsc
#SBATCH --gpus-per-node=1
#SBATCH -A amsc011
#SBATCH -t 00:10:00
#SBATCH -J wa_smoke_gpu
#SBATCH -o /global/u2/a/arghyara/work/wa_hls4ml_models/slurm/logs/%x_%j.out
#SBATCH -e /global/u2/a/arghyara/work/wa_hls4ml_models/slurm/logs/%x_%j.err

set -euo pipefail
echo "host=$(hostname)  date=$(date)"

module load pytorch/2.8.0
python -c "import torch; print('torch', torch.__version__, '| cuda_available', torch.cuda.is_available(), '|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'))"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd /global/u2/a/arghyara/work/wa_hls4ml_models
make train EPOCHS=2 BATCH=16 DATA_OUT="$SCRATCH/catapult_asic_smoke"
echo "SMOKE_OK"
