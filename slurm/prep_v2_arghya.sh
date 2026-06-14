#!/bin/bash
# Build the depth-extrapolation v2 dataset (1,2,3,4-layer) from the reorganized
# nangate45 archive. Parallel converter (memory-light) + layer-aware splits.
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -A amsc011
#SBATCH -c 32
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -J wa_prep_v2
#SBATCH -o /global/u2/a/arghyara/work/wa_hls4ml_models/slurm/logs/%x_%j.out
#SBATCH -e /global/u2/a/arghyara/work/wa_hls4ml_models/slurm/logs/%x_%j.err

set -euo pipefail
REPO=/global/u2/a/arghyara/work/wa_hls4ml_models
ARCHIVE=/global/cfs/cdirs/amsc011/shared/wa-hls4ml-catapult/nangate45
RAW=${SCRATCH}/catapult_v2/raw
OUT=${SCRATCH}/catapult_v2
EXCLUDE="run_20260504 run_20260505 run_20260506 run_20260507_143427"

module load pytorch/2.8.0
source "${SCRATCH}/venv_wa_hls4ml_models/bin/activate"
export WA_NUM_WORKERS=32
cd "${REPO}"

for N in 1 2 3 4; do
  echo "===== converting mlp-${N}layer ====="
  python dataset/convert_parallel.py --archive "${ARCHIVE}/mlp-${N}layer" \
    --exclude ${EXCLUDE} --output "${RAW}" --prefix "L${N}" --workers 32
done

echo "===== building layer splits ====="
python dataset/build_layer_splits.py --raw "${RAW}" --out "${OUT}"
echo "PREP_V2_DONE"
