#!/bin/bash
# Convert 22nm gf22fdx 1/2/3-layer reports -> npy, then build the transfer-study splits.
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -A amsc011
#SBATCH -c 32
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -J wa_prep_22nm
#SBATCH -o /global/u2/a/arghyara/work/wa_hls4ml_models/slurm/logs/%x_%j.out
#SBATCH -e /global/u2/a/arghyara/work/wa_hls4ml_models/slurm/logs/%x_%j.err

set -euo pipefail
REPO=/global/u2/a/arghyara/work/wa_hls4ml_models
ARCHIVE=/global/cfs/cdirs/amsc011/shared/wa-hls4ml-catapult/gf22fdx
RAW=${SCRATCH}/catapult_22nm/raw
OUT=${SCRATCH}/catapult_22nm

module load pytorch/2.8.0
source "${SCRATCH}/venv_wa_hls4ml_models/bin/activate"
export WA_NUM_WORKERS=32
cd "${REPO}"

for N in 1 2 3; do
  echo "===== converting 22nm L${N} from ${ARCHIVE}/mlp-${N}layer ====="
  python dataset/convert_parallel.py --archive "${ARCHIVE}/mlp-${N}layer" \
    --output "${RAW}" --prefix "L${N}" --workers 32
done

echo "===== building 22nm splits ====="
python dataset/build_22nm_splits.py --raw "${RAW}" --out "${OUT}"
echo "PREP_22NM_DONE"
