#!/bin/bash
# Convert the FULL 6-layer (10k, all RF) + new 8-layer (10k) + 10-layer (10k) sets
# from staged symlink dirs (underlying gdg:amsc011 run dirs; the mlp-Nlayer symlink
# dirs themselves remain gdg:gdg-blocked). Then build split_123456 for the new
# deepest training config (train {1..6} -> internal test {8}).
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -A amsc011
#SBATCH -c 32
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -J wa_prep_6_8_10
#SBATCH -o /global/u2/a/arghyara/work/wa_hls4ml_models/slurm/logs/%x_%j.out
#SBATCH -e /global/u2/a/arghyara/work/wa_hls4ml_models/slurm/logs/%x_%j.err

set -euo pipefail
REPO=/global/u2/a/arghyara/work/wa_hls4ml_models
STAGE=${SCRATCH}/catapult_v2/stage
RAW=${SCRATCH}/catapult_v2/raw
OUT=${SCRATCH}/catapult_v2

module load pytorch/2.8.0
source "${SCRATCH}/venv_wa_hls4ml_models/bin/activate"
export WA_NUM_WORKERS=32
cd "${REPO}"

for N in 6 8 10; do
  echo "===== converting L${N} (full) from ${STAGE}/L${N} ====="
  python dataset/convert_parallel.py --archive "${STAGE}/L${N}" \
    --output "${RAW}" --prefix "L${N}" --workers 32
done

echo "===== building split_123456 (train {1,2,3,4,5,6} -> test {8}) ====="
python dataset/build_split_generic.py --raw "${RAW}" --out "${OUT}/split_123456" \
  --train-groups 1 2 3 4 5 6 --test-groups 8

echo "PREP_6_8_10_DONE"
