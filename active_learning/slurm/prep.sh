#!/bin/bash
# Convert the nangate45 depth groups for the active-learning replay (experiment A).
# Usage: sbatch -q <qos> active_learning/slurm/prep.sh [groups...]   (default 1 2 3 4 5 6 8 10)
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -A amsc011
#SBATCH -t 01:30:00
#SBATCH -J al_prep
#SBATCH -o /pscratch/sd/g/gdg/al_replay/logs/%x_%j.out
set -euo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
[ -f "$REPO/active_learning/prep_groups.py" ] || REPO=$SLURM_SUBMIT_DIR
OUT=${OUT:-$SCRATCH/al_replay/raw}
GROUPS_=${*:-1 2 3 4 5 6 8 10}
module load pytorch/2.8.0
cd "$REPO/active_learning"
python prep_groups.py --archive /global/cfs/cdirs/amsc011/shared/wa-hls4ml-catapult/nangate45 \
  --groups $GROUPS_ --out "$OUT" --workers 64
echo AL_PREP_DONE
