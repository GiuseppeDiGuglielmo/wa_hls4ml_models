#!/bin/bash
# Run the active-learning replay (experiment A) on one GPU.
# Usage: sbatch -q <qos> [-t HH:MM:SS] active_learning/slurm/replay.sh <out name> [replay.py args...]
#   e.g. sbatch -q express_amsc active_learning/slurm/replay.sh timing --seeds 0 --max-rounds 3
#   COST="cost_L12_v2.tsv cost_L3pool.tsv" sbatch ... (cost tables under $AL; default cost_L12.tsv)
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH --gpus-per-node=1
#SBATCH -c 32
#SBATCH -A amsc011
#SBATCH -t 02:00:00
#SBATCH -J al_replay
#SBATCH -o /pscratch/sd/g/gdg/al_replay/logs/%x_%j.out
set -euo pipefail
REPO=$SLURM_SUBMIT_DIR
[ -f "$REPO/active_learning/replay.py" ] || { echo "submit from the wa-hls4ml-models repo root"; exit 2; }
NAME=$1; shift
AL=${AL:-$SCRATCH/al_replay}
# COST: space-separated design_cost.py tables under $AL (wait up to 60 min for them)
COST=${COST:-cost_L12.tsv}
for c in $COST; do
  for i in $(seq 360); do [ -s "$AL/$c" ] && break; sleep 10; done
  [ -s "$AL/$c" ] || { echo "no cost table at $AL/$c"; exit 3; }
done
module load pytorch/2.8.0
cd "$REPO/active_learning"
echo "commit $(git -C "$REPO" rev-parse --short HEAD)$([ -z "$(git -C "$REPO" status --porcelain)" ] || echo +dirty)"
python replay.py --raw "$AL/raw" --cost $(for c in $COST; do echo "$AL/$c"; done) --out "$AL/A/$NAME" "$@"
echo AL_REPLAY_DONE
