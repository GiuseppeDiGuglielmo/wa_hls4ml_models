#!/bin/bash
# Eval the NEW deepest config checkpoints (train {1,2,3,4,5,6}, both models x4 seeds)
# on the 6/8/10-layer test sets. Eval-only. Writes JSON+npz under eval_results/ with
# config=train123456 so aggregate_6_8_10.py picks them up alongside the rest.
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q express_amsc
#SBATCH --gpus-per-node=1
#SBATCH -c 32
#SBATCH -A amsc011
#SBATCH -t 02:00:00
#SBATCH -J wa_eval_123456
#SBATCH -o /global/u2/a/arghyara/work/wa_hls4ml_models/slurm/logs/%x_%j.out
#SBATCH -e /global/u2/a/arghyara/work/wa_hls4ml_models/slurm/logs/%x_%j.err

set -euo pipefail
REPO=/global/u2/a/arghyara/work/wa_hls4ml_models
CKROOT=${REPO}/GNN/gnn_results_plots
RAW=${SCRATCH}/catapult_v2/raw
RES=${SCRATCH}/catapult_v2/eval_results
mkdir -p "${RES}"

module load pytorch/2.8.0
source "${SCRATCH}/venv_wa_hls4ml_models/bin/activate"
export WA_NUM_WORKERS=8
cd "${REPO}/GNN"

# variant | seedtag | glob_suffix   (config is train123456 for all)
ROWS=(
  "plain|a|_x123456a" "plain|b|_x123456b" "plain|c|_x123456c" "plain|d|_x123456d"
  "gatv2|a|_gx123456a" "gatv2|b|_gx123456b" "gatv2|c|_gx123456c" "gatv2|d|_gx123456d"
)
TESTSETS=(L6 L8 L10)
config=train123456

for row in "${ROWS[@]}"; do
  IFS='|' read -r variant seed suffix <<< "$row"
  ckdir=$(ls -d ${CKROOT}/*"${suffix}"/best_model 2>/dev/null | head -1)
  if [ -z "${ckdir}" ] || [ ! -f "${ckdir}/model.pt" ]; then
    echo "MISSING ckpt for ${config} ${seed} (${suffix})"; continue
  fi
  for ts in "${TESTSETS[@]}"; do
    tag="${variant}__${config}__${seed}__${ts}"
    echo "===== eval ${tag} <- ${ckdir} ====="
    python eval_ckpt.py --ckpt-dir "${ckdir}" --variant "${variant}" \
      --features "${RAW}/${ts}_features.npy" --labels "${RAW}/${ts}_labels.npy" \
      --tag "${tag}" --json-out "${RES}/${tag}.json" \
      --save-preds "${RES}/${tag}.npz" || echo "FAILED ${tag}"
  done
done
echo "EVAL_123456_DONE"
