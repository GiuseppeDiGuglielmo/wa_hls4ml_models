#!/bin/bash
# Eval all existing sum-decomp checkpoints (small plain + large gatv2; 4 train-configs x 4 seeds)
# on the full 6-layer (10k), 8-layer (10k) and 10-layer (10k) test sets. Eval-only, no training.
# Writes one JSON + one preds .npz per (config, seed, testset) under eval_results/.
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q express_amsc
#SBATCH --gpus-per-node=1
#SBATCH -c 32
#SBATCH -A amsc011
#SBATCH -t 02:00:00
#SBATCH -J wa_eval_6_8_10
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

# manifest rows: variant | config | seedtag | glob_suffix
ROWS=(
  "plain|train12|a|_s12"   "plain|train12|b|_s12b"   "plain|train12|c|_s12c"   "plain|train12|d|_s12d"
  "plain|train123|a|_s123" "plain|train123|b|_s123b" "plain|train123|c|_s123c" "plain|train123|d|_s123d"
  "plain|train1234|a|_s1234a" "plain|train1234|b|_s1234b" "plain|train1234|c|_s1234c" "plain|train1234|d|_s1234d"
  "plain|train12345|a|_j12345a" "plain|train12345|b|_j12345b" "plain|train12345|c|_j12345c" "plain|train12345|d|_j12345d"
  "gatv2|train12|a|_g12a"  "gatv2|train12|b|_g12b"   "gatv2|train12|c|_g12c"   "gatv2|train12|d|_g12d"
  "gatv2|train123|a|_s123gat2" "gatv2|train123|b|_g123b" "gatv2|train123|c|_g123c" "gatv2|train123|d|_g123d"
  "gatv2|train1234|a|_g1234a" "gatv2|train1234|b|_g1234b" "gatv2|train1234|c|_g1234c" "gatv2|train1234|d|_g1234d"
  "gatv2|train12345|a|_gj12345a" "gatv2|train12345|b|_gj12345b" "gatv2|train12345|c|_gj12345c" "gatv2|train12345|d|_gj12345d"
)
TESTSETS=(L6 L8 L10)

for row in "${ROWS[@]}"; do
  IFS='|' read -r variant config seed suffix <<< "$row"
  ckdir=$(ls -d ${CKROOT}/*"${suffix}"/best_model 2>/dev/null | head -1)
  if [ -z "${ckdir}" ] || [ ! -f "${ckdir}/model.pt" ]; then
    echo "MISSING ckpt for ${config} ${seed} (${suffix})"; continue
  fi
  for ts in "${TESTSETS[@]}"; do
    tag="${variant}__${config}__${seed}__${ts}"
    echo "===== eval ${variant} ${tag} <- ${ckdir} ====="
    python eval_ckpt.py --ckpt-dir "${ckdir}" --variant "${variant}" \
      --features "${RAW}/${ts}_features.npy" --labels "${RAW}/${ts}_labels.npy" \
      --tag "${tag}" --json-out "${RES}/${tag}.json" \
      --save-preds "${RES}/${tag}.npz" || echo "FAILED ${tag}"
  done
done
echo "EVAL_6_8_10_DONE"
