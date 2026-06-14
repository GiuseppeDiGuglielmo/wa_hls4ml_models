#!/bin/bash
# One 22nm transfer config = 4 seeds (scratch OR fine-tune), each trained then evaluated on the
# fixed 22nm 3-layer test set. Env: VARIANT OPT LR EP PAT SPLIT SRC FREEZE CONFIG.
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q express_amsc
#SBATCH --gpus-per-node=1
#SBATCH -c 32
#SBATCH -A amsc011
#SBATCH -t 04:00:00
#SBATCH -J wa_22nm
#SBATCH -o /global/u2/a/arghyara/work/wa_hls4ml_models/slurm/logs/%x_%j.out
#SBATCH -e /global/u2/a/arghyara/work/wa_hls4ml_models/slurm/logs/%x_%j.err

set -euo pipefail
REPO=/global/u2/a/arghyara/work/wa_hls4ml_models
CKROOT=${REPO}/GNN/gnn_results_plots
RES=${SCRATCH}/catapult_22nm/eval_results
mkdir -p "${RES}"
VARIANT=${VARIANT:-plain}; OPT=${OPT:-adamw}; LR=${LR:-1e-3}; EP=${EP:-200}; PAT=${PAT:-30}
SPLIT=${SPLIT:?}; SRC=${SRC:-none}; FREEZE=${FREEZE:-0}; CONFIG=${CONFIG:?}
vsuf=""; [ "$VARIANT" = "gatv2" ] && vsuf="gatv2"

module load pytorch/2.8.0
source "${SCRATCH}/venv_wa_hls4ml_models/bin/activate"
export WA_NUM_WORKERS=16
cd "${REPO}/GNN"

ft_args=""
[ "$SRC" != "none" ] && ft_args="--finetune-from ${SRC}"
[ "$FREEZE" = "1" ] && ft_args="${ft_args} --freeze-encoder"

for seed in a b c d; do
  tag="n22_${CONFIG}_${seed}"
  echo "===== TRAIN ${VARIANT} ${tag}  (split=$(basename ${SPLIT}) src=$(basename ${SRC}) freeze=${FREEZE}) ====="
  python run_sumdecomp.py --base-dir "${SPLIT}" --variant "${VARIANT}" --optimizer "${OPT}" \
    --lr "${LR}" --epochs "${EP}" --patience "${PAT}" --batch-size 256 --drop-throughput \
    --run-tag "${tag}" ${ft_args}
  ck=$(ls -dt ${CKROOT}/sumdecomp${vsuf}_${OPT}_*_${tag}/best_model 2>/dev/null | head -1)
  if [ -z "${ck}" ]; then echo "NO CKPT for ${tag}"; continue; fi
  echo "===== EVAL ${tag} <- ${ck} ====="
  python eval_ckpt.py --ckpt-dir "${ck}" --variant "${VARIANT}" \
    --features "${SPLIT}/test_features.npy" --labels "${SPLIT}/test_labels.npy" \
    --tag "${VARIANT}__${CONFIG}__${seed}" \
    --json-out "${RES}/${VARIANT}__${CONFIG}__${seed}.json" \
    --save-preds "${RES}/${VARIANT}__${CONFIG}__${seed}.npz" || echo "EVAL FAILED ${tag}"
done
echo "DONE_22NM ${VARIANT} ${CONFIG}"
