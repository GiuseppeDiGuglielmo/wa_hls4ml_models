#!/bin/bash
# Refresh the stale "GNN GATv2 (pooling, 54.47M) extrap" rows with the new catapult_v2 data + 4 seeds.
# Trains the PLAIN POOLING FPGA_GNN_GATv2 (NOT SumDecomp) on:
#   train12  = 45nm {1,2}    -> eval L3 (the {1,2}->3L refresh) + L6/L8/L10 (deep extrap)
#   train123 = 45nm {1,2,3}  -> eval L6/L8/L10 (L10 = the Run N comparison vs SumDecomp)
# 2 configs x 4 seeds = 8 jobs.  Run from repo root: bash slurm/submit_poolgat.sh
set -euo pipefail
REPO=/global/u2/a/arghyara/work/wa_hls4ml_models
V2=${SCRATCH}/catapult_v2
cd "$REPO"

submit_cfg() {  # CONFIG SPLIT TESTSETS
  local cfg=$1 split=$2 testsets=$3
  for seed in a b c d; do
    echo "Submitting poolgat ${cfg} seed ${seed}  (eval: ${testsets})"
    sbatch --export=ALL,SPLIT=${V2}/${split},CONFIG=${cfg},SEED=${seed},LR=1e-3,EP=200,PAT=30,TESTSETS="${testsets}" \
      slurm/train_poolgat_one.sh
  done
}

submit_cfg train12  split_12  "L3 L6 L8 L10"
submit_cfg train123 split_123 "L6 L8 L10"

echo ""
echo "Submitted 8 jobs (poolgat: 2 configs x 4 seeds)."
echo "Monitor: squeue --me"
echo "Results -> \${SCRATCH}/catapult_v2/eval_results/poolgat__*.json"
