#!/bin/bash
# Refresh the §1.1 interpolation numbers (stale: old smaller dataset, single seed) for the two POOLING
# baselines on catapult_v2 + 4 seeds. Train+test both on random {1,2,3} (split_rand123). 2x4 = 8 jobs.
# Run from repo root: bash slurm/submit_pool_interp.sh
set -euo pipefail
REPO=/global/u2/a/arghyara/work/wa_hls4ml_models
cd "$REPO"

# ARCH | VARIANT(eval_ckpt) | LR | EP
for row in "gnn|poolsage|1e-3|120" "gatv2|poolgat|1e-3|120"; do
  IFS='|' read -r arch variant lr ep <<< "$row"
  for seed in a b c d; do
    echo "Submitting interp ${arch} (${variant}) seed ${seed}"
    sbatch --export=ALL,ARCH=${arch},VARIANT=${variant},SEED=${seed},LR=${lr},EP=${ep},PAT=25 \
      slurm/train_pool_interp_one.sh
  done
done
echo ""
echo "Submitted 8 interpolation jobs (FPGA_GNN + FPGA_GNN_GATv2, 4 seeds each)."
echo "Results -> \${SCRATCH}/catapult_v2/eval_results_interp/{poolsage,poolgat}__interp__{a,b,c,d}.json"