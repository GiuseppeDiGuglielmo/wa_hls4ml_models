#!/bin/bash
# Launch the 22nm in-distribution (rand123) comparison study.
# Conditions: {scratch, FT-full from 45nm-{1,2,3}} × {small-1.7M, GATv2-54M} × 4 seeds = 16 jobs
# All use the same rand123 split (random 70/15/15 of all 22nm L1+L2+L3).
# Run from the repo root: bash slurm/submit_22nm_rand123.sh
set -euo pipefail
REPO=/global/u2/a/arghyara/work/wa_hls4ml_models
CK=$REPO/GNN/gnn_results_plots
V2=${SCRATCH}/catapult_22nm
cd "$REPO"

# ---- Build rand123 split if not already done ----
if [ ! -f "${V2}/rand123/train_features.npy" ]; then
  echo "Building rand123 split..."
  module load pytorch/2.8.0
  source "${SCRATCH}/venv_wa_hls4ml_models/bin/activate"
  python dataset/build_22nm_rand123.py --raw "${V2}/raw" --out "${V2}"
  echo "rand123 split built."
fi

# ---- Source checkpoints for FT ----
SRC_SMALL=$(ls -d $CK/sumdecomp_adamw_*_s123/best_model 2>/dev/null | head -1)
SRC_GATV2=$(ls -d $CK/sumdecompgatv2_nadam_*_g123b/best_model 2>/dev/null | head -1)

echo "FT source small : ${SRC_SMALL}"
echo "FT source GATv2 : ${SRC_GATV2}"

if [ -z "$SRC_SMALL" ] || [ -z "$SRC_GATV2" ]; then
  echo "ERROR: 45nm {1,2,3} source checkpoints not found — check gnn_results_plots/ for s123 / g123b"
  exit 1
fi

# ---- Submit 8 jobs: small configs ----
for row in \
  "scratch|none|0|1e-3|200" \
  "ftfull_s123|${SRC_SMALL}|0|1e-4|120"; do
  IFS='|' read -r cfg src frz lr ep <<< "$row"
  echo "Submitting plain ${cfg}..."
  sbatch --export=ALL,VARIANT=plain,OPT=adamw,LR=$lr,EP=$ep,PAT=30,SRC=$src,FREEZE=$frz,CONFIG=$cfg \
    slurm/train_22nm_rand123_one.sh
done

# ---- Submit 8 jobs: GATv2 configs ----
for row in \
  "scratch|none|0|1e-4|120" \
  "ftfull_g123|${SRC_GATV2}|0|5e-5|80"; do
  IFS='|' read -r cfg src frz lr ep <<< "$row"
  echo "Submitting gatv2 ${cfg}..."
  sbatch --export=ALL,VARIANT=gatv2,OPT=nadam,LR=$lr,EP=$ep,PAT=30,SRC=$src,FREEZE=$frz,CONFIG=$cfg \
    slurm/train_22nm_rand123_one.sh
done

echo "Submitted 4 jobs (each loops 4 seeds) = 16 total runs."
echo "Monitor: squeue --me"
echo "Results will appear in: \${SCRATCH}/catapult_22nm/eval_results_rand123/"
