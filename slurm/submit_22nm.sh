#!/bin/bash
# Launch the full 22nm transfer matrix: both backbones × 12 configs × 4 seeds (each config = one job
# that loops 4 seeds, trains then evals on the fixed 22nm 3-layer test). Run from the repo root.
set -euo pipefail
REPO=/global/u2/a/arghyara/work/wa_hls4ml_models
CK=$REPO/GNN/gnn_results_plots
V2=${SCRATCH}/catapult_22nm
cd "$REPO"

declare -A SRC
SRC[s12]=$(ls -d $CK/*_s12d/best_model 2>/dev/null | head -1)        # 45nm small {1,2}
SRC[sdeep]=$(ls -d $CK/*_j12345a/best_model 2>/dev/null | head -1)   # 45nm small {1..5}
SRC[g12]=$(ls -d $CK/*_g12c/best_model 2>/dev/null | head -1)        # 45nm GATv2 {1,2}
SRC[gdeep]=$(ls -d $CK/*_gj12345b/best_model 2>/dev/null | head -1)  # 45nm GATv2 {1..5}

# CONFIG|SPLIT|SRCKEY|FREEZE|LR|EP   (SRCKEY none = from scratch)
small_rows=(
 "scratch|ft22_all|none|0|1e-3|200"      "fthead12|ft22_all|s12|1|1e-3|120"
 "ftfull12|ft22_all|s12|0|1e-4|120"      "ftheaddeep|ft22_all|sdeep|1|1e-3|120"
 "ftfulldeep|ft22_all|sdeep|0|1e-4|120"  "ceiling|ceil22|none|0|1e-3|200"
 "scratch100|ft22_100|none|0|1e-3|200"   "scratch500|ft22_500|none|0|1e-3|200"
 "scratch2000|ft22_2000|none|0|1e-3|200" "ftfull100|ft22_100|s12|0|1e-4|120"
 "ftfull500|ft22_500|s12|0|1e-4|120"     "ftfull2000|ft22_2000|s12|0|1e-4|120")
gat_rows=(
 "scratch|ft22_all|none|0|1e-4|120"      "fthead12|ft22_all|g12|1|1e-4|80"
 "ftfull12|ft22_all|g12|0|5e-5|80"       "ftheaddeep|ft22_all|gdeep|1|1e-4|80"
 "ftfulldeep|ft22_all|gdeep|0|5e-5|80"   "ceiling|ceil22|none|0|1e-4|120"
 "scratch100|ft22_100|none|0|1e-4|120"   "scratch500|ft22_500|none|0|1e-4|120"
 "scratch2000|ft22_2000|none|0|1e-4|120" "ftfull100|ft22_100|g12|0|5e-5|80"
 "ftfull500|ft22_500|g12|0|5e-5|80"      "ftfull2000|ft22_2000|g12|0|5e-5|80")

submit_set() { local variant=$1 opt=$2; shift 2
  for row in "$@"; do
    IFS='|' read -r cfg split srckey frz lr ep <<< "$row"
    local src=none; [ "$srckey" != "none" ] && src="${SRC[$srckey]}"
    sbatch --export=ALL,VARIANT=$variant,OPT=$opt,LR=$lr,EP=$ep,PAT=30,SPLIT=$V2/$split,SRC=$src,FREEZE=$frz,CONFIG=$cfg \
      slurm/train_22nm_one.sh
  done
}
submit_set plain adamw "${small_rows[@]}"
submit_set gatv2 nadam "${gat_rows[@]}"
