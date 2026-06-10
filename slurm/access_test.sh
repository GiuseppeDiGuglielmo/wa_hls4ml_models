#!/bin/bash
# Test whether the CFS dataset reports/ dir is readable from INSIDE a compute-node job.
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -A amsc011
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:05:00
#SBATCH -J wa_access_test
#SBATCH -o /global/u2/a/arghyara/work/wa_hls4ml_models/slurm/logs/%x_%j.out
#SBATCH -e /global/u2/a/arghyara/work/wa_hls4ml_models/slurm/logs/%x_%j.err

set -uo pipefail
echo "host=$(hostname)"
echo "=== id on the compute node ==="
id
D=/global/cfs/cdirs/amsc011/shared/wa-hls4ml-catapult
RUN=run_20260507_150243_e6c8c415
echo "=== ls run dir (group amsc011) ==="
ls "$D/$RUN" 2>&1
echo "=== ls reports subdir (group gdg) ==="
ls "$D/$RUN/reports" 2>&1 | head -3
echo "ls_reports_rc=$?"
echo "=== count readable json in reports ==="
ls "$D/$RUN/reports/"*.json 2>/dev/null | wc -l
echo "=== try reading first json ==="
f=$(ls "$D/$RUN/reports/"*.json 2>/dev/null | head -1)
echo "first_json=$f"
head -c 150 "$f" 2>&1
echo
echo "=== ACCESS_TEST_DONE ==="
