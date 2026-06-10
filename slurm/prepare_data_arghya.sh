#!/bin/bash
# Catapult-ASIC surrogate: build numpy dataset + train/val/test split from the
# shared CFS archive. CPU-only (no GPU needed). Arghya's paths.
#
# NOTE: requires read access to ${ARCHIVE}/run_*/reports/*.json. As of 2026-06-01
# those subdirs are group 'gdg' mode 770 and are NOT readable by amsc011 members.
# Ask Giuseppe to chgrp -R amsc011 + chmod -R g+rX the run_*/reports (and tarballs)
# dirs before submitting this.
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -A amsc011
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -J wa_prepare_data
#SBATCH -o /global/u2/a/arghyara/work/wa_hls4ml_models/slurm/logs/%x_%j.out
#SBATCH -e /global/u2/a/arghyara/work/wa_hls4ml_models/slurm/logs/%x_%j.err

set -euo pipefail

ARCHIVE=/global/cfs/cdirs/amsc011/shared/wa-hls4ml-catapult
REPO=/global/u2/a/arghyara/work/wa_hls4ml_models

module load pytorch/2.8.0

cd "${REPO}"
# make data  -> dataset/catapult_asic_to_numpy.py --archive ... (excludes legacy fixed-weight runs)
# make split -> dataset/split_data.py (random 70/15/15, seed 42)
# Outputs land under $SCRATCH/catapult_asic_data/{,split}/
# NOTE: deliberately NOT running 'make clean' (which rm -rf's $DATA_OUT) — build is additive.
make data split ARCHIVE="${ARCHIVE}"
