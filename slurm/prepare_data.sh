#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -A amsc011
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -J prepare_data
#SBATCH -o /global/homes/g/gdg/research/projects/genesis/wa-hls4ml-paper/wa-hls4ml-models/slurm/logs/%x_%j.out
#SBATCH -e /global/homes/g/gdg/research/projects/genesis/wa-hls4ml-paper/wa-hls4ml-models/slurm/logs/%x_%j.err

set -euo pipefail

ARCHIVE=/global/cfs/cdirs/amsc011/shared/wa-hls4ml-catapult
REPO=/global/homes/g/gdg/research/projects/genesis/wa-hls4ml-paper/wa-hls4ml-models

module load pytorch/2.8.0

cd "${REPO}"
make clean data split ARCHIVE="${ARCHIVE}"
