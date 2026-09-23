#!/bin/bash
#SBATCH --job-name=matcha-submit
#SBATCH --partition=gpu-long
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/%x-%j.out

# For a site-specific partition, export SBATCH_PARTITION before calling sbatch.
# SLURM reads directives before this shell starts; MATCHA_PARTITION alone can
# only configure child submissions. See MATCHA_EXPERIMENT_HANDOVER.md.
set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
mkdir -p "$PROJECT_ROOT/logs"
exec bash "$PROJECT_ROOT/matcha_exp/slurm/submit_all.sh"
