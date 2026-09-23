#!/bin/bash
#SBATCH --partition=gpu-long
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
PROJECT_ROOT="${MATCHA_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT_ROOT/matcha_exp/slurm/common.sh"
matcha_setup_env
PORT="${MATCHA_TB_PORT:-6006}"
NODE="$(hostname -f)"
echo "SSH tunnel: ssh -N -L ${PORT}:${NODE}:${PORT} <nus-user>@<nus-login-node>"
echo "Open: http://localhost:${PORT}"
tensorboard --logdir "$MATCHA_TB_ROOT" --port "$PORT" --bind_all --reload_multifile=true
