#!/bin/bash
# Called by the root-level `sbatch submit_all.sh` orchestration job.
set -euo pipefail

cd "${MATCHA_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source matcha_exp/slurm/common.sh
matcha_preflight
mkdir -p "$MATCHA_OUTPUT_ROOT" "$MATCHA_TB_ROOT" "$MATCHA_PROJECT_ROOT/logs"

RUNS=(baseline jepa_a0 canopy_a2)
queue="$(squeue -u "$USER" -h -o "%j")" || {
    echo "[fatal] squeue failed; refusing submissions with unknown scheduler state" >&2
    exit 2
}
echo "Submitting all runs independently (eligible to execute in parallel):"
for run_id in "${RUNS[@]}"; do
    gpu="$(matcha_gpu_for "$run_id")"
    job_name="$(matcha_job_name "$run_id")"
    if matcha_is_complete "$run_id" || grep -qx "$job_name" <<< "$queue"; then
        echo "  $run_id already complete or queued; skipping"
        continue
    fi
    job_id="$(sbatch --parsable \
        --partition="$MATCHA_PARTITION" \
        --gres="gpu:${gpu}:1" \
        --job-name="$job_name" \
        --export="ALL,MATCHA_PROJECT_ROOT=$MATCHA_PROJECT_ROOT,MATCHA_RUN_ID=$run_id" \
        matcha_exp/slurm/train_one.slurm)"
    echo "  $run_id -> $job_id [$gpu]"
done

if ! grep -qx "matcha-tb-s${MATCHA_SEED}" <<< "$queue"; then
    tb_id="$(sbatch --parsable --partition="$MATCHA_PARTITION" --job-name="matcha-tb-s${MATCHA_SEED}" \
        --export="ALL,MATCHA_PROJECT_ROOT=$MATCHA_PROJECT_ROOT" \
        matcha_exp/slurm/tensorboard.slurm)"
    echo "  tensorboard -> $tb_id"
fi

if ! grep -qx "matcha-wd-s${MATCHA_SEED}" <<< "$queue"; then
    wd_id="$(sbatch --parsable --partition="$MATCHA_PARTITION" --job-name="matcha-wd-s${MATCHA_SEED}" \
        --export="ALL,MATCHA_PROJECT_ROOT=$MATCHA_PROJECT_ROOT" \
        matcha_exp/slurm/watchdog.slurm)"
    echo "  watchdog -> $wd_id"
fi

echo "TensorBoard root: $MATCHA_TB_ROOT"
