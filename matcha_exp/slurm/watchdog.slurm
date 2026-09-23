#!/bin/bash
#SBATCH --partition=gpu-long
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/%x-%j.out
#SBATCH --signal=B:USR1@120

set -uo pipefail
PROJECT_ROOT="${MATCHA_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT_ROOT/matcha_exp/slurm/common.sh"
cd "$PROJECT_ROOT"

INTERVAL="${MATCHA_WATCHDOG_INTERVAL:-120}"
MAX_RETRIES="${MATCHA_WATCHDOG_MAX_RETRIES:-5}"
STATE_DIR="$MATCHA_OUTPUT_ROOT/watchdog"
mkdir -p "$STATE_DIR" "$MATCHA_PROJECT_ROOT/logs"
RUNS=(baseline jepa_a0 canopy_a2)

resubmit_watchdog() {
    echo "[watchdog] time limit approaching; submitting replacement"
    sbatch --partition="$MATCHA_PARTITION" --job-name="matcha-wd-s${MATCHA_SEED}" \
        --export="ALL,MATCHA_PROJECT_ROOT=$MATCHA_PROJECT_ROOT" \
        matcha_exp/slurm/watchdog.slurm || true
    exit 0
}
trap resubmit_watchdog USR1

echo "[watchdog] interval=${INTERVAL}s max_retries=$MAX_RETRIES"
while true; do
    all_done=1
    if ! queue="$(squeue -u "${SLURM_JOB_USER:-$USER}" -h -o "%j")"; then
        echo "[watchdog] squeue failed; scheduler state unknown, skipping this poll" >&2
        sleep "$INTERVAL"
        continue
    fi
    for run_id in "${RUNS[@]}"; do
        if matcha_is_complete "$run_id"; then
            continue
        fi
        all_done=0
        job_name="$(matcha_job_name "$run_id")"
        if echo "$queue" | grep -qx "$job_name"; then
            continue
        fi

        retry_file="$STATE_DIR/$run_id.retries"
        retries=0
        [ -f "$retry_file" ] && retries="$(cat "$retry_file")"
        if [ "$retries" -ge "$MAX_RETRIES" ]; then
            echo "[watchdog] $run_id reached retry limit $MAX_RETRIES; manual diagnosis required" >&2
            continue
        fi
        retries=$((retries + 1))
        gpu="$(matcha_gpu_for "$run_id")"
        previous="$(sacct -n -X --name "$job_name" --format=State -S now-2days 2>/dev/null | tail -1 | xargs || true)"
        echo "[watchdog] $(date -Iseconds) $run_id absent (last=$previous), resubmit $retries/$MAX_RETRIES"
        if sbatch --partition="$MATCHA_PARTITION" \
            --gres="gpu:${gpu}:1" \
            --job-name="$job_name" \
            --export="ALL,MATCHA_PROJECT_ROOT=$MATCHA_PROJECT_ROOT,MATCHA_RUN_ID=$run_id" \
            matcha_exp/slurm/train_one.slurm; then
            echo "$retries" > "$retry_file"
        else
            echo "[watchdog] submission failed; retry budget unchanged" >&2
        fi
    done
    if [ "$all_done" -eq 1 ]; then
        echo "[watchdog] all three SUCCESS markers found; exiting"
        exit 0
    fi
    sleep "$INTERVAL"
done
