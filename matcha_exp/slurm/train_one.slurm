#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --output=logs/%x-%j.out
#SBATCH --signal=B:USR1@300
#SBATCH --requeue

set -uo pipefail
PROJECT_ROOT="${MATCHA_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "$PROJECT_ROOT/matcha_exp/slurm/common.sh"
RUN_ID="${MATCHA_RUN_ID:?MATCHA_RUN_ID is required}"
EXPERIMENT="$(matcha_experiment_for "$RUN_ID")" || exit 2
RUN_DIR="$(matcha_run_dir "$RUN_ID")"
mkdir -p "$RUN_DIR/checkpoints" "$MATCHA_PROJECT_ROOT/logs"

matcha_setup_env
matcha_preflight || exit 2

LAST="$RUN_DIR/checkpoints/last.ckpt"
if [ -f "$LAST" ]; then
    RESUME="ckpt_path=$LAST"
    echo "[$RUN_ID] resuming from $LAST"
else
    RESUME="ckpt_path=null"
    echo "[$RUN_ID] initializing from $MATCHA_BASE_CKPT"
fi

TRAIN_PID=""
on_signal() {
    echo "[$RUN_ID] received pre-timeout signal; terminating trainer after its latest checkpoint"
    if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
        kill -TERM "$TRAIN_PID" 2>/dev/null || true
        wait "$TRAIN_PID" || true
    fi
    exit 99
}
trap on_signal USR1 TERM

python -m matcha.train \
    experiment="$EXPERIMENT" \
    seed="$MATCHA_SEED" \
    "$RESUME" \
    "hydra.run.dir=$RUN_DIR" \
    "paths.output_dir=$RUN_DIR" \
    "callbacks.model_checkpoint.dirpath=$RUN_DIR/checkpoints" \
    callbacks.model_checkpoint.every_n_train_steps=2000 \
    callbacks.model_checkpoint.every_n_epochs=null \
    "logger.tensorboard.save_dir=$MATCHA_TB_ROOT" \
    "logger.tensorboard.name=$RUN_ID-s$MATCHA_SEED" \
    trainer.devices=1 \
    trainer.accelerator=gpu \
    trainer.precision=bf16-mixed \
    test=false &
TRAIN_PID=$!
wait "$TRAIN_PID"
status=$?
if [ "$status" -eq 0 ]; then
    touch "$RUN_DIR/SUCCESS"
    echo "[$RUN_ID] complete"
else
    echo "[$RUN_ID] failed with status $status; watchdog will resubmit" >&2
fi
exit "$status"
