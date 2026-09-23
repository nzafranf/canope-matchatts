#!/bin/bash

export MATCHA_PROJECT_ROOT="${MATCHA_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export MATCHA_UPSTREAM_ROOT="${MATCHA_UPSTREAM_ROOT:-$MATCHA_PROJECT_ROOT/third_party/Matcha-TTS}"
export MATCHA_CONDA_ENV="${MATCHA_CONDA_ENV:-matcha-canon}"
export MATCHA_DATA_ROOT="${MATCHA_DATA_ROOT:-$MATCHA_PROJECT_ROOT/data_hpc}"
export MATCHA_AUG_ROOT="${MATCHA_AUG_ROOT:-$MATCHA_DATA_ROOT/augmentation}"
export MATCHA_OUTPUT_ROOT="${MATCHA_OUTPUT_ROOT:-$MATCHA_PROJECT_ROOT/outputs/matcha_augmented}"
export MATCHA_TB_ROOT="${MATCHA_TB_ROOT:-$MATCHA_OUTPUT_ROOT/tensorboard}"
export MATCHA_BASE_CKPT="${MATCHA_BASE_CKPT:-$MATCHA_DATA_ROOT/assets/checkpoints/matcha_ljspeech.ckpt}"
export CANON_A0_CKPT="${CANON_A0_CKPT:-$MATCHA_DATA_ROOT/assets/checkpoints/a0_jepa_s42/latest.pt}"
export CANON_A2_CKPT="${CANON_A2_CKPT:-$MATCHA_DATA_ROOT/assets/checkpoints/a2_visreg_s42/latest.pt}"
export MATCHA_SEED="${MATCHA_SEED:-42}"
export MATCHA_PARTITION="${MATCHA_PARTITION:-${SBATCH_PARTITION:-gpu-long}}"
export MATCHA_GPU_H100="${MATCHA_GPU_H100:-h100-96}"
export MATCHA_GPU_A100="${MATCHA_GPU_A100:-a100-80}"

matcha_find_conda_sh() {
    local candidate
    for candidate in "${MATCHA_CONDA_SH:-}" \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/anaconda3/etc/profile.d/conda.sh" \
        "$HOME/miniforge3/etc/profile.d/conda.sh" \
        "$HOME/mambaforge/etc/profile.d/conda.sh"; do
        if [ -n "$candidate" ] && [ -f "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

matcha_setup_env() {
    local conda_sh
    conda_sh="$(matcha_find_conda_sh)" || {
        echo "[fatal] conda.sh not found; set MATCHA_CONDA_SH" >&2
        exit 2
    }
    source "$conda_sh"
    conda activate "$MATCHA_CONDA_ENV"
    export PROJECT_ROOT="$MATCHA_UPSTREAM_ROOT"
    export PYTHONPATH="$MATCHA_PROJECT_ROOT:$MATCHA_UPSTREAM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    export PYTHONUNBUFFERED=1
    export TOKENIZERS_PARALLELISM=false
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    cd "$MATCHA_UPSTREAM_ROOT"
}

matcha_experiment_for() {
    case "$1" in
        baseline)  echo augmented_baseline ;;
        jepa_a0)   echo augmented_jepa_a0 ;;
        canopy_a2) echo augmented_canopy_a2 ;;
        *) return 1 ;;
    esac
}

matcha_gpu_for() {
    case "$1" in
        baseline) echo "$MATCHA_GPU_A100" ;;
        jepa_a0|canopy_a2) echo "$MATCHA_GPU_H100" ;;
        *) return 1 ;;
    esac
}

matcha_job_name() {
    echo "mt-$1-s${MATCHA_SEED}"
}

matcha_run_dir() {
    echo "$MATCHA_OUTPUT_ROOT/$1-s${MATCHA_SEED}"
}

matcha_preflight() {
    local failed=0 path
    for path in \
        "$MATCHA_AUG_ROOT/APPROVED" \
        "$MATCHA_AUG_ROOT/filelists/train_augmented.txt" \
        "$MATCHA_AUG_ROOT/filelists/val_augmented.txt" \
        "$MATCHA_BASE_CKPT" \
        "$CANON_A0_CKPT" \
        "$CANON_A2_CKPT"; do
        if [ ! -f "$path" ]; then
            echo "[preflight] missing: $path" >&2
            failed=1
        fi
    done
    if [ "$failed" -ne 0 ]; then
        echo "[preflight] retrieve/render approved data and checkpoints before sbatch" >&2
        return 1
    fi
    if [ "$(tr -d '[:space:]' < "$MATCHA_AUG_ROOT/APPROVED")" != "3f3c6a85b71d3732" ]; then
        echo "[preflight] augmentation approval digest mismatch" >&2
        return 1
    fi
}

matcha_is_complete() {
    [ -f "$(matcha_run_dir "$1")/SUCCESS" ]
}
