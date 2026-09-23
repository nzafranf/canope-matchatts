#!/bin/bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"
python "$PROJECT_ROOT/matcha_exp/scripts/hpc_bundle.py" verify --project-root "$PROJECT_ROOT"

if command -v mamba >/dev/null 2>&1; then
    mamba env create -f "$PROJECT_ROOT/matcha_exp/environment.yml"
elif command -v conda >/dev/null 2>&1; then
    conda env create -f "$PROJECT_ROOT/matcha_exp/environment.yml"
else
    echo "Neither mamba nor conda was found; load your site's Conda module first" >&2
    exit 2
fi
echo "Environment created. Activate with: conda activate matcha-canon"
