#!/bin/bash
# Download the approved non-code assets and dataset from Drive.
set -euo pipefail

if [ "$#" -ne 3 ] || [[ "$1" != *: ]]; then
    echo "Usage: $0 <rclone-drive-remote:> <destination-data-root> <extracted-project-root>" >&2
    exit 2
fi
REMOTE="$1"
DATA_ROOT="$(mkdir -p "$2" && cd "$2" && pwd)"
PROJECT_ROOT="$(cd "$3" && pwd)"
FOLDER_ID="${MATCHA_DRIVE_FOLDER_ID:-1eFzRECP1BWAvc1OZHUkRRk-kdDrp8qBZ}"

command -v rclone >/dev/null || { echo "rclone is required" >&2; exit 2; }
command -v python >/dev/null || { echo "python is required" >&2; exit 2; }
python "$PROJECT_ROOT/matcha_exp/scripts/hpc_bundle.py" verify --project-root "$PROJECT_ROOT"

(cd "$DATA_ROOT" && mkdir -p assets LJSpeech-1.1/wavs drive_filelists)
rclone copy "${REMOTE}assets" "$DATA_ROOT/assets" \
    --drive-root-folder-id "$FOLDER_ID" --progress
rclone check "${REMOTE}assets" "$DATA_ROOT/assets" \
    --drive-root-folder-id "$FOLDER_ID" --one-way
python "$PROJECT_ROOT/matcha_exp/scripts/hpc_bundle.py" verify-assets \
    --project-root "$PROJECT_ROOT" --data-root "$DATA_ROOT"
rclone lsf "${REMOTE}dataset" --drive-root-folder-id "$FOLDER_ID" --max-depth 1
rclone copy "${REMOTE}dataset" "$DATA_ROOT/drive_filelists" \
    --drive-root-folder-id "$FOLDER_ID" --include '/*.txt' --max-depth 1 --progress
rclone copy "${REMOTE}dataset/wavs" "$DATA_ROOT/LJSpeech-1.1/wavs" \
    --drive-root-folder-id "$FOLDER_ID" --progress
rclone check "${REMOTE}dataset" "$DATA_ROOT/drive_filelists" \
    --drive-root-folder-id "$FOLDER_ID" --include '/*.txt' --max-depth 1 --one-way
rclone check "${REMOTE}dataset/wavs" "$DATA_ROOT/LJSpeech-1.1/wavs" \
    --drive-root-folder-id "$FOLDER_ID" --one-way

python "$PROJECT_ROOT/matcha_exp/scripts/hpc_bundle.py" stage \
    --project-root "$PROJECT_ROOT" \
    --data-root "$DATA_ROOT" \
    --drive-filelists "$DATA_ROOT/drive_filelists"
echo "Assets ready. Export MATCHA_DATA_ROOT=$DATA_ROOT and MATCHA_AUG_ROOT=$DATA_ROOT/augmentation"
