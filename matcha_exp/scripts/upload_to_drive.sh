#!/bin/bash
# Upload only the approved non-code assets to a writable Drive remote.
set -euo pipefail

if [ "$#" -ne 2 ] || [[ "$1" != *: ]]; then
    echo "Usage: $0 <writable-rclone-drive-remote:> <project-root>" >&2
    exit 2
fi
REMOTE="$1"
PROJECT_ROOT="$(cd "$2" && pwd)"
FOLDER_ID="${MATCHA_DRIVE_FOLDER_ID:-1eFzRECP1BWAvc1OZHUkRRk-kdDrp8qBZ}"
command -v rclone >/dev/null || { echo "rclone is required" >&2; exit 2; }
python "$PROJECT_ROOT/matcha_exp/scripts/hpc_bundle.py" verify-source-assets \
    --project-root "$PROJECT_ROOT"

rclone copy "$PROJECT_ROOT/artifacts/ljspeech_augmented_medium" "${REMOTE}assets/augmentation" \
    --drive-root-folder-id "$FOLDER_ID" --progress
rclone check "$PROJECT_ROOT/artifacts/ljspeech_augmented_medium" "${REMOTE}assets/augmentation" \
    --drive-root-folder-id "$FOLDER_ID" --one-way

upload_one() {
    local source="$1" destination="$2"
    local source_dir="${source%/*}" destination_dir="${destination%/*}"
    local filename="${source##*/}"
    rclone copyto "$PROJECT_ROOT/$source" "${REMOTE}assets/$destination" \
        --drive-root-folder-id "$FOLDER_ID" --progress
    rclone check "$PROJECT_ROOT/$source_dir" "${REMOTE}assets/$destination_dir" \
        --drive-root-folder-id "$FOLDER_ID" --include "/$filename" --one-way
}
upload_one "a0_jepa_s42/latest.pt" "checkpoints/a0_jepa_s42/latest.pt"
upload_one "a2_visreg_s42/latest.pt" "checkpoints/a2_visreg_s42/latest.pt"
upload_one "checkpoints/matcha_ljspeech.ckpt" "checkpoints/matcha_ljspeech.ckpt"
upload_one "ljspeech/LJSpeech-1.1/metadata.csv" "ljspeech/metadata.csv"
upload_one "lexicon/abbrev-lexicon.json" "lexicon/abbrev-lexicon.json"
echo "Uploaded and checked approved assets under ${REMOTE}assets/"
