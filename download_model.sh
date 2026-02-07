#!/bin/bash
# Download Voxtral Realtime 4B model from HuggingFace
#
# Usage: ./download_model.sh [--dir DIR]
#   --dir DIR   Download to DIR (default: voxtral-model)

set -euo pipefail

MODEL_ID="mistralai/Voxtral-Mini-4B-Realtime-2602"
MODEL_DIR="voxtral-model"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dir) MODEL_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "Downloading Voxtral Realtime 4B to ${MODEL_DIR}/"
echo "Model: ${MODEL_ID}"
echo ""

mkdir -p "${MODEL_DIR}"

# Files to download
FILES=(
    "consolidated.safetensors"
    "params.json"
    "tekken.json"
)

BASE_URL="https://huggingface.co/${MODEL_ID}/resolve/main"

for file in "${FILES[@]}"; do
    dest="${MODEL_DIR}/${file}"
    url="${BASE_URL}/${file}"

    local_size=0
    if [ -f "${dest}" ]; then
        local_size="$(stat -c%s "${dest}" 2>/dev/null || echo 0)"
    fi

    remote_size="$(curl -sIL "${url}" | awk 'BEGIN{IGNORECASE=1} /^content-length:/ {print $2}' | tail -n 1 | tr -d '\r' || true)"

    if [[ "${local_size}" != "0" && -n "${remote_size}" && "${local_size}" == "${remote_size}" ]]; then
        echo "  [skip] ${file} (already complete)"
        continue
    fi

    if [[ "${local_size}" != "0" ]]; then
        echo "  [resume] ${file} (${local_size} bytes present)..."
    else
        echo "  [download] ${file}..."
    fi

    curl -L --fail --retry 5 --retry-delay 2 --continue-at - \
      -o "${dest}" "${url}" --progress-bar
    echo "  [done] ${file}"
done

echo ""
echo "Download complete. Model files in ${MODEL_DIR}/"
ls -lh "${MODEL_DIR}/"
