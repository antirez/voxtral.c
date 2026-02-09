#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-voxtral-model}"
SAMPLE_FILE="${2:-samples/antirez_speaking_italian_short.ogg}"

make cuda

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "[warn] model directory '$MODEL_DIR' not found. Skipping runtime pipeline compact test."
  exit 0
fi

SAMPLE_WAV="$SAMPLE_FILE"
tmp_wav=""
lower="${SAMPLE_FILE,,}"
if [[ "$lower" != *.wav ]]; then
  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[err] input is not .wav and ffmpeg is not installed: '$SAMPLE_FILE'"
    exit 1
  fi
  tmp_wav="/tmp/voxtral_pipeline_compact_${$}.wav"
  ffmpeg -y -hide_banner -loglevel error -i "$SAMPLE_FILE" -ac 1 -ar 16000 "$tmp_wav"
  SAMPLE_WAV="$tmp_wav"
  trap 'rm -f "$tmp_wav"' EXIT
fi

cap="${VOX_CUDA_ADAPTER_CAP_TOKENS:-256}"
interval="${VOX_VALIDATE_INTERVAL_S:-0.05}"

echo "[info] VOX_CUDA_PIPELINE_FULL=1 ring-cap=${cap} interval=${interval}s"

VOX_CUDA_PIPELINE_FULL=1 VOX_CUDA_ADAPTER_CAP_TOKENS="$cap" \
  ./voxtral -d "$MODEL_DIR" -i "$SAMPLE_WAV" -I "$interval" --silent >/tmp/voxtral_cuda_pipeline_compact.txt
printf "[ok] pipeline compact output bytes: %s\n" "$(wc -c </tmp/voxtral_cuda_pipeline_compact.txt)"

VOX_CUDA_PIPELINE_FULL=1 VOX_CUDA_FAST=1 VOX_CUDA_ADAPTER_CAP_TOKENS="$cap" \
  ./voxtral -d "$MODEL_DIR" -i "$SAMPLE_WAV" -I "$interval" --silent >/tmp/voxtral_cuda_pipeline_compact_fast.txt
printf "[ok] pipeline compact+fast output bytes: %s\n" "$(wc -c </tmp/voxtral_cuda_pipeline_compact_fast.txt)"
