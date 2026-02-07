#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-voxtral-model}"
SAMPLE_FILE="${2:-samples/test_speech.wav}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[warn] nvidia-smi not found; CUDA runtime visibility cannot be verified"
else
  nvidia-smi
fi

make cuda

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "[warn] model directory '$MODEL_DIR' not found. Skipping runtime transcription smoke test."
  exit 0
fi

./voxtral -d "$MODEL_DIR" -i "$SAMPLE_FILE" --silent >/tmp/voxtral_cuda_smoke.txt
printf "[ok] CUDA smoke output bytes: %s\n" "$(wc -c </tmp/voxtral_cuda_smoke.txt)"

# stdin smoke
cat "$SAMPLE_FILE" | ./voxtral -d "$MODEL_DIR" --stdin --silent >/tmp/voxtral_cuda_stdin_smoke.txt
printf "[ok] CUDA stdin smoke output bytes: %s\n" "$(wc -c </tmp/voxtral_cuda_stdin_smoke.txt)"
