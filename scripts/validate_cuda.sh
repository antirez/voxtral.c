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

# "All opts on" convenience knob (best-effort; respects per-feature disables).
VOX_CUDA_FAST=1 ./voxtral -d "$MODEL_DIR" -i "$SAMPLE_FILE" --silent >/tmp/voxtral_cuda_fast_smoke.txt
printf "[ok] CUDA fast smoke output bytes: %s\n" "$(wc -c </tmp/voxtral_cuda_fast_smoke.txt)"

# CUDA Graph smoke (opt-in). Also validates the graph capture path.
VOX_CUDA_GRAPHS=1 ./voxtral -d "$MODEL_DIR" -i "$SAMPLE_FILE" --silent >/tmp/voxtral_cuda_graph_smoke.txt
printf "[ok] CUDA graphs smoke output bytes: %s\n" "$(wc -c </tmp/voxtral_cuda_graph_smoke.txt)"

# Graphs + force-disable attention v3 (should still run; falls back to v2/v1).
VOX_CUDA_GRAPHS=1 VOX_DISABLE_CUDA_ATTN_V3=1 ./voxtral -d "$MODEL_DIR" -i "$SAMPLE_FILE" --silent >/tmp/voxtral_cuda_graph_nov3_smoke.txt
printf "[ok] CUDA graphs (no v3) smoke output bytes: %s\n" "$(wc -c </tmp/voxtral_cuda_graph_nov3_smoke.txt)"

# Optional GPU conv stem smoke (encoder front-end).
VOX_CUDA_CONV_STEM=1 ./voxtral -d "$MODEL_DIR" -i "$SAMPLE_FILE" --silent >/tmp/voxtral_cuda_conv_stem_smoke.txt
printf "[ok] CUDA conv-stem smoke output bytes: %s\n" "$(wc -c </tmp/voxtral_cuda_conv_stem_smoke.txt)"

# Optional full CUDA pipeline smoke (encoder adapter stays on device, decoder
# consumes adapter embeddings directly). Enable with: VOX_VALIDATE_PIPELINE_FULL=1
if [[ "${VOX_VALIDATE_PIPELINE_FULL:-0}" != "0" ]]; then
  VOX_CUDA_PIPELINE_FULL=1 ./voxtral -d "$MODEL_DIR" -i "$SAMPLE_FILE" --silent >/tmp/voxtral_cuda_pipeline_smoke.txt
  printf "[ok] CUDA pipeline smoke output bytes: %s\n" "$(wc -c </tmp/voxtral_cuda_pipeline_smoke.txt)"

  VOX_CUDA_PIPELINE_FULL=1 VOX_CUDA_GRAPHS=1 ./voxtral -d "$MODEL_DIR" -i "$SAMPLE_FILE" --silent >/tmp/voxtral_cuda_pipeline_graph_smoke.txt
  printf "[ok] CUDA pipeline+graphs smoke output bytes: %s\n" "$(wc -c </tmp/voxtral_cuda_pipeline_graph_smoke.txt)"
fi

# stdin smoke
cat "$SAMPLE_FILE" | ./voxtral -d "$MODEL_DIR" --stdin --silent >/tmp/voxtral_cuda_stdin_smoke.txt
printf "[ok] CUDA stdin smoke output bytes: %s\n" "$(wc -c </tmp/voxtral_cuda_stdin_smoke.txt)"
