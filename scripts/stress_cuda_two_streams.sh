#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-voxtral-model}"
SAMPLE_FILE="${2:-samples/test_speech.wav}"

make cuda

SAMPLE_WAV="$SAMPLE_FILE"
tmp_wav=""
lower="${SAMPLE_FILE,,}"
if [[ "$lower" != *.wav ]]; then
  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[err] input is not .wav and ffmpeg is not installed: '$SAMPLE_FILE'"
    exit 1
  fi
  tmp_wav="/tmp/voxtral_stress_two_streams_${$}.wav"
  ffmpeg -y -hide_banner -loglevel error -i "$SAMPLE_FILE" -ac 1 -ar 16000 "$tmp_wav"
  SAMPLE_WAV="$tmp_wav"
  trap 'rm -f "$tmp_wav"' EXIT
fi

BIN="/tmp/voxtral_stress_two_streams"
OBJ="/tmp/voxtral_stress_two_streams.o"

gcc -O2 -Wall -Wextra -pthread -I. -c -o "$OBJ" scripts/stress_two_streams.c
gcc -o "$BIN" \
  "$OBJ" \
  voxtral.o voxtral_kernels.o voxtral_audio.o voxtral_encoder.o voxtral_decoder.o voxtral_tokenizer.o \
  voxtral_safetensors.o voxtral_mic_macos.o voxtral_cuda_stub.o voxtral_cuda.o \
  -lm -fopenmp -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs -lcublasLt -lcublas -lcuda

echo "[info] two-stream stress: VOX_CUDA_PIPELINE_FULL=1 VOX_CUDA_FAST=1"
VOX_CUDA_PIPELINE_FULL=1 VOX_CUDA_FAST=1 "$BIN" "$MODEL_DIR" "$SAMPLE_WAV"
echo "[ok] two-stream stress passed"

