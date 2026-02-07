#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-voxtral-model}"
SAMPLE_FILE="${2:-samples/test_speech.wav}"

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "model dir '$MODEL_DIR' missing"
  exit 1
fi

SAMPLE_WAV="$SAMPLE_FILE"
tmp_wav=""
lower="${SAMPLE_FILE,,}"
if [[ "$lower" != *.wav ]]; then
  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[err] input is not .wav and ffmpeg is not installed: '$SAMPLE_FILE'"
    exit 1
  fi
  tmp_wav="/tmp/voxtral_bench_${$}.wav"
  ffmpeg -y -hide_banner -loglevel error -i "$SAMPLE_FILE" -ac 1 -ar 16000 "$tmp_wav"
  SAMPLE_WAV="$tmp_wav"
  trap 'rm -f "$tmp_wav"' EXIT
fi

run_case() {
  local backend="$1"
  echo "== backend: $backend =="
  make "$backend"
  VOX_PRINT_TIMINGS=1 /usr/bin/time -f "elapsed=%E cpu=%P maxrss_kb=%M" -o /tmp/voxtral_${backend}.time \
    ./voxtral -d "$MODEL_DIR" -i "$SAMPLE_WAV" --silent >/tmp/voxtral_${backend}.txt 2>/tmp/voxtral_${backend}.err
  cat /tmp/voxtral_${backend}.time
  rg --no-line-number --no-heading '^(Model load:|Wall transcribe:|Encoder:|Decoder:)' /tmp/voxtral_${backend}.err || true
  # Some benchmark comparisons include model load in "total time". Provide an
  # apples-to-apples derived figure without requiring external math.
  load_ms="$(awk '/^Model load:/ {print $3}' /tmp/voxtral_${backend}.err | head -n1 || true)"
  wall_ms="$(awk '/^Wall transcribe:/ {print $3}' /tmp/voxtral_${backend}.err | head -n1 || true)"
  if [[ -n "${load_ms:-}" && -n "${wall_ms:-}" ]]; then
    echo "Total (load+transcribe): $((load_ms + wall_ms)) ms"
  fi
  echo "output_bytes=$(wc -c </tmp/voxtral_${backend}.txt)"
  echo
}

ran_blas=0
if make blas >/dev/null 2>&1; then
  run_case blas
  ran_blas=1
else
  echo "[warn] BLAS backend build failed (missing OpenBLAS headers/libs?). Skipping BLAS benchmark."
  echo "[hint] On Ubuntu: sudo apt-get install libopenblas-dev"
  echo
fi
run_case cuda

if [[ "$ran_blas" == "1" ]]; then
  echo "Done. Compare /tmp/voxtral_blas.txt and /tmp/voxtral_cuda.txt for transcript diffs."
else
  echo "Done. CUDA output at /tmp/voxtral_cuda.txt"
fi
