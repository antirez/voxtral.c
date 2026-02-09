#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-voxtral-model}"
SAMPLE_FILE="${2:-samples/test_speech.wav}"

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "model dir '$MODEL_DIR' missing"
  exit 1
fi

get_audio_duration_s() {
  local wav="$1"
  if command -v ffprobe >/dev/null 2>&1; then
    # ffprobe is the most accurate across formats/headers.
    ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$wav"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    # Fallback for .wav only.
    python3 - "$wav" <<'PY'
import sys, wave
with wave.open(sys.argv[1], "rb") as w:
    print(w.getnframes() / float(w.getframerate()))
PY
    return 0
  fi
  return 1
}

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

audio_s="$(get_audio_duration_s "$SAMPLE_WAV" || true)"
if [[ -n "${audio_s:-}" ]]; then
  echo "Audio duration: ${audio_s}s"
else
  echo "[warn] could not determine audio duration (missing ffprobe/python3?)"
fi
echo

run_case() {
  local backend="$1"
  local label="${2:-$backend}"
  shift 2 || true
  local env_kv=("$@") # zero or more VAR=VALUE pairs

  local slug="${label//[^a-zA-Z0-9]/_}"
  echo "== backend: $label =="
  make "$backend"
  env VOX_PRINT_TIMINGS=1 "${env_kv[@]}" \
    /usr/bin/time -f "elapsed=%E cpu=%P maxrss_kb=%M" -o "/tmp/voxtral_${slug}.time" \
    ./voxtral -d "$MODEL_DIR" -i "$SAMPLE_WAV" --silent >"/tmp/voxtral_${slug}.txt" 2>"/tmp/voxtral_${slug}.err"
  cat "/tmp/voxtral_${slug}.time"
  if command -v rg >/dev/null 2>&1; then
    rg --no-line-number --no-heading '^(Model load:|Wall transcribe:|Encoder:|Decoder:)' "/tmp/voxtral_${slug}.err" || true
  else
    grep -E '^(Model load:|Wall transcribe:|Encoder:|Decoder:)' "/tmp/voxtral_${slug}.err" || true
  fi
  # Some benchmark comparisons include model load in "total time". Provide an
  # apples-to-apples derived figure without requiring external math.
  load_ms="$(awk '/^Model load:/ {print $3}' "/tmp/voxtral_${slug}.err" | head -n1 || true)"
  wall_ms="$(awk '/^Wall transcribe:/ {print $3}' "/tmp/voxtral_${slug}.err" | head -n1 || true)"
  if [[ -n "${load_ms:-}" && -n "${wall_ms:-}" ]]; then
    total_ms="$((load_ms + wall_ms))"
    echo "Total (load+transcribe): ${total_ms} ms"
    if [[ -n "${audio_s:-}" ]]; then
      xrt_wall="$(awk -v a="$audio_s" -v w="$wall_ms" 'BEGIN{ if(w>0) printf "%.2f", (a*1000.0)/w; else printf "nan"; }')"
      xrt_total="$(awk -v a="$audio_s" -v t="$total_ms" 'BEGIN{ if(t>0) printf "%.2f", (a*1000.0)/t; else printf "nan"; }')"
      echo "xRT (wall): ${xrt_wall}x"
      echo "xRT (total): ${xrt_total}x"
    fi
  fi
  out_bytes="$(wc -c <"/tmp/voxtral_${slug}.txt")"
  echo "output_bytes=${out_bytes}"
  echo
}

ran_blas=0
if [[ "${VOX_BENCH_SKIP_BLAS:-0}" == "1" ]]; then
  echo "[info] VOX_BENCH_SKIP_BLAS=1: skipping BLAS benchmark"
  echo
elif make blas >/dev/null 2>&1; then
  run_case blas blas
  ran_blas=1
else
  echo "[warn] BLAS backend build failed (missing OpenBLAS headers/libs?). Skipping BLAS benchmark."
  echo "[hint] On Ubuntu: sudo apt-get install libopenblas-dev"
  echo
fi
run_case cuda cuda

if [[ "${VOX_BENCH_CUDA_OPTS:-0}" == "1" ]]; then
  echo "== extra CUDA variants (VOX_BENCH_CUDA_OPTS=1) =="
  run_case cuda "cuda+fast" VOX_CUDA_FAST=1
  run_case cuda "cuda+graphs" VOX_CUDA_GRAPHS=1
  run_case cuda "cuda+attn_v3" VOX_CUDA_ATTN_V3=1
  run_case cuda "cuda+graphs+attn_v3" VOX_CUDA_GRAPHS=1 VOX_CUDA_ATTN_V3=1
  run_case cuda "cuda+fast+attn_v5" VOX_CUDA_FAST=1 VOX_CUDA_ATTN_V5=1
  run_case cuda "cuda+merged_weights" VOX_CUDA_MERGE_WEIGHTS=1
  run_case cuda "cuda+graphs+merged_weights" VOX_CUDA_GRAPHS=1 VOX_CUDA_MERGE_WEIGHTS=1
  run_case cuda "cuda+graphs+merged_weights+rope_dev" VOX_CUDA_GRAPHS=1 VOX_CUDA_MERGE_WEIGHTS=1 VOX_CUDA_ROPE_DEV=1
  run_case cuda "cuda+opt_all" VOX_CUDA_GRAPHS=1 VOX_CUDA_ATTN_V3=1 VOX_CUDA_MERGE_WEIGHTS=1 VOX_CUDA_ROPE_DEV=1
fi

if [[ "$ran_blas" == "1" ]]; then
  echo "Done. Compare /tmp/voxtral_blas.txt and /tmp/voxtral_cuda.txt for transcript diffs."
else
  echo "Done. CUDA output at /tmp/voxtral_cuda.txt"
fi
