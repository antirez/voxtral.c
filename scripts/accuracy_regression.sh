#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-voxtral-model}"
SAMPLE_FILE="${2:-samples/test_speech.wav}"
TOLERANCE_RATIO="${3:-0.005}"   # 0.5% token mismatch tolerance

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "model dir '$MODEL_DIR' missing"
  exit 1
fi

normalize() {
  tr '[:upper:]' '[:lower:]' | tr -s '[:space:]' ' '
}

echo "[1/4] build BLAS"
if make blas >/dev/null 2>&1; then
  have_blas=1
else
  have_blas=0
  echo "[warn] BLAS backend build failed (missing OpenBLAS headers/libs?). Falling back to expected-text regression."
  echo "[hint] On Ubuntu: sudo apt-get install libopenblas-dev"
fi

if [[ "$have_blas" == "1" ]]; then
  echo "[2/4] run BLAS transcript"
  ./voxtral -d "$MODEL_DIR" -i "$SAMPLE_FILE" --silent | normalize > /tmp/voxtral_ref_blas.txt
else
  echo "[2/4] skip BLAS transcript"
fi

echo "[3/4] build CUDA"
make cuda >/dev/null

echo "[4/4] run CUDA transcript"
./voxtral -d "$MODEL_DIR" -i "$SAMPLE_FILE" --silent | normalize > /tmp/voxtral_ref_cuda.txt

python3 - <<PY
from pathlib import Path
import difflib

have_blas = int("${have_blas}")
cuda_text = Path('/tmp/voxtral_ref_cuda.txt').read_text().strip()
cuda_tokens = cuda_text.split()

if have_blas:
    ref = Path('/tmp/voxtral_ref_blas.txt').read_text().split()
    sm = difflib.SequenceMatcher(a=ref, b=cuda_tokens)
    ratio = 1.0 - sm.ratio()
    print(f"token_mismatch_ratio={ratio:.6f}")
    print(f"ref_tokens={len(ref)} cuda_tokens={len(cuda_tokens)}")
    tol = float('${TOLERANCE_RATIO}')
    if ratio > tol:
        raise SystemExit(f"FAIL: mismatch ratio {ratio:.6f} exceeds tolerance {tol:.6f}")
    print("PASS")
else:
    # Fallback smoke check: assert transcript contains a few anchor words.
    anchors = ["hello", "test", "speech-to-text", "system"]
    missing = [a for a in anchors if a not in cuda_text]
    print(f"cuda_tokens={len(cuda_tokens)}")
    if missing:
        raise SystemExit(f"FAIL: missing anchor words: {missing}")
    print("PASS (smoke)")
PY
