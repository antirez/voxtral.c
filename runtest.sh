#!/bin/bash
#
# runtest.sh - Voxtral regression test
#
# Runs batch and streaming transcription of a 60-second Italian audio clip
# and checks that key phrases appear in the output. GPU floating-point
# non-determinism means minor punctuation can vary between runs, so we
# check substrings rather than exact match.
# Requires: ffmpeg, voxtral binary, voxtral-model directory.

set -e

# MODEL_DIR="voxtral-model"
MODEL_DIR="voxtral-model-q8"
OGG="samples/antirez_speaking_italian_short.ogg"
VOXTRAL="./voxtral"
TMPWAV="/tmp/voxtral_test_$$.wav"
PASS=0
FAIL=0

# Key phrases that must appear in the transcription. Covers beginning,
# middle, and end of the 60-second clip. If the output is truncated or
# garbled, at least one of these will be missing.
#
# Known near-tied tokens (from --alt 0.5) that can flip between runs:
#   Farmer[,|.]    personalmente[ ma|,]    distribuzione[ così|,]
# All phrases below are chosen to avoid spanning these flip points.
PHRASES=(
    "Forse qualcuno"
    "arance rosse"
    "Mr. Farmer"
    "agricoltore siciliano"
    "non conosco personalmente"
    "consigliarvelo"
    "pubblicizzare i suoi prodotti"
    "alcuno sponsor"
    "grande distribuzione"
    "mercati ortofrutticoli"
    "dove si trova"
)

cleanup() { rm -f "$TMPWAV"; }
trap cleanup EXIT

# Check prerequisites
if [ ! -x "$VOXTRAL" ]; then
    echo "FAIL: $VOXTRAL not found. Run 'make mps' or 'make blas' first."
    exit 1
fi
if [ ! -d "$MODEL_DIR" ]; then
    echo "FAIL: $MODEL_DIR not found. Run ./download_model.sh first."
    exit 1
fi
if ! command -v ffmpeg &>/dev/null; then
    echo "FAIL: ffmpeg not found."
    exit 1
fi
if [ ! -f "$OGG" ]; then
    echo "FAIL: $OGG not found."
    exit 1
fi

# Convert OGG to WAV once
ffmpeg -i "$OGG" -ar 16000 -ac 1 "$TMPWAV" -y 2>/dev/null

check_output() {
    local name="$1"
    local got="$2"
    local ok=1

    for phrase in "${PHRASES[@]}"; do
        if [[ "$got" != *"$phrase"* ]]; then
            echo "  MISSING: \"$phrase\""
            ok=0
        fi
    done

    if [ "$ok" -eq 1 ]; then
        echo "PASS: $name (all ${#PHRASES[@]} phrases found)"
        PASS=$((PASS + 1))
    else
        echo "FAIL: $name"
        FAIL=$((FAIL + 1))
    fi
    echo ""
}

run_test() {
    local name="$1"
    shift
    echo "=== Test: $name ==="
    local got
    got=$("$@" 2>&1 | tee /dev/stderr | grep -v '^Loading\|^Metal\|^Model\|^Audio:\|^Encoder:\|^Decoder:')
    echo ""
    check_output "$name" "$got"
}

# Test 1: Batch mode (whole file at once)
run_test "batch" $VOXTRAL -d "$MODEL_DIR" -i "$TMPWAV"

# Test 2: Streaming mode with small chunks (exercises KV cache compaction)
run_test "streaming -I 0.1" \
    bash -c "ffmpeg -i '$OGG' -f s16le -ar 16000 -ac 1 - 2>/dev/null | $VOXTRAL -d $MODEL_DIR --stdin -I 0.1"

echo "=== Results: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ] || exit 1
