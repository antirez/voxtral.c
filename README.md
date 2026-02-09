# Voxtral Realtime 4B Pure C Implementation

This is a C implementation of the inference pipeline for the [Mistral AI's Voxtral Realtime 4B model](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602). It has zero external dependencies beyond the C standard library. The MPS inference is decently fast, while the BLAS acceleration is usable but slow (it continuously convert the bf16 weights to fp32).

Audio processing uses a chunked encoder with overlapping windows, bounding memory usage regardless of input length. Audio can also be piped from stdin (`--stdin`), or captured live from the microphone (`--from-mic`, macOS), making it easy to transcode and transcribe any format via ffmpeg. A streaming C API (`vox_stream_t`) lets you feed audio incrementally and receive token strings as they become available.

**More testing needed:** please note that this project was mostly tested against few samples, and likely requires some more work to be production quality. However the hard part, to understand the model inference and reproduce the inference pipeline, is here, so the rest likely can be done easily. Testing it against very long transcriptions, able to stress the KV cache circular buffer, will be a useful task.

![demo](samples/demo.gif)

## Motivations (and some rant)

**Thank you to Mistral** for releasing such a great model in an Open Weights fashion. However, the author of this project believes that limiting the inference to a partnership with vLLM, without providing a self-contained reference implementation in Python, limits the model's actual reach and the potential good effects it could have. For this reason, this project was created: it provides both a pure C inference engine and a simple, self-contained Python reference implementation (`python_simple_implementation.py`) that anyone can read and understand without digging through the vLLM codebase.

## Quick Start

```bash
# Build (choose your backend)
make mps       # Apple Silicon (fastest)
# or: make blas    # Intel Mac / Linux with OpenBLAS
# or: make cuda    # NVIDIA CUDA/cuBLAS (Linux/WSL2)

# Download the model (~8.9GB)
./download_model.sh

# Transcribe audio (tokens stream to stdout as generated)
./voxtral -d voxtral-model -i audio.wav

# Live microphone transcription (macOS, Ctrl+C to stop)
./voxtral -d voxtral-model --from-mic

# Pipe any format via ffmpeg
ffmpeg -i audio.mp3 -f s16le -ar 16000 -ac 1 - 2>/dev/null | \
    ./voxtral -d voxtral-model --stdin

# Real-time streaming with low latency
ffmpeg -i audio.mp3 -f s16le -ar 16000 -ac 1 - 2>/dev/null | \
    ./voxtral -d voxtral-model --stdin -I 0.5
```

That's it. No Python runtime, no CUDA toolkit, no `mistral_common` or vLLM required at inference time.

### Python Reference Implementation

A self-contained Python implementation is also provided for reading and understanding the model:

```bash
pip install torch safetensors soundfile soxr
python python_simple_implementation.py voxtral-model audio.wav
```

This requires just PyTorch and a few standard libraries.

## Features

- **Zero dependencies**: Pure C implementation, works standalone for MPS. BLAS required for other targets (OpenBLAS on Linux).
- **Metal GPU acceleration**: Automatic on Apple Silicon Macs with fused GPU operations and batched attention.
- **Streaming output**: Tokens are printed to stdout as they are generated, word by word.
- **Streaming C API**: Feed audio incrementally, get token strings back as they become available.
- **Memory-mapped weights**: BF16 weights are mmap'd directly from safetensors, loading is near-instant.
- **Live microphone input**: `--from-mic` captures and transcribes from the default microphone (macOS) with automatic silence detection.
- **WAV input**: Supports 16-bit PCM WAV files at any sample rate (auto-resampled to 16kHz).
- **Chunked encoder**: Processes audio in overlapping chunks, bounding memory regardless of length.
- **Rolling KV cache**: Decoder KV cache is automatically compacted when it exceeds the sliding window (8192 positions), capping memory usage and allowing unlimited-length audio.

## Usage

### Basic Transcription

```bash
./voxtral -d voxtral-model -i recording.wav
```

Tokens stream to stdout as they are generated. By default, timing info is printed to stderr. Use `--silent` or `--debug` to control verbosity:

```bash
./voxtral -d voxtral-model -i samples/test_speech.wav --silent    # no stderr output
./voxtral -d voxtral-model -i samples/test_speech.wav --debug     # per-layer/per-chunk details
./voxtral -d voxtral-model -i samples/test_speech.wav --alt 0.5   # show alternative tokens
```

### Alternative Tokens

When the model is uncertain between similar-sounding words, `--alt <cutoff>` shows the competing candidates inline:

```
./voxtral -d voxtral-model -i audio.wav --alt 0.95
Hello, this is a test of the[ V| Vo]ox[T|tral]roll speech-to-text system.
```

The cutoff (0.0–1.0) controls how close an alternative must be to the best token. A token qualifies if `1 - prob[i]/prob[0] <= cutoff`. Lower values show only very close alternatives, higher values are more permissive.

### Processing Interval (`-I`)

The `-I <seconds>` flag controls how often the encoder processes accumulated audio. This is the key latency/efficiency tradeoff:

```bash
./voxtral -d voxtral-model --stdin -I 0.5    # low latency (responsive, more GPU overhead)
./voxtral -d voxtral-model --stdin -I 5.0    # high efficiency (batches more audio per encoder call)
```

The default is 2.0 seconds. Lower values make streaming more responsive (text appears sooner after speech) but increase GPU overhead because each encoder call has a fixed startup cost (~50ms). Higher values batch more audio into fewer, larger encoder calls, improving GPU utilization.

The overhead is significant: on a 60-second clip, batch mode takes ~2.9s for the encoder, while `-I 0.1` takes ~15.8s (5.4x slower) because of hundreds of small encoder calls each paying the fixed cost. For **real-time streaming**, values between 1.0 and 2.0 work well. Going below 0.5 wastes most of the GPU time on per-call overhead. For **offline file transcription** the interval is irrelevant since all audio is available at once.

### Reading Audio from Stdin

The **`--stdin` flag** reads audio from standard input instead of a file. The format is auto-detected: if the data starts with a RIFF header it is parsed as WAV, otherwise it is treated as **raw signed 16-bit little-endian, 16 kHz, mono** (`s16le`).

This makes it trivial to transcode any audio/video format on the fly with ffmpeg:

```bash
# Transcribe an MP3 file
ffmpeg -i podcast.mp3 -f s16le -ar 16000 -ac 1 - 2>/dev/null | \
    ./voxtral -d voxtral-model --stdin

# Pipe a WAV directly (auto-detected)
cat recording.wav | ./voxtral -d voxtral-model --stdin
```

### Live Microphone Input

The **`--from-mic` flag** captures audio from the default microphone (macOS only, uses AudioQueue Services). Press Ctrl+C to stop. Silence is automatically detected and stripped to reduce encoder/decoder work when you pause speaking — only actual speech is processed.

```bash
./voxtral -d voxtral-model --from-mic                # default 2s processing interval
./voxtral -d voxtral-model --from-mic -I 1.0          # lower latency
./voxtral -d voxtral-model --from-mic --silent         # no stderr status
```

If the model falls behind real-time, a warning is printed and audio is skipped to catch up.

`--from-mic`, `--stdin`, and `-i` are mutually exclusive.

If you are piping raw PCM and want to reduce overhead (process in larger chunks), you can set:

```bash
VOX_STDIN_INTERVAL_SEC=2 ./voxtral -d voxtral-model --stdin
```

To convert files to WAV format, just use `ffmpeg`:

    ffmpeg -i input.ogg output.wav

The above command line works for many file types, not just for OGG files, of course.
There are two example wave files under the `samples` directory.

### C API

The library exposes a streaming API (`vox_stream_t`) that works for both offline and real-time use. You feed audio samples and retrieve decoded token strings as they become available.

**Offline transcription** — feed all audio, then collect results:

```c
#include "voxtral.h"

vox_ctx_t *ctx = vox_load("voxtral-model");

/* Load audio (your own code, or use vox_load_wav) */
int n_samples;
float *samples = vox_load_wav("audio.wav", &n_samples);

/* Transcribe */
vox_stream_t *s = vox_stream_init(ctx);
vox_stream_feed(s, samples, n_samples);
vox_stream_finish(s);

/* Collect token strings */
const char *tokens[64];
int n;
while ((n = vox_stream_get(s, tokens, 64)) > 0) {
    for (int i = 0; i < n; i++)
        printf("%s", tokens[i]);
}
printf("\n");

vox_stream_free(s);
free(samples);
vox_free(ctx);
```

**Real-time streaming** — feed audio incrementally, retrieve tokens as they arrive:

```c
vox_stream_t *s = vox_stream_init(ctx);

while (have_more_audio()) {
    float chunk[4096];
    int n_read = read_audio(chunk, 4096);
    vox_stream_feed(s, chunk, n_read);

    const char *tokens[16];
    int n;
    while ((n = vox_stream_get(s, tokens, 16)) > 0) {
        for (int i = 0; i < n; i++)
            printf("%s", tokens[i]);
        fflush(stdout);
    }
}

vox_stream_finish(s);
const char *tokens[16];
int n;
while ((n = vox_stream_get(s, tokens, 16)) > 0) {
    for (int i = 0; i < n; i++)
        printf("%s", tokens[i]);
}
printf("\n");

vox_stream_free(s);
```

`feed()` runs the mel spectrogram, encoder, and decoder on available data, queuing output tokens. `finish()` adds padding and processes remaining audio. `get()` retrieves pending tokens — call it after each `feed()` or whenever convenient. Token string pointers returned by `vox_stream_get()` are valid until `vox_stream_free()`.

`vox_stream_flush(s)` forces the encoder to process whatever audio is buffered, regardless of the processing interval, and feeds right-padding so the decoder emits tokens that are behind the delay window. Unlike `finish()`, the stream stays open — you can continue feeding audio afterwards. This is useful for silence detection: when the speaker pauses, flush to get the pending transcription without ending the stream.

Use `vox_set_processing_interval(s, seconds)` to control the latency/efficiency tradeoff (equivalent to `-I` on the CLI). When set, `feed()` accumulates audio but only runs the encoder/decoder after at least the specified duration of new audio has been fed. Lower values give more responsive streaming (text appears sooner), higher values batch more audio per encoder call for better GPU utilization. Default is 2.0 seconds. See the `-I` flag documentation above for guidance on choosing values.

**Alternative tokens** — when the model is uncertain, retrieve competing candidates:

```c
vox_stream_set_alt(s, 3, 0.5);  /* up to 3 alternatives, cutoff 0.5 */

const int n_alt = 3;
const char *tokens[16 * 3];
int n;
while ((n = vox_stream_get_alt(s, tokens, 16, n_alt)) > 0) {
    for (int i = 0; i < n; i++) {
        printf("%s", tokens[i * n_alt]);  /* best token */
        for (int a = 1; a < n_alt && tokens[i * n_alt + a]; a++)
            printf(" [alt: %s]", tokens[i * n_alt + a]);
    }
}
```

`vox_stream_get()` is unaffected — it always returns just the best token.

There is also a one-shot convenience function if you don't need streaming:

```c
char *text = vox_transcribe(ctx, "audio.wav");
printf("%s\n", text);
free(text);
```

## Building

Choose a backend when building:

```bash
make            # Show available backends
make blas       # BLAS acceleration (Accelerate on macOS, OpenBLAS on Linux)
make cuda       # NVIDIA CUDA/cuBLAS acceleration (Linux/WSL2)
make mps        # Apple Silicon Metal GPU (fastest, macOS only)
```

**Recommended:**
- macOS Apple Silicon: `make mps`
- macOS Intel: `make blas`
- Linux with NVIDIA GPU (including WSL2): `make cuda`
- Linux CPU-only / OpenBLAS: `make blas`

For `make blas` on Linux, install OpenBLAS first:
```bash
# Ubuntu/Debian
sudo apt install libopenblas-dev

# Fedora
sudo dnf install openblas-devel
```

Other targets:
```bash
make clean      # Clean build artifacts
make info       # Show available backends for this platform
make inspect    # Build safetensors weight inspector
```

## Model Download

Download model weights (~8.9GB) from HuggingFace:

```bash
./download_model.sh
```

This downloads to `./voxtral-model/` containing:
- `consolidated.safetensors` — all weights, BF16 (~8.9GB)
- `tekken.json` — Tekken tokenizer vocabulary (~15MB)
- `params.json` — model configuration

The model is [Apache-2.0 licensed](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602).

## How Fast Is It?

Benchmarks on **Apple M3 Max** (40-core GPU, 128GB RAM, 400 GB/s bandwidth):

| Backend | Encoder (3.6s audio) | Prefill | Decoder |
|---------|---------------------|---------|---------|
| MPS | 284 ms | 252 ms | 23.5 ms/step (short) |
| BLAS | ~8s | ~1.2s | 335 ms/step |

The MPS backend runs the entire decoder in a single Metal command buffer per token, with custom GPU kernels for attention, RoPE, and KV cache management. All weights are pre-converted to f16 on GPU at load time. The BLAS backend uses Accelerate's multi-threaded sgemm with on-the-fly BF16→F32 conversion.

Decoder speed depends on sequence length: attention scans the full KV cache each step, so longer transcriptions are slower per token. For a 60-second clip (~760 steps), the average is ~31.6 ms/step. For short clips (~15 steps) it's ~23.5 ms/step. Either way, the decoder generates one token per ~80ms of audio, so even at 31.6 ms/step transcription runs ~2.5x faster than real-time.

Longer audio scales linearly with the encoder (O(n) with sliding window attention) and the decoder (one token per 80ms of audio).

## Model Architecture

Voxtral Realtime 4B is a streaming speech-to-text model with ~4B parameters:

**Pipeline:**
```
WAV → 16kHz → Mel Spectrogram → Conv Stem → Encoder → Downsample 4x → Adapter → Decoder → Tokens
```

| Component | Architecture |
|-----------|-------------|
| Audio Encoder | 32-layer causal transformer, 1280 dim, 32 heads, sliding window 750 |
| Adapter | Linear(5120→3072) → GELU → Linear(3072→3072) |
| LLM Decoder | 26-layer transformer (Ministral-3 based), 3072 dim, GQA (32 heads / 8 KV) |

| Parameter | Value |
|-----------|-------|
| Total parameters | ~4B (0.6B encoder + 3.4B decoder) |
| Weight format | BF16 |
| Vocab size | 131,072 (Tekken tokenizer) |
| Audio frame rate | 12.5 Hz (1 token = 80ms) |
| Max audio length | Unlimited (rolling KV cache) |
| Supported languages | EN, ES, FR, PT, HI, DE, NL, IT, AR, RU, ZH, JA, KO |

## Memory Requirements

| Component | Size |
|-----------|------|
| Model weights (mmap'd) | 8.9 GB on disk, mapped on-demand |
| MPS GPU weight cache | ~8.4 GB (BF16→F16 cached on GPU) |
| KV cache (decoder) | ~1.8 GB max (rolling, capped at sliding window) |
| Working buffers | ~200 MB |

## License

MIT

## CUDA on Windows 11 + WSL2 (Ubuntu)

This repository now includes a CUDA backend that uses **cuBLAS** for large matrix multiplies (`make cuda`).
The CUDA backend uses the CUDA **driver API** (`libcuda`) and does not require linking against `libcudart`.

### Prerequisites

1. Install the latest NVIDIA Windows driver with WSL CUDA support.
2. Install WSL2 + Ubuntu 22.04+.
3. Inside Ubuntu, install the CUDA toolkit (includes `nvcc`, `cublas`, and runtime libraries).

### Build

```bash
make cuda
# or if CUDA toolkit is installed elsewhere:
make cuda CUDA_HOME=/usr/local/cuda
```

The build now performs a preflight check for `cuda.h` + `cublas_v2.h` under `${CUDA_HOME}/include`.

Optional (Linux): enable OpenMP for faster CPU-side kernels:

```bash
make cuda OPENMP=1
```

### Verify CUDA visibility in WSL2

```bash
nvidia-smi
```

If your RTX 3080 Ti is visible there, `make cuda` should work and Voxtral will offload large GEMMs to the GPU.


### Recommended WSL2 versions (known-good baseline)

- Windows 11 with recent NVIDIA Game Ready or Studio driver that includes WSL CUDA support.
- Ubuntu 22.04+ in WSL2.
- CUDA toolkit in Ubuntu (`cuda.h`, `libcublas`).

### Troubleshooting

- `cuda.h not found`: install CUDA toolkit in Ubuntu and/or set `CUDA_HOME`.
- `error while loading shared libraries: libcublas.so`: ensure `${CUDA_HOME}/lib64` is in linker path (or build with correct `CUDA_HOME`).
- `nvidia-smi` fails in WSL2: update Windows NVIDIA driver and verify WSL GPU support is enabled.
- OOM during long runs: reduce concurrent workloads and close GPU-intensive apps on host Windows.

### Validation and benchmark helpers

```bash
# Build + optional smoke test
./scripts/validate_cuda.sh voxtral-model samples/test_speech.wav

# Compare BLAS vs CUDA timing + output files
./scripts/benchmark_backends.sh voxtral-model samples/test_speech.wav
```

Fast CUDA config (best-effort, can be overridden by per-feature env vars):

```bash
VOX_CUDA_FAST=1 ./voxtral -d voxtral-model -i samples/test_speech.wav
```

Notes:
- `VOX_CUDA_FAST=1` enables a fused top1-only logits path by default when alternatives are disabled (`--alt` not used). Disable it with `VOX_DISABLE_CUDA_LOGITS_FUSED=1` if you want to benchmark the baseline logits+argmax path.
- `VOX_CUDA_FAST=1` also enables the chunked attention v4 path by default (fused KV append into the v3 partial kernel, best-effort). Disable it with `VOX_DISABLE_CUDA_ATTN_V4=1`.
- `VOX_CUDA_ATTN_V5=1` enables an experimental decoder attention variant that skips inactive chunks (best-effort; default off until validated broadly).
- `VOX_CUDA_FAST=1` also enables cuBLASLt autotune for the `M=1` decoder GEMMs (best-effort). Disable it with `VOX_DISABLE_CUBLASLT_AUTOTUNE=1`.
- `VOX_CUDA_FAST=1` also enables a cuBLASLt “transpose-B view” for `M=1` decoder GEMMs (best-effort). Disable it with `VOX_DISABLE_CUBLASLT_TRANSPOSE_B=1` (or force it with `VOX_CUDA_CUBLASLT_TRANSPOSE_B=0/1`).
- `VOX_CUDA_LT_COMPUTE=32F_FAST_16BF` (or similar) opts into alternate cuBLASLt compute modes for BF16 GEMMs (default: `32F`). This may change outputs slightly; use `./scripts/accuracy_regression.sh` to validate.

To run the extra CUDA benchmark variants (graphs/v3/merged/etc):

```bash
VOX_BENCH_CUDA_OPTS=1 ./scripts/benchmark_backends.sh voxtral-model samples/test_speech.wav
```


### Driver / Toolkit matrix (known-good starting point)

| Layer | Recommended |
|---|---|
| Windows host | Windows 11 23H2+ |
| NVIDIA driver | R550+ with WSL CUDA support |
| WSL | WSL2 (`wsl --update`) |
| Ubuntu guest | 22.04 LTS or 24.04 LTS |
| CUDA toolkit in Ubuntu | 12.4+ |
| GPU class tested target | RTX 3080 Ti |

### Real-time microphone pipeline recipe (WSL2)

```bash
# Windows host: capture mic with ffmpeg (or equivalent) and pipe raw PCM into WSL voxtral
# Example run *inside* WSL when mic source is exposed by ffmpeg:
ffmpeg -f pulse -i default -f s16le -ar 16000 -ac 1 - 2>/dev/null |   ./voxtral -d voxtral-model --stdin
```

If `pulse` is unavailable in your WSL distribution, use a host-side capture path and stream PCM into WSL over stdin or TCP.

### Accuracy regression helper

```bash
# Compares BLAS vs CUDA transcripts with token mismatch tolerance (default: 0.5%)
./scripts/accuracy_regression.sh voxtral-model samples/test_speech.wav 0.005
```

### CUDA / WSL2 Notes

For detailed bringup, benchmark results, and profiling notes, see `PR_NOTES_CUDA_WSL2.md`.
