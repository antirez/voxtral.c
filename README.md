# Voxtral Realtime 4B Pure C Implementation

This is a C implementation of the inference pipeline for the [Mistral AI's Voxtral Realtime 4B model](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602). It has zero external dependencies beyond the C standard library. The MPS inference is decently fast, while the BLAS acceleration is usable but slow (it continuously convert the bf16 weights to fp32).

Audio processing uses a chunked encoder with overlapping windows, bounding memory usage regardless of input length. Audio can also be piped from stdin (`--stdin`), making it easy to transcode and transcribe any format via ffmpeg. A streaming C API (`vox_stream_t`) lets you feed audio incrementally and receive token strings as they become available.

**More testing needed:** please note that this project was mostly tested against few samples, and likely requires some more work to be production quality. However the hard part, to understand the model inference and reproduce the inference pipeline, is here, so the rest likely can be done easily. Testing it against very long transcriptions, able to stress the KV cache circular buffer, will be a useful task.

![demo](samples/demo.gif)

## Motivations (and some rant)

**Thank you to Mistral** for releasing such a great model in an Open Weights fashion. However, the author of this project believes that limiting the inference to a partnership with vLLM, without providing a self-contained reference implementation in Python, limits the model's actual reach and the potential good effects it could have. For this reason, this project was created: it provides both a pure C inference engine and a simple, self-contained Python reference implementation (`python_simple_implementation.py`) that anyone can read and understand without digging through the vLLM codebase.

## Quick Start

```bash
# Build (choose your backend)
make mps       # Apple Silicon (fastest)
# or: make blas    # Intel Mac / Linux with OpenBLAS

# Download the model (~8.9GB)
./download_model.sh

# Transcribe audio (tokens stream to stdout as generated)
./voxtral -d voxtral-model -i audio.wav

# Pipe any format via ffmpeg
ffmpeg -i audio.mp3 -f s16le -ar 16000 -ac 1 - 2>/dev/null | \
    ./voxtral -d voxtral-model --stdin

# Real-time streaming with low latency
ffmpeg -i audio.mp3 -f s16le -ar 16000 -ac 1 - 2>/dev/null | \
    ./voxtral -d voxtral-model --stdin -I 0.5

# Live microphone transcription on macOS
./voxtral -d voxtral-model --mic
```

That's it. No Python runtime, no CUDA toolkit, no `mistral_common` or vLLM required at inference time.

## Code Organization

- `main.c`: CLI argument parsing and streaming transcription loop.
- `voxtral_audio.c/.h`: WAV parsing/resampling and mel spectrogram (offline + incremental).
- `voxtral_mic.c/.h`: microphone capture on macOS (`--mic`), device reporting, stop handling.
- `voxtral.c/.h`: model loading, encoder/decoder pipeline, and streaming API (`vox_stream_t`).

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
./voxtral -d voxtral-model --mic --mic-secs 10                    # mic capture for 10 seconds (macOS)
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

`--stdin` and `-i` are mutually exclusive.

### Live Microphone Input (macOS)

Use `--mic` to capture from the default system microphone and transcribe live:

```bash
./voxtral -d voxtral-model --mic
```

By default this runs until you press `Ctrl+C`. You can stop automatically after a fixed duration:

```bash
./voxtral -d voxtral-model --mic --mic-secs 15
```

`--mic` is mutually exclusive with `-i` and `--stdin`.
At start, the CLI prints the selected input device. If no audio is captured, it prints a permission/routing hint.
On non-macOS platforms, `--mic` returns an explicit error; file/stdin transcription works unchanged.

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
make mps        # Apple Silicon Metal GPU (fastest, macOS only)
```

**Recommended:**
- macOS Apple Silicon: `make mps`
- macOS Intel: `make blas`
- Linux with OpenBLAS: `make blas`

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
