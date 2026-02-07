# CUDA (WSL2) Notes, Findings, and Benchmarks

This PR adds a production-oriented CUDA backend for Voxtral that works reliably under Windows 11 + WSL2 (Ubuntu) on an NVIDIA RTX 3080 Ti, and it pushes the two main hot paths fully onto the GPU:

- Encoder + adapter (GPU resident, BF16 weights + cuBLAS GEMMs + CUDA elementwise kernels)
- Decoder single-token generation (GPU resident, device KV cache + cuBLAS GEMMs + CUDA attention + GPU argmax)

The CUDA runtime uses the CUDA Driver API (`libcuda`) and embeds a CUBIN for custom kernels to avoid PTX JIT issues under WSL2.

## What Changed (High Level)

- CUDA build target: `make cuda` with `CUDA_HOME` override and preflight checks.
- CUDA runtime init uses:
  - `cuInit`, primary context, non-blocking stream
  - cuBLAS + (optional) cuBLASLt for small `M=1` GEMMs
- Custom CUDA kernels:
  - Built via `nvcc -cubin` and embedded as a C header (no PTX JIT at runtime).
  - Implements RMSNorm, RoPE, BF16/FP16 casts, SwiGLU/GELU, downsample concat, argmax, etc.
- BF16 weight caching on device:
  - Host BF16 pointers (mmap-backed) are used as stable cache keys.
  - Device cache is LRU-ish and sized conservatively based on free VRAM.
- Encoder full path:
  - CPU conv stem remains on CPU (small).
  - Transformer layers + adapter run on GPU; intermediates stay on device.
- Decoder full path:
  - Device-side KV cache (FP16 by default) and device-only intermediates.
  - Faster per-token attention kernel (online softmax, warp-synchronous).
  - Optional logits copy: if `logits==NULL`, logits stay on GPU and only the best token id is copied back.

## Build

### CUDA

```bash
make cuda
```

Notes:
- Requires CUDA toolkit headers + `nvcc` (used only to compile the embedded CUBIN).
- Links against `-lcublasLt -lcublas -lcuda` (Driver API; no `-lcudart` dependency).

### BLAS (Baseline)

```bash
sudo apt-get install -y libopenblas-dev
make blas
```

## Validation

```bash
./download_model.sh

make cuda
./scripts/validate_cuda.sh voxtral-model samples/test_speech.wav

./scripts/accuracy_regression.sh voxtral-model samples/test_speech.wav 0
./scripts/benchmark_backends.sh voxtral-model samples/test_speech.wav
```

## Benchmarks (WSL2 RTX 3080 Ti)

All runs below are from the CLI. Stage timings are printed with `VOX_PRINT_TIMINGS=1`:
- `Model load:` is safetensors mmap + init.
- `Encoder:` is the cumulative encoder+adapter time.
- `Decoder:` is the cumulative decoder time.
- `Wall transcribe:` is total transcription wall time (excluding `Model load:`).
- `Total (load+transcribe):` is a derived sum printed by `scripts/benchmark_backends.sh` for comparisons that include model load in the end-to-end time.

Audio durations:
- `samples/test_speech.wav`: `3.641750s` (ffprobe)
- `samples/I_have_a_dream.ogg`: `180.021438s` after conversion to WAV (ffprobe)

### `samples/test_speech.wav`

BLAS (`./scripts/benchmark_backends.sh voxtral-model samples/test_speech.wav`):
- Model load: `75 ms`
- Wall transcribe: `40918 ms`
- Total (load+transcribe): `40993 ms`
- Encoder: `760 mel -> 95 tokens (13864 ms)`
- Decoder: `17 text tokens (57 steps) in 27046 ms (prefill 7772 ms + 344.2 ms/step)`

CUDA (`./scripts/benchmark_backends.sh voxtral-model samples/test_speech.wav`):
- Model load: `31 ms`
- Wall transcribe: `3018 ms`
- Total (load+transcribe): `3049 ms`
- Encoder: `760 mel -> 95 tokens (646 ms)`
- Decoder: `17 text tokens (57 steps) in 2159 ms (prefill 1435 ms + 12.9 ms/step)`

### `samples/I_have_a_dream.ogg` (180s)

Convert once:

```bash
ffmpeg -y -hide_banner -loglevel error -i samples/I_have_a_dream.ogg -ac 1 -ar 16000 /tmp/I_have_a_dream.wav
```

BLAS:
- Model load: `68 ms`
- Wall transcribe: `1468788 ms` (24:29)
- Total (load+transcribe): `1468856 ms`
- Encoder: `18400 mel -> 2300 tokens (541742 ms)` (9:02)
- Decoder: `311 text tokens (2262 steps) in 926821 ms (prefill 7398 ms + 406.6 ms/step)` (15:27)

CUDA:
- Model load: `39 ms`
- Wall transcribe: `81686 ms` (1:22)
- Total (load+transcribe): `81725 ms`
- Encoder: `18400 mel -> 2299 tokens (2588 ms)`
- Decoder: `310 text tokens (2261 steps) in 78625 ms (prefill 1466 ms + 34.1 ms/step)` (1:19)

BF16 cache stats at exit (same run):
- `uploaded=8.23 GiB`, `misses=409`, `hits=415,796`

## Profiling Notes

Nsight Systems (`nsys`) on a short run shows heavy use of tensor-core BF16 GEMM kernels (cutlass/ampere BF16 paths), and confirms:
- Decoder attention is a major knob for long sequences (seq grows to ~2300 on the 180s sample).
- Avoiding large host copies (logits, intermediates) is important for throughput.

## Debug / Escape Hatches

- Disable full CUDA encoder+adapter:
  - `VOX_DISABLE_CUDA_ENCODER_FULL=1`
- Disable full CUDA decoder path:
  - `VOX_DISABLE_CUDA_DECODER_FULL=1`
- Use the optional direct windowed attention kernel for encoder attention (currently slower; opt-in):
  - `VOX_CUDA_DIRECT_ATTN=1`
- Disable cuBLASLt (force cuBLAS GEMMEx):
  - `VOX_DISABLE_CUBLASLT=1`
- Disable FP16 KV cache (use FP32 KV cache):
  - `VOX_CUDA_KV_FP16=0`
