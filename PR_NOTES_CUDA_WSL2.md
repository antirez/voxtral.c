# CUDA (WSL2) Notes, Findings, and Benchmarks

This PR adds a production-oriented CUDA backend for Voxtral that works reliably under Windows 11 + WSL2 (Ubuntu) on an NVIDIA RTX 3080 Ti, and it pushes the two main hot paths fully onto the GPU:

- Encoder + adapter (GPU resident, BF16 weights + cuBLAS GEMMs + CUDA elementwise kernels)
- Decoder single-token generation (GPU resident, device KV cache + cuBLAS GEMMs + CUDA attention + GPU argmax)
- Decoder prefill (prompt prefill, seq_len > 1): runs fully on GPU when possible

The CUDA runtime uses the CUDA Driver API (`libcuda`) and embeds a CUBIN for custom kernels to avoid PTX JIT issues under WSL2.

## What Changed (High Level)

- CUDA build target: `make cuda` with `CUDA_HOME` override and preflight checks.
- CUDA runtime init uses:
  - `cuInit`, primary context, non-blocking stream
  - cuBLAS + (optional) cuBLASLt for small `M=1` GEMMs
  - Optional cuBLASLt autotune for `M=1` decoder GEMMs (enabled by `VOX_CUDA_FAST=1`; disable with `VOX_DISABLE_CUBLASLT_AUTOTUNE=1`)
  - Optional cuBLASLt transpose-B view for `M=1` decoder GEMMs (enabled by `VOX_CUDA_FAST=1`; disable with `VOX_DISABLE_CUBLASLT_TRANSPOSE_B=1`)
- Custom CUDA kernels:
  - Built via `nvcc -cubin` and embedded as a C header (no PTX JIT at runtime).
  - Implements RMSNorm, RoPE, BF16/FP16 casts, SwiGLU/GELU, downsample concat, argmax, etc.
- BF16 weight caching on device:
  - Host BF16 pointers (mmap-backed) are used as stable cache keys.
  - Device cache is LRU-ish and sized conservatively based on free VRAM.
- Cold-start improvements (weight upload / allocator overhead):
  - Async device allocator + mempool (`cuMemAllocAsync`/`cuMemFreeAsync`): enabled by default when supported (disable with `VOX_DISABLE_CUDA_MEMPOOL=1` or `VOX_CUDA_MEMPOOL=0`).
  - Optional host page registration for hot weight ranges: `VOX_CUDA_HOSTREG_GIB=<GiB>` (0 disables; best-effort).
  - Optional prefetch at model load: `VOX_CUDA_PREFETCH=1` (shifts weight uploads out of the first transcription call).
- Encoder full path:
  - Transformer layers + adapter run on GPU; intermediates stay on device.
- Decoder full path:
  - Device-side KV cache (FP16 by default) and device-only intermediates.
  - Faster per-token attention kernel (online softmax, warp-synchronous).
  - Optional attention v2 kernel variant (vectorized loads/stores; opt-in).
  - Optional attention v3 kernel variant (chunked reduction + GQA shared-load; opt-in).
  - Optional merged decoder projections (QKV and FFN W1+W3) to reduce GEMM launches (opt-in).
  - Optional CUDA Graph capture for the single-token decoder step (reduces CPU launch overhead; opt-in).
  - Optional device RoPE freqs generation for CUDA Graph mode (reduces CPU overhead per step; opt-in).
  - Optional logits copy: if `logits==NULL`, logits stay on GPU and only the best token id is copied back.
  - Optional fused top1-only logits projection (enabled by `VOX_CUDA_FAST=1`): avoids materializing the full-vocab logits buffer when only the best token id is needed.
  - Prefill is attempted on GPU (seq_len > 1) and falls back to the CPU prefill implementation if unsupported/disabled.

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
- `prefill` (in the `Decoder:` line) includes prompt prefill plus the first generated token step (the timing block wraps both).

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
- Wall transcribe: `3045 ms`
- Total (load+transcribe): `3076 ms`
- Encoder: `760 mel -> 95 tokens (683 ms)`
- Decoder: `17 text tokens (57 steps) in 2146 ms (prefill 1396 ms + 13.4 ms/step)`

### CUDA Graphs (opt-in)

Enable with:

```bash
VOX_CUDA_GRAPHS=1
```

On `samples/antirez_speaking_italian_short.ogg` (converted to WAV; ~60s), CUDA Graphs reduce CPU launch overhead. On this setup they also auto-select attention v3 for graph capture when available (unless disabled):

- Without graphs: `Wall transcribe 16916 ms`, decoder `18.7 ms/step`
- With graphs: `Wall transcribe 12696 ms`, decoder `13.2 ms/step`

### Attention v2 (opt-in)

Enable with:

```bash
VOX_CUDA_ATTN_V2=1
```

Attention v2 is experimental and can regress depending on GPU/driver/toolkit. On this setup (RTX 3080 Ti + WSL2) it was significantly slower, so keep it disabled unless you benchmark it on your hardware:

- Without graphs: `Wall transcribe 107635 ms`, decoder `74.8 ms/step`
- With graphs: `Wall transcribe 102259 ms`, decoder `58.2 ms/step`

### Attention v3 (opt-in)

Enable with:

```bash
VOX_CUDA_ATTN_V3=1
```

Notes:
- v3 is currently implemented for FP16 KV cache only (`VOX_CUDA_KV_FP16=1`, which is the default).
- v3 uses a 2-stage chunked reduction and reduces redundant KV loads under GQA by having one block compute 4 query heads that share the same KV head.
- When CUDA Graphs are enabled (`VOX_CUDA_GRAPHS=1`), v3 is auto-selected for the graph capture path if available (unless disabled via `VOX_DISABLE_CUDA_ATTN_V3=1`).

On `samples/antirez_speaking_italian_short.ogg` (converted to WAV; 60s), v3 is a win for the non-graph decoder path (numbers from `VOX_PRINT_TIMINGS=1`):

- Without v3: `Wall transcribe 16916 ms`, decoder `18.7 ms/step`
- With v3: `Wall transcribe 14364 ms`, decoder `15.4 ms/step`

When CUDA Graphs are enabled, v3 is auto-selected for the graph capture path if available (unless disabled via `VOX_DISABLE_CUDA_ATTN_V3=1`).

### Attention v5 (opt-in)

Enable with:

```bash
VOX_CUDA_ATTN_V5=1
```

Notes:
- v5 is currently implemented for FP16 KV cache only (`VOX_CUDA_KV_FP16=1`, which is the default).
- v5 keeps the same kernel grid shape as v4 (graph-capture safe), but reduces wasted work for shorter sequences by:
  - skipping inactive chunks in the partial kernel (no zero-filling)
  - iterating only the active chunks in the reduce kernel (instead of all `VOX_DEC_WINDOW`-derived chunks)

On `/tmp/vox_iad.wav` (~180s WAV) with `VOX_CUDA_FAST=1`:

- Baseline (v4): `Wall transcribe 33470 ms`, decoder `13.1 ms/step`
- With v5: `Wall transcribe 33037 ms`, decoder `12.9 ms/step`

### Merged Decoder Projections (opt-in)

Enable with:

```bash
VOX_CUDA_MERGE_WEIGHTS=1
```

Notes:
- `VOX_CUDA_MERGE_WEIGHTS=1` enables both:
  - merged QKV projection (one GEMM per layer instead of 3)
  - merged FFN W1+W3 projection (one GEMM per layer instead of 2)
- You can also enable them individually:
  - `VOX_CUDA_MERGE_QKV=1`
  - `VOX_CUDA_MERGE_FFN13=1`

On `samples/antirez_speaking_italian_short.ogg` (~60s), combined with CUDA Graphs, merged projections reduced per-step decoder time further (numbers from `VOX_PRINT_TIMINGS=1`):

- Graphs (no merged weights): decoder ~`13.2 ms/step`
- Graphs + merged weights: decoder ~`12.7 ms/step`

### Device RoPE For CUDA Graphs (opt-in)

Enable with:

```bash
VOX_CUDA_ROPE_DEV=1
```

When enabled (and if the optional kernel is available), CUDA Graph mode generates the RoPE freqs on-device inside the captured graph:
- Upload `logical_pos` (4 bytes) instead of computing trig on CPU and uploading the RoPE freqs (~512 bytes) per step.
Note: this is primarily about reducing host-side work; end-to-end speed impact can be neutral depending on GPU/driver.

### GPU Conv Stem (opt-in)

Enable with:

```bash
VOX_CUDA_CONV_STEM=1
```

This runs the encoder conv stem (conv0/conv1 + GELU) on GPU via custom CUDA kernels + cuBLAS SGEMM (no cuDNN). It mainly reduces CPU-side `im2col` overhead in the encoder front-end.

### Full CUDA Streaming Pipeline (opt-in)

Enable with:

```bash
VOX_CUDA_PIPELINE_FULL=1
```

This keeps streaming adapter embeddings on-device and lets CUDA build the per-step decoder input embedding directly from the device-side adapter buffer:
- Avoids a large adapter `DtoH` copy for every encoder chunk (streaming mode).
- Avoids uploading a new step embedding (`HtoD`) every generated token.

Notes:
- Experimental and currently **not thread-safe** (uses a global device-side adapter buffer).
- If it fails mid-run, we currently fail-fast rather than attempting a CPU fallback.
- Prompt prefill still copies only the first prompt window from device to host to reuse the existing prefill path.
- In pipeline mode, GPU conv stem is attempted by default unless disabled (`VOX_DISABLE_CUDA_CONV_STEM=1`).

Related env vars:
- `VOX_DISABLE_CUDA_PIPELINE_FULL=1` disables the pipeline.
- `VOX_CUDA_ADAPTER_CAP_TOKENS=<int>` sets the initial adapter buffer capacity (default: 8192).

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

- Enable CUDA Graphs for the decoder single-token step (opt-in):
  - `VOX_CUDA_GRAPHS=1`
- Disable CUDA Graphs (force non-graph path):
  - `VOX_DISABLE_CUDA_GRAPHS=1`
- Enable merged decoder weights (reduces GEMM launches; opt-in):
  - `VOX_CUDA_MERGE_WEIGHTS=1`
  - `VOX_CUDA_MERGE_QKV=1`
  - `VOX_CUDA_MERGE_FFN13=1`
- Enable device RoPE freqs generation for CUDA Graph mode (opt-in):
  - `VOX_CUDA_ROPE_DEV=1`
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
- Disable CUDA decoder prefill fast path (force CPU prefill):
  - `VOX_DISABLE_CUDA_PREFILL=1`
- Disable RMSNorm->BF16 fused kernel (debug fallback):
  - `VOX_DISABLE_CUDA_RMSNORM_BF16_FUSED=1`
- Enable attention v2 kernel variant (opt-in):
  - `VOX_CUDA_ATTN_V2=1`
- Disable attention v2 kernel variant (force v1):
  - `VOX_DISABLE_CUDA_ATTN_V2=1`
- Enable attention v3 kernel variant (opt-in):
  - `VOX_CUDA_ATTN_V3=1`
- Disable attention v3 kernel variant (force v1/v2):
  - `VOX_DISABLE_CUDA_ATTN_V3=1`
- Enable GPU conv stem (opt-in):
  - `VOX_CUDA_CONV_STEM=1`
- Disable GPU conv stem (force CPU conv stem):
  - `VOX_DISABLE_CUDA_CONV_STEM=1`
