# Voxtral Realtime 4B — Model Reference

Model: `mistralai/Voxtral-Mini-4B-Realtime-2602`

This document describes the model architecture, weight format, tokenizer layout,
and inference algorithm needed to implement Voxtral Realtime 4B from scratch.
The Python reference implementation (`python_simple_implementation.py`) is the
executable version of this document.

---

## Architecture Overview

Voxtral Realtime is a streaming speech-to-text model with ~4B parameters:
- **Audio Encoder**: ~0.6B params (causal transformer, 32 layers)
- **LLM Decoder**: ~3.4B params (Ministral-3 based, 26 layers)

**Pipeline:**
```
WAV → 16kHz → Mel Spectrogram → Conv Stem → Encoder → Downsample 4x → Adapter → Decoder → Tokens
```

### Audio Preprocessing
| Parameter | Value |
|-----------|-------|
| Sample rate | 16000 Hz |
| Frame rate | 12.5 Hz |
| Mel bins | 128 |
| Hop length | 160 samples (10ms) |
| Window size | 400 samples (25ms) |
| global_log_mel_max | 1.5 |
| 1 text token | 80ms of audio |

**Exact mel computation** (must match for correctness):
1. Window: `hann(window_size=400)`
2. STFT: `torch.stft(audio, n_fft=400, hop_length=160, window=window, return_complex=True)`
3. Power: `magnitudes = stft[..., :-1].abs() ** 2` (drops last bin)
4. Mel filter bank: Slaney-style, 128 bins, 0-8000 Hz
5. `mel_spec = mel_filters.T @ magnitudes`
6. `log_spec = log10(clamp(mel_spec, min=1e-10))`
7. Clamp: `log_spec = max(log_spec, global_log_mel_max - 8.0)`
8. Scale: `log_spec = (log_spec + 4.0) / 4.0`

### Audio Encoder (Causal Transformer)
| Parameter | Value |
|-----------|-------|
| dim | 1280 |
| n_layers | 32 |
| n_heads | 32 (MHA) |
| head_dim | 64 |
| hidden_dim (MLP) | 5120 |
| FFN type | SwiGLU |
| Norm | RMSNorm (eps=1e-5) |
| Position | RoPE (theta=1e6, **interleaved**) |
| Attention | causal, sliding window=750 |
| Biases | YES (wq, wv, wo, w2 only; wk has NO bias) |

Conv stem before the transformer:
- `conv1d(128→1280, kernel=3, stride=1, causal)` → GELU
- `conv1d(1280→1280, kernel=3, stride=2, causal)` → GELU

### Audio-Language Adapter
```
Input: [seq/4, 5120] (encoder output reshaped via 4x downsample)
       ↓
Linear(5120 → 3072, no bias)
       ↓
GELU
       ↓
Linear(3072 → 3072, no bias)
       ↓
Output: [seq/4, 3072]
```

### LLM Decoder
| Parameter | Value |
|-----------|-------|
| dim | 3072 |
| n_layers | 26 |
| n_heads | 32 |
| n_kv_heads | 8 (GQA 4:1) |
| head_dim | 128 |
| hidden_dim (MLP) | 9216 |
| Norm | RMSNorm (eps=1e-5) |
| Position | RoPE (theta=1e6) |
| Attention | causal, sliding window=8192 |
| Biases | NO |
| Vocab size | 131,072 |
| Tied embeddings | yes (tok_embeddings used as LM head) |

### Adaptive RMS Norm (ada_rms_norm_t_cond)

The decoder uses adaptive normalization conditioned on transcription delay:
```python
# Applied AFTER ffn_norm, BEFORE FFN
ada_scale = ada_up(gelu(ada_down(t_cond)))  # [3072]
h_norm = h_norm * (1 + ada_scale)
```
- `ada_down`: Linear(3072 → 32, no bias)
- `ada_up`: Linear(32 → 3072, no bias)
- `t_cond`: TimeEmbedding(n_delay_tokens) where n_delay_tokens=6 for 480ms delay

**Time embedding:**
```python
inv_freq = exp(-log(10000) * arange(dim/2) / (dim/2))
emb = t * inv_freq  # t = float(n_delay_tokens)
t_cond = cat([cos(emb), sin(emb)])  # [3072]
```

---

## Tokenizer (Tekken)

### Special token IDs
```
BOS = 1
EOS = 2
[STREAMING_PAD] = 32
[AUDIO] = 24          (not used in streaming transcription)
[BEGIN_AUDIO] = 25    (not used in streaming transcription)
```

### Token ID layout

`tekken.json` stores:
- `special_tokens` (1000 entries): token IDs `0..999`
- `vocab[]` (150k entries): base vocab pieces, model token IDs are offset by `+1000`

So for decoding:
- Token IDs `0..999` → special tokens (skip in output)
- Token IDs `1000..131071` → `vocab[token_id - 1000].token_bytes` (base64 → bytes → UTF-8)

### Audio streaming config (from `tekken.json`)
| Parameter | Value |
|-----------|-------|
| sampling_rate | 16000 |
| frame_rate | 12.5 (80ms per token) |
| num_mel_bins | 128 |
| hop_length | 160 |
| window_size | 400 |
| transcription_delay_ms | 480 (= 6 delay tokens) |
| n_left_pad_tokens | 32 |
| n_right_pad_tokens (offline) | 17 = (6+1) + 10 |

---

## Weight Format

### Files
- `consolidated.safetensors`: 8.86 GB, all BF16, 711 tensors
- `tekken.json`: 14.9 MB tokenizer
- `params.json`: model config

### Tensor Names

**Token Embeddings** (tied with LM head):
```
mm_streams_embeddings.embedding_module.tok_embeddings.weight  [131072, 3072]
```

**Audio Encoder** (prefix: `mm_streams_embeddings.embedding_module.whisper_encoder.`):
```
conv_layers.0.conv.weight       [1280, 128, 3] + bias [1280]
conv_layers.1.conv.weight       [1280, 1280, 3] + bias [1280]  # stride=2
transformer.layers.{i}.attention.wq.weight  [2048, 1280] + bias
transformer.layers.{i}.attention.wk.weight  [2048, 1280]  # NO bias
transformer.layers.{i}.attention.wv.weight  [2048, 1280] + bias
transformer.layers.{i}.attention.wo.weight  [1280, 2048] + bias
transformer.layers.{i}.attention_norm.weight [1280]
transformer.layers.{i}.feed_forward.w1.weight [5120, 1280]  # no bias
transformer.layers.{i}.feed_forward.w2.weight [1280, 5120] + bias
transformer.layers.{i}.feed_forward.w3.weight [5120, 1280]  # no bias
transformer.layers.{i}.ffn_norm.weight [1280]
transformer.norm.weight         [1280]  # final encoder norm
```

**Adapter** (prefix: `mm_streams_embeddings.embedding_module.`):
```
audio_language_projection.0.weight  [3072, 5120]  # no bias
audio_language_projection.2.weight  [3072, 3072]  # no bias
```

**LLM Decoder** (prefix: `layers.{i}.`):
```
attention_norm.weight           [3072]
attention.wq.weight             [4096, 3072]  # 32 heads x 128
attention.wk.weight             [1024, 3072]  # 8 KV heads x 128
attention.wv.weight             [1024, 3072]
attention.wo.weight             [3072, 4096]
ffn_norm.weight                 [3072]
feed_forward.w1.weight          [9216, 3072]  # gate
feed_forward.w2.weight          [3072, 9216]  # down
feed_forward.w3.weight          [9216, 3072]  # up
ada_rms_norm_t_cond.0.weight    [32, 3072]    # ada_down
ada_rms_norm_t_cond.2.weight    [3072, 32]    # ada_up
```
Plus `norm.weight [3072]` (final norm). NO biases in decoder.

---

## Decoder Forward Pass

The Metal backend quantizes all decoder matmul weights from BF16 to per-group symmetric
INT8 at runtime (group_size=128, scales cached). The encoder stays F16/MPS.

Per-layer computation for hidden state `h` at positions `pos .. pos+seq-1`:

1. **Attention RMSNorm**: `x = RMSNorm(h, attention_norm, eps=1e-5)`
2. **QKV projections (GQA)**:
   - `q = x @ wq^T` → `[seq, 4096]` → reshape `[seq, 32, 128]`
   - `k = x @ wk^T` → `[seq, 1024]` → reshape `[seq, 8, 128]`
   - `v = x @ wv^T` → `[seq, 1024]` → reshape `[seq, 8, 128]`
3. **RoPE** on Q and K (theta=1e6, head_dim=128)
4. **KV cache**: append K, V to per-layer cache
5. **Causal sliding-window attention**: window=8192, scale=1/sqrt(128), GQA repeat 4:1
6. **Output projection + residual**: `h = h + attn_out @ wo^T`
7. **FFN RMSNorm**: `h_norm = RMSNorm(h, ffn_norm, eps=1e-5)`
8. **Ada norm**: `h_norm = h_norm * (1 + ada_scale[layer])` (precomputed from t_cond)
9. **SwiGLU FFN + residual**:
   - `gate = silu(h_norm @ w1^T)`
   - `up = h_norm @ w3^T`
   - `h = h + (gate * up) @ w2^T`

After last layer: `h = RMSNorm(h, norm.weight)`, then `logits = h @ tok_embeddings^T`.

### RoPE note

The encoder uses interleaved (GPT-J style) RoPE: `x1 = x[..., ::2], x2 = x[..., 1::2]`.
vLLM permutes decoder `wq/wk` weights at load time and uses NeoX-style split-halves RoPE.
These are mathematically equivalent representations — pick one and be consistent.

---

## Offline Decode Schedule

This is the critical algorithm for correct transcription. The model combines audio and
text embeddings at every position and expects multimodal input at every step.

Given:
- `audio_text_embeds`: `[N_audio, 3072]` (adapter output)
- `prompt_ids`: length `L` (default 39) = `[BOS] + [STREAMING_PAD] * (n_left_pad + n_delay)`

Algorithm:
1. **Prefill** positions `0..L-2`:
   - feed `audio_text_embeds[pos] + tok_embed(prompt_ids[pos])` to build KV caches
2. **First generated token** at position `L-1`:
   - logits from `audio_text_embeds[L-1] + tok_embed(prompt_ids[L-1])`
   - greedy argmax → `token_0`
3. **Decode within audio span**, for `pos = L .. N_audio-1`:
   - `embed = audio_text_embeds[pos] + tok_embed(prev_token)`
   - logits → next token (greedy, temperature=0.0)
   - stop on EOS

The offline right-padding provides extra audio-token budget so the model can finish
emitting text and produce EOS within the audio span.

---

## Online Decode Schedule (not yet implemented in C)

For real-time streaming, the model processes audio incrementally:

1. First chunk: use `[BOS] + [STREAMING_PAD]*38` prefix with left-padded audio
2. Subsequent chunks: feed `prev_token` + next audio embedding
3. Audio chunks must be multiples of 40 samples (`|window_size/2 - hop_length|`)
4. No right padding (lower latency, slightly worse accuracy at utterance boundaries)

The encoder is fully causal (sliding window=750), so past outputs don't change when
new audio arrives. Implementation requires adding KV cache to the encoder and handling
conv stem boundary state across chunks.
