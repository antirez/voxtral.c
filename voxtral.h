/*
 * voxtral.h - Voxtral Realtime 4B Pure C Inference Engine
 *
 * Main API header for the Voxtral speech-to-text model.
 */

#ifndef VOXTRAL_H
#define VOXTRAL_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/* ========================================================================
 * Model Constants
 * ======================================================================== */

/* Audio preprocessing */
#define VOX_SAMPLE_RATE      16000
#define VOX_MEL_BINS         128
#define VOX_HOP_LENGTH       160
#define VOX_WINDOW_SIZE      400
#define VOX_FRAME_RATE       12.5f
#define VOX_LOG_MEL_MAX      1.5f

/* Audio encoder */
#define VOX_ENC_DIM          1280
#define VOX_ENC_LAYERS       32
#define VOX_ENC_HEADS        32
#define VOX_ENC_KV_HEADS     32
#define VOX_ENC_HEAD_DIM     64
#define VOX_ENC_HIDDEN       5120
#define VOX_ENC_WINDOW       750
#define VOX_ENC_NORM_EPS     1e-5f

/* Downsampling */
#define VOX_DOWNSAMPLE       4

/* LLM decoder */
#define VOX_DEC_DIM          3072
#define VOX_DEC_LAYERS       26
#define VOX_DEC_HEADS        32
#define VOX_DEC_KV_HEADS     8
#define VOX_DEC_HEAD_DIM     128
#define VOX_DEC_HIDDEN       9216
#define VOX_DEC_WINDOW       8192
#define VOX_DEC_NORM_EPS     1e-5f
#define VOX_VOCAB_SIZE       131072
#define VOX_ADA_NORM_DIM     32
#define VOX_ROPE_THETA       1000000.0f

/* ========================================================================
 * Audio Encoder Layer
 * ======================================================================== */

typedef struct {
    /* Attention weights (all have biases except wk) */
    float *wq_weight;        /* [2048, 1280] - f32 (NULL if bf16) */
    uint16_t *wq_weight_bf16;/* [2048, 1280] - bf16 mmap direct */
    float *wq_bias;          /* [2048] */
    float *wk_weight;        /* [2048, 1280] - f32 (NULL if bf16) */
    uint16_t *wk_weight_bf16;/* [2048, 1280] - bf16 mmap direct */
    /* wk has NO bias */
    float *wv_weight;        /* [2048, 1280] - f32 (NULL if bf16) */
    uint16_t *wv_weight_bf16;/* [2048, 1280] - bf16 mmap direct */
    float *wv_bias;          /* [2048] */
    float *wo_weight;        /* [1280, 2048] - f32 (NULL if bf16) */
    uint16_t *wo_weight_bf16;/* [1280, 2048] - bf16 mmap direct */
    float *wo_bias;          /* [1280] */
    float *attention_norm;   /* [1280] */

    /* Feed-forward (w1, w3 have no bias, w2 has bias) */
    float *w1_weight;        /* [5120, 1280] gate - f32 (NULL if bf16) */
    uint16_t *w1_weight_bf16;/* [5120, 1280] - bf16 mmap direct */
    float *w2_weight;        /* [1280, 5120] down - f32 (NULL if bf16) */
    uint16_t *w2_weight_bf16;/* [1280, 5120] - bf16 mmap direct */
    float *w2_bias;          /* [1280] */
    float *w3_weight;        /* [5120, 1280] up - f32 (NULL if bf16) */
    uint16_t *w3_weight_bf16;/* [5120, 1280] - bf16 mmap direct */
    float *ffn_norm;         /* [1280] */
} vox_enc_layer_t;

typedef struct {
    /* Conv stem */
    float *conv0_weight;     /* [1280, 128, 3] */
    float *conv0_bias;       /* [1280] */
    float *conv1_weight;     /* [1280, 1280, 3] */
    float *conv1_bias;       /* [1280] */

    /* Transformer layers */
    vox_enc_layer_t layers[VOX_ENC_LAYERS];

    /* Final norm */
    float *norm;             /* [1280] */
} vox_encoder_t;

/* ========================================================================
 * LLM Decoder Layer
 * ======================================================================== */

typedef struct {
    /* Adaptive RMS norm conditioning MLP (small, always f32) */
    float *ada_norm_down;    /* [32, 3072] Linear(3072->32) */
    float *ada_norm_up;      /* [3072, 32] Linear(32->3072) */

    /* Attention (no biases in decoder) */
    float *wq_weight;        /* [4096, 3072] - f32 (NULL if bf16) */
    uint16_t *wq_weight_bf16;/* [4096, 3072] - bf16 mmap direct */
    float *wk_weight;        /* [1024, 3072] - f32 (NULL if bf16) */
    uint16_t *wk_weight_bf16;/* [1024, 3072] - bf16 mmap direct */
    float *wv_weight;        /* [1024, 3072] - f32 (NULL if bf16) */
    uint16_t *wv_weight_bf16;/* [1024, 3072] - bf16 mmap direct */
    float *wo_weight;        /* [3072, 4096] - f32 (NULL if bf16) */
    uint16_t *wo_weight_bf16;/* [3072, 4096] - bf16 mmap direct */
    float *attention_norm;   /* [3072] */

    /* Feed-forward */
    float *w1_weight;        /* [9216, 3072] gate - f32 (NULL if bf16) */
    uint16_t *w1_weight_bf16;/* [9216, 3072] - bf16 mmap direct */
    float *w2_weight;        /* [3072, 9216] down - f32 (NULL if bf16) */
    uint16_t *w2_weight_bf16;/* [3072, 9216] - bf16 mmap direct */
    float *w3_weight;        /* [9216, 3072] up - f32 (NULL if bf16) */
    uint16_t *w3_weight_bf16;/* [9216, 3072] - bf16 mmap direct */
    float *ffn_norm;         /* [3072] */
} vox_dec_layer_t;

typedef struct {
    /* Token embeddings (shared with output projection) */
    float *tok_embeddings;   /* [131072, 3072] - f32 (NULL if bf16) */
    uint16_t *tok_embeddings_bf16; /* [131072, 3072] - bf16 mmap direct */

    /* Transformer layers */
    vox_dec_layer_t layers[VOX_DEC_LAYERS];

    /* Final norm */
    float *norm;             /* [3072] */
} vox_decoder_t;

/* ========================================================================
 * Audio-Language Adapter
 * ======================================================================== */

typedef struct {
    float *linear0_weight;   /* [3072, 5120] - f32 (NULL if bf16) */
    uint16_t *linear0_weight_bf16; /* [3072, 5120] - bf16 mmap direct */
    float *linear1_weight;   /* [3072, 3072] - f32 (NULL if bf16) */
    uint16_t *linear1_weight_bf16; /* [3072, 3072] - bf16 mmap direct */
} vox_adapter_t;

/* ========================================================================
 * Main Context
 * ======================================================================== */

typedef struct {
    vox_encoder_t encoder;
    vox_adapter_t adapter;
    vox_decoder_t decoder;

    /* Model file (kept open for mmap) */
    void *safetensors;       /* safetensors_file_t* */
    char model_dir[512];

    /* KV cache for decoder (rolling: compacted when full) */
    float *kv_cache_k;       /* [layers, max_seq, kv_heads * head_dim] */
    float *kv_cache_v;       /* [layers, max_seq, kv_heads * head_dim] */
    uint16_t *kv_cache_k_f16;/* Optional fp16 KV cache (Metal path) */
    uint16_t *kv_cache_v_f16;/* Optional fp16 KV cache (Metal path) */
    int kv_cache_fp16;       /* 1 if decoder KV cache uses fp16 storage */
    int kv_cache_len;        /* Current physical cache length */
    int kv_cache_max;        /* Maximum cache size */
    int kv_pos_offset;       /* Logical position offset (positions discarded by compaction) */

    /* Transcription delay in tokens (1 token = 80ms) */
    int delay_tokens;        /* Default: 6 (480ms) */

    /* Precomputed timing conditioning for the decoder (vLLM ada_rms_norm_t_cond) */
    float t_cond[VOX_DEC_DIM];      /* TimeEmbedding(delay_tokens) */
    float *ada_scale;               /* [VOX_DEC_LAYERS * VOX_DEC_DIM] */

    /* BF16 direct mmap mode (weights stay as bf16, convert on-the-fly) */
    int use_bf16;

    /* Encoder KV cache (rolling: compacted at window=750) */
    float *enc_kv_cache_k;    /* [ENC_LAYERS, max_seq, enc_kv_dim] */
    float *enc_kv_cache_v;    /* [ENC_LAYERS, max_seq, enc_kv_dim] */
    int enc_kv_cache_len;     /* physical cache length */
    int enc_kv_cache_max;     /* allocated capacity */
    int enc_kv_cache_is_shared; /* allocated with vox_metal_shared_alloc */
    int enc_kv_pos_offset;    /* logical offset from rolling compaction */

    /* Persistent incremental-encoder scratch (allocated/grown on demand). */
    int enc_inc_cap;          /* max new_len supported by buffers below */
    float *enc_inc_x_norm, *enc_inc_q, *enc_inc_k, *enc_inc_v;
    float *enc_inc_attn_out, *enc_inc_proj_out;
    float *enc_inc_gate, *enc_inc_up, *enc_inc_ffn_out;
    int *enc_inc_positions;
    float *enc_inc_rope_freqs;

    /* Persistent single-token decoder buffers (allocated on first forward) */
    float *dec_x, *dec_x_norm, *dec_q, *dec_k, *dec_v;
    float *dec_attn_out, *dec_proj_out;
    float *dec_gate, *dec_up, *dec_ffn_out;
    float *dec_rope_freqs;
} vox_ctx_t;

/* ========================================================================
 * Alternative Tokens
 * ======================================================================== */

#define VOX_MAX_ALT 4

/* ========================================================================
 * API Functions
 * ======================================================================== */

/* Load model from directory containing consolidated.safetensors + tekken.json */
vox_ctx_t *vox_load(const char *model_dir);

/* Free all resources */
void vox_free(vox_ctx_t *ctx);

/* Set transcription delay in milliseconds (80-2400, default 480) */
void vox_set_delay(vox_ctx_t *ctx, int delay_ms);

/* ========================================================================
 * Streaming API — works for both real-time and offline transcription
 *
 * Usage:
 *   vox_stream_t *s = vox_stream_init(ctx);
 *   while (have_audio) {
 *       vox_stream_feed(s, chunk, n);       // runs encoder+decoder
 *       while ((n = vox_stream_get(s, tokens, 16)) > 0) { ... }
 *   }
 *   vox_stream_finish(s);                   // process remaining audio
 *   while ((n = vox_stream_get(s, tokens, 16)) > 0) { ... }
 *   vox_stream_free(s);
 * ======================================================================== */

typedef struct vox_stream vox_stream_t;

/* Create a streaming transcription context. */
vox_stream_t *vox_stream_init(vox_ctx_t *ctx);

/* Feed audio samples (mono float32, 16kHz, [-1,1]).
 * Runs encoder/decoder on available data and queues output tokens.
 * Returns 0 on success, -1 on error. */
int vox_stream_feed(vox_stream_t *s, const float *samples, int n_samples);

/* Signal end of audio. Triggers right-padding, final encoder chunks,
 * and remaining token generation. Returns 0 on success, -1 on error. */
int vox_stream_finish(vox_stream_t *s);

/* Retrieve pending decoded token strings. Fills out_tokens with up to max
 * pointers to token text. Pointers are valid until vox_stream_free().
 * Returns number of tokens written (0 = nothing pending). */
int vox_stream_get(vox_stream_t *s, const char **out_tokens, int max);

/* Configure alternative token tracking.
 * n_alt: max alternatives per position (1-VOX_MAX_ALT, default 1 = no alts).
 * cutoff: max distance from top token (0.0-1.0). A token qualifies if
 *         1 - prob[i]/prob[0] <= cutoff. */
void vox_stream_set_alt(vox_stream_t *s, int n_alt, float cutoff);

/* Retrieve pending tokens with alternatives. out_tokens has max_tokens * n_alt
 * slots. For each token position, n_alt consecutive entries: [0]=best, rest=
 * alternatives or NULL. n_alt is clamped to VOX_MAX_ALT.
 * Returns number of token positions dequeued. */
int vox_stream_get_alt(vox_stream_t *s, const char **out_tokens,
                       int max_tokens, int n_alt);

/* Set minimum time between encoder runs, in seconds.
 * Lower = more responsive streaming (higher GPU overhead).
 * Higher = more efficient batching (higher latency).
 * Default: 2.0. First chunk always waits for ~3s (decoder prompt needs 312 mel).
 * finish() always processes all remaining data regardless. */
void vox_set_processing_interval(vox_stream_t *s, float seconds);

/* Enable continuous (live) mode: auto-restart decoder on EOS, KV overflow,
 * or prolonged non-text decode streaks. Restarts are hard resets (no decoder
 * context carry-over). Without this, EOS ends decoding and KV grows unbounded
 * (fine for finite files). */
void vox_stream_set_continuous(vox_stream_t *s, int enable);

/* Force the encoder to process whatever audio is buffered, regardless of the
 * processing interval. Useful for flushing on silence detection. */
int vox_stream_flush(vox_stream_t *s);

/* Free streaming context and all resources. */
void vox_stream_free(vox_stream_t *s);

/* ========================================================================
 * Convenience Functions (built on streaming API)
 * ======================================================================== */

/* Transcribe a WAV file, returns allocated string (caller must free) */
char *vox_transcribe(vox_ctx_t *ctx, const char *wav_path);

/* Transcribe from raw audio samples (mono, 16kHz, float32 [-1,1]) */
char *vox_transcribe_audio(vox_ctx_t *ctx, const float *samples, int n_samples);

/* Transcribe from stdin (auto-detect WAV vs raw s16le, streaming for raw) */
char *vox_transcribe_stdin(vox_ctx_t *ctx);

/* ========================================================================
 * Internal Functions (used by encoder/decoder implementations)
 * ======================================================================== */

/* Audio encoder forward pass (full, non-incremental) */
float *vox_encoder_forward(vox_ctx_t *ctx, const float *mel,
                           int mel_frames, int *out_seq_len);

/* Incremental encoder forward pass (processes new_len post-conv-stem positions
 * through transformer layers using encoder KV cache). Returns [new_len, 1280].
 * Caller must free the returned buffer. */
float *vox_encoder_forward_incremental(vox_ctx_t *ctx, const float *x_new,
                                        int new_len, int *out_len);

/* Adapter forward pass */
float *vox_adapter_forward(vox_ctx_t *ctx, const float *enc_out,
                           int enc_seq_len, int *out_seq_len);

/* Decoder forward pass (single token, uses KV cache) */
int vox_decoder_forward(vox_ctx_t *ctx, const float *input_embeds, float *logits);

/* Decoder forward pass for prefill (multiple tokens) */
void vox_decoder_prefill(vox_ctx_t *ctx, const float *input_embeds, int seq_len);
int vox_decoder_kv_cache_preallocate(vox_ctx_t *ctx, int max_seq);
int vox_encoder_kv_cache_preallocate(vox_ctx_t *ctx, int max_pos);

#endif /* VOXTRAL_H */
