/*
 * voxtral.c - Main API for Voxtral Realtime 4B inference
 *
 * Orchestrates the full pipeline:
 *   Load weights -> WAV -> Mel -> Encoder -> Adapter -> Decoder -> Tokenizer -> Text
 */

#include "voxtral.h"
#include "voxtral_kernels.h"
#include "voxtral_safetensors.h"
#include "voxtral_audio.h"
#include "voxtral_tokenizer.h"
#ifdef USE_METAL
#include "voxtral_metal.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <sys/time.h>

/* Global verbose flag */
int vox_verbose = 0;
int vox_monitor = 0;

/* ========================================================================
 * Decoder timing conditioning (t_cond + per-layer ada_scale)
 * ======================================================================== */

static void vox_compute_time_embedding(float *out, float t_value) {
    /* Matches vLLM MistralDecoderLayer time embedding:
     * inv_freq = exp(-log(10000) * arange(dim/2) / (dim/2))
     * emb = t * inv_freq
     * out = [cos(emb), sin(emb)] */
    int dim = VOX_DEC_DIM;
    int half = dim / 2;
    float log_theta = logf(10000.0f);
    for (int i = 0; i < half; i++) {
        float inv_freq = expf(-log_theta * (float)i / (float)half);
        float emb = t_value * inv_freq;
        out[i] = cosf(emb);
        out[i + half] = sinf(emb);
    }
}

static void vox_update_time_conditioning(vox_ctx_t *ctx) {
    if (!ctx) return;
    vox_compute_time_embedding(ctx->t_cond, (float)ctx->delay_tokens);

    size_t n = (size_t)VOX_DEC_LAYERS * VOX_DEC_DIM;
    if (!ctx->ada_scale) {
        ctx->ada_scale = (float *)malloc(n * sizeof(float));
    }
    if (!ctx->ada_scale) return;

    /* Precompute per-layer ada_scale = ada_up(GELU(ada_down(t_cond))). */
    float hidden[VOX_ADA_NORM_DIM];
    for (int layer = 0; layer < VOX_DEC_LAYERS; layer++) {
        const vox_dec_layer_t *l = &ctx->decoder.layers[layer];

        /* hidden = ada_down @ t_cond  (32 x 3072) */
        for (int i = 0; i < VOX_ADA_NORM_DIM; i++) {
            float sum = 0.0f;
            const float *row = l->ada_norm_down + (size_t)i * VOX_DEC_DIM;
            for (int j = 0; j < VOX_DEC_DIM; j++) sum += row[j] * ctx->t_cond[j];
            hidden[i] = sum;
        }
        vox_gelu(hidden, VOX_ADA_NORM_DIM);

        float *scale = ctx->ada_scale + (size_t)layer * VOX_DEC_DIM;
        /* scale = ada_up @ hidden  (3072 x 32) */
        for (int i = 0; i < VOX_DEC_DIM; i++) {
            float sum = 0.0f;
            const float *row = l->ada_norm_up + (size_t)i * VOX_ADA_NORM_DIM;
            for (int j = 0; j < VOX_ADA_NORM_DIM; j++) sum += row[j] * hidden[j];
            scale[i] = sum;
        }
    }
}

/* ========================================================================
 * Internal load functions (defined in encoder/decoder .c files)
 * ======================================================================== */

extern int vox_encoder_load(vox_encoder_t *enc, safetensors_file_t *sf);
extern int vox_decoder_load(vox_decoder_t *dec, safetensors_file_t *sf);

/* ========================================================================
 * Adapter Weight Loading
 * ======================================================================== */

static uint16_t *load_bf16_direct(safetensors_file_t *sf, const char *name) {
    const safetensor_t *t = safetensors_find(sf, name);
    if (!t) {
        fprintf(stderr, "adapter: weight not found: %s\n", name);
        return NULL;
    }
    return safetensors_get_bf16_direct(sf, t);
}

static int vox_adapter_load(vox_adapter_t *adapter, safetensors_file_t *sf) {
    adapter->linear0_weight_bf16 = load_bf16_direct(sf,
        "mm_streams_embeddings.embedding_module.audio_language_projection.0.weight");
    adapter->linear1_weight_bf16 = load_bf16_direct(sf,
        "mm_streams_embeddings.embedding_module.audio_language_projection.2.weight");

    if (!adapter->linear0_weight_bf16 || !adapter->linear1_weight_bf16) return -1;
    return 0;
}

/* ========================================================================
 * Model Loading
 * ======================================================================== */

vox_ctx_t *vox_load(const char *model_dir) {
    vox_ctx_t *ctx = (vox_ctx_t *)calloc(1, sizeof(vox_ctx_t));
    if (!ctx) return NULL;

    strncpy(ctx->model_dir, model_dir, sizeof(ctx->model_dir) - 1);
    ctx->delay_tokens = 6; /* 480ms default */
    ctx->use_bf16 = 1; /* bf16 mmap direct mode */
    ctx->kv_cache_fp16 = 0; /* default: fp32 KV cache */

    /* Open safetensors file */
    char path[1024];
    snprintf(path, sizeof(path), "%s/consolidated.safetensors", model_dir);

    if (vox_verbose >= 2)
        fprintf(stderr, "Loading model from %s\n", path);

    safetensors_file_t *sf = safetensors_open(path);
    if (!sf) {
        fprintf(stderr, "vox_load: cannot open %s\n", path);
        free(ctx);
        return NULL;
    }
    ctx->safetensors = sf;

    if (vox_verbose >= 1)
        fprintf(stderr, "Loading weights...\n");
    if (vox_encoder_load(&ctx->encoder, sf) != 0) {
        fprintf(stderr, "vox_load: failed to load encoder\n");
        vox_free(ctx);
        return NULL;
    }

    if (vox_adapter_load(&ctx->adapter, sf) != 0) {
        fprintf(stderr, "vox_load: failed to load adapter\n");
        vox_free(ctx);
        return NULL;
    }

    if (vox_decoder_load(&ctx->decoder, sf) != 0) {
        fprintf(stderr, "vox_load: failed to load decoder\n");
        vox_free(ctx);
        return NULL;
    }

    /* Precompute time conditioning for the decoder (t_cond + per-layer ada_scale). */
    vox_update_time_conditioning(ctx);

#ifdef USE_METAL
    /* Pre-warm Metal bf16->f16 weight cache to avoid first-token spike */
    if (vox_metal_available()) {
        const char *kv_env = getenv("VOX_DECODER_KV_FP16");
        ctx->kv_cache_fp16 = (!kv_env || atoi(kv_env) != 0) ? 1 : 0;

        if (vox_verbose >= 2)
            fprintf(stderr, "Pre-warming Metal weight cache...\n");

        /* Encoder weights: merged QKV + merged w1+w3 (replaces individual caching)
         * wo and w2 still cached individually. */
        for (int i = 0; i < VOX_ENC_LAYERS; i++) {
            vox_enc_layer_t *l = &ctx->encoder.layers[i];
            size_t enc_attn = (size_t)(VOX_ENC_HEADS * VOX_ENC_HEAD_DIM) * VOX_ENC_DIM;
            size_t enc_wo   = (size_t)VOX_ENC_DIM * (VOX_ENC_HEADS * VOX_ENC_HEAD_DIM);
            size_t enc_ffn1 = (size_t)VOX_ENC_HIDDEN * VOX_ENC_DIM;
            size_t enc_ffn2 = (size_t)VOX_ENC_DIM * VOX_ENC_HIDDEN;
            /* Merged QKV and w1+w3 (internally caches individual f16 buffers too) */
            vox_metal_warmup_merged_3(
                l->wq_weight_bf16, enc_attn,
                l->wk_weight_bf16, enc_attn,
                l->wv_weight_bf16, enc_attn);
            vox_metal_warmup_merged_2(
                l->w1_weight_bf16, enc_ffn1,
                l->w3_weight_bf16, enc_ffn1);
            /* wo and w2 used individually */
            vox_metal_warmup_bf16(l->wo_weight_bf16, enc_wo);
            vox_metal_warmup_bf16(l->w2_weight_bf16, enc_ffn2);
        }

        /* Adapter weights */
        vox_metal_warmup_bf16(ctx->adapter.linear0_weight_bf16,
                              (size_t)VOX_DEC_DIM * (VOX_ENC_DIM * VOX_DOWNSAMPLE));
        vox_metal_warmup_bf16(ctx->adapter.linear1_weight_bf16,
                              (size_t)VOX_DEC_DIM * VOX_DEC_DIM);

        /* Decoder weights (26 layers) */
        for (int i = 0; i < VOX_DEC_LAYERS; i++) {
            vox_dec_layer_t *l = &ctx->decoder.layers[i];
            size_t dec_q  = (size_t)(VOX_DEC_HEADS * VOX_DEC_HEAD_DIM) * VOX_DEC_DIM;
            size_t dec_kv = (size_t)(VOX_DEC_KV_HEADS * VOX_DEC_HEAD_DIM) * VOX_DEC_DIM;
            size_t dec_wo = (size_t)VOX_DEC_DIM * (VOX_DEC_HEADS * VOX_DEC_HEAD_DIM);
            size_t dec_f1 = (size_t)VOX_DEC_HIDDEN * VOX_DEC_DIM;
            size_t dec_f2 = (size_t)VOX_DEC_DIM * VOX_DEC_HIDDEN;
            vox_metal_warmup_bf16(l->wq_weight_bf16, dec_q);
            vox_metal_warmup_bf16(l->wk_weight_bf16, dec_kv);
            vox_metal_warmup_bf16(l->wv_weight_bf16, dec_kv);
            vox_metal_warmup_bf16(l->wo_weight_bf16, dec_wo);
            vox_metal_warmup_bf16(l->w1_weight_bf16, dec_f1);
            vox_metal_warmup_bf16(l->w2_weight_bf16, dec_f2);
            vox_metal_warmup_bf16(l->w3_weight_bf16, dec_f1);
        }

        /* Token embeddings (also used as logits projection) */
        vox_metal_warmup_bf16(ctx->decoder.tok_embeddings_bf16,
                              (size_t)VOX_VOCAB_SIZE * VOX_DEC_DIM);

        /* Pre-warm merged weight buffers for monolithic decoder step */
        for (int i = 0; i < VOX_DEC_LAYERS; i++) {
            vox_dec_layer_t *l = &ctx->decoder.layers[i];
            /* Merged QKV = wq + wk + wv */
            vox_metal_warmup_merged_3(
                l->wq_weight_bf16, (size_t)(VOX_DEC_HEADS * VOX_DEC_HEAD_DIM) * VOX_DEC_DIM,
                l->wk_weight_bf16, (size_t)(VOX_DEC_KV_HEADS * VOX_DEC_HEAD_DIM) * VOX_DEC_DIM,
                l->wv_weight_bf16, (size_t)(VOX_DEC_KV_HEADS * VOX_DEC_HEAD_DIM) * VOX_DEC_DIM);
            /* Merged w1+w3 */
            vox_metal_warmup_merged_2(
                l->w1_weight_bf16, (size_t)VOX_DEC_HIDDEN * VOX_DEC_DIM,
                l->w3_weight_bf16, (size_t)VOX_DEC_HIDDEN * VOX_DEC_DIM);
        }

        /* Pre-warm decoder MPS ops and f32 weight caches */
        vox_metal_warmup_decoder_ops(ctx);

        /* Pre-allocate KV cache (shared GPU memory) */
        vox_decoder_kv_cache_preallocate(ctx, VOX_DEC_WINDOW + 1024);

        if (vox_verbose >= 1)
            fprintf(stderr, "Decoder KV cache: %s\n",
                    ctx->kv_cache_fp16 ? "fp16" : "fp32");

        /* Pre-allocate encoder KV cache (shared GPU memory for monolithic step) */
        vox_encoder_kv_cache_preallocate(ctx, VOX_ENC_WINDOW + 256);

        if (vox_verbose >= 1)
            fprintf(stderr, "Metal GPU: %.1f MB\n",
                    vox_metal_memory_used() / (1024.0 * 1024.0));
    }
#endif

    if (vox_verbose >= 1)
        fprintf(stderr, "Model loaded.\n");
    return ctx;
}

/* ========================================================================
 * Model Free
 * ======================================================================== */

void vox_free(vox_ctx_t *ctx) {
    if (!ctx) return;

    /* Free f32 weights that were converted/copied from safetensors. */
    #define FREE0(p) do { free(p); (p) = NULL; } while (0)

    /* Encoder conv stem */
    FREE0(ctx->encoder.conv0_weight);
    FREE0(ctx->encoder.conv0_bias);
    FREE0(ctx->encoder.conv1_weight);
    FREE0(ctx->encoder.conv1_bias);

    for (int i = 0; i < VOX_ENC_LAYERS; i++) {
        vox_enc_layer_t *l = &ctx->encoder.layers[i];
        FREE0(l->wq_bias);
        FREE0(l->wv_bias);
        FREE0(l->wo_bias);
        FREE0(l->attention_norm);
        FREE0(l->w2_bias);
        FREE0(l->ffn_norm);
    }
    FREE0(ctx->encoder.norm);

    /* Decoder */
    for (int i = 0; i < VOX_DEC_LAYERS; i++) {
        vox_dec_layer_t *l = &ctx->decoder.layers[i];
        FREE0(l->ada_norm_down);
        FREE0(l->ada_norm_up);
        FREE0(l->attention_norm);
        FREE0(l->ffn_norm);
    }
    FREE0(ctx->decoder.norm);

    #undef FREE0

#ifdef USE_METAL
    vox_metal_shared_free(ctx->kv_cache_k);
    vox_metal_shared_free(ctx->kv_cache_v);
    vox_metal_shared_free(ctx->kv_cache_k_f16);
    vox_metal_shared_free(ctx->kv_cache_v_f16);
#else
    free(ctx->kv_cache_k);
    free(ctx->kv_cache_v);
    free(ctx->kv_cache_k_f16);
    free(ctx->kv_cache_v_f16);
#endif
#ifdef USE_METAL
    if (ctx->enc_kv_cache_is_shared) {
        vox_metal_shared_free(ctx->enc_kv_cache_k);
        vox_metal_shared_free(ctx->enc_kv_cache_v);
    } else
#endif
    {
        free(ctx->enc_kv_cache_k);
        free(ctx->enc_kv_cache_v);
    }
    free(ctx->ada_scale);
    free(ctx->enc_inc_x_norm);
    free(ctx->enc_inc_q);
    free(ctx->enc_inc_k);
    free(ctx->enc_inc_v);
    free(ctx->enc_inc_attn_out);
    free(ctx->enc_inc_proj_out);
    free(ctx->enc_inc_gate);
    free(ctx->enc_inc_up);
    free(ctx->enc_inc_ffn_out);
    free(ctx->enc_inc_positions);
    free(ctx->enc_inc_rope_freqs);

    /* Persistent decoder buffers */
    free(ctx->dec_x);
    free(ctx->dec_x_norm);
    free(ctx->dec_q);
    free(ctx->dec_k);
    free(ctx->dec_v);
    free(ctx->dec_attn_out);
    free(ctx->dec_proj_out);
    free(ctx->dec_gate);
    free(ctx->dec_up);
    free(ctx->dec_ffn_out);
    free(ctx->dec_rope_freqs);

    if (ctx->safetensors) {
        safetensors_close((safetensors_file_t *)ctx->safetensors);
    }

    free(ctx);
}

/* ========================================================================
 * Transcription Constants
 * ======================================================================== */

/*
 * Special token IDs (Tekken):
 *   BOS = 1, EOS = 2, [STREAMING_PAD] = 32
 *
 * Decoder input: inputs_embeds[pos] = adapter_out[pos] + tok_embed(input_id)
 * Prompt: [BOS] + [STREAMING_PAD] * (32 + delay_tokens)
 */
#define TOKEN_BOS          1
#define TOKEN_EOS          2
#define TOKEN_STREAMING_PAD 32
#define TOKEN_TEXT_MIN     1000

#define RAW_AUDIO_LENGTH_PER_TOK 1280
#define OFFLINE_STREAMING_BUFFER_TOKENS 10

/* First chunk minimum mel frames (enough for 39 prompt adapter tokens) */
#define STREAM_FIRST_CHUNK_MIN_MEL  312

/* Default processing interval in seconds (mel rate = 100 fps) */
#define STREAM_DEFAULT_INTERVAL  2.0f

/* Max decoder KV cache entries before forced restart on live streams.
 * Bounds attention cost to keep decode steps under control. */
#define STREAM_MAX_DECODE_KV  2000
/* If the decoder keeps generating non-text/control tokens for too long,
 * force a hard restart so live transcription can recover. */
#define STREAM_MAX_NON_TEXT_STREAK  64
/* If incoming audio advances but decoder emits no tokens for too long,
 * force a full stream reset (mel+encoder+decoder state). */
#define STREAM_MAX_NO_DECODE_SAMPLES  (VOX_SAMPLE_RATE * 20)
/* Escalate to full stream reset after repeated no-text restarts. */
#define STREAM_EMPTY_RESTARTS_FOR_FULL_RESET  2

static void trim_ascii_whitespace(char *s) {
    if (!s) return;
    size_t len = strlen(s);
    size_t start = 0;
    while (start < len && isspace((unsigned char)s[start])) start++;
    size_t end = len;
    while (end > start && isspace((unsigned char)s[end - 1])) end--;
    if (start > 0) memmove(s, s + start, end - start);
    s[end - start] = '\0';
}

/* Convert a single token embedding from bf16 to f32 */
static void tok_embed_bf16_to_f32(float *dst, const uint16_t *tok_emb_bf16,
                                  int token_id, int dim) {
    const uint16_t *src = tok_emb_bf16 + (size_t)token_id * dim;
    for (int i = 0; i < dim; i++) {
        uint32_t f32_bits = ((uint32_t)src[i]) << 16;
        memcpy(&dst[i], &f32_bits, sizeof(float));
    }
}

/* ========================================================================
 * Streaming API Implementation
 * ======================================================================== */

struct vox_stream {
    vox_ctx_t *ctx;
    vox_tokenizer_t *tokenizer;

    /* Incremental mel */
    vox_mel_ctx_t *mel_ctx;
    int64_t real_samples_fed;

    /* Encoder chunk tracking */
    int mel_cursor;

    /* Incremental conv stem state */
    float *mel_tail;           /* [128 * 2] last 2 mel frames (column-major: [128, 2]) */
    float *conv0_tail;         /* [1280 * 2] last 2 conv0 outputs consumed by conv1 */
    float *conv0_residual;     /* [1280] 0 or 1 conv0 output pending stride alignment */
    int conv0_residual_count;  /* 0 or 1 */
    int conv_stem_initialized; /* 0 for first chunk */

    /* Residual encoder positions for 4x downsample alignment */
    float *enc_residual;       /* [1280 * 3] leftover positions */
    int enc_residual_count;    /* 0-3 */

    /* Adapter output buffer (growing) */
    float *adapter_buf;
    int total_adapter;      /* total logical tokens produced */
    int adapter_cap;
    int adapter_pos_offset; /* logical offset from compactions */

    /* Decoder state */
    int decoder_started;
    int gen_pos;        /* next adapter position for generation */
    int prev_token;
    int eos_seen;
    int nontext_streak;
    int text_since_restart;
    int empty_restarts;    /* consecutive restarts with no text emitted */
    int waiting_prompt;
    int64_t last_decode_sample; /* real_samples_fed at last decoder token */
    int finished;       /* vox_stream_finish() called */
    int continuous;     /* live stream: auto-restart decoder on EOS / KV overflow */

    /* Pending token queue (circular buffer, VOX_MAX_ALT strings per position) */
    const char **token_queue;   /* [queue_cap * VOX_MAX_ALT] */
    int queue_head;     /* next position to read */
    int queue_tail;     /* next position to write */
    int queue_cap;      /* capacity in token positions */

    /* Alternative token settings */
    int n_alt;           /* max alternatives to track (default 1 = no alternatives) */
    float alt_cutoff;    /* max distance from top token (0.0-1.0) */

    /* Decoder working buffers */
    float *logits;
    float *step_embed;
    float *tok_tmp;

    /* Processing interval: minimum new mel frames before encoder triggers */
    int min_new_mel;            /* derived from interval in seconds (mel rate = 100 fps) */

    /* Timing */
    double encoder_ms;
    double decoder_ms;
    double prefill_ms;
    int n_generated;
    int n_text_tokens;          /* tokens emitted as visible text */
};

typedef enum stream_tok_class {
    STREAM_TOK_TEXT = 0,    /* decodes to a non-empty piece */
    STREAM_TOK_CONTROL = 1, /* token ID below text range */
    STREAM_TOK_INVALID = 2, /* text-range token with empty/invalid decode */
    STREAM_TOK_EOS = 3
} stream_tok_class_t;

/* In Tekken, token 1000 is raw byte 0x00. As a C string this is empty.
 * Treat empty decodes as non-text to avoid "decode activity but no output". */
static stream_tok_class_t stream_classify_token(vox_stream_t *s, int token_id) {
    if (token_id == TOKEN_EOS) return STREAM_TOK_EOS;
    if (token_id < TOKEN_TEXT_MIN) return STREAM_TOK_CONTROL;
    const char *piece = vox_tokenizer_decode(s->tokenizer, token_id);
    if (!piece || piece[0] == '\0') return STREAM_TOK_INVALID;
    return STREAM_TOK_TEXT;
}

/* Enqueue one token position. alts[0]=best, alts[1..VOX_MAX_ALT-1]=alternatives or NULL. */
static void stream_enqueue_token(vox_stream_t *s, const char *alts[VOX_MAX_ALT]) {
    /* Grow queue if full */
    int next_tail = (s->queue_tail + 1) % s->queue_cap;
    if (next_tail == s->queue_head) {
        int old_cap = s->queue_cap;
        int new_cap = old_cap * 2;
        const char **new_q = (const char **)calloc((size_t)new_cap * VOX_MAX_ALT, sizeof(const char *));
        if (!new_q) return;
        /* Copy old entries in order */
        int n = 0;
        for (int i = s->queue_head; i != s->queue_tail; i = (i + 1) % old_cap) {
            memcpy(&new_q[n * VOX_MAX_ALT], &s->token_queue[i * VOX_MAX_ALT],
                   VOX_MAX_ALT * sizeof(const char *));
            n++;
        }
        free(s->token_queue);
        s->token_queue = new_q;
        s->queue_head = 0;
        s->queue_tail = n;
        s->queue_cap = new_cap;
        next_tail = (s->queue_tail + 1) % s->queue_cap;
    }
    memcpy(&s->token_queue[s->queue_tail * VOX_MAX_ALT], alts,
           VOX_MAX_ALT * sizeof(const char *));
    s->queue_tail = next_tail;
}

/*
 * Incremental conv stem: process new mel frames through conv0/conv1,
 * using tail buffers for boundary correctness.
 *
 * Conv1 has stride=2, so we must feed it an even number of conv0 outputs
 * to avoid right-side zero-padding that would corrupt the last output and
 * shift stride alignment for all subsequent chunks. Any odd remainder is
 * saved as conv0_residual and prepended to the next chunk.
 *
 * Returns a newly-allocated buffer of [*out_len, 1280] post-conv positions.
 * Caller must free. Returns NULL if out_len == 0.
 */
static float *stream_conv_stem(vox_stream_t *s, const float *mel_new,
                                int n_new_mel, int *out_len) {
    vox_encoder_t *enc = &s->ctx->encoder;
    int dim = VOX_ENC_DIM; /* 1280 */
    *out_len = 0;

    if (n_new_mel <= 0) return NULL;

    int is_first = 0;

    /* === Phase 1: Conv0 — produce new conv0 outputs [dim, conv0_new_len] === */
    int conv0_new_len;
    float *conv0_new; /* [dim, conv0_new_len] column-major, caller frees */

    if (!s->conv_stem_initialized) {
        is_first = 1;

        /* Transpose mel [n_new_mel, 128] -> [128, n_new_mel] */
        float *conv_in = (float *)malloc((size_t)VOX_MEL_BINS * n_new_mel * sizeof(float));
        for (int f = 0; f < n_new_mel; f++)
            for (int m = 0; m < VOX_MEL_BINS; m++)
                conv_in[m * n_new_mel + f] = mel_new[f * VOX_MEL_BINS + m];

        conv0_new_len = n_new_mel;
        conv0_new = (float *)malloc((size_t)dim * conv0_new_len * sizeof(float));
        vox_causal_conv1d(conv0_new, conv_in, enc->conv0_weight, enc->conv0_bias,
                          VOX_MEL_BINS, dim, n_new_mel, 3, 1);
        vox_gelu(conv0_new, dim * conv0_new_len);
        free(conv_in);

        /* Save last 2 mel frames (column-major [128, 2]) */
        if (!s->mel_tail) s->mel_tail = (float *)calloc((size_t)VOX_MEL_BINS * 2, sizeof(float));
        int ts = n_new_mel >= 2 ? n_new_mel - 2 : 0;
        int tc = n_new_mel >= 2 ? 2 : n_new_mel;
        memset(s->mel_tail, 0, (size_t)VOX_MEL_BINS * 2 * sizeof(float));
        for (int f = 0; f < tc; f++)
            for (int m = 0; m < VOX_MEL_BINS; m++)
                s->mel_tail[m * 2 + (2 - tc + f)] = mel_new[(ts + f) * VOX_MEL_BINS + m];

        s->conv_stem_initialized = 1;
    } else {
        /* Subsequent chunks: prepend mel_tail for conv0 boundary */
        int padded_mel_len = 2 + n_new_mel;
        float *conv_in = (float *)malloc((size_t)VOX_MEL_BINS * padded_mel_len * sizeof(float));
        for (int m = 0; m < VOX_MEL_BINS; m++) {
            conv_in[m * padded_mel_len + 0] = s->mel_tail[m * 2 + 0];
            conv_in[m * padded_mel_len + 1] = s->mel_tail[m * 2 + 1];
            for (int f = 0; f < n_new_mel; f++)
                conv_in[m * padded_mel_len + 2 + f] = mel_new[f * VOX_MEL_BINS + m];
        }

        float *conv0_full = (float *)malloc((size_t)dim * padded_mel_len * sizeof(float));
        vox_causal_conv1d(conv0_full, conv_in, enc->conv0_weight, enc->conv0_bias,
                          VOX_MEL_BINS, dim, padded_mel_len, 3, 1);
        vox_gelu(conv0_full, dim * padded_mel_len);
        free(conv_in);

        /* Discard first 2 (from overlap, contaminated by zero-pad) */
        conv0_new_len = n_new_mel;
        conv0_new = (float *)malloc((size_t)dim * conv0_new_len * sizeof(float));
        for (int d = 0; d < dim; d++)
            memcpy(conv0_new + (size_t)d * conv0_new_len,
                   conv0_full + (size_t)d * padded_mel_len + 2,
                   (size_t)conv0_new_len * sizeof(float));
        free(conv0_full);

        /* Update mel_tail */
        int ts = n_new_mel >= 2 ? n_new_mel - 2 : 0;
        int tc = n_new_mel >= 2 ? 2 : n_new_mel;
        memset(s->mel_tail, 0, (size_t)VOX_MEL_BINS * 2 * sizeof(float));
        for (int f = 0; f < tc; f++)
            for (int m = 0; m < VOX_MEL_BINS; m++)
                s->mel_tail[m * 2 + (2 - tc + f)] = mel_new[(ts + f) * VOX_MEL_BINS + m];
    }

    /* === Phase 2: Stride alignment — ensure even count for conv1 === */
    int prev_res = s->conv0_residual_count;
    int total_avail = prev_res + conv0_new_len;
    int new_res = total_avail & 1; /* 1 if odd, 0 if even */
    int feed_from_new = conv0_new_len - new_res;
    int feed_total = prev_res + feed_from_new; /* always even */

    if (feed_total <= 0) {
        /* Not enough to feed conv1 — just save residual */
        if (new_res && conv0_new_len > 0) {
            if (!s->conv0_residual)
                s->conv0_residual = (float *)malloc((size_t)dim * sizeof(float));
            for (int d = 0; d < dim; d++)
                s->conv0_residual[d] = conv0_new[(size_t)d * conv0_new_len + conv0_new_len - 1];
        }
        s->conv0_residual_count = new_res;
        free(conv0_new);
        return NULL;
    }

    /* Build feed buffer [dim, feed_total] column-major */
    float *feed = (float *)malloc((size_t)dim * feed_total * sizeof(float));
    int fpos = 0;

    /* Copy old residual first (before overwriting with new) */
    if (prev_res == 1) {
        for (int d = 0; d < dim; d++)
            feed[(size_t)d * feed_total + 0] = s->conv0_residual[d];
        fpos = 1;
    }

    /* Copy feed_from_new values from conv0_new */
    for (int d = 0; d < dim; d++)
        memcpy(feed + (size_t)d * feed_total + fpos,
               conv0_new + (size_t)d * conv0_new_len,
               (size_t)feed_from_new * sizeof(float));

    /* Save new residual (last value of conv0_new) if total was odd */
    if (new_res) {
        if (!s->conv0_residual)
            s->conv0_residual = (float *)malloc((size_t)dim * sizeof(float));
        for (int d = 0; d < dim; d++)
            s->conv0_residual[d] = conv0_new[(size_t)d * conv0_new_len + conv0_new_len - 1];
    }
    s->conv0_residual_count = new_res;
    free(conv0_new);

    /* === Phase 3: Conv1 === */
    float *conv1_in;
    int conv1_in_len;
    int conv1_discard; /* outputs to discard at front */

    if (is_first) {
        /* First chunk: zero left-pad is correct for start of sequence */
        conv1_in = feed; /* aliased, freed after conv1 */
        conv1_in_len = feed_total;
        conv1_discard = 0;
    } else {
        /* Subsequent: prepend conv0_tail (2 frames) for boundary context */
        conv1_in_len = 2 + feed_total;
        conv1_in = (float *)malloc((size_t)dim * conv1_in_len * sizeof(float));
        for (int d = 0; d < dim; d++) {
            conv1_in[(size_t)d * conv1_in_len + 0] = s->conv0_tail[d * 2 + 0];
            conv1_in[(size_t)d * conv1_in_len + 1] = s->conv0_tail[d * 2 + 1];
            memcpy(conv1_in + (size_t)d * conv1_in_len + 2,
                   feed + (size_t)d * feed_total,
                   (size_t)feed_total * sizeof(float));
        }
        conv1_discard = 1;
    }

    /* Update conv0_tail from last 2 of feed (before freeing feed) */
    if (!s->conv0_tail) s->conv0_tail = (float *)calloc((size_t)dim * 2, sizeof(float));
    for (int d = 0; d < dim; d++) {
        s->conv0_tail[d * 2 + 0] = feed[(size_t)d * feed_total + feed_total - 2];
        s->conv0_tail[d * 2 + 1] = feed[(size_t)d * feed_total + feed_total - 1];
    }
    if (!is_first) free(feed);

    /* conv1_in_len is always even → conv1 output count = conv1_in_len / 2 */
    int conv1_out_len = conv1_in_len / 2;
    float *conv1_out = (float *)malloc((size_t)dim * conv1_out_len * sizeof(float));
    vox_causal_conv1d(conv1_out, conv1_in, enc->conv1_weight, enc->conv1_bias,
                      dim, dim, conv1_in_len, 3, 2);
    vox_gelu(conv1_out, dim * conv1_out_len);
    if (is_first) free(feed); /* was aliased to conv1_in */
    else free(conv1_in);

    int result_len = conv1_out_len - conv1_discard;
    if (result_len <= 0) {
        free(conv1_out);
        return NULL;
    }

    /* Transpose [dim, result_len] -> [result_len, dim] (row-major) */
    float *result = (float *)malloc((size_t)result_len * dim * sizeof(float));
    for (int si = 0; si < result_len; si++)
        for (int d = 0; d < dim; d++)
            result[(size_t)si * dim + d] = conv1_out[(size_t)d * conv1_out_len + conv1_discard + si];
    free(conv1_out);

    *out_len = result_len;
    return result;
}

/* Compact adapter buffer: discard tokens the decoder has already consumed */
static void stream_adapter_compact(vox_stream_t *s) {
    int consumed = s->gen_pos - s->adapter_pos_offset;
    if (consumed <= 0) return;

    int dim = VOX_DEC_DIM;
    int remaining = (s->total_adapter - s->adapter_pos_offset) - consumed;

    if (remaining > 0)
        memmove(s->adapter_buf,
                s->adapter_buf + (size_t)consumed * dim,
                (size_t)remaining * dim * sizeof(float));

    s->adapter_pos_offset += consumed;
}

/* Reset decoder-side streaming state (hard reset: drop adapter backlog/context). */
static void stream_reset_decoder_state(vox_stream_t *s) {
    if (!s) return;

    s->ctx->kv_cache_len = 0;
    s->ctx->kv_pos_offset = 0;

    s->total_adapter = 0;
    s->adapter_pos_offset = 0;
    s->gen_pos = 0;
    s->decoder_started = 0;
    s->prev_token = TOKEN_BOS;
    s->eos_seen = 0;
    s->n_generated = 0;
    s->nontext_streak = 0;
    s->text_since_restart = 0;
    s->waiting_prompt = 0;
}

/* Reset full live-stream state (mel/conv/encoder/decoder). */
static int stream_reset_full_state(vox_stream_t *s) {
    if (!s) return -1;

    vox_mel_ctx_t *new_mel = vox_mel_ctx_init(32 * RAW_AUDIO_LENGTH_PER_TOK);
    if (!new_mel) return -1;

    vox_mel_free(s->mel_ctx);
    s->mel_ctx = new_mel;
    s->mel_cursor = 0;

    s->conv_stem_initialized = 0;
    s->conv0_residual_count = 0;
    s->enc_residual_count = 0;
    if (s->mel_tail)
        memset(s->mel_tail, 0, (size_t)VOX_MEL_BINS * 2 * sizeof(float));
    if (s->conv0_tail)
        memset(s->conv0_tail, 0, (size_t)VOX_ENC_DIM * 2 * sizeof(float));
    if (s->conv0_residual)
        memset(s->conv0_residual, 0, (size_t)VOX_ENC_DIM * sizeof(float));
    if (s->enc_residual)
        memset(s->enc_residual, 0, (size_t)3 * VOX_ENC_DIM * sizeof(float));

    s->ctx->enc_kv_cache_len = 0;
    s->ctx->enc_kv_pos_offset = 0;

    stream_reset_decoder_state(s);
    return 0;
}

/* Run encoder incrementally on available mel, append adapter tokens */
static void stream_run_encoder(vox_stream_t *s) {
    int mel_frames = 0;
    int mel_offset = vox_mel_frame_offset(s->mel_ctx);
    float *mel_data = vox_mel_data(s->mel_ctx, &mel_frames);
    int total_mel = mel_offset + mel_frames;
    int dim = VOX_DEC_DIM;

    if (s->mel_cursor < mel_offset) s->mel_cursor = mel_offset;
    int mel_start = s->mel_cursor - mel_offset;
    int new_mel = total_mel - s->mel_cursor;
    int need_mel = (!s->conv_stem_initialized) ? STREAM_FIRST_CHUNK_MIN_MEL : s->min_new_mel;

    if (new_mel < need_mel && !s->finished) return;
    if (new_mel <= 0) return;

    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

    /* 1. Run incremental conv stem on new mel -> post-conv positions */
    int conv_out_len = 0;
    float *conv_out = stream_conv_stem(s, mel_data + (size_t)mel_start * VOX_MEL_BINS,
                                        new_mel, &conv_out_len);
    s->mel_cursor = total_mel;

    if (!conv_out || conv_out_len <= 0) {
        free(conv_out);
        vox_mel_discard_before(s->mel_ctx, s->mel_cursor);
        return;
    }

    /* 2. Run incremental encoder transformer with KV cache */
    int enc_out_len = 0;
    float *enc_out = vox_encoder_forward_incremental(s->ctx, conv_out, conv_out_len, &enc_out_len);
    free(conv_out);

    if (!enc_out || enc_out_len <= 0) {
        free(enc_out);
        vox_mel_discard_before(s->mel_ctx, s->mel_cursor);
        return;
    }

    /* 3. Combine with residual, align to 4x for downsample */
    int total_enc = s->enc_residual_count + enc_out_len;
    int usable = (total_enc / VOX_DOWNSAMPLE) * VOX_DOWNSAMPLE;
    int leftover = total_enc - usable;

    if (usable > 0) {
        /* Build a combined buffer: residual + new encoder output */
        float *combined = (float *)malloc((size_t)usable * VOX_ENC_DIM * sizeof(float));
        int pos = 0;

        /* Copy residual positions first */
        if (s->enc_residual_count > 0 && s->enc_residual) {
            int from_residual = s->enc_residual_count;
            if (from_residual > usable) from_residual = usable;
            memcpy(combined, s->enc_residual, (size_t)from_residual * VOX_ENC_DIM * sizeof(float));
            pos = from_residual;
        }

        /* Copy from new encoder output */
        int from_enc = usable - pos;
        if (from_enc > 0) {
            memcpy(combined + (size_t)pos * VOX_ENC_DIM,
                   enc_out, (size_t)from_enc * VOX_ENC_DIM * sizeof(float));
        }

        /* Run adapter on usable positions */
        int chunk_tokens = 0;
        float *adapter_chunk = vox_adapter_forward(s->ctx, combined, usable, &chunk_tokens);
        free(combined);

        if (adapter_chunk && chunk_tokens > 0) {
            /* Append to adapter buffer (physical offsets account for compaction) */
            int phys_len = s->total_adapter - s->adapter_pos_offset;
            if (phys_len + chunk_tokens > s->adapter_cap) {
                int new_cap = s->adapter_cap ? s->adapter_cap * 2 : 256;
                while (new_cap < phys_len + chunk_tokens) new_cap *= 2;
                float *tmp = (float *)realloc(s->adapter_buf,
                    (size_t)new_cap * dim * sizeof(float));
                if (!tmp) {
                    free(adapter_chunk);
                    free(enc_out);
                    vox_mel_discard_before(s->mel_ctx, s->mel_cursor);
                    return;
                }
                s->adapter_buf = tmp;
                s->adapter_cap = new_cap;
            }
            memcpy(s->adapter_buf + (size_t)phys_len * dim,
                   adapter_chunk, (size_t)chunk_tokens * dim * sizeof(float));
            s->total_adapter += chunk_tokens;
            free(adapter_chunk);
        } else {
            free(adapter_chunk);
        }
    }

    /* 4. Save leftover encoder positions to residual */
    if (leftover > 0) {
        if (!s->enc_residual)
            s->enc_residual = (float *)malloc(3 * VOX_ENC_DIM * sizeof(float));
        /* The leftover comes from the end of enc_out (after what we used from enc_out) */
        int enc_used = usable - s->enc_residual_count;
        if (enc_used < 0) enc_used = 0;
        memcpy(s->enc_residual, enc_out + (size_t)enc_used * VOX_ENC_DIM,
               (size_t)leftover * VOX_ENC_DIM * sizeof(float));
    }
    s->enc_residual_count = leftover;

    free(enc_out);

    gettimeofday(&t1, NULL);
    s->encoder_ms += (t1.tv_sec - t0.tv_sec) * 1000.0 +
                     (t1.tv_usec - t0.tv_usec) / 1000.0;

    if (vox_monitor) {
        fprintf(stderr, "\xe2\x96\xb6");  /* ▶ = encoder chunk */
        fflush(stderr);
    }
    if (vox_verbose >= 2)
        fprintf(stderr, "  Encoder inc: %d mel -> %d conv -> %d usable (total adapter: %d, residual: %d)\n",
                new_mel, conv_out_len, usable, s->total_adapter, leftover);

    vox_mel_discard_before(s->mel_ctx, s->mel_cursor);
}

/* Build alternatives array from logits. alts[0]=best (already decoded as best_token).
 * Fills alts[1..] with qualifying alternatives. */
static void stream_fill_alts(vox_stream_t *s, int best_token,
                              const char *alts[VOX_MAX_ALT]) {
    memset(alts, 0, VOX_MAX_ALT * sizeof(const char *));
    alts[0] = vox_tokenizer_decode(s->tokenizer, best_token);

    if (s->n_alt <= 1) return;

    /* Softmax over logits to get probabilities */
    float *logits = s->logits;
    float max_val = logits[0];
    for (int i = 1; i < VOX_VOCAB_SIZE; i++)
        if (logits[i] > max_val) max_val = logits[i];

    float sum = 0;
    for (int i = 0; i < VOX_VOCAB_SIZE; i++) {
        logits[i] = expf(logits[i] - max_val);
        sum += logits[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < VOX_VOCAB_SIZE; i++)
        logits[i] *= inv_sum;

    float best_prob = logits[best_token];
    if (best_prob <= 0) return;

    /* Find top alternatives by repeated scan (n_alt is small, <=4) */
    int found = 1; /* alts[0] already set */
    int used[VOX_MAX_ALT];
    used[0] = best_token;

    while (found < s->n_alt) {
        int best_idx = -1;
        float best_p = -1;
        for (int i = TOKEN_TEXT_MIN; i < VOX_VOCAB_SIZE; i++) {
            if (i == best_token) continue;
            /* Skip already-picked */
            int skip = 0;
            for (int j = 1; j < found; j++)
                if (used[j] == i) { skip = 1; break; }
            if (skip) continue;

            if (logits[i] > best_p) {
                best_p = logits[i];
                best_idx = i;
            }
        }
        if (best_idx < 0) break;

        float r = 1.0f - best_p / best_prob;
        if (r > s->alt_cutoff) break;

        used[found] = best_idx;
        alts[found] = vox_tokenizer_decode(s->tokenizer, best_idx);
        found++;
    }
}

/* Run decoder: prefill if needed, then generate tokens while adapter available */
static void stream_run_decoder(vox_stream_t *s) {
    struct timeval t0, t1;
    int dim = VOX_DEC_DIM;
    int prompt_len = 1 + 32 + s->ctx->delay_tokens;
    uint16_t *tok_emb_bf16 = s->ctx->decoder.tok_embeddings_bf16;

    /* Prefill when current adapter can cover the prompt. */
    int cur_adapter = s->total_adapter - s->adapter_pos_offset;
    if (!s->decoder_started && cur_adapter < prompt_len) {
        if (vox_monitor && !s->waiting_prompt) {
            /* ⌛ = waiting for enough adapter tokens to prefill prompt */
            fprintf(stderr, "\xe2\x8c\x9b");
            fflush(stderr);
            s->waiting_prompt = 1;
        }
        return;
    }
    if (!s->decoder_started && cur_adapter >= prompt_len) {
        s->waiting_prompt = 0;
        gettimeofday(&t0, NULL);

        float *prompt_embeds = (float *)malloc((size_t)prompt_len * dim * sizeof(float));
        if (!prompt_embeds) return;

        for (int i = 0; i < prompt_len; i++) {
            int tok = (i == 0) ? TOKEN_BOS : TOKEN_STREAMING_PAD;
            tok_embed_bf16_to_f32(s->tok_tmp, tok_emb_bf16, tok, dim);
            const float *a = s->adapter_buf + (size_t)i * dim;
            float *dst = prompt_embeds + (size_t)i * dim;
            for (int j = 0; j < dim; j++) dst[j] = a[j] + s->tok_tmp[j];
        }

        s->ctx->kv_cache_len = 0;
        s->ctx->kv_pos_offset = 0;
        /* Keep KV cache allocated — vox_decoder_prefill will reuse or grow it */

        int prefill_count = prompt_len - 1;
        vox_decoder_prefill(s->ctx, prompt_embeds, prefill_count);

        memcpy(s->step_embed, prompt_embeds + (size_t)prefill_count * dim,
               (size_t)dim * sizeof(float));
        free(prompt_embeds);

        s->prev_token = vox_decoder_forward(s->ctx, s->step_embed, s->logits);
        s->n_generated++;
        s->last_decode_sample = s->real_samples_fed;

        /* Enqueue only if this token decodes to visible text */
        stream_tok_class_t cls = stream_classify_token(s, s->prev_token);
        if (cls == STREAM_TOK_TEXT) {
            const char *alts[VOX_MAX_ALT];
            stream_fill_alts(s, s->prev_token, alts);
            if (alts[0]) {
                stream_enqueue_token(s, alts);
                s->n_text_tokens++;
                s->text_since_restart = 1;
                s->empty_restarts = 0;
            }
            s->nontext_streak = 0;
        } else if (cls != STREAM_TOK_EOS) {
            s->nontext_streak++;
        }
        if (s->prev_token == TOKEN_EOS) s->eos_seen = 1;

        s->gen_pos = s->adapter_pos_offset + prompt_len;
        s->decoder_started = 1;

        gettimeofday(&t1, NULL);
        double pf_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                       (t1.tv_usec - t0.tv_usec) / 1000.0;
        s->decoder_ms += pf_ms;
        s->prefill_ms += pf_ms;

        if (vox_monitor) {
            fprintf(stderr, "\xc2\xb7");  /* · = prefill */
            fflush(stderr);
        }
    }

    /* Generate tokens while adapter tokens are available */
    if (s->decoder_started && !s->eos_seen) {
        gettimeofday(&t0, NULL);
        int gen_before = s->n_generated;
        int text_steps = 0;
        int control_steps = 0;
        int invalid_steps = 0;
        int eos_step = 0;
        while (s->gen_pos < s->total_adapter) {
            tok_embed_bf16_to_f32(s->tok_tmp, tok_emb_bf16, s->prev_token, dim);
            int phys_pos = s->gen_pos - s->adapter_pos_offset;
            const float *a = s->adapter_buf + (size_t)phys_pos * dim;
            for (int j = 0; j < dim; j++)
                s->step_embed[j] = a[j] + s->tok_tmp[j];

            s->prev_token = vox_decoder_forward(s->ctx, s->step_embed, s->logits);
            s->n_generated++;
            s->last_decode_sample = s->real_samples_fed;

            stream_tok_class_t cls = stream_classify_token(s, s->prev_token);
            if (cls == STREAM_TOK_TEXT) {
                const char *alts[VOX_MAX_ALT];
                stream_fill_alts(s, s->prev_token, alts);
                if (alts[0]) {
                    stream_enqueue_token(s, alts);
                    s->n_text_tokens++;
                    s->text_since_restart = 1;
                    s->empty_restarts = 0;
                }
                s->nontext_streak = 0;
                text_steps++;
            } else if (cls == STREAM_TOK_CONTROL) {
                s->nontext_streak++;
                control_steps++;
            } else if (cls == STREAM_TOK_INVALID) {
                s->nontext_streak++;
                invalid_steps++;
            }

            s->gen_pos++;
            if (s->prev_token == TOKEN_EOS) {
                s->eos_seen = 1;
                eos_step = 1;
                break;
            }
        }
        if (s->n_generated > gen_before) {
            gettimeofday(&t1, NULL);
            double dec_ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                            (t1.tv_usec - t0.tv_usec) / 1000.0;
            s->decoder_ms += dec_ms;
            if (vox_monitor) {
                int steps = s->n_generated - gen_before;
                int slow = (dec_ms / steps > 40);
                const char *sym;
                const char *sev = "";
                if (text_steps > 0) {
                    /* ▪ = decode chunk with text, ▸ = slow text decode */
                    sym = slow ? "\xe2\x96\xb8" : "\xe2\x96\xaa";
                } else if (invalid_steps > 0) {
                    /* ✗ = text-range token(s) with empty/invalid decode */
                    sym = slow ? "\xe2\x9c\x98" : "\xe2\x9c\x97";
                } else if (control_steps > 0) {
                    /* ▫ = control chunk (token<1000), ▹ = slow control chunk */
                    sym = slow ? "\xe2\x96\xb9" : "\xe2\x96\xab";
                } else if (eos_step) {
                    /* ◦ = EOS-only step */
                    sym = "\xe2\x97\xa6";
                } else {
                    sym = "\xe2\x96\xaa";
                }
                if (text_steps == 0 && (control_steps > 0 || invalid_steps > 0)) {
                    if (s->nontext_streak >= STREAM_MAX_NON_TEXT_STREAK - 8) {
                        /* ☠ = critical non-text streak (restart imminent) */
                        sev = "\xe2\x98\xa0";
                    } else if (s->nontext_streak >= STREAM_MAX_NON_TEXT_STREAK / 2) {
                        /* ⚠ = elevated non-text streak */
                        sev = "\xe2\x9a\xa0";
                    }
                }
                fprintf(stderr, "%s%s", sym, sev);
                fflush(stderr);
            }
        }
    }

    /* Reclaim adapter buffer space for tokens the decoder has consumed */
    stream_adapter_compact(s);

    /* Restart decoder on live (non-finished) streams when:
     * - EOS: model decided the segment is done
     * - KV cache too large: bounds attention cost to keep real-time pace
     * - non-text/invalid streak: catches decoder loops on control tokens
     * - no decoder progress despite advancing audio: catches malformed state */
    int need_restart = 0;
    int full_reset = 0;
    if (s->continuous) {
        if (s->eos_seen)
            need_restart = 1;
        else if (s->decoder_started &&
                 s->ctx->kv_cache_len > STREAM_MAX_DECODE_KV)
            need_restart = 2;
        else if (s->decoder_started &&
                 s->nontext_streak >= STREAM_MAX_NON_TEXT_STREAK)
            need_restart = 3;
        else if (!s->finished &&
                 (s->real_samples_fed - s->last_decode_sample) >=
                 STREAM_MAX_NO_DECODE_SAMPLES)
            need_restart = 4;
    }
    if (need_restart) {
        if (s->text_since_restart) s->empty_restarts = 0;
        else s->empty_restarts++;
        if (need_restart >= 2 ||
            s->empty_restarts >= STREAM_EMPTY_RESTARTS_FOR_FULL_RESET)
            full_reset = 1;

        if (vox_monitor) {
            /* ↺ = restart on EOS, ⟳ = restart on KV overflow,
             * ↯ = non-text stall, ⌚ = no-decode watchdog,
             * ✂ = decoder hard reset, ♻ = full stream reset */
            const char *sym = (need_restart == 1) ? "\xe2\x86\xba" :
                              (need_restart == 2) ? "\xe2\x9f\xb3" :
                              (need_restart == 3) ? "\xe2\x86\xaf" :
                                                   "\xe2\x8c\x9a";
            fprintf(stderr, "%s%s", sym,
                    full_reset ? "\xe2\x99\xbb" : "\xe2\x9c\x82");
            fflush(stderr);
        }
        if (full_reset) {
            if (stream_reset_full_state(s) != 0) {
                /* If mel reinit fails, keep live path running with decoder reset. */
                stream_reset_decoder_state(s);
            }
            s->empty_restarts = 0;
        } else {
            stream_reset_decoder_state(s);
        }
        s->last_decode_sample = s->real_samples_fed;
    }
}

vox_stream_t *vox_stream_init(vox_ctx_t *ctx) {
    vox_stream_t *s = (vox_stream_t *)calloc(1, sizeof(vox_stream_t));
    if (!s) return NULL;

    s->ctx = ctx;

    /* Load tokenizer */
    char tok_path[1024];
    snprintf(tok_path, sizeof(tok_path), "%s/tekken.json", ctx->model_dir);
    s->tokenizer = vox_tokenizer_load(tok_path);
    if (!s->tokenizer) { free(s); return NULL; }

    /* Initialize incremental mel with 32 left-pad tokens of silence */
    s->mel_ctx = vox_mel_ctx_init(32 * RAW_AUDIO_LENGTH_PER_TOK);
    if (!s->mel_ctx) {
        vox_tokenizer_free(s->tokenizer);
        free(s);
        return NULL;
    }

    /* Token queue (VOX_MAX_ALT strings per position) */
    s->queue_cap = 256;
    s->token_queue = (const char **)calloc((size_t)s->queue_cap * VOX_MAX_ALT, sizeof(const char *));
    s->n_alt = 1;

    /* Decoder working buffers */
    int dim = VOX_DEC_DIM;
    s->logits = (float *)malloc(VOX_VOCAB_SIZE * sizeof(float));
    s->step_embed = (float *)malloc(dim * sizeof(float));
    s->tok_tmp = (float *)malloc(dim * sizeof(float));

    if (!s->token_queue || !s->logits || !s->step_embed || !s->tok_tmp) {
        vox_stream_free(s);
        return NULL;
    }

    /* Reset encoder KV cache for new transcription */
    ctx->enc_kv_cache_len = 0;
    ctx->enc_kv_pos_offset = 0;

    /* Default processing interval: 2 seconds (200 mel frames) */
    s->min_new_mel = (int)(STREAM_DEFAULT_INTERVAL * 100.0f);

    return s;
}

int vox_stream_feed(vox_stream_t *s, const float *samples, int n_samples) {
    if (!s || s->finished || n_samples <= 0) return -1;

    vox_mel_feed(s->mel_ctx, samples, n_samples);
    s->real_samples_fed += n_samples;

    stream_run_encoder(s);
    stream_run_decoder(s);
    return 0;
}

int vox_stream_finish(vox_stream_t *s) {
    if (!s || s->finished) return -1;

    /* Flush with right padding (shared with vox_stream_flush) */
    vox_stream_flush(s);

    s->finished = 1;
    vox_mel_finish(s->mel_ctx, 0);

    if (vox_verbose >= 2)
        fprintf(stderr, "Stream finished: %lld real samples (%.1f sec)\n",
                (long long)s->real_samples_fed,
                (double)s->real_samples_fed / VOX_SAMPLE_RATE);

    /* Final pass after mel finalization */
    stream_run_encoder(s);
    stream_run_decoder(s);
    return 0;
}

int vox_stream_get(vox_stream_t *s, const char **out_tokens, int max) {
    if (!s || max <= 0) return 0;
    int count = 0;
    while (count < max && s->queue_head != s->queue_tail) {
        out_tokens[count++] = s->token_queue[s->queue_head * VOX_MAX_ALT];
        s->queue_head = (s->queue_head + 1) % s->queue_cap;
    }
    return count;
}

void vox_stream_set_alt(vox_stream_t *s, int n_alt, float cutoff) {
    if (!s) return;
    if (n_alt < 1) n_alt = 1;
    if (n_alt > VOX_MAX_ALT) n_alt = VOX_MAX_ALT;
    if (cutoff < 0) cutoff = 0;
    if (cutoff > 1) cutoff = 1;
    s->n_alt = n_alt;
    s->alt_cutoff = cutoff;
}

int vox_stream_get_alt(vox_stream_t *s, const char **out_tokens,
                       int max_tokens, int n_alt) {
    if (!s || max_tokens <= 0 || n_alt <= 0) return 0;
    if (n_alt > VOX_MAX_ALT) n_alt = VOX_MAX_ALT;
    int count = 0;
    while (count < max_tokens && s->queue_head != s->queue_tail) {
        const char **src = &s->token_queue[s->queue_head * VOX_MAX_ALT];
        const char **dst = &out_tokens[count * n_alt];
        for (int a = 0; a < n_alt; a++)
            dst[a] = src[a];
        count++;
        s->queue_head = (s->queue_head + 1) % s->queue_cap;
    }
    return count;
}

void vox_stream_free(vox_stream_t *s) {
    if (!s) return;

    /* Print stats after caller has drained all tokens */
    if (vox_verbose >= 1) {
        fprintf(stderr, "Encoder: %d mel -> %d tokens (%.0f ms)\n",
                s->mel_cursor, s->total_adapter, s->encoder_ms);
        if (s->n_text_tokens > 0) {
            double gen_ms = s->decoder_ms - s->prefill_ms;
            fprintf(stderr, "Decoder: %d text tokens (%d steps) in %.0f ms "
                    "(prefill %.0f ms + %.1f ms/step)\n",
                    s->n_text_tokens, s->n_generated, s->decoder_ms,
                    s->prefill_ms,
                    s->n_generated > 1 ? gen_ms / (s->n_generated - 1) : 0);
        }
    }

    vox_mel_free(s->mel_ctx);
    if (s->tokenizer) vox_tokenizer_free(s->tokenizer);
    free(s->adapter_buf);
    free(s->token_queue);
    free(s->logits);
    free(s->step_embed);
    free(s->tok_tmp);
    free(s->mel_tail);
    free(s->conv0_tail);
    free(s->conv0_residual);
    free(s->enc_residual);
    free(s);
}

/* ========================================================================
 * Convenience Functions (built on streaming API)
 * ======================================================================== */

char *vox_transcribe_audio(vox_ctx_t *ctx, const float *samples, int n_samples) {
    vox_stream_t *s = vox_stream_init(ctx);
    if (!s) return NULL;

    vox_stream_feed(s, samples, n_samples);
    vox_stream_finish(s);

    /* Collect all tokens into a string */
    size_t text_cap = 1024;
    size_t text_len = 0;
    char *text = (char *)malloc(text_cap);
    text[0] = '\0';

    const char *tokens[64];
    int n;
    while ((n = vox_stream_get(s, tokens, 64)) > 0) {
        for (int i = 0; i < n; i++) {
            size_t piece_len = strlen(tokens[i]);
            if (text_len + piece_len + 1 > text_cap) {
                while (text_len + piece_len + 1 > text_cap) text_cap *= 2;
                text = (char *)realloc(text, text_cap);
            }
            memcpy(text + text_len, tokens[i], piece_len);
            text_len += piece_len;
            text[text_len] = '\0';
        }
    }

    vox_stream_free(s);
    trim_ascii_whitespace(text);
    return text;
}

char *vox_transcribe_stdin(vox_ctx_t *ctx) {
    /* Read first 4 bytes to detect format */
    uint8_t header[4];
    size_t hdr_read = fread(header, 1, 4, stdin);
    if (hdr_read < 4) {
        fprintf(stderr, "vox_transcribe_stdin: not enough data on stdin\n");
        return NULL;
    }

    /* WAV detected: buffer all, parse, transcribe as offline */
    if (memcmp(header, "RIFF", 4) == 0) {
        if (vox_verbose >= 2)
            fprintf(stderr, "Detected WAV format on stdin\n");

        size_t capacity = 1024 * 1024;
        size_t size = 4;
        uint8_t *buf = (uint8_t *)malloc(capacity);
        if (!buf) return NULL;
        memcpy(buf, header, 4);
        while (1) {
            if (size == capacity) {
                capacity *= 2;
                uint8_t *tmp = (uint8_t *)realloc(buf, capacity);
                if (!tmp) { free(buf); return NULL; }
                buf = tmp;
            }
            size_t n = fread(buf + size, 1, capacity - size, stdin);
            if (n == 0) break;
            size += n;
        }
        if (vox_verbose >= 2)
            fprintf(stderr, "Read %zu bytes from stdin\n", size);

        if (size < 44 || memcmp(buf + 8, "WAVE", 4) != 0) {
            fprintf(stderr, "Invalid WAV data on stdin\n");
            free(buf);
            return NULL;
        }

        int channels = 0, sample_rate = 0, bits_per_sample = 0, audio_format = 0;
        const uint8_t *pcm_data = NULL;
        int pcm_size = 0;
        const uint8_t *p = buf + 12;
        const uint8_t *end = buf + size;
        while (p + 8 <= end) {
            uint32_t chunk_size = (uint32_t)p[4] | ((uint32_t)p[5]<<8) |
                                  ((uint32_t)p[6]<<16) | ((uint32_t)p[7]<<24);
            if (p + 8 + chunk_size > end) break;
            if (memcmp(p, "fmt ", 4) == 0 && chunk_size >= 16) {
                audio_format = p[8] | (p[9]<<8);
                channels = p[10] | (p[11]<<8);
                sample_rate = (int)((uint32_t)p[12] | ((uint32_t)p[13]<<8) |
                              ((uint32_t)p[14]<<16) | ((uint32_t)p[15]<<24));
                bits_per_sample = p[22] | (p[23]<<8);
            } else if (memcmp(p, "data", 4) == 0) {
                pcm_data = p + 8;
                pcm_size = (int)chunk_size;
                if (pcm_data + pcm_size > end) pcm_size = (int)(end - pcm_data);
            }
            p += 8 + chunk_size;
            if (chunk_size & 1) p++;
        }

        if (audio_format != 1 || bits_per_sample != 16 || !pcm_data || channels < 1) {
            fprintf(stderr, "Unsupported WAV format on stdin\n");
            free(buf);
            return NULL;
        }

        int n_frames = pcm_size / (channels * 2);
        float *samples = (float *)malloc((size_t)n_frames * sizeof(float));
        if (!samples) { free(buf); return NULL; }
        const int16_t *src = (const int16_t *)pcm_data;
        for (int i = 0; i < n_frames; i++) {
            if (channels == 1) {
                samples[i] = src[i] / 32768.0f;
            } else {
                float sum = 0;
                for (int c = 0; c < channels; c++) {
                    int16_t val;
                    memcpy(&val, &src[i * channels + c], sizeof(int16_t));
                    sum += val;
                }
                samples[i] = (sum / channels) / 32768.0f;
            }
        }
        free(buf);

        if (sample_rate != VOX_SAMPLE_RATE) {
            int new_n = (int)((long long)n_frames * VOX_SAMPLE_RATE / sample_rate);
            float *resampled = (float *)malloc((size_t)new_n * sizeof(float));
            if (!resampled) { free(samples); return NULL; }
            for (int i = 0; i < new_n; i++) {
                float src_pos = (float)i * sample_rate / VOX_SAMPLE_RATE;
                int idx = (int)src_pos;
                float frac = src_pos - idx;
                if (idx + 1 < n_frames)
                    resampled[i] = samples[idx] * (1-frac) + samples[idx+1] * frac;
                else
                    resampled[i] = (idx < n_frames) ? samples[idx] : 0.0f;
            }
            free(samples);
            samples = resampled;
            n_frames = new_n;
        }

        if (vox_verbose >= 1)
            fprintf(stderr, "Audio: %d samples (%.1f seconds)\n",
                    n_frames, (float)n_frames / VOX_SAMPLE_RATE);

        /* Use stream API so tokens are emitted incrementally */
        vox_stream_t *s = vox_stream_init(ctx);
        if (!s) { free(samples); return NULL; }
        vox_stream_feed(s, samples, n_frames);
        vox_stream_finish(s);
        free(samples);

        /* Collect and optionally stream tokens */
        size_t text_cap = 1024;
        size_t text_len = 0;
        char *text = (char *)malloc(text_cap);
        text[0] = '\0';

        const char *tokens[64];
        int n;
        while ((n = vox_stream_get(s, tokens, 64)) > 0) {
            for (int i = 0; i < n; i++) {
                size_t piece_len = strlen(tokens[i]);
                if (text_len + piece_len + 1 > text_cap) {
                    while (text_len + piece_len + 1 > text_cap) text_cap *= 2;
                    text = (char *)realloc(text, text_cap);
                }
                memcpy(text + text_len, tokens[i], piece_len);
                text_len += piece_len;
                text[text_len] = '\0';
            }
        }
        vox_stream_free(s);
        trim_ascii_whitespace(text);
        return text;
    }

    /* Raw s16le streaming mode */
    if (vox_verbose >= 2)
        fprintf(stderr, "Streaming raw s16le 16kHz mono from stdin\n");

    vox_stream_t *s = vox_stream_init(ctx);
    if (!s) return NULL;

    /* Feed the 4 peeked header bytes as 2 s16le samples */
    {
        int16_t sv[2];
        memcpy(sv, header, 4);
        float f[2] = { sv[0] / 32768.0f, sv[1] / 32768.0f };
        vox_stream_feed(s, f, 2);
    }

    /* Collect text for non-streaming mode, or emit for streaming */
    size_t text_cap = 1024;
    size_t text_len = 0;
    char *text = (char *)malloc(text_cap);
    text[0] = '\0';

    /* Read loop */
    int16_t raw_buf[4096];
    float fbuf[4096];
    const char *tokens[64];
    int eof_reached = 0;

    while (!eof_reached) {
        size_t nread = fread(raw_buf, sizeof(int16_t), 4096, stdin);
        if (nread == 0) {
            eof_reached = 1;
            vox_stream_finish(s);
        } else {
            for (size_t i = 0; i < nread; i++)
                fbuf[i] = raw_buf[i] / 32768.0f;
            vox_stream_feed(s, fbuf, (int)nread);
        }

        /* Drain pending tokens */
        int n;
        while ((n = vox_stream_get(s, tokens, 64)) > 0) {
            for (int i = 0; i < n; i++) {
                size_t piece_len = strlen(tokens[i]);
                if (text_len + piece_len + 1 > text_cap) {
                    while (text_len + piece_len + 1 > text_cap) text_cap *= 2;
                    text = (char *)realloc(text, text_cap);
                }
                memcpy(text + text_len, tokens[i], piece_len);
                text_len += piece_len;
                text[text_len] = '\0';
            }
        }
    }

    vox_stream_free(s);
    trim_ascii_whitespace(text);
    return text;
}

char *vox_transcribe(vox_ctx_t *ctx, const char *wav_path) {
    int n_samples = 0;
    float *samples = vox_load_wav(wav_path, &n_samples);
    if (!samples) {
        fprintf(stderr, "vox_transcribe: cannot load %s\n", wav_path);
        return NULL;
    }
    if (vox_verbose >= 1)
        fprintf(stderr, "Audio: %d samples (%.1f seconds)\n",
                n_samples, (float)n_samples / VOX_SAMPLE_RATE);

    char *text = vox_transcribe_audio(ctx, samples, n_samples);
    free(samples);
    return text;
}

int vox_stream_flush(vox_stream_t *s) {
    if (!s || s->finished) return -1;

    /* Feed the same right padding that finish() uses, so the decoder can
     * push out tokens that are behind the delay window. */
    int n_delay_tokens = s->ctx->delay_tokens;
    int align_pad = (int)((RAW_AUDIO_LENGTH_PER_TOK -
        (s->real_samples_fed % RAW_AUDIO_LENGTH_PER_TOK)) % RAW_AUDIO_LENGTH_PER_TOK);
    int n_right_pad_tokens = (n_delay_tokens + 1) + OFFLINE_STREAMING_BUFFER_TOKENS;
    int right_pad = align_pad + n_right_pad_tokens * RAW_AUDIO_LENGTH_PER_TOK;

    float zero_buf[4096];
    memset(zero_buf, 0, sizeof(zero_buf));
    int remaining = right_pad;
    while (remaining > 0) {
        int chunk = remaining > 4096 ? 4096 : remaining;
        vox_mel_feed(s->mel_ctx, zero_buf, chunk);
        remaining -= chunk;
    }

    /* Force encoder to process all buffered mel, then run decoder */
    int saved = s->min_new_mel;
    s->min_new_mel = 1;
    stream_run_encoder(s);
    stream_run_decoder(s);
    s->min_new_mel = saved;
    return 0;
}

void vox_set_processing_interval(vox_stream_t *s, float seconds) {
    if (!s) return;
    if (seconds <= 0) seconds = 0;
    /* mel rate = sample_rate / hop_length = 16000/160 = 100 fps */
    s->min_new_mel = (int)(seconds * 100.0f);
    if (s->min_new_mel < 1) s->min_new_mel = 1;
}

void vox_stream_set_continuous(vox_stream_t *s, int enable) {
    if (s) s->continuous = enable;
}

void vox_set_delay(vox_ctx_t *ctx, int delay_ms) {
    /* Each token represents 80ms (frame_rate=12.5Hz) */
    if (delay_ms < 80) delay_ms = 80;
    if (delay_ms > 2400) delay_ms = 2400;
    ctx->delay_tokens = delay_ms / 80;
    vox_update_time_conditioning(ctx);
}
