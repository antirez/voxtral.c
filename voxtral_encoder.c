/*
 * voxtral_encoder.c - Audio encoder (causal transformer)
 *
 * Architecture:
 *   Conv stem: conv1d(128->1280, k=3, s=1, p=1) -> GELU
 *              conv1d(1280->1280, k=3, s=2, p=1) -> GELU
 *   32 transformer layers (causal, sliding window=750):
 *     - RMSNorm -> Attention (MHA, 32 heads, head_dim=64, with biases)
 *     - RMSNorm -> SwiGLU FFN (dim=1280, hidden=5120, w2 has bias)
 *   Final RMSNorm
 *   Downsample 4x: reshape [seq, 1280] -> [seq/4, 5120]
 *   Adapter: Linear(5120->3072) -> GELU -> Linear(3072->3072)
 */

#include "voxtral.h"
#include "voxtral_kernels.h"
#include "voxtral_safetensors.h"
#ifdef USE_METAL
#include "voxtral_metal.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========================================================================
 * Weight Loading
 * ======================================================================== */

#define ENC_PREFIX "mm_streams_embeddings.embedding_module.whisper_encoder"

static float *load_f32(safetensors_file_t *sf, const char *name) {
    const safetensor_t *t = safetensors_find(sf, name);
    if (!t) {
        fprintf(stderr, "encoder: weight not found: %s\n", name);
        return NULL;
    }
    return safetensors_get_f32(sf, t);
}

static uint16_t *load_bf16_direct(safetensors_file_t *sf, const char *name) {
    const safetensor_t *t = safetensors_find(sf, name);
    if (!t) {
        fprintf(stderr, "encoder: weight not found: %s\n", name);
        return NULL;
    }
    return safetensors_get_bf16_direct(sf, t);
}

/* Load INT8 quantized weight if available, otherwise return NULL. */
static const int8_t *load_int8_direct(safetensors_file_t *sf, const char *name, float *scale_out) {
    const safetensor_t *t = safetensors_find(sf, name);
    if (!t || !safetensor_is_int8(t)) {
        return NULL;
    }
    float s = safetensors_get_scale(sf, name);
    if (s <= 0.0f) {
        fprintf(stderr, "encoder: INT8 per-channel scales not supported: %s\n", name);
        return NULL;
    }
    *scale_out = s;
    int64_t count;
    return safetensors_get_int8_direct(sf, t, &count);
}

/* Try INT8 first, fall back to BF16. Returns 1 if INT8, 0 if BF16. */
static int load_weight_with_int8_fallback(safetensors_file_t *sf, const char *name,
                                           const int8_t **int8_out, float *scale_out,
                                           uint16_t **bf16_out) {
    *int8_out = load_int8_direct(sf, name, scale_out);
    if (*int8_out) {
        *bf16_out = NULL;
        return 1;
    }
    /* Fall back to BF16 */
    *bf16_out = load_bf16_direct(sf, name);
    *scale_out = 1.0f;
    return 0;
}

int vox_encoder_load(vox_encoder_t *enc, safetensors_file_t *sf) {
    char name[512];

    /* Conv stem (small, always f32) */
    snprintf(name, sizeof(name), "%s.conv_layers.0.conv.weight", ENC_PREFIX);
    enc->conv0_weight = load_f32(sf, name);
    snprintf(name, sizeof(name), "%s.conv_layers.0.conv.bias", ENC_PREFIX);
    enc->conv0_bias = load_f32(sf, name);
    snprintf(name, sizeof(name), "%s.conv_layers.1.conv.weight", ENC_PREFIX);
    enc->conv1_weight = load_f32(sf, name);
    snprintf(name, sizeof(name), "%s.conv_layers.1.conv.bias", ENC_PREFIX);
    enc->conv1_bias = load_f32(sf, name);

    if (!enc->conv0_weight || !enc->conv1_weight) return -1;

    /* Transformer layers */
    for (int i = 0; i < VOX_ENC_LAYERS; i++) {
        vox_enc_layer_t *l = &enc->layers[i];
        const char *lp = ENC_PREFIX ".transformer.layers";

        /* Large matmul weights: try INT8, fall back to bf16 */
        int use_int8 = 0;
        snprintf(name, sizeof(name), "%s.%d.attention.wq.weight", lp, i);
        use_int8 += load_weight_with_int8_fallback(sf, name,
            &l->wq_int8, &l->wq_scale, &l->wq_weight_bf16);
        snprintf(name, sizeof(name), "%s.%d.attention.wk.weight", lp, i);
        use_int8 += load_weight_with_int8_fallback(sf, name,
            &l->wk_int8, &l->wk_scale, &l->wk_weight_bf16);
        snprintf(name, sizeof(name), "%s.%d.attention.wv.weight", lp, i);
        use_int8 += load_weight_with_int8_fallback(sf, name,
            &l->wv_int8, &l->wv_scale, &l->wv_weight_bf16);
        snprintf(name, sizeof(name), "%s.%d.attention.wo.weight", lp, i);
        use_int8 += load_weight_with_int8_fallback(sf, name,
            &l->wo_int8, &l->wo_scale, &l->wo_weight_bf16);
        snprintf(name, sizeof(name), "%s.%d.feed_forward.w1.weight", lp, i);
        use_int8 += load_weight_with_int8_fallback(sf, name,
            &l->w1_int8, &l->w1_scale, &l->w1_weight_bf16);
        /* w2 always BF16 for accuracy */
        snprintf(name, sizeof(name), "%s.%d.feed_forward.w2.weight", lp, i);
        l->w2_weight_bf16 = load_bf16_direct(sf, name);
        snprintf(name, sizeof(name), "%s.%d.feed_forward.w3.weight", lp, i);
        use_int8 += load_weight_with_int8_fallback(sf, name,
            &l->w3_int8, &l->w3_scale, &l->w3_weight_bf16);

        /* Small weights: biases and norms (always f32) */
        snprintf(name, sizeof(name), "%s.%d.attention.wq.bias", lp, i);
        l->wq_bias = load_f32(sf, name);
        /* wk has NO bias */
        snprintf(name, sizeof(name), "%s.%d.attention.wv.bias", lp, i);
        l->wv_bias = load_f32(sf, name);
        snprintf(name, sizeof(name), "%s.%d.attention.wo.bias", lp, i);
        l->wo_bias = load_f32(sf, name);
        snprintf(name, sizeof(name), "%s.%d.attention_norm.weight", lp, i);
        l->attention_norm = load_f32(sf, name);
        snprintf(name, sizeof(name), "%s.%d.feed_forward.w2.bias", lp, i);
        l->w2_bias = load_f32(sf, name);
        snprintf(name, sizeof(name), "%s.%d.ffn_norm.weight", lp, i);
        l->ffn_norm = load_f32(sf, name);

        /* Check if any required weight is missing (allow mixed BF16/INT8) */
        int has_wq = l->wq_weight_bf16 || l->wq_int8;
        int has_wk = l->wk_weight_bf16 || l->wk_int8;
        int has_wv = l->wv_weight_bf16 || l->wv_int8;
        int has_wo = l->wo_weight_bf16 || l->wo_int8;
        if (!has_wq || !has_wk || !has_wv || !has_wo) {
            fprintf(stderr, "encoder: failed to load layer %d weights\n", i);
            return -1;
        }

        if (vox_verbose >= 2) {
            fprintf(stderr, "  Encoder layer %d/%d loaded%s\n", i + 1, VOX_ENC_LAYERS,
                    use_int8 > 0 ? " (INT8)" : " (BF16)");
        }
    }

    /* Final norm */
    snprintf(name, sizeof(name), "%s.transformer.norm.weight", ENC_PREFIX);
    enc->norm = load_f32(sf, name);

    if (!enc->norm) return -1;
    return 0;
}

/* ========================================================================
 * Forward Pass
 * ======================================================================== */

/* GELU activation */
static void gelu_inplace(float *x, int n) {
    vox_gelu(x, n);
}

static int causal_conv1d_out_len(int length, int kernel_size, int stride) {
    int padding_total = kernel_size - stride;
    float n_frames = ((float)length - kernel_size + padding_total) / (float)stride + 1.0f;
    int out_len = (int)ceilf(n_frames);
    return out_len < 0 ? 0 : out_len;
}

float *vox_encoder_forward(vox_ctx_t *ctx, const float *mel,
                           int mel_frames, int *out_seq_len) {
    vox_encoder_t *enc = &ctx->encoder;
    int dim = VOX_ENC_DIM;        /* 1280 */
    int n_heads = VOX_ENC_HEADS;  /* 32 */
    int head_dim = VOX_ENC_HEAD_DIM; /* 64 */
    int hidden = VOX_ENC_HIDDEN;  /* 5120 */
    int qkv_dim = n_heads * head_dim; /* 2048 */

    if (vox_verbose >= 2)
        fprintf(stderr, "Encoder: %d mel frames\n", mel_frames);

    /* ---- Conv stem ---- */
    /* mel: [mel_frames, 128] -> transpose to [128, mel_frames] for conv1d */
    float *conv_in = (float *)malloc(VOX_MEL_BINS * mel_frames * sizeof(float));
    for (int f = 0; f < mel_frames; f++) {
        for (int m = 0; m < VOX_MEL_BINS; m++) {
            conv_in[m * mel_frames + f] = mel[f * VOX_MEL_BINS + m];
        }
    }

    /* Conv0: [128, mel_frames] -> [1280, mel_frames] (stride=1, causal) */
    int conv0_out_len = causal_conv1d_out_len(mel_frames, 3, 1);
    float *conv0_out = (float *)malloc(dim * conv0_out_len * sizeof(float));
    vox_causal_conv1d(conv0_out, conv_in, enc->conv0_weight, enc->conv0_bias,
                      VOX_MEL_BINS, dim, mel_frames, 3, 1);
    gelu_inplace(conv0_out, dim * conv0_out_len);
    free(conv_in);

    /* Conv1: [1280, mel_frames] -> [1280, ceil(mel_frames/2)] (stride=2, causal) */
    int conv1_out_len = causal_conv1d_out_len(conv0_out_len, 3, 2);
    float *conv1_out = (float *)malloc(dim * conv1_out_len * sizeof(float));
    vox_causal_conv1d(conv1_out, conv0_out, enc->conv1_weight, enc->conv1_bias,
                      dim, dim, conv0_out_len, 3, 2);
    gelu_inplace(conv1_out, dim * conv1_out_len);
    free(conv0_out);

    int seq_len = conv1_out_len;

    /* Transpose: [1280, seq_len] -> [seq_len, 1280] */
    float *x = (float *)malloc(seq_len * dim * sizeof(float));
    for (int s = 0; s < seq_len; s++) {
        for (int d = 0; d < dim; d++) {
            x[s * dim + d] = conv1_out[d * seq_len + s];
        }
    }
    free(conv1_out);

    if (vox_verbose >= 2)
        fprintf(stderr, "  Conv stem: %d frames -> %d\n", mel_frames, seq_len);

    /* ---- Transformer layers ---- */
    float *x_norm = (float *)malloc(seq_len * dim * sizeof(float));
    float *q = (float *)malloc(seq_len * qkv_dim * sizeof(float));
    float *k = (float *)malloc(seq_len * qkv_dim * sizeof(float));
    float *v = (float *)malloc(seq_len * qkv_dim * sizeof(float));
    float *attn_out = (float *)malloc(seq_len * qkv_dim * sizeof(float));
    float *proj_out = (float *)malloc(seq_len * dim * sizeof(float));
    float *gate = (float *)malloc(seq_len * hidden * sizeof(float));
    float *up = (float *)malloc(seq_len * hidden * sizeof(float));
    float *ffn_out = (float *)malloc(seq_len * dim * sizeof(float));

    /* RoPE frequencies */
    int *positions = (int *)malloc(seq_len * sizeof(int));
    for (int i = 0; i < seq_len; i++) positions[i] = i;
    float *rope_freqs = (float *)malloc(seq_len * (head_dim / 2) * 2 * sizeof(float));
    vox_compute_rope_freqs(rope_freqs, positions, seq_len, head_dim, VOX_ROPE_THETA);

    for (int layer = 0; layer < VOX_ENC_LAYERS; layer++) {
        vox_enc_layer_t *l = &enc->layers[layer];

        /* ---- Self-attention ---- */
        vox_rms_norm(x_norm, x, l->attention_norm, seq_len, dim, VOX_ENC_NORM_EPS);

        /* Q, K, V projections (bf16 weights, f32 biases) */
#ifdef USE_METAL
        if (vox_metal_available()) {
            vox_metal_fused_qkv_bf16(seq_len, dim, x_norm,
                                      l->wq_weight_bf16, qkv_dim,
                                      l->wk_weight_bf16, qkv_dim,
                                      l->wv_weight_bf16, qkv_dim,
                                      q, k, v);
            /* Add biases on CPU (wq has bias, wk has NO bias, wv has bias) */
            for (int s = 0; s < seq_len; s++) {
                for (int j = 0; j < qkv_dim; j++) {
                    q[s * qkv_dim + j] += l->wq_bias[j];
                    v[s * qkv_dim + j] += l->wv_bias[j];
                }
            }
        } else {
#endif
            vox_linear_bf16(q, x_norm, l->wq_weight_bf16, l->wq_bias, seq_len, dim, qkv_dim);
            vox_linear_nobias_bf16(k, x_norm, l->wk_weight_bf16, seq_len, dim, qkv_dim);
            vox_linear_bf16(v, x_norm, l->wv_weight_bf16, l->wv_bias, seq_len, dim, qkv_dim);
#ifdef USE_METAL
        }
#endif

        /* Apply RoPE to Q and K */
        vox_apply_rope(q, rope_freqs, seq_len, n_heads, head_dim);
        vox_apply_rope(k, rope_freqs, seq_len, n_heads, head_dim);

        /* Causal attention with sliding window */
        float scale = 1.0f / sqrtf((float)head_dim);
#ifdef USE_METAL
        if (vox_metal_available()) {
            vox_metal_encoder_attention(attn_out, q, k, v,
                                         seq_len, seq_len, n_heads, VOX_ENC_KV_HEADS,
                                         head_dim, scale, VOX_ENC_WINDOW, 0);
        } else {
#endif
            vox_causal_attention(attn_out, q, k, v,
                                 seq_len, seq_len, n_heads, VOX_ENC_KV_HEADS,
                                 head_dim, scale, VOX_ENC_WINDOW, 0);
#ifdef USE_METAL
        }
#endif

        /* Output projection + residual */
#ifdef USE_METAL
        if (vox_metal_available()) {
            vox_metal_sgemm_bf16(seq_len, dim, qkv_dim, attn_out,
                                   l->wo_weight_bf16, proj_out);
            /* Add wo bias on CPU */
            for (int s = 0; s < seq_len; s++)
                for (int j = 0; j < dim; j++)
                    proj_out[s * dim + j] += l->wo_bias[j];
        } else {
#endif
            vox_linear_bf16(proj_out, attn_out, l->wo_weight_bf16, l->wo_bias, seq_len, qkv_dim, dim);
#ifdef USE_METAL
        }
#endif
        vox_add_inplace(x, proj_out, seq_len * dim);

        /* ---- FFN ---- */
        vox_rms_norm(x_norm, x, l->ffn_norm, seq_len, dim, VOX_ENC_NORM_EPS);

        /* SwiGLU: gate = silu(w1(x)), up = w3(x), ffn = w2(gate * up) + bias */
#ifdef USE_METAL
        if (vox_metal_available()) {
            vox_metal_fused_ffn_bf16(seq_len, dim, hidden, x_norm,
                                      l->w1_weight_bf16, l->w3_weight_bf16,
                                      l->w2_weight_bf16, ffn_out);
            /* Add w2 bias on CPU */
            for (int s = 0; s < seq_len; s++)
                for (int j = 0; j < dim; j++)
                    ffn_out[s * dim + j] += l->w2_bias[j];
        } else {
#endif
            vox_linear_nobias_bf16(gate, x_norm, l->w1_weight_bf16, seq_len, dim, hidden);
            vox_silu(gate, seq_len * hidden);
            vox_linear_nobias_bf16(up, x_norm, l->w3_weight_bf16, seq_len, dim, hidden);
            vox_mul_inplace(gate, up, seq_len * hidden);
            vox_linear_bf16(ffn_out, gate, l->w2_weight_bf16, l->w2_bias, seq_len, hidden, dim);
#ifdef USE_METAL
        }
#endif

        /* Residual */
        vox_add_inplace(x, ffn_out, seq_len * dim);

        if (vox_verbose >= 2 && ((layer + 1) % 8 == 0 || layer == VOX_ENC_LAYERS - 1))
            fprintf(stderr, "  Encoder layer %d/%d\n", layer + 1, VOX_ENC_LAYERS);
    }

    /* Final norm */
    vox_rms_norm(x, x, enc->norm, seq_len, dim, VOX_ENC_NORM_EPS);

    /* Clean up working buffers */
    free(x_norm); free(q); free(k); free(v);
    free(attn_out); free(proj_out);
    free(gate); free(up); free(ffn_out);
    free(positions); free(rope_freqs);

    *out_seq_len = seq_len;
    return x;
}

/* ========================================================================
 * Incremental Encoder KV Cache
 * ======================================================================== */

#define ENC_KV_DIM (VOX_ENC_KV_HEADS * VOX_ENC_HEAD_DIM)  /* 32 * 64 = 2048 */

static float *enc_kv_cache_k_at(vox_ctx_t *ctx, int layer, int pos) {
    return ctx->enc_kv_cache_k + ((size_t)layer * ctx->enc_kv_cache_max + pos) * ENC_KV_DIM;
}

static float *enc_kv_cache_v_at(vox_ctx_t *ctx, int layer, int pos) {
    return ctx->enc_kv_cache_v + ((size_t)layer * ctx->enc_kv_cache_max + pos) * ENC_KV_DIM;
}

int vox_encoder_kv_cache_preallocate(vox_ctx_t *ctx, int max_pos) {
    if (ctx->enc_kv_cache_k) return 0; /* already allocated */

    size_t total = (size_t)VOX_ENC_LAYERS * max_pos * ENC_KV_DIM * sizeof(float);

#ifdef USE_METAL
    if (vox_metal_available()) {
        ctx->enc_kv_cache_k = (float *)vox_metal_shared_alloc(total);
        ctx->enc_kv_cache_v = (float *)vox_metal_shared_alloc(total);
        ctx->enc_kv_cache_is_shared = 1;
    } else
#endif
    {
        ctx->enc_kv_cache_k = (float *)calloc(1, total);
        ctx->enc_kv_cache_v = (float *)calloc(1, total);
    }

    if (!ctx->enc_kv_cache_k || !ctx->enc_kv_cache_v) return -1;
    ctx->enc_kv_cache_max = max_pos;
    return 0;
}

static int enc_kv_cache_grow(vox_ctx_t *ctx, int required) {
    if (ctx->enc_kv_cache_max >= required) return 0;

    /* Shared GPU memory cannot be grown; should not happen with proper pre-allocation */
    if (ctx->enc_kv_cache_is_shared) {
        fprintf(stderr, "encoder: KV cache too small (%d < %d), cannot grow shared buffer\n",
                ctx->enc_kv_cache_max, required);
        return -1;
    }

    int new_max = ctx->enc_kv_cache_max ? ctx->enc_kv_cache_max : 256;
    while (new_max < required) new_max *= 2;

    size_t new_stride = (size_t)new_max * ENC_KV_DIM;
    size_t total = (size_t)VOX_ENC_LAYERS * new_stride * sizeof(float);

    float *new_k = (float *)calloc(1, total);
    float *new_v = (float *)calloc(1, total);
    if (!new_k || !new_v) { free(new_k); free(new_v); return -1; }

    /* Copy existing data */
    if (ctx->enc_kv_cache_len > 0 && ctx->enc_kv_cache_k) {
        size_t old_stride = (size_t)ctx->enc_kv_cache_max * ENC_KV_DIM;
        size_t copy = (size_t)ctx->enc_kv_cache_len * ENC_KV_DIM * sizeof(float);
        for (int l = 0; l < VOX_ENC_LAYERS; l++) {
            memcpy(new_k + l * new_stride, ctx->enc_kv_cache_k + l * old_stride, copy);
            memcpy(new_v + l * new_stride, ctx->enc_kv_cache_v + l * old_stride, copy);
        }
    }

    free(ctx->enc_kv_cache_k);
    free(ctx->enc_kv_cache_v);
    ctx->enc_kv_cache_k = new_k;
    ctx->enc_kv_cache_v = new_v;
    ctx->enc_kv_cache_max = new_max;
    return 0;
}

static void enc_kv_cache_compact(vox_ctx_t *ctx) {
    int keep = VOX_ENC_WINDOW;
    if (ctx->enc_kv_cache_len <= keep) return;

    int discard = ctx->enc_kv_cache_len - keep;
    size_t keep_bytes = (size_t)keep * ENC_KV_DIM * sizeof(float);

    for (int l = 0; l < VOX_ENC_LAYERS; l++) {
        float *k_base = enc_kv_cache_k_at(ctx, l, 0);
        float *k_src  = enc_kv_cache_k_at(ctx, l, discard);
        float *v_base = enc_kv_cache_v_at(ctx, l, 0);
        float *v_src  = enc_kv_cache_v_at(ctx, l, discard);
        memmove(k_base, k_src, keep_bytes);
        memmove(v_base, v_src, keep_bytes);
    }

    ctx->enc_kv_pos_offset += discard;
    ctx->enc_kv_cache_len = keep;
}

static int enc_realloc_float(float **ptr, size_t elems) {
    float *tmp = (float *)realloc(*ptr, elems * sizeof(float));
    if (!tmp) return -1;
    *ptr = tmp;
    return 0;
}

static int enc_realloc_int(int **ptr, size_t elems) {
    int *tmp = (int *)realloc(*ptr, elems * sizeof(int));
    if (!tmp) return -1;
    *ptr = tmp;
    return 0;
}

/* Grow persistent incremental-encoder scratch buffers for new_len positions. */
static int enc_inc_ensure_buffers(vox_ctx_t *ctx, int new_len) {
    if (new_len <= ctx->enc_inc_cap) return 0;

    int dim = VOX_ENC_DIM;
    int head_dim = VOX_ENC_HEAD_DIM;
    int qkv_dim = VOX_ENC_HEADS * VOX_ENC_HEAD_DIM;
    int hidden = VOX_ENC_HIDDEN;
    size_t rope_elems = (size_t)new_len * (head_dim / 2) * 2;

    if (enc_realloc_float(&ctx->enc_inc_x_norm, (size_t)new_len * dim) != 0) return -1;
    if (enc_realloc_float(&ctx->enc_inc_q, (size_t)new_len * qkv_dim) != 0) return -1;
    if (enc_realloc_float(&ctx->enc_inc_k, (size_t)new_len * qkv_dim) != 0) return -1;
    if (enc_realloc_float(&ctx->enc_inc_v, (size_t)new_len * qkv_dim) != 0) return -1;
    if (enc_realloc_float(&ctx->enc_inc_attn_out, (size_t)new_len * qkv_dim) != 0) return -1;
    if (enc_realloc_float(&ctx->enc_inc_proj_out, (size_t)new_len * dim) != 0) return -1;
    if (enc_realloc_float(&ctx->enc_inc_gate, (size_t)new_len * hidden) != 0) return -1;
    if (enc_realloc_float(&ctx->enc_inc_up, (size_t)new_len * hidden) != 0) return -1;
    if (enc_realloc_float(&ctx->enc_inc_ffn_out, (size_t)new_len * dim) != 0) return -1;
    if (enc_realloc_int(&ctx->enc_inc_positions, (size_t)new_len) != 0) return -1;
    if (enc_realloc_float(&ctx->enc_inc_rope_freqs, rope_elems) != 0) return -1;

    ctx->enc_inc_cap = new_len;
    return 0;
}

/* ========================================================================
 * Incremental Encoder Forward Pass
 * ======================================================================== */

float *vox_encoder_forward_incremental(vox_ctx_t *ctx, const float *x_new,
                                        int new_len, int *out_len) {
    vox_encoder_t *enc = &ctx->encoder;
    int dim = VOX_ENC_DIM;        /* 1280 */
    int n_heads = VOX_ENC_HEADS;  /* 32 */
    int head_dim = VOX_ENC_HEAD_DIM; /* 64 */
    int hidden = VOX_ENC_HIDDEN;  /* 5120 */
    int qkv_dim = n_heads * head_dim; /* 2048 */

    if (new_len <= 0) { *out_len = 0; return NULL; }

    /* Compact if needed before adding new positions */
    if (ctx->enc_kv_cache_len + new_len > VOX_ENC_WINDOW) {
        enc_kv_cache_compact(ctx);
    }

    /* Grow cache if needed */
    if (enc_kv_cache_grow(ctx, ctx->enc_kv_cache_len + new_len) != 0) {
        *out_len = 0;
        return NULL;
    }

    int cache_len = ctx->enc_kv_cache_len;

    if (vox_verbose >= 2)
        fprintf(stderr, "  Encoder incremental: %d new positions (cache: %d, offset: %d)\n",
                new_len, cache_len, ctx->enc_kv_pos_offset);

    /* Output/working state for new positions */
    float *x = (float *)malloc((size_t)new_len * dim * sizeof(float));
    if (!x) { *out_len = 0; return NULL; }
    memcpy(x, x_new, (size_t)new_len * dim * sizeof(float));

    if (enc_inc_ensure_buffers(ctx, new_len) != 0) {
        free(x);
        *out_len = 0;
        return NULL;
    }
    float *x_norm = ctx->enc_inc_x_norm;
    float *q = ctx->enc_inc_q;
    float *k = ctx->enc_inc_k;
    float *v = ctx->enc_inc_v;
    float *attn_out = ctx->enc_inc_attn_out;
    float *proj_out = ctx->enc_inc_proj_out;
    float *gate = ctx->enc_inc_gate;
    float *up = ctx->enc_inc_up;
    float *ffn_out = ctx->enc_inc_ffn_out;

    /* RoPE frequencies for logical positions */
    int logical_start = ctx->enc_kv_pos_offset + cache_len;
    int *positions = ctx->enc_inc_positions;
    for (int i = 0; i < new_len; i++) positions[i] = logical_start + i;
    float *rope_freqs = ctx->enc_inc_rope_freqs;
    vox_compute_rope_freqs(rope_freqs, positions, new_len, head_dim, VOX_ROPE_THETA);

    /* GPU monolithic path: all 32 layers in one command buffer */
#ifdef USE_METAL
    if (vox_metal_available() && ctx->enc_kv_cache_is_shared) {
        if (vox_metal_encoder_full_step(ctx, x, new_len, rope_freqs, cache_len) == 0) {
            ctx->enc_kv_cache_len = cache_len + new_len;
            *out_len = new_len;
            return x;
        }
        /* Fall through to CPU path on failure */
    }
#endif

    for (int layer = 0; layer < VOX_ENC_LAYERS; layer++) {
        vox_enc_layer_t *l = &enc->layers[layer];

        /* ---- Self-attention ---- */
        vox_rms_norm(x_norm, x, l->attention_norm, new_len, dim, VOX_ENC_NORM_EPS);

        /* Q, K, V projections on new positions only */
#ifdef USE_METAL
        if (vox_metal_available()) {
            vox_metal_fused_qkv_bf16(new_len, dim, x_norm,
                                      l->wq_weight_bf16, qkv_dim,
                                      l->wk_weight_bf16, qkv_dim,
                                      l->wv_weight_bf16, qkv_dim,
                                      q, k, v);
            /* Add biases (wq has bias, wk has NO bias, wv has bias) */
            for (int s = 0; s < new_len; s++) {
                for (int j = 0; j < qkv_dim; j++) {
                    q[s * qkv_dim + j] += l->wq_bias[j];
                    v[s * qkv_dim + j] += l->wv_bias[j];
                }
            }
        } else {
#endif
            vox_linear_bf16(q, x_norm, l->wq_weight_bf16, l->wq_bias, new_len, dim, qkv_dim);
            vox_linear_nobias_bf16(k, x_norm, l->wk_weight_bf16, new_len, dim, qkv_dim);
            vox_linear_bf16(v, x_norm, l->wv_weight_bf16, l->wv_bias, new_len, dim, qkv_dim);
#ifdef USE_METAL
        }
#endif

        /* Apply RoPE to Q and K */
        vox_apply_rope(q, rope_freqs, new_len, n_heads, head_dim);
        vox_apply_rope(k, rope_freqs, new_len, n_heads, head_dim);

        /* Copy new K, V into cache */
        for (int s = 0; s < new_len; s++) {
            memcpy(enc_kv_cache_k_at(ctx, layer, cache_len + s),
                   k + (size_t)s * qkv_dim, qkv_dim * sizeof(float));
            memcpy(enc_kv_cache_v_at(ctx, layer, cache_len + s),
                   v + (size_t)s * qkv_dim, qkv_dim * sizeof(float));
        }

        /* Attention: q=[new_len], kv=[cache_len + new_len] */
        int total_kv = cache_len + new_len;
        float *full_k = enc_kv_cache_k_at(ctx, layer, 0);
        float *full_v = enc_kv_cache_v_at(ctx, layer, 0);
        float scale = 1.0f / sqrtf((float)head_dim);

#ifdef USE_METAL
        if (vox_metal_available()) {
            vox_metal_encoder_attention(attn_out, q, full_k, full_v,
                                         new_len, total_kv, n_heads, VOX_ENC_KV_HEADS,
                                         head_dim, scale, VOX_ENC_WINDOW, cache_len);
        } else {
#endif
            vox_causal_attention(attn_out, q, full_k, full_v,
                                 new_len, total_kv, n_heads, VOX_ENC_KV_HEADS,
                                 head_dim, scale, VOX_ENC_WINDOW, cache_len);
#ifdef USE_METAL
        }
#endif

        /* Output projection + residual */
#ifdef USE_METAL
        if (vox_metal_available()) {
            vox_metal_sgemm_bf16(new_len, dim, qkv_dim, attn_out,
                                   l->wo_weight_bf16, proj_out);
            /* Add wo bias on CPU */
            for (int s = 0; s < new_len; s++)
                for (int j = 0; j < dim; j++)
                    proj_out[s * dim + j] += l->wo_bias[j];
        } else {
#endif
            vox_linear_bf16(proj_out, attn_out, l->wo_weight_bf16, l->wo_bias, new_len, qkv_dim, dim);
#ifdef USE_METAL
        }
#endif
        vox_add_inplace(x, proj_out, new_len * dim);

        /* ---- FFN ---- */
        vox_rms_norm(x_norm, x, l->ffn_norm, new_len, dim, VOX_ENC_NORM_EPS);

#ifdef USE_METAL
        if (vox_metal_available()) {
            vox_metal_fused_ffn_bf16(new_len, dim, hidden, x_norm,
                                      l->w1_weight_bf16, l->w3_weight_bf16,
                                      l->w2_weight_bf16, ffn_out);
            /* Add w2 bias on CPU */
            for (int s = 0; s < new_len; s++)
                for (int j = 0; j < dim; j++)
                    ffn_out[s * dim + j] += l->w2_bias[j];
        } else {
#endif
            vox_linear_nobias_bf16(gate, x_norm, l->w1_weight_bf16, new_len, dim, hidden);
            vox_silu(gate, new_len * hidden);
            vox_linear_nobias_bf16(up, x_norm, l->w3_weight_bf16, new_len, dim, hidden);
            vox_mul_inplace(gate, up, new_len * hidden);
            vox_linear_bf16(ffn_out, gate, l->w2_weight_bf16, l->w2_bias, new_len, hidden, dim);
#ifdef USE_METAL
        }
#endif

        /* Residual */
        vox_add_inplace(x, ffn_out, new_len * dim);

        if (vox_verbose >= 2 && ((layer + 1) % 8 == 0 || layer == VOX_ENC_LAYERS - 1))
            fprintf(stderr, "  Encoder inc layer %d/%d\n", layer + 1, VOX_ENC_LAYERS);
    }

    /* Final norm */
    vox_rms_norm(x, x, enc->norm, new_len, dim, VOX_ENC_NORM_EPS);

    /* Update cache length */
    ctx->enc_kv_cache_len = cache_len + new_len;

    *out_len = new_len;
    return x;
}

/* ========================================================================
 * Adapter Forward Pass
 * ======================================================================== */

float *vox_adapter_forward(vox_ctx_t *ctx, const float *enc_out,
                           int enc_seq_len, int *out_seq_len) {
    /* Downsample 4x: [enc_seq_len, 1280] -> [enc_seq_len/4, 5120] */
    int ds_len = enc_seq_len / VOX_DOWNSAMPLE;
    int ds_dim = VOX_ENC_DIM * VOX_DOWNSAMPLE; /* 5120 */

    float *ds = (float *)malloc(ds_len * ds_dim * sizeof(float));
    for (int i = 0; i < ds_len; i++) {
        for (int j = 0; j < VOX_DOWNSAMPLE; j++) {
            memcpy(ds + i * ds_dim + j * VOX_ENC_DIM,
                   enc_out + (i * VOX_DOWNSAMPLE + j) * VOX_ENC_DIM,
                   VOX_ENC_DIM * sizeof(float));
        }
    }

    if (vox_verbose >= 2)
        fprintf(stderr, "  Adapter: %d -> %d (downsample %dx)\n",
                enc_seq_len, ds_len, VOX_DOWNSAMPLE);

    /* Linear(5120 -> 3072) -> GELU -> Linear(3072 -> 3072) */
    float *mid = (float *)malloc(ds_len * VOX_DEC_DIM * sizeof(float));
    vox_linear_nobias_bf16(mid, ds, ctx->adapter.linear0_weight_bf16, ds_len, ds_dim, VOX_DEC_DIM);
    vox_gelu(mid, ds_len * VOX_DEC_DIM);

    float *out = (float *)malloc(ds_len * VOX_DEC_DIM * sizeof(float));
    vox_linear_nobias_bf16(out, mid, ctx->adapter.linear1_weight_bf16, ds_len, VOX_DEC_DIM, VOX_DEC_DIM);

    free(ds);
    free(mid);

    *out_seq_len = ds_len;
    return out;
}
