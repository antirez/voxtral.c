/*
 * voxtral_decoder.c - LLM decoder (26 layers, GQA)
 *
 * Architecture (per layer):
 *   RMSNorm -> Attention (GQA: 32 heads, 8 KV heads)
 *   RMSNorm -> (optional) ada_rms_norm_t_cond -> SwiGLU FFN (dim=3072, hidden=9216)
 *
 * ada_rms_norm_t_cond (vLLM MistralDecoderLayer):
 *   hidden_states = hidden_states * (1 + ada_mlp(t_cond))
 * where:
 *   ada_mlp = Linear(3072->32, bias=False) -> GELU -> Linear(32->3072, bias=False)
 *
 * Per-layer `ctx->ada_scale[layer, :]` is precomputed once in voxtral.c at load time.
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

static float *load_f32(safetensors_file_t *sf, const char *name) {
    const safetensor_t *t = safetensors_find(sf, name);
    if (!t) {
        fprintf(stderr, "decoder: weight not found: %s\n", name);
        return NULL;
    }
    return safetensors_get_f32(sf, t);
}

static uint16_t *load_bf16_direct(safetensors_file_t *sf, const char *name) {
    const safetensor_t *t = safetensors_find(sf, name);
    if (!t) {
        fprintf(stderr, "decoder: weight not found: %s\n", name);
        return NULL;
    }
    return safetensors_get_bf16_direct(sf, t);
}

/* Load INT8 quantized weight if available, otherwise return NULL.
 * Sets *scale_out to the dequantization scale (per-tensor). */
static const int8_t *load_int8_direct(safetensors_file_t *sf, const char *name,
                                       float *scale_out) {
    const safetensor_t *t = safetensors_find(sf, name);
    if (!t || !safetensor_is_int8(t)) {
        return NULL;
    }
    float s = safetensors_get_scale(sf, name);
    if (s <= 0.0f) {
        fprintf(stderr, "decoder: INT8 per-channel scales not supported: %s\n", name);
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

int vox_decoder_load(vox_decoder_t *dec, safetensors_file_t *sf) {
    char name[512];

    /* Token embeddings (large, bf16 mmap direct) */
    dec->tok_embeddings_bf16 = load_bf16_direct(sf,
        "mm_streams_embeddings.embedding_module.tok_embeddings.weight");
    if (!dec->tok_embeddings_bf16) return -1;

    /* Transformer layers */
    for (int i = 0; i < VOX_DEC_LAYERS; i++) {
        vox_dec_layer_t *l = &dec->layers[i];

        /* Ada RMS norm MLP (small, always f32) */
        snprintf(name, sizeof(name), "layers.%d.ada_rms_norm_t_cond.0.weight", i);
        l->ada_norm_down = load_f32(sf, name);
        snprintf(name, sizeof(name), "layers.%d.ada_rms_norm_t_cond.2.weight", i);
        l->ada_norm_up = load_f32(sf, name);

        /* Attention (large matmul weights: try INT8, fall back to bf16) */
        int use_int8 = 0;
        snprintf(name, sizeof(name), "layers.%d.attention.wq.weight", i);
        use_int8 += load_weight_with_int8_fallback(sf, name,
            &l->wq_int8, &l->wq_scale, &l->wq_weight_bf16);
        snprintf(name, sizeof(name), "layers.%d.attention.wk.weight", i);
        use_int8 += load_weight_with_int8_fallback(sf, name,
            &l->wk_int8, &l->wk_scale, &l->wk_weight_bf16);
        snprintf(name, sizeof(name), "layers.%d.attention.wv.weight", i);
        use_int8 += load_weight_with_int8_fallback(sf, name,
            &l->wv_int8, &l->wv_scale, &l->wv_weight_bf16);
        snprintf(name, sizeof(name), "layers.%d.attention.wo.weight", i);
        use_int8 += load_weight_with_int8_fallback(sf, name,
            &l->wo_int8, &l->wo_scale, &l->wo_weight_bf16);

        /* Norms (small, always f32) */
        snprintf(name, sizeof(name), "layers.%d.attention_norm.weight", i);
        l->attention_norm = load_f32(sf, name);

        /* FFN (large matmul weights: try INT8, fall back to bf16) */
        snprintf(name, sizeof(name), "layers.%d.feed_forward.w1.weight", i);
        use_int8 += load_weight_with_int8_fallback(sf, name,
            &l->w1_int8, &l->w1_scale, &l->w1_weight_bf16);
        /* w2 always BF16 for accuracy (kept in quantize_weights.py) */
        snprintf(name, sizeof(name), "layers.%d.feed_forward.w2.weight", i);
        l->w2_weight_bf16 = load_bf16_direct(sf, name);
        snprintf(name, sizeof(name), "layers.%d.feed_forward.w3.weight", i);
        use_int8 += load_weight_with_int8_fallback(sf, name,
            &l->w3_int8, &l->w3_scale, &l->w3_weight_bf16);

        /* Norms (small, always f32) */
        snprintf(name, sizeof(name), "layers.%d.ffn_norm.weight", i);
        l->ffn_norm = load_f32(sf, name);

        /* Check if any required weight is missing (allow mixed BF16/INT8) */
        int has_wq = l->wq_weight_bf16 || l->wq_int8;
        int has_wk = l->wk_weight_bf16 || l->wk_int8;
        int has_wv = l->wv_weight_bf16 || l->wv_int8;
        int has_wo = l->wo_weight_bf16 || l->wo_int8;
        if (!has_wq || !has_wk || !has_wv || !has_wo) {
            fprintf(stderr, "decoder: failed to load layer %d\n", i);
            return -1;
        }

        if (vox_verbose >= 2) {
            fprintf(stderr, "  Decoder layer %d/%d loaded%s\n", i + 1, VOX_DEC_LAYERS,
                    use_int8 > 0 ? " (INT8)" : " (BF16)");
        }
    }

    /* Final norm */
    dec->norm = load_f32(sf, "norm.weight");
    if (!dec->norm) return -1;

    return 0;
}

/* ========================================================================
 * KV Cache Management
 * ======================================================================== */

static int kv_cache_init(vox_ctx_t *ctx, int max_seq) {
    int kv_dim = VOX_DEC_KV_HEADS * VOX_DEC_HEAD_DIM; /* 8 * 128 = 1024 */
    size_t cache_size = (size_t)VOX_DEC_LAYERS * max_seq * kv_dim * sizeof(float);

#ifdef USE_METAL
    if (vox_metal_available()) {
        ctx->kv_cache_k = (float *)vox_metal_shared_alloc(cache_size);
        ctx->kv_cache_v = (float *)vox_metal_shared_alloc(cache_size);
    } else
#endif
    {
        ctx->kv_cache_k = (float *)calloc(1, cache_size);
        ctx->kv_cache_v = (float *)calloc(1, cache_size);
    }
    ctx->kv_cache_len = 0;
    ctx->kv_cache_max = max_seq;
    /* kv_pos_offset is NOT reset here — caller manages it */

    if (!ctx->kv_cache_k || !ctx->kv_cache_v) return -1;
    return 0;
}

int vox_decoder_kv_cache_preallocate(vox_ctx_t *ctx, int max_seq) {
    if (ctx->kv_cache_k) return 0; /* already allocated */
    return kv_cache_init(ctx, max_seq);
}

/* Grow KV cache to fit at least `required` positions */
static int kv_cache_grow(vox_ctx_t *ctx, int required) {
    if (required <= ctx->kv_cache_max) return 0;

    int kv_dim = VOX_DEC_KV_HEADS * VOX_DEC_HEAD_DIM;
    int new_max = ctx->kv_cache_max;
    while (new_max < required) new_max *= 2;

    size_t new_stride = (size_t)new_max * kv_dim;
    size_t old_stride = (size_t)ctx->kv_cache_max * kv_dim;
    size_t total = (size_t)VOX_DEC_LAYERS * new_stride * sizeof(float);

    float *new_k, *new_v;
#ifdef USE_METAL
    if (vox_metal_available()) {
        new_k = (float *)vox_metal_shared_alloc(total);
        new_v = (float *)vox_metal_shared_alloc(total);
    } else
#endif
    {
        new_k = (float *)calloc(1, total);
        new_v = (float *)calloc(1, total);
    }
    if (!new_k || !new_v) {
#ifdef USE_METAL
        vox_metal_shared_free(new_k);
        vox_metal_shared_free(new_v);
#else
        free(new_k); free(new_v);
#endif
        return -1;
    }

    size_t copy = (size_t)ctx->kv_cache_len * kv_dim * sizeof(float);
    for (int l = 0; l < VOX_DEC_LAYERS; l++) {
        memcpy(new_k + l * new_stride, ctx->kv_cache_k + l * old_stride, copy);
        memcpy(new_v + l * new_stride, ctx->kv_cache_v + l * old_stride, copy);
    }

#ifdef USE_METAL
    vox_metal_shared_free(ctx->kv_cache_k);
    vox_metal_shared_free(ctx->kv_cache_v);
#else
    free(ctx->kv_cache_k);
    free(ctx->kv_cache_v);
#endif
    ctx->kv_cache_k = new_k;
    ctx->kv_cache_v = new_v;
    ctx->kv_cache_max = new_max;
    return 0;
}

/* Get K cache pointer for layer at position */
static float *kv_cache_k_at(vox_ctx_t *ctx, int layer, int pos) {
    int kv_dim = VOX_DEC_KV_HEADS * VOX_DEC_HEAD_DIM;
    return ctx->kv_cache_k + ((size_t)layer * ctx->kv_cache_max + pos) * kv_dim;
}

static float *kv_cache_v_at(vox_ctx_t *ctx, int layer, int pos) {
    int kv_dim = VOX_DEC_KV_HEADS * VOX_DEC_HEAD_DIM;
    return ctx->kv_cache_v + ((size_t)layer * ctx->kv_cache_max + pos) * kv_dim;
}

/* Compact KV cache: discard entries older than the sliding window.
 * Keeps the last VOX_DEC_WINDOW entries, moves them to position 0,
 * and updates kv_pos_offset so RoPE positions remain correct.
 * RoPE is already baked into cached K vectors, so no re-encoding needed. */
static void kv_cache_compact(vox_ctx_t *ctx) {
    int keep = VOX_DEC_WINDOW;
    if (ctx->kv_cache_len <= keep) return;

    int discard = ctx->kv_cache_len - keep;
    int kv_dim = VOX_DEC_KV_HEADS * VOX_DEC_HEAD_DIM;
    size_t keep_bytes = (size_t)keep * kv_dim * sizeof(float);

    for (int l = 0; l < VOX_DEC_LAYERS; l++) {
        float *k_base = kv_cache_k_at(ctx, l, 0);
        float *k_src  = kv_cache_k_at(ctx, l, discard);
        float *v_base = kv_cache_v_at(ctx, l, 0);
        float *v_src  = kv_cache_v_at(ctx, l, discard);
        memmove(k_base, k_src, keep_bytes);
        memmove(v_base, v_src, keep_bytes);
    }

    ctx->kv_pos_offset += discard;
    ctx->kv_cache_len = keep;
}

/* ========================================================================
 * Decoder Forward Pass (Prefill)
 * ======================================================================== */

void vox_decoder_prefill(vox_ctx_t *ctx, const float *input_embeds, int seq_len) {
    vox_decoder_t *dec = &ctx->decoder;
    int dim = VOX_DEC_DIM;
    int n_heads = VOX_DEC_HEADS;
    int n_kv_heads = VOX_DEC_KV_HEADS;
    int head_dim = VOX_DEC_HEAD_DIM;
    int hidden = VOX_DEC_HIDDEN;
    int q_dim = n_heads * head_dim;     /* 4096 */
    int kv_dim = n_kv_heads * head_dim; /* 1024 */

    /* Ensure KV cache is allocated and large enough */
    if (!ctx->kv_cache_k) {
        if (kv_cache_init(ctx, VOX_DEC_WINDOW + seq_len + 1024) != 0) return;
    } else if (ctx->kv_cache_len + seq_len > ctx->kv_cache_max) {
        if (kv_cache_grow(ctx, ctx->kv_cache_len + seq_len + 1024) != 0) return;
    }

    /* Working buffers */
    float *x = (float *)malloc(seq_len * dim * sizeof(float));
    memcpy(x, input_embeds, seq_len * dim * sizeof(float));

    float *x_norm = (float *)malloc(seq_len * dim * sizeof(float));
    float *q = (float *)malloc(seq_len * q_dim * sizeof(float));
    float *k = (float *)malloc(seq_len * kv_dim * sizeof(float));
    float *v = (float *)malloc(seq_len * kv_dim * sizeof(float));
    float *attn_out = (float *)malloc(seq_len * q_dim * sizeof(float));
    float *proj_out = (float *)malloc(seq_len * dim * sizeof(float));
    float *ffn_out = (float *)malloc(seq_len * dim * sizeof(float));

    /* RoPE frequencies (logical positions include offset from compactions) */
    int start_pos = ctx->kv_cache_len;
    int logical_start = ctx->kv_pos_offset + start_pos;
    int *positions = (int *)malloc(seq_len * sizeof(int));
    for (int i = 0; i < seq_len; i++) positions[i] = logical_start + i;
    float *rope_freqs = (float *)malloc(seq_len * (head_dim / 2) * 2 * sizeof(float));
    vox_compute_rope_freqs(rope_freqs, positions, seq_len, head_dim, VOX_ROPE_THETA);

    /* GPU monolithic prefill: all 26 layers in one command buffer */
#ifdef USE_METAL
    if (vox_metal_available()) {
        if (vox_metal_decoder_prefill_step(ctx, x, seq_len, rope_freqs)) {
            /* GPU prefill succeeded */
            free(x); free(x_norm); free(q); free(k); free(v);
            free(attn_out); free(proj_out); free(ffn_out);
            free(positions); free(rope_freqs);
            return;
        }
        /* GPU prefill not available (e.g., INT8 weights).
         * Check if CPU fallback is possible (requires BF16 weights). */
        if (dec->layers[0].wq_weight_bf16 == NULL) {
            /* INT8 model without GPU prefill: process tokens one at a time.
             * This is slower but works with the per-step INT8 GPU decoder. */
            float *logits = (float *)malloc(VOX_VOCAB_SIZE * sizeof(float));
            for (int i = 0; i < seq_len; i++) {
                vox_decoder_forward(ctx, input_embeds + (size_t)i * dim, logits);
            }
            free(logits);
            free(x); free(x_norm); free(q); free(k); free(v);
            free(attn_out); free(proj_out); free(ffn_out);
            free(positions); free(rope_freqs);
            return;
        }
    }
#endif

    for (int layer = 0; layer < VOX_DEC_LAYERS; layer++) {
        vox_dec_layer_t *l = &dec->layers[layer];

        /* ---- Self-attention ---- */
        vox_rms_norm(x_norm, x, l->attention_norm, seq_len, dim, VOX_DEC_NORM_EPS);

        /* Q, K, V projections (no bias in decoder, bf16 weights) */
#ifdef USE_METAL
        if (vox_metal_available()) {
            vox_metal_fused_qkv_bf16(seq_len, dim, x_norm,
                                      l->wq_weight_bf16, q_dim,
                                      l->wk_weight_bf16, kv_dim,
                                      l->wv_weight_bf16, kv_dim,
                                      q, k, v);
        } else
#endif
        {
            vox_linear_nobias_bf16(q, x_norm, l->wq_weight_bf16, seq_len, dim, q_dim);
            vox_linear_nobias_bf16(k, x_norm, l->wk_weight_bf16, seq_len, dim, kv_dim);
            vox_linear_nobias_bf16(v, x_norm, l->wv_weight_bf16, seq_len, dim, kv_dim);
        }

        /* Apply RoPE */
        vox_apply_rope(q, rope_freqs, seq_len, n_heads, head_dim);
        vox_apply_rope(k, rope_freqs, seq_len, n_kv_heads, head_dim);

        /* Store K, V in cache */
        for (int s = 0; s < seq_len; s++) {
            memcpy(kv_cache_k_at(ctx, layer, start_pos + s),
                   k + s * kv_dim, kv_dim * sizeof(float));
            memcpy(kv_cache_v_at(ctx, layer, start_pos + s),
                   v + s * kv_dim, kv_dim * sizeof(float));
        }

        /* Causal attention over full cached sequence */
        int total_seq = start_pos + seq_len;
        float *full_k = kv_cache_k_at(ctx, layer, 0);
        float *full_v = kv_cache_v_at(ctx, layer, 0);

        float scale = 1.0f / sqrtf((float)head_dim);
        vox_causal_attention(attn_out, q, full_k, full_v,
                             seq_len, total_seq, n_heads, n_kv_heads,
                             head_dim, scale, VOX_DEC_WINDOW, start_pos);

        /* Output projection + residual */
        vox_linear_nobias_bf16(proj_out, attn_out, l->wo_weight_bf16, seq_len, q_dim, dim);
        vox_add_inplace(x, proj_out, seq_len * dim);

        /* ---- FFN ---- */
        vox_rms_norm(x_norm, x, l->ffn_norm, seq_len, dim, VOX_DEC_NORM_EPS);

        /* Time conditioning (ada_rms_norm_t_cond): h_norm *= (1 + ada_scale[layer]) */
        if (ctx->ada_scale) {
            const float *ada = ctx->ada_scale + (size_t)layer * dim;
            for (int s = 0; s < seq_len; s++) {
                float *row = x_norm + (size_t)s * dim;
                for (int i = 0; i < dim; i++) row[i] *= (1.0f + ada[i]);
            }
        }

        /* SwiGLU */
#ifdef USE_METAL
        if (vox_metal_available()) {
            vox_metal_fused_ffn_bf16(seq_len, dim, hidden, x_norm,
                                      l->w1_weight_bf16, l->w3_weight_bf16,
                                      l->w2_weight_bf16, ffn_out);
        } else
#endif
        {
            /* CPU path needs separate gate/up buffers */
            float *gate = (float *)malloc(seq_len * hidden * sizeof(float));
            float *up = (float *)malloc(seq_len * hidden * sizeof(float));
            vox_linear_nobias_bf16(gate, x_norm, l->w1_weight_bf16, seq_len, dim, hidden);
            vox_silu(gate, seq_len * hidden);
            vox_linear_nobias_bf16(up, x_norm, l->w3_weight_bf16, seq_len, dim, hidden);
            vox_mul_inplace(gate, up, seq_len * hidden);
            vox_linear_nobias_bf16(ffn_out, gate, l->w2_weight_bf16, seq_len, hidden, dim);
            free(gate); free(up);
        }

        /* Residual */
        vox_add_inplace(x, ffn_out, seq_len * dim);

        if (vox_verbose >= 2 && ((layer + 1) % 8 == 0 || layer == VOX_DEC_LAYERS - 1))
            fprintf(stderr, "  Decoder prefill layer %d/%d\n", layer + 1, VOX_DEC_LAYERS);
    }

    ctx->kv_cache_len = start_pos + seq_len;

    free(x); free(x_norm); free(q); free(k); free(v);
    free(attn_out); free(proj_out); free(ffn_out);
    free(positions); free(rope_freqs);
}

/* ========================================================================
 * Decoder Forward Pass (Single Token Generation)
 * ======================================================================== */

/* Lazy-init persistent single-token decoder buffers */
static void ensure_dec_buffers(vox_ctx_t *ctx) {
    if (ctx->dec_x) return; /* already allocated */
    int dim = VOX_DEC_DIM;
    int q_dim = VOX_DEC_HEADS * VOX_DEC_HEAD_DIM;
    int kv_dim = VOX_DEC_KV_HEADS * VOX_DEC_HEAD_DIM;
    int hidden = VOX_DEC_HIDDEN;
    int head_dim = VOX_DEC_HEAD_DIM;

    ctx->dec_x        = (float *)malloc(dim * sizeof(float));
    ctx->dec_x_norm   = (float *)malloc(dim * sizeof(float));
    ctx->dec_q        = (float *)malloc(q_dim * sizeof(float));
    ctx->dec_k        = (float *)malloc(kv_dim * sizeof(float));
    ctx->dec_v        = (float *)malloc(kv_dim * sizeof(float));
    ctx->dec_attn_out = (float *)malloc(q_dim * sizeof(float));
    ctx->dec_proj_out = (float *)malloc(dim * sizeof(float));
    ctx->dec_gate     = (float *)malloc(hidden * sizeof(float));
    ctx->dec_up       = (float *)malloc(hidden * sizeof(float));
    ctx->dec_ffn_out  = (float *)malloc(dim * sizeof(float));
    ctx->dec_rope_freqs = (float *)malloc((head_dim / 2) * 2 * sizeof(float));
}

int vox_decoder_forward(vox_ctx_t *ctx, const float *input_embeds, float *logits) {
    vox_decoder_t *dec = &ctx->decoder;
    int dim = VOX_DEC_DIM;
    int n_heads = VOX_DEC_HEADS;
    int n_kv_heads = VOX_DEC_KV_HEADS;
    int head_dim = VOX_DEC_HEAD_DIM;
    int hidden = VOX_DEC_HIDDEN;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;

    /* Persistent working buffers (allocated once, reused across tokens) */
    ensure_dec_buffers(ctx);
    float *x = ctx->dec_x;
    float *x_norm = ctx->dec_x_norm;
    float *q = ctx->dec_q;
    float *k = ctx->dec_k;
    float *v = ctx->dec_v;
    float *attn_out = ctx->dec_attn_out;
    float *proj_out = ctx->dec_proj_out;
    float *gate_buf = ctx->dec_gate;
    float *up_buf = ctx->dec_up;
    float *ffn_out = ctx->dec_ffn_out;
    float *rope_freqs = ctx->dec_rope_freqs;

    memcpy(x, input_embeds, dim * sizeof(float));

    int pos = ctx->kv_cache_len;

    /* Rolling KV cache: compact instead of growing when possible */
    if (pos >= ctx->kv_cache_max) {
        if (ctx->kv_cache_len > VOX_DEC_WINDOW) {
            kv_cache_compact(ctx);
            pos = ctx->kv_cache_len;
        }
        if (pos >= ctx->kv_cache_max) {
            if (kv_cache_grow(ctx, pos + 1024) != 0) return 2; /* EOS on OOM */
        }
    }

    /* RoPE uses logical position (physical + offset from compactions) */
    int logical_pos = ctx->kv_pos_offset + pos;
    int positions[1] = { logical_pos };
    vox_compute_rope_freqs(rope_freqs, positions, 1, head_dim, VOX_ROPE_THETA);

    float scale = 1.0f / sqrtf((float)head_dim);

#ifdef USE_METAL
    if (vox_metal_available()) {
        /* Try monolithic GPU path: all 26 layers + logits in ONE command buffer.
         * RoPE, KV cache writes, and attention all run on GPU.
         * Requires shared KV cache (allocated via vox_metal_shared_alloc). */
        vox_metal_decoder_start(x, dim);
        int token = vox_metal_decoder_full_step(ctx, rope_freqs, logits);
        vox_metal_decoder_end();
        if (token >= 0) return token;

        /* full_step returned -1 (shared KV cache not available).
         * Fall through to CPU path. */
    }
#endif

    /* CPU fallback path */
    for (int layer = 0; layer < VOX_DEC_LAYERS; layer++) {
        vox_dec_layer_t *l = &dec->layers[layer];

        vox_rms_norm(x_norm, x, l->attention_norm, 1, dim, VOX_DEC_NORM_EPS);
        vox_linear_nobias_bf16(q, x_norm, l->wq_weight_bf16, 1, dim, q_dim);
        vox_linear_nobias_bf16(k, x_norm, l->wk_weight_bf16, 1, dim, kv_dim);
        vox_linear_nobias_bf16(v, x_norm, l->wv_weight_bf16, 1, dim, kv_dim);

        vox_apply_rope(q, rope_freqs, 1, n_heads, head_dim);
        vox_apply_rope(k, rope_freqs, 1, n_kv_heads, head_dim);

        memcpy(kv_cache_k_at(ctx, layer, pos), k, kv_dim * sizeof(float));
        memcpy(kv_cache_v_at(ctx, layer, pos), v, kv_dim * sizeof(float));

        int total_seq = pos + 1;
        float *full_k = kv_cache_k_at(ctx, layer, 0);
        float *full_v = kv_cache_v_at(ctx, layer, 0);

        vox_causal_attention(attn_out, q, full_k, full_v,
                             1, total_seq, n_heads, n_kv_heads,
                             head_dim, scale, VOX_DEC_WINDOW, pos);

        vox_linear_nobias_bf16(proj_out, attn_out, l->wo_weight_bf16, 1, q_dim, dim);
        vox_add_inplace(x, proj_out, dim);

        vox_rms_norm(x_norm, x, l->ffn_norm, 1, dim, VOX_DEC_NORM_EPS);
        if (ctx->ada_scale) {
            const float *ada_s = ctx->ada_scale + (size_t)layer * dim;
            for (int i = 0; i < dim; i++) x_norm[i] *= (1.0f + ada_s[i]);
        }

        vox_linear_nobias_bf16(gate_buf, x_norm, l->w1_weight_bf16, 1, dim, hidden);
        vox_silu(gate_buf, hidden);
        vox_linear_nobias_bf16(up_buf, x_norm, l->w3_weight_bf16, 1, dim, hidden);
        vox_mul_inplace(gate_buf, up_buf, hidden);
        vox_linear_nobias_bf16(ffn_out, gate_buf, l->w2_weight_bf16, 1, hidden, dim);
        vox_add_inplace(x, ffn_out, dim);
    }

    ctx->kv_cache_len = pos + 1;

    vox_rms_norm(x, x, dec->norm, 1, dim, VOX_DEC_NORM_EPS);
    vox_matmul_t_bf16(logits, x, dec->tok_embeddings_bf16, 1, dim, VOX_VOCAB_SIZE);

    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < VOX_VOCAB_SIZE; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}
