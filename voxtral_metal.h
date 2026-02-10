/*
 * voxtral_metal.h - Metal GPU acceleration for Voxtral inference
 *
 * Provides MPS-accelerated matrix multiplication with bf16->f16 weight caching,
 * plus GPU compute shaders for element-wise operations.
 * Ported from flux-2-4b (same author, same license).
 */

#ifndef VOXTRAL_METAL_H
#define VOXTRAL_METAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize Metal acceleration. Returns 1 on success, 0 if unavailable. */
int vox_metal_init(void);

/* Check if Metal is initialized and available. */
int vox_metal_available(void);

/* Cleanup all Metal resources. */
void vox_metal_shutdown(void);

/*
 * GPU-accelerated matrix multiplication with bf16 weights.
 * C[M,N] = alpha * A[M,K] @ B^T[N,K] + beta * C[M,N]
 *
 * A is f32 (activations), B is bf16 (weights, converted to f16 for MPS),
 * C is f32 (output). B is always transposed (row-major weight layout).
 *
 * Weight buffers are cached on GPU after first use (bf16->f16 conversion
 * happens once per unique weight pointer).
 */
void vox_metal_sgemm_bf16(int M, int N, int K,
                           const float *A,
                           const uint16_t *B_bf16,
                           float *C);

/*
 * GPU-accelerated f32 matrix multiplication.
 * C[M,N] = A[M,K] @ B^T[N,K]
 */
void vox_metal_sgemm(int M, int N, int K,
                     const float *A,
                     const float *B,
                     float *C);

/*
 * Fused QKV: three matmuls in one command buffer with shared input.
 * q[M,Nq] = input[M,K] @ wq[Nq,K]^T
 * k[M,Nk] = input[M,K] @ wk[Nk,K]^T
 * v[M,Nv] = input[M,K] @ wv[Nv,K]^T
 * Saves 2 command buffer round-trips vs 3 separate calls.
 */
void vox_metal_fused_qkv_bf16(int M, int K,
                                const float *input,
                                const uint16_t *wq_bf16, int Nq,
                                const uint16_t *wk_bf16, int Nk,
                                const uint16_t *wv_bf16, int Nv,
                                float *q, float *k, float *v);

/*
 * Fused RMSNorm + QKV: norm + three matmuls in one command buffer.
 * x_norm = rms_norm(x, norm_weight, eps), then QKV projections.
 */
void vox_metal_fused_norm_qkv_bf16(int M, int K,
                                     const float *x,
                                     const float *norm_weight, float eps,
                                     const uint16_t *wq_bf16, int Nq,
                                     const uint16_t *wk_bf16, int Nk,
                                     const uint16_t *wv_bf16, int Nv,
                                     float *q, float *k, float *v);

/*
 * Fused SwiGLU FFN: w1+w3+silu+mul+w2 in one command buffer.
 * gate = silu(input @ w1^T)
 * up = input @ w3^T
 * output = (gate * up) @ w2^T
 * All intermediate data stays on GPU. Saves 2 round-trips + eliminates
 * intermediate CPU memcpy for silu/mul.
 */
void vox_metal_fused_ffn_bf16(int M, int dim, int hidden,
                               const float *input,
                               const uint16_t *w1_bf16,
                               const uint16_t *w3_bf16,
                               const uint16_t *w2_bf16,
                               float *output);

/*
 * Fused wo projection + residual + RMSNorm + ada_scale + SwiGLU FFN + residual.
 * All in one command buffer. Saves 1 round-trip per layer vs separate wo + FFN.
 *
 * attn_out[M, q_dim]: attention output (input for wo)
 * wo_bf16[dim, q_dim]: output projection weights
 * x[M, dim]: current residual (modified in-place: += wo_out, then += ffn_out)
 * ffn_norm[dim]: RMS norm weights for FFN
 * ada_scale[dim]: adaptive conditioning (NULL to skip)
 * w1,w3,w2: FFN weights
 */
void vox_metal_fused_wo_ffn_bf16(int M, int dim, int q_dim, int hidden,
                                   float *x,
                                   const float *attn_out,
                                   const uint16_t *wo_bf16,
                                   const float *ffn_norm, float eps,
                                   const float *ada_scale,
                                   const uint16_t *w1_bf16,
                                   const uint16_t *w3_bf16,
                                   const uint16_t *w2_bf16);

/*
 * GPU batched attention (all heads in one command buffer).
 * Performs QK^T matmul, causal+window masked softmax, scores*V matmul
 * entirely on GPU. Uses strided MPS matrix views (no per-head copies).
 *
 * Q:   [seq_q, n_heads * head_dim]   f32
 * K:   [seq_k, n_kv_heads * head_dim] f32
 * V:   [seq_k, n_kv_heads * head_dim] f32
 * out: [seq_q, n_heads * head_dim]   f32
 */
void vox_metal_batched_attention(float *out,
                                  const float *Q, const float *K, const float *V,
                                  int seq_q, int seq_k,
                                  int n_heads, int n_kv_heads,
                                  int head_dim, float scale,
                                  int window_size, int q_offset);

/*
 * Fused encoder attention: single compute dispatch for all heads.
 * Replaces per-head MPS matmul encodes with a single kernel.
 * Same interface as vox_metal_batched_attention.
 */
void vox_metal_encoder_attention(float *out,
                                   const float *Q, const float *K, const float *V,
                                   int seq_q, int seq_k,
                                   int n_heads, int n_kv_heads,
                                   int head_dim, float scale,
                                   int window_size, int q_offset);

/*
 * Fused final RMSNorm + logits matmul + argmax.
 * Computes: x_norm = rms_norm(x, norm, eps), logits = x_norm @ tok_emb^T, argmax.
 * Returns best token ID. logits_out may be NULL if not needed.
 */
int vox_metal_fused_logits_bf16(int dim, int vocab_size,
                                  const float *x,
                                  const float *norm_weight, float eps,
                                  const uint16_t *tok_emb_bf16,
                                  float *logits_out);

/*
 * Persistent-x decoder step API.
 * Keeps x on GPU across all 26 layers to fuse wo_ffn[i] + norm_qkv[i+1]
 * in one command buffer. Halves command buffer count: 53 → 27 per token.
 */

/* Upload x to persistent GPU buffer (call before decoder loop). */
void vox_metal_decoder_start(const float *x, int dim);

/* Release persistent GPU x (call after decoder loop). */
void vox_metal_decoder_end(void);

/* First layer: rms_norm + QKV from persistent GPU x. (1 cmd buf) */
void vox_metal_decoder_norm_qkv(int K,
                                  const float *norm_weight, float eps,
                                  const uint16_t *wq_bf16, int Nq,
                                  const uint16_t *wk_bf16, int Nk,
                                  const uint16_t *wv_bf16, int Nv,
                                  float *q, float *k, float *v);

/* Cross-layer: wo+FFN (updates GPU x) + norm+QKV for next layer. (1 cmd buf)
 * Fuses 10 wo+FFN steps + 4 norm+QKV steps into a single command buffer. */
void vox_metal_decoder_wo_ffn_next_qkv(int dim, int q_dim, int hidden,
                                         const float *attn_out,
                                         const uint16_t *wo_bf16,
                                         const float *ffn_norm, float eps,
                                         const float *ada_scale,
                                         const uint16_t *w1_bf16,
                                         const uint16_t *w3_bf16,
                                         const uint16_t *w2_bf16,
                                         const float *next_attn_norm,
                                         const uint16_t *next_wq_bf16, int next_Nq,
                                         const uint16_t *next_wk_bf16, int next_Nk,
                                         const uint16_t *next_wv_bf16, int next_Nv,
                                         float *q, float *k, float *v);

/* Final layer: wo+FFN (updates GPU x) + logits + argmax. (1 cmd buf)
 * Returns token ID. logits_out may be NULL. */
int vox_metal_decoder_wo_ffn_logits(int dim, int q_dim, int hidden, int vocab_size,
                                      const float *attn_out,
                                      const uint16_t *wo_bf16,
                                      const float *ffn_norm, float eps,
                                      const float *ada_scale,
                                      const uint16_t *w1_bf16,
                                      const uint16_t *w3_bf16,
                                      const uint16_t *w2_bf16,
                                      const float *final_norm,
                                      const uint16_t *tok_emb_bf16,
                                      float *logits_out);

/*
 * GPU-shared memory allocation (zero-copy between CPU and GPU).
 * Returns a CPU pointer backed by a Metal shared buffer.
 * Falls back to calloc if Metal is not available.
 */
void *vox_metal_shared_alloc(size_t size);
void vox_metal_shared_free(void *ptr);

/*
 * Monolithic decoder step: all 26 layers + logits in ONE command buffer.
 * Requires KV cache allocated with vox_metal_shared_alloc().
 * GPU kernels for RoPE, KV cache write, and attention eliminate all
 * CPU round-trips between layers. ctx is cast to vox_ctx_t* internally.
 * Returns token ID. logits_out may be NULL.
 */
int vox_metal_decoder_full_step(void *ctx, const float *rope_freqs, float *logits);

/*
 * Pre-warm the bf16->f16 cache for a weight tensor.
 * Call during model loading to avoid first-use latency.
 */
void vox_metal_warmup_bf16(const uint16_t *bf16_weights, size_t num_elements);

/* Pre-warm MPS matmul ops and f32 weight caches for decoder. */
void vox_metal_warmup_decoder_ops(void *ctx);

/* Pre-warm merged weight buffers (used by monolithic decoder step). */
void vox_metal_warmup_merged_2(const uint16_t *a, size_t a_n,
                                const uint16_t *b, size_t b_n);
void vox_metal_warmup_merged_3(const uint16_t *a, size_t a_n,
                                const uint16_t *b, size_t b_n,
                                const uint16_t *c, size_t c_n);

/*
 * Monolithic encoder step: all 32 layers + final norm in ONE command buffer.
 * Requires encoder KV cache allocated with vox_metal_shared_alloc().
 * x is [new_len, VOX_ENC_DIM] float, modified in-place with the output.
 * rope_freqs: [new_len, head_dim/2, 2] precomputed frequencies.
 * cache_len: current number of positions in the KV cache.
 * Returns 0 on success, -1 on failure.
 */
int vox_metal_encoder_full_step(void *ctx, float *x, int new_len,
                                 const float *rope_freqs, int cache_len);

/*
 * Monolithic decoder prefill: all 26 layers in ONE command buffer (M>1).
 * x is [seq_len, VOX_DEC_DIM] float, modified in-place.
 * rope_freqs: [seq_len, head_dim/2, 2] precomputed frequencies.
 * Updates ctx->kv_cache_len internally.
 * Returns 1 on success, 0 if caller should fall back to CPU (e.g., INT8 weights).
 */
int vox_metal_decoder_prefill_step(void *ctx, float *x, int seq_len,
                                    const float *rope_freqs);

/* GPU memory usage (for debugging). */
size_t vox_metal_memory_used(void);

#ifdef __cplusplus
}
#endif

#endif /* VOXTRAL_METAL_H */
