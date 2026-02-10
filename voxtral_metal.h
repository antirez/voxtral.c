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
 * Persistent-x decoder step API.
 * Keeps x on GPU across all 26 layers to fuse wo_ffn[i] + norm_qkv[i+1]
 * in one command buffer. Halves command buffer count: 53 → 27 per token.
 */

/* Upload x to persistent GPU buffer (call before decoder loop). */
void vox_metal_decoder_start(const float *x, int dim);

/* Release persistent GPU x (call after decoder loop). */
void vox_metal_decoder_end(void);

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
 */
void vox_metal_decoder_prefill_step(void *ctx, float *x, int seq_len,
                                      const float *rope_freqs);

/* GPU memory usage (for debugging). */
size_t vox_metal_memory_used(void);

/* Pre-warm INT8 weight cache for a decoder weight tensor. */
void vox_metal_warmup_int8(const uint16_t *bf16_weights, size_t num_elements, int K);


#ifdef __cplusplus
}
#endif

#endif /* VOXTRAL_METAL_H */
