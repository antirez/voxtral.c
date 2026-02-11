/*
 * voxtral_cuda.h - CUDA GPU acceleration for Voxtral inference
 *
 * Provides cuBLAS-accelerated matrix multiplication with bf16 weight caching,
 * plus custom CUDA kernels for element-wise operations.
 */

#ifndef VOXTRAL_CUDA_H
#define VOXTRAL_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize CUDA acceleration. Returns 1 on success, 0 if unavailable.
 * Requires SM 8.0+ for native BF16 support. */
int vox_cuda_init(void);

/* Check if CUDA is initialized and available. */
int vox_cuda_available(void);

/* Cleanup all CUDA resources. */
void vox_cuda_shutdown(void);

/*
 * GPU-accelerated matrix multiplication with bf16 weights.
 * C[M,N] = A[M,K] @ B^T[N,K]
 *
 * A is f32 (activations), B is bf16 (weights, cached on GPU after first use),
 * C is f32 (output). B is always transposed (row-major weight layout).
 *
 * Weight buffers are cached on GPU after first use (uploaded once per unique
 * bf16 pointer). cuBLAS uses native BF16 (no conversion needed).
 */
void vox_cuda_sgemm_bf16(int M, int N, int K,
                          const float *A,
                          const uint16_t *B_bf16,
                          float *C);

/*
 * GPU-accelerated f32 matrix multiplication.
 * C[M,N] = A[M,K] @ B^T[N,K]
 */
void vox_cuda_sgemm(int M, int N, int K,
                     const float *A,
                     const float *B,
                     float *C);

/*
 * GPU causal attention with online softmax.
 * out[seq_q, n_heads * head_dim] = softmax(Q @ K^T * scale) @ V
 *
 * Q and out are in CPU memory (uploaded/downloaded automatically).
 * K and V must be in CUDA managed memory (KV cache).
 * Supports GQA (n_heads > n_kv_heads), sliding window, and causal masking.
 */
void vox_cuda_causal_attention(float *out, const float *Q, const float *K, const float *V,
                                int seq_q, int seq_k, int n_heads, int n_kv_heads,
                                int head_dim, float scale, int window_size, int q_offset);

/*
 * Pre-warm the bf16 weight cache by uploading to GPU.
 * Call during model loading to avoid first-use latency.
 */
void vox_cuda_warmup_bf16(const uint16_t *bf16_weights, size_t num_elements);

/*
 * Pre-warm the f32 weight cache (norms, biases, ada_scale).
 */
void vox_cuda_warmup_f32(const float *f32_weights, size_t num_elements);

/*
 * Monolithic decoder step: all 26 layers + logits on GPU, single sync.
 * ctx is cast to vox_ctx_t* internally.
 * input_embeds: [DEC_DIM] f32 on CPU.
 * rope_freqs: [HEAD_DIM] f32 on CPU (pre-computed for this position).
 * logits: [VOCAB_SIZE] f32 on CPU (output).
 * Returns argmax token ID, or -1 on failure.
 */
int vox_cuda_decoder_step(void *ctx, const float *input_embeds,
                           const float *rope_freqs, float *logits);

/*
 * Monolithic encoder step: all 32 layers + final norm on GPU, single sync.
 * ctx is cast to vox_ctx_t* internally.
 * x: [new_len, ENC_DIM] f32 on CPU (in/out: overwritten with result).
 * rope_freqs: [new_len, HEAD_DIM/2, 2] f32 on CPU.
 * cache_len: current KV cache length (before this chunk).
 * Returns 0 on success, -1 on failure.
 */
int vox_cuda_encoder_step(void *ctx, float *x, int new_len,
                           const float *rope_freqs, int cache_len);

/*
 * GPU-shared memory allocation (unified memory, accessible from CPU and GPU).
 * Returns a pointer usable from both CPU and GPU.
 * Falls back to calloc if CUDA is not available.
 */
void *vox_cuda_shared_alloc(size_t size);
void  vox_cuda_shared_free(void *ptr);

/* GPU memory usage in bytes (for debugging). */
size_t vox_cuda_memory_used(void);

#ifdef __cplusplus
}
#endif

#endif /* VOXTRAL_CUDA_H */
