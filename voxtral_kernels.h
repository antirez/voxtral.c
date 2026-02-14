/*
 * voxtral_kernels.h - Math kernels for Voxtral inference
 *
 * Low-level math operations. All operate on float32 tensors in row-major order.
 * Adapted from flux-2-4b project.
 */

#ifndef VOXTRAL_KERNELS_H
#define VOXTRAL_KERNELS_H

#include <stddef.h>
#include <stdint.h>

/* ========================================================================
 * Basic Operations
 * ======================================================================== */

void vox_add_inplace(float *a, const float *b, int n);
void vox_mul_inplace(float *a, const float *b, int n);
void vox_axpy(float *a, float scale, const float *b, int n);
void vox_scale(float *x, float s, int n);
void vox_copy(float *dst, const float *src, int n);

/* ========================================================================
 * Matrix Operations
 * ======================================================================== */

/*
 * General matrix multiplication: C = A @ B
 * A: [M, K], B: [K, N], C: [M, N]
 */
void vox_matmul(float *C, const float *A, const float *B, int M, int K, int N);

/*
 * Matrix multiplication with transposed B: C = A @ B^T
 * A: [M, K], B: [N, K], C: [M, N]
 */
void vox_matmul_t(float *C, const float *A, const float *B, int M, int K, int N);

/*
 * Linear layer: y = x @ W^T + b (if b != NULL)
 * x: [seq_len, in_dim], W: [out_dim, in_dim], b: [out_dim], y: [seq_len, out_dim]
 */
void vox_linear(float *y, const float *x, const float *W, const float *b,
                int seq_len, int in_dim, int out_dim);

void vox_linear_nobias(float *y, const float *x, const float *W,
                       int seq_len, int in_dim, int out_dim);

/*
 * Linear layer with bf16 weights (no bias)
 * x: [seq_len, in_dim] (f32), W: [out_dim, in_dim] (bf16), y: [seq_len, out_dim] (f32)
 */
void vox_linear_nobias_bf16(float *y, const float *x, const uint16_t *W_bf16,
                            int seq_len, int in_dim, int out_dim);

/*
 * Linear layer with bf16 weights and f32 bias
 * x: [seq_len, in_dim] (f32), W: [out_dim, in_dim] (bf16), b: [out_dim] (f32)
 */
void vox_linear_bf16(float *y, const float *x, const uint16_t *W_bf16,
                     const float *b, int seq_len, int in_dim, int out_dim);

/*
 * Matrix multiplication with transposed bf16 B: C = A @ B^T
 * A: [M, K] (f32), B: [N, K] (bf16), C: [M, N] (f32)
 */
void vox_matmul_t_bf16(float *C, const float *A, const uint16_t *B_bf16,
                       int M, int K, int N);

/* ========================================================================
 * 1D Convolution (for audio encoder conv stem)
 * ======================================================================== */

/*
 * 1D Convolution: out = conv1d(in, weight, bias)
 * in: [channels_in, length]
 * weight: [channels_out, channels_in, kernel_size]
 * bias: [channels_out] (can be NULL)
 * out: [channels_out, out_length]
 */
void vox_conv1d(float *out, const float *in, const float *weight, const float *bias,
                int channels_in, int channels_out, int length,
                int kernel_size, int stride, int padding);

/*
 * Causal conv1d used by WhisperCausalEncoder (matches vLLM WhisperCausalConv1d):
 * - left padding = kernel_size - stride
 * - right padding = extra padding to reach the target length implied by ceil(n_frames)
 */
void vox_causal_conv1d(float *out, const float *in, const float *weight, const float *bias,
                       int channels_in, int channels_out, int length,
                       int kernel_size, int stride);

/* ========================================================================
 * Normalization
 * ======================================================================== */

/*
 * RMS Normalization
 * x: [seq_len, hidden], weight: [hidden]
 */
void vox_rms_norm(float *out, const float *x, const float *weight,
                  int seq_len, int hidden, float eps);

/* ========================================================================
 * Activation Functions
 * ======================================================================== */

/* SiLU / Swish activation: x * sigmoid(x) */
void vox_silu(float *x, int n);

/* GELU activation */
void vox_gelu(float *x, int n);

/* Softmax over last dimension */
void vox_softmax(float *x, int rows, int cols);

/* ========================================================================
 * Attention Operations
 * ======================================================================== */

/*
 * Causal attention with optional sliding window.
 * Q: [seq_q, n_heads * head_dim]
 * K: [seq_k, n_kv_heads * head_dim]
 * V: [seq_k, n_kv_heads * head_dim]
 * out: [seq_q, n_heads * head_dim]
 *
 * Supports GQA: n_heads can be a multiple of n_kv_heads.
 * window_size <= 0 means no sliding window (full causal).
 */
/*
 * q_offset: global position of the first query token. Used for causal mask:
 *   query at local position i can attend to K positions 0..(q_offset+i).
 *   For prefill from scratch, q_offset=0. For generation, q_offset=cache_len.
 */
void vox_causal_attention(float *out, const float *Q, const float *K, const float *V,
                          int seq_q, int seq_k, int n_heads, int n_kv_heads,
                          int head_dim, float scale, int window_size,
                          int q_offset);

/* ========================================================================
 * Rotary Position Embeddings
 * ======================================================================== */

/*
 * Compute RoPE frequencies for 1D positions
 * pos: position indices [seq]
 * freqs: output [seq, dim/2, 2] (cos, sin pairs)
 */
void vox_compute_rope_freqs(float *freqs, const int *pos, int seq, int dim, float theta);

/*
 * Apply RoPE to Q/K tensors
 * x: [seq, heads * head_dim] (in-place)
 * freqs: [seq, head_dim/2, 2]
 */
void vox_apply_rope(float *x, const float *freqs, int seq, int heads, int head_dim);

/* Global verbose flag */
extern int vox_verbose;
extern int vox_monitor;

#endif /* VOXTRAL_KERNELS_H */
