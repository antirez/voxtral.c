#ifndef VOXTRAL_CUDA_H
#define VOXTRAL_CUDA_H

#include <stddef.h>
#include <stdint.h>

#include "voxtral.h"

int vox_cuda_available(void);
int vox_cuda_matmul(float *C, const float *A, const float *B, int M, int K, int N);
int vox_cuda_matmul_t(float *C, const float *A, const float *B, int M, int K, int N);
int vox_cuda_matmul_t_bf16(float *C, const float *A, const uint16_t *B_bf16, int M, int K, int N);

/* Convenience wrapper for y = x @ W^T (+bias), where W is BF16 [out_dim, in_dim]. */
int vox_cuda_linear_bf16(float *y, const float *x, const uint16_t *W_bf16, const float *b,
                         int seq_len, int in_dim, int out_dim);

/* Specialized helper for decoder hot path: compute two BF16 linear projections
 * with the same input and dimensions while sharing the input upload and stream
 * sync point.
 *
 * y0 = x[1,in_dim] @ W0^T[out_dim,in_dim]
 * y1 = x[1,in_dim] @ W1^T[out_dim,in_dim]
 *
 * Returns 1 on success, 0 on fallback.
 */
int vox_cuda_linear2_bf16(float *y0, float *y1,
                          const float *x,
                          const uint16_t *W0_bf16,
                          const uint16_t *W1_bf16,
                          int in_dim,
                          int out_dim);

/* Decoder attention acceleration (seq_q=1 path).
 * - Appends this layer's K/V at `pos` into a device-side cache.
 * - Computes attention for Q against cached K/V for this layer and returns
 *   attn_out (float32, shape [VOX_DEC_HEADS*VOX_DEC_HEAD_DIM]).
 *
 * Returns 1 on success, 0 on fallback.
 */
int vox_cuda_attention_step(float *attn_out,
                            const float *q,
                            const float *k,
                            const float *v,
                            int layer,
                            int pos,
                            int total_seq,
                            int window_size);

/* Keep CUDA-side KV cache in sync with CPU KV cache compactions/resets. */
void vox_cuda_kv_cache_compact(int discard, int keep, int kv_dim, int max_seq);
void vox_cuda_kv_cache_reset(void);
void vox_cuda_kv_cache_append_block(int layer, int start_pos, int seq_len,
                                    int kv_dim, int window_size,
                                    const float *k, const float *v);

/* Generic causal attention on GPU (float32 Q,K,V).
 * Returns 1 on success, 0 on fallback.
 */
int vox_cuda_causal_attention(float *out,
                              const float *Q,
                              const float *K,
                              const float *V,
                              int seq_q,
                              int seq_k,
                              int n_heads,
                              int n_kv_heads,
                              int head_dim,
                              float scale,
                              int window_size,
                              int q_offset);

/* CUDA fast paths that keep encoder/decoder intermediates on-device. */
int vox_cuda_encode_adapter(float **out, int *out_tokens,
                            vox_ctx_t *ctx,
                            const float *mel,
                            int mel_frames,
                            int overlap_mel);

/* Optional: full CUDA streaming pipeline (encoder+adapter outputs kept on
 * device, decoder consumes adapter embeddings directly). Opt-in via
 * VOX_CUDA_PIPELINE_FULL=1. */
void vox_cuda_stream_adapter_reset(void);

/* Copy the first `n_tokens` adapter embeddings from the device-side adapter
 * buffer into `out_host` (float32, shape [n_tokens, VOX_DEC_DIM]). Used to
 * build the initial decoder prompt on CPU without copying the full adapter. */
int vox_cuda_stream_adapter_copy_prompt(float *out_host, int n_tokens);

/* Run CUDA full encoder+adapter and append the resulting adapter embeddings
 * to the internal device-side adapter buffer. Returns 1 on success. */
int vox_cuda_encode_adapter_stream_append(int *out_tokens,
                                          vox_ctx_t *ctx,
                                          const float *mel,
                                          int mel_frames,
                                          int overlap_mel);

/* Decoder single-token step that pulls the current adapter embedding from the
 * device-side adapter buffer (using logical position = kv_pos_offset+kv_cache_len).
 * prev_token is the previous generated token id used for the input embedding.
 * Returns 1 on success, 0 on fallback. */
int vox_cuda_decoder_forward_from_stream_adapter(int *out_token,
                                                 float *logits_or_null,
                                                 vox_ctx_t *ctx,
                                                 int prev_token);

int vox_cuda_decoder_forward_full(int *out_token,
                                  float *logits_or_null,
                                  vox_ctx_t *ctx,
                                  const float *input_embeds);

/* CUDA prefill fast path (seq_len > 1).
 * Computes the full transformer prefill on GPU and keeps both the device KV
 * cache and the host KV cache in sync. Returns 1 on success, 0 on fallback.
 *
 * rope_freqs is the precomputed RoPE frequency table for the prefill tokens:
 *   shape = [seq_len, (head_dim/2)*2].
 */
int vox_cuda_decoder_prefill_full(vox_ctx_t *ctx,
                                  const float *input_embeds,
                                  int seq_len,
                                  const float *rope_freqs);

const char *vox_cuda_device_name(void);
void vox_cuda_shutdown(void);

#endif
