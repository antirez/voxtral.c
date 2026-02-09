#include "voxtral_cuda.h"

/* This file is compiled for non-CUDA builds to satisfy references and keep the
 * real CUDA implementation (voxtral_cuda.c) out of CPU/Metal builds. */
#ifndef USE_CUDA

int vox_cuda_available(void) { return 0; }
const char *vox_cuda_device_name(void) { return "disabled"; }

int vox_cuda_matmul(float *C, const float *A, const float *B, int M, int K, int N) {
    (void)C; (void)A; (void)B; (void)M; (void)K; (void)N;
    return 0;
}

int vox_cuda_matmul_t(float *C, const float *A, const float *B, int M, int K, int N) {
    (void)C; (void)A; (void)B; (void)M; (void)K; (void)N;
    return 0;
}

int vox_cuda_matmul_t_bf16(float *C, const float *A, const uint16_t *B_bf16, int M, int K, int N) {
    (void)C; (void)A; (void)B_bf16; (void)M; (void)K; (void)N;
    return 0;
}

int vox_cuda_linear_bf16(float *y, const float *x, const uint16_t *W_bf16, const float *b,
                         int seq_len, int in_dim, int out_dim) {
    (void)y; (void)x; (void)W_bf16; (void)b; (void)seq_len; (void)in_dim; (void)out_dim;
    return 0;
}

int vox_cuda_linear2_bf16(float *y0, float *y1,
                          const float *x,
                          const uint16_t *W0_bf16,
                          const uint16_t *W1_bf16,
                          int in_dim,
                          int out_dim) {
    (void)y0; (void)y1; (void)x; (void)W0_bf16; (void)W1_bf16; (void)in_dim; (void)out_dim;
    return 0;
}

int vox_cuda_attention_step(vox_ctx_t *ctx,
                            float *attn_out,
                            const float *q,
                            const float *k,
                            const float *v,
                            int layer,
                            int pos,
                            int total_seq,
                            int window_size) {
    (void)ctx; (void)attn_out; (void)q; (void)k; (void)v; (void)layer; (void)pos; (void)total_seq; (void)window_size;
    return 0;
}

void vox_cuda_kv_cache_compact(vox_ctx_t *ctx, int discard, int keep, int kv_dim, int max_seq) {
    (void)ctx; (void)discard; (void)keep; (void)kv_dim; (void)max_seq;
}

void vox_cuda_kv_cache_reset(vox_ctx_t *ctx) { (void)ctx; }

void vox_cuda_kv_cache_append_block(vox_ctx_t *ctx, int layer, int start_pos, int seq_len,
                                    int kv_dim, int window_size,
                                    const float *k, const float *v) {
    (void)ctx; (void)layer; (void)start_pos; (void)seq_len; (void)kv_dim; (void)window_size; (void)k; (void)v;
}

int vox_cuda_kv_cache_download_host(vox_ctx_t *ctx, int start_pos, int n_pos) {
    (void)ctx; (void)start_pos; (void)n_pos;
    return 0;
}

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
                              int q_offset) {
    (void)out; (void)Q; (void)K; (void)V;
    (void)seq_q; (void)seq_k; (void)n_heads; (void)n_kv_heads;
    (void)head_dim; (void)scale; (void)window_size; (void)q_offset;
    return 0;
}

int vox_cuda_encode_adapter(float **out, int *out_tokens,
                            vox_ctx_t *ctx,
                            const float *mel,
                            int mel_frames,
                            int overlap_mel) {
    (void)out; (void)out_tokens; (void)ctx; (void)mel; (void)mel_frames; (void)overlap_mel;
    return 0;
}

void vox_cuda_stream_adapter_reset(vox_ctx_t *ctx) { (void)ctx; }

int vox_cuda_stream_adapter_copy_prompt(vox_ctx_t *ctx, float *out_host, int n_tokens) {
    (void)ctx; (void)out_host; (void)n_tokens;
    return 0;
}

void vox_cuda_stream_adapter_compact(vox_ctx_t *ctx, int consumed_tokens) {
    (void)ctx; (void)consumed_tokens;
}

int vox_cuda_encode_adapter_stream_append(int *out_tokens,
                                          vox_ctx_t *ctx,
                                          const float *mel,
                                          int mel_frames,
                                          int overlap_mel) {
    (void)out_tokens; (void)ctx; (void)mel; (void)mel_frames; (void)overlap_mel;
    return 0;
}

int vox_cuda_decoder_forward_from_stream_adapter(int *out_token,
                                                 float *logits_or_null,
                                                 vox_ctx_t *ctx,
                                                 int prev_token) {
    (void)out_token; (void)logits_or_null; (void)ctx; (void)prev_token;
    return 0;
}

int vox_cuda_decoder_forward_full(int *out_token,
                                  float *logits_or_null,
                                  vox_ctx_t *ctx,
                                  const float *input_embeds) {
    (void)out_token; (void)logits_or_null; (void)ctx; (void)input_embeds;
    return 0;
}

int vox_cuda_decoder_prefill_full(vox_ctx_t *ctx,
                                  const float *input_embeds,
                                  int seq_len,
                                  const float *rope_freqs) {
    (void)ctx; (void)input_embeds; (void)seq_len; (void)rope_freqs;
    return 0;
}

int vox_cuda_prefetch_weights(vox_ctx_t *ctx) {
    (void)ctx;
    return 0;
}

void vox_cuda_shutdown(void) {}

void vox_cuda_ctx_free(vox_ctx_t *ctx) { (void)ctx; }

#endif /* !USE_CUDA */
