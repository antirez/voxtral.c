// CUDA kernels compiled to a CUBIN and launched via the CUDA driver API.
// This avoids a libcudart dependency (which has been unreliable under WSL2
// for this project) while still letting us write kernels in CUDA C.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

static __device__ __forceinline__ float warp_reduce_sum(float x) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(0xffffffff, x, offset);
    }
    return x;
}

/* Reduce across up to 8 warps (256 threads). Assumes blockDim.x is a multiple of 32. */
static __device__ __forceinline__ float block_reduce_sum_256(float x, float *shmem) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5; /* 0..7 for 256-thread blocks */

    x = warp_reduce_sum(x);
    if (lane == 0) shmem[warp] = x;
    __syncthreads();

    float sum = 0.0f;
    if (warp == 0) {
        int nwarps = blockDim.x >> 5;
        sum = (lane < nwarps) ? shmem[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
    }
    __syncthreads();

    if (warp == 0 && lane == 0) shmem[0] = sum;
    __syncthreads();
    return shmem[0];
}

extern "C" __global__ void vox_attn_q4_kv8_fp16(float *out_q,          /* [32*128] */
                                                const float *q,        /* [32*128] */
                                                const __half *k_cache,  /* [max_seq*8*128] */
                                                const __half *v_cache,  /* [max_seq*8*128] */
                                                int total_seq,
                                                int window_size,
                                                float scale) {
    /* One block per query head (32). This increases parallelism vs. kv-head blocks
     * and avoids block-wide __syncthreads() inside the hot per-token loop.
     *
     * Each block is expected to be launched with exactly one warp (32 threads),
     * and each lane owns 4 dims (128-d head). */
    int h = (int)blockIdx.x;     /* 0..31 */
    int lane = (int)threadIdx.x; /* 0..31 */
    if (h >= 32 || lane >= 32) return;

    int kv_h = h >> 2; /* 4 query heads share 1 KV head */

    float qv0 = q[h * 128 + (lane + 0 * 32)];
    float qv1 = q[h * 128 + (lane + 1 * 32)];
    float qv2 = q[h * 128 + (lane + 2 * 32)];
    float qv3 = q[h * 128 + (lane + 3 * 32)];

    int end = total_seq;
    int start = 0;
    if (window_size > 0) {
        int s = end - window_size;
        if (s > 0) start = s;
    }
    if (start < 0) start = 0;
    if (end < start) end = start;

    /* Online softmax state (scalar) lives on lane 0; values broadcast via shfl. */
    float max_score = -1.0e30f;
    float sum_exp = 0.0f;

    /* Output vector per lane (4 dims). */
    float out0 = 0.0f;
    float out1 = 0.0f;
    float out2 = 0.0f;
    float out3 = 0.0f;

    for (int j = start; j < end; j++) {
        const __half *k_row = k_cache + ((size_t)j * 8 + (size_t)kv_h) * 128;
        float k0 = __half2float(k_row[lane + 0 * 32]);
        float k1 = __half2float(k_row[lane + 1 * 32]);
        float k2 = __half2float(k_row[lane + 2 * 32]);
        float k3 = __half2float(k_row[lane + 3 * 32]);

        float partial = qv0 * k0 + qv1 * k1 + qv2 * k2 + qv3 * k3;
        float sum = warp_reduce_sum(partial);
        sum = __shfl_sync(0xffffffff, sum, 0);

        float score = sum * scale;

        /* Lane 0 updates the online softmax scalars. */
        float w = 0.0f;
        float corr = 1.0f;
        int new_max = 0;
        if (lane == 0) {
            if (score > max_score) {
                corr = __expf(max_score - score);
                sum_exp = sum_exp * corr + 1.0f;
                max_score = score;
                w = 1.0f;
                new_max = 1;
            } else {
                w = __expf(score - max_score);
                sum_exp += w;
                corr = 1.0f;
                new_max = 0;
            }
        }
        w = __shfl_sync(0xffffffff, w, 0);
        corr = __shfl_sync(0xffffffff, corr, 0);
        new_max = __shfl_sync(0xffffffff, new_max, 0);

        const __half *v_row = v_cache + ((size_t)j * 8 + (size_t)kv_h) * 128;
        float v0 = __half2float(v_row[lane + 0 * 32]);
        float v1 = __half2float(v_row[lane + 1 * 32]);
        float v2 = __half2float(v_row[lane + 2 * 32]);
        float v3 = __half2float(v_row[lane + 3 * 32]);

        if (new_max) {
            out0 = out0 * corr + v0;
            out1 = out1 * corr + v1;
            out2 = out2 * corr + v2;
            out3 = out3 * corr + v3;
        } else {
            out0 += w * v0;
            out1 += w * v1;
            out2 += w * v2;
            out3 += w * v3;
        }
    }

    float inv_sum = 0.0f;
    if (lane == 0) inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    inv_sum = __shfl_sync(0xffffffff, inv_sum, 0);

    out_q[h * 128 + (lane + 0 * 32)] = out0 * inv_sum;
    out_q[h * 128 + (lane + 1 * 32)] = out1 * inv_sum;
    out_q[h * 128 + (lane + 2 * 32)] = out2 * inv_sum;
    out_q[h * 128 + (lane + 3 * 32)] = out3 * inv_sum;
}

extern "C" __global__ void vox_attn_q4_kv8_f32(float *out_q,      /* [32*128] */
                                               const float *q,    /* [32*128] */
                                               const float *k_cache, /* [max_seq*8*128] */
                                               const float *v_cache, /* [max_seq*8*128] */
                                               int total_seq,
                                               int window_size,
                                               float scale) {
    int h = (int)blockIdx.x;     /* 0..31 */
    int lane = (int)threadIdx.x; /* 0..31 */
    if (h >= 32 || lane >= 32) return;

    int kv_h = h >> 2;

    float qv0 = q[h * 128 + (lane + 0 * 32)];
    float qv1 = q[h * 128 + (lane + 1 * 32)];
    float qv2 = q[h * 128 + (lane + 2 * 32)];
    float qv3 = q[h * 128 + (lane + 3 * 32)];

    int end = total_seq;
    int start = 0;
    if (window_size > 0) {
        int s = end - window_size;
        if (s > 0) start = s;
    }
    if (start < 0) start = 0;
    if (end < start) end = start;

    float max_score = -1.0e30f;
    float sum_exp = 0.0f;

    float out0 = 0.0f;
    float out1 = 0.0f;
    float out2 = 0.0f;
    float out3 = 0.0f;

    for (int j = start; j < end; j++) {
        const float *k_row = k_cache + ((size_t)j * 8 + (size_t)kv_h) * 128;
        float k0 = k_row[lane + 0 * 32];
        float k1 = k_row[lane + 1 * 32];
        float k2 = k_row[lane + 2 * 32];
        float k3 = k_row[lane + 3 * 32];

        float partial = qv0 * k0 + qv1 * k1 + qv2 * k2 + qv3 * k3;
        float sum = warp_reduce_sum(partial);
        sum = __shfl_sync(0xffffffff, sum, 0);

        float score = sum * scale;

        float w = 0.0f;
        float corr = 1.0f;
        int new_max = 0;
        if (lane == 0) {
            if (score > max_score) {
                corr = __expf(max_score - score);
                sum_exp = sum_exp * corr + 1.0f;
                max_score = score;
                w = 1.0f;
                new_max = 1;
            } else {
                w = __expf(score - max_score);
                sum_exp += w;
                corr = 1.0f;
                new_max = 0;
            }
        }
        w = __shfl_sync(0xffffffff, w, 0);
        corr = __shfl_sync(0xffffffff, corr, 0);
        new_max = __shfl_sync(0xffffffff, new_max, 0);

        const float *v_row = v_cache + ((size_t)j * 8 + (size_t)kv_h) * 128;
        float v0 = v_row[lane + 0 * 32];
        float v1 = v_row[lane + 1 * 32];
        float v2 = v_row[lane + 2 * 32];
        float v3 = v_row[lane + 3 * 32];

        if (new_max) {
            out0 = out0 * corr + v0;
            out1 = out1 * corr + v1;
            out2 = out2 * corr + v2;
            out3 = out3 * corr + v3;
        } else {
            out0 += w * v0;
            out1 += w * v1;
            out2 += w * v2;
            out3 += w * v3;
        }
    }

    float inv_sum = 0.0f;
    if (lane == 0) inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    inv_sum = __shfl_sync(0xffffffff, inv_sum, 0);

    out_q[h * 128 + (lane + 0 * 32)] = out0 * inv_sum;
    out_q[h * 128 + (lane + 1 * 32)] = out1 * inv_sum;
    out_q[h * 128 + (lane + 2 * 32)] = out2 * inv_sum;
    out_q[h * 128 + (lane + 3 * 32)] = out3 * inv_sum;
}

extern "C" __global__ void vox_causal_attn_f32(float *out,
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
    /* Warp-level implementation: one warp (32 threads) computes one (head, query)
     * output vector. This avoids the massive scores matrix and eliminates the
     * heavy __syncthreads() usage of the original block-reduction kernel. */
    int h = (int)blockIdx.x;
    int i = (int)blockIdx.y;
    int lane = (int)threadIdx.x;

    if (lane >= 32) return;
    if (h >= n_heads || i >= seq_q) return;
    if (head_dim <= 0 || head_dim > 128) return;

    int heads_per_kv = n_heads / n_kv_heads;
    int kv_h = h / heads_per_kv;

    int q_hidden = n_heads * head_dim;
    int kv_hidden = n_kv_heads * head_dim;

    /* Load Q once. Each lane owns up to 4 elements (head_dim<=128). */
    float qv[4] = {0.f, 0.f, 0.f, 0.f};
    int elems_per_lane = (head_dim + 31) >> 5; /* ceil(head_dim/32) */
#pragma unroll
    for (int e = 0; e < 4; e++) {
        if (e < elems_per_lane) {
            int idx = lane + (e << 5);
            if (idx < head_dim) qv[e] = Q[i * q_hidden + h * head_dim + idx];
        }
    }

    int global_pos = q_offset + i;
    int k_start = 0;
    if (window_size > 0) {
        int s = global_pos - window_size + 1;
        if (s > 0) k_start = s;
    }
    int k_end = global_pos + 1;
    if (k_end > seq_k) k_end = seq_k;
    if (k_start < 0) k_start = 0;
    if (k_end < k_start) k_end = k_start;

    float max_score = -1.0e30f;

    /* Pass 1: max score */
    for (int j = k_start; j < k_end; j++) {
        float partial = 0.0f;
#pragma unroll
        for (int e = 0; e < 4; e++) {
            if (e < elems_per_lane) {
                int idx = lane + (e << 5);
                if (idx < head_dim) {
                    float kv = K[j * kv_hidden + kv_h * head_dim + idx];
                    partial += qv[e] * kv;
                }
            }
        }

        float sum = warp_reduce_sum(partial);
        sum = __shfl_sync(0xffffffff, sum, 0);

        if (lane == 0) {
            float sc = sum * scale;
            if (sc > max_score) max_score = sc;
        }
    }

    max_score = __shfl_sync(0xffffffff, max_score, 0);

    /* Pass 2: sumexp + weighted sum */
    float outv[4] = {0.f, 0.f, 0.f, 0.f};
    float sumexp = 0.0f;

    for (int j = k_start; j < k_end; j++) {
        float partial = 0.0f;
#pragma unroll
        for (int e = 0; e < 4; e++) {
            if (e < elems_per_lane) {
                int idx = lane + (e << 5);
                if (idx < head_dim) {
                    float kv = K[j * kv_hidden + kv_h * head_dim + idx];
                    partial += qv[e] * kv;
                }
            }
        }

        float sum = warp_reduce_sum(partial);
        sum = __shfl_sync(0xffffffff, sum, 0);

        float w = 0.0f;
        if (lane == 0) {
            float sc = sum * scale;
            w = __expf(sc - max_score);
            sumexp += w;
        }
        w = __shfl_sync(0xffffffff, w, 0);

#pragma unroll
        for (int e = 0; e < 4; e++) {
            if (e < elems_per_lane) {
                int idx = lane + (e << 5);
                if (idx < head_dim) {
                    float vv = V[j * kv_hidden + kv_h * head_dim + idx];
                    outv[e] += w * vv;
                }
            }
        }
    }

    float inv_sum = 0.0f;
    if (lane == 0) inv_sum = (sumexp > 0.0f) ? (1.0f / sumexp) : 0.0f;
    inv_sum = __shfl_sync(0xffffffff, inv_sum, 0);

#pragma unroll
    for (int e = 0; e < 4; e++) {
        if (e < elems_per_lane) {
            int idx = lane + (e << 5);
            if (idx < head_dim) {
                out[i * q_hidden + h * head_dim + idx] = outv[e] * inv_sum;
            }
        }
    }
}

extern "C" __global__ void vox_pack_heads_f32(float *dst,
                                              const float *src,
                                              int seq,
                                              int n_heads,
                                              int head_dim) {
    /* Reorder:
     * src: [seq, n_heads*head_dim] (interleaved by head per row)
     * dst: [n_heads, seq, head_dim] (contiguous per head)
     */
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = seq * n_heads * head_dim;
    if (idx >= total) return;

    int d = idx % head_dim;
    int t = idx / head_dim;
    int i = t % seq;
    int h = t / seq;

    int src_stride = n_heads * head_dim;
    dst[(h * seq + i) * head_dim + d] = src[i * src_stride + h * head_dim + d];
}

extern "C" __global__ void vox_unpack_heads_f32(float *dst,
                                                const float *src,
                                                int seq,
                                                int n_heads,
                                                int head_dim) {
    /* Reverse of vox_pack_heads_f32. */
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = seq * n_heads * head_dim;
    if (idx >= total) return;

    int d = idx % head_dim;
    int t = idx / head_dim;
    int i = t % seq;
    int h = t / seq;

    int dst_stride = n_heads * head_dim;
    dst[i * dst_stride + h * head_dim + d] = src[(h * seq + i) * head_dim + d];
}

extern "C" __global__ void vox_expand_kv_heads_f32(float *dst,
                                                   const float *src,
                                                   int seq,
                                                   int n_heads,
                                                   int n_kv_heads,
                                                   int head_dim) {
    /* Replicate KV heads into per-query-head layout:
     * src: [n_kv_heads, seq, head_dim]
     * dst: [n_heads,    seq, head_dim]
     */
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = seq * n_heads * head_dim;
    if (idx >= total) return;

    int d = idx % head_dim;
    int t = idx / head_dim;
    int i = t % seq;
    int h = t / seq;

    int heads_per_kv = n_heads / n_kv_heads;
    int kv_h = h / heads_per_kv;

    dst[(h * seq + i) * head_dim + d] = src[(kv_h * seq + i) * head_dim + d];
}

static __device__ __forceinline__ float block_reduce_max(float x, float *shmem) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    for (int offset = 16; offset > 0; offset >>= 1) {
        float y = __shfl_down_sync(0xffffffff, x, offset);
        x = (y > x) ? y : x;
    }
    if (lane == 0) shmem[warp] = x;
    __syncthreads();

    float vmax = -1.0e30f;
    if (warp == 0) {
        vmax = (lane < (blockDim.x >> 5)) ? shmem[lane] : -1.0e30f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float y = __shfl_down_sync(0xffffffff, vmax, offset);
            vmax = (y > vmax) ? y : vmax;
        }
    }
    __syncthreads();
    if (warp == 0 && lane == 0) shmem[0] = vmax;
    __syncthreads();
    return shmem[0];
}

extern "C" __global__ void vox_rms_norm_f32(float *out,
                                            const float *x,
                                            const float *weight,
                                            int rows,
                                            int hidden,
                                            float eps) {
    int r = (int)blockIdx.x;
    if (r >= rows) return;

    const float *x_row = x + (size_t)r * (size_t)hidden;
    float *o_row = out + (size_t)r * (size_t)hidden;

    __shared__ float sh[256];
    float sum = 0.0f;
    for (int i = (int)threadIdx.x; i < hidden; i += (int)blockDim.x) {
        float v = x_row[i];
        sum += v * v;
    }
    sh[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
        __syncthreads();
    }

    float inv_rms = rsqrtf(sh[0] / (float)hidden + eps);
    for (int i = (int)threadIdx.x; i < hidden; i += (int)blockDim.x) {
        o_row[i] = x_row[i] * inv_rms * weight[i];
    }
}

extern "C" __global__ void vox_add_bias_f32(float *x,
                                            const float *bias,
                                            int rows,
                                            int cols) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = rows * cols;
    if (idx >= total) return;
    int c = idx % cols;
    x[idx] += bias[c];
}

extern "C" __global__ void vox_add_inplace_f32(float *x,
                                               const float *y,
                                               int n) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) return;
    x[idx] += y[idx];
}

extern "C" __global__ void vox_mul_inplace_f32(float *x,
                                               const float *y,
                                               int n) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) return;
    x[idx] *= y[idx];
}

extern "C" __global__ void vox_mul_1p_inplace_f32(float *x,
                                                  const float *scale,
                                                  int n) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) return;
    x[idx] *= (1.0f + scale[idx]);
}

extern "C" __global__ void vox_silu_inplace_f32(float *x, int n) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) return;
    float v = x[idx];
    x[idx] = v / (1.0f + __expf(-v));
}

extern "C" __global__ void vox_gelu_inplace_f32(float *x, int n) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) return;
    float v = x[idx];
    /* 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3))) */
    float x3 = v * v * v;
    float inner = 0.7978845608028654f * (v + 0.044715f * x3);
    x[idx] = 0.5f * v * (1.0f + tanhf(inner));
}

extern "C" __global__ void vox_f32_to_bf16(uint16_t *dst,
                                           const float *src,
                                           int n) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) return;
    __nv_bfloat16 h = __float2bfloat16_rn(src[idx]);
    (( __nv_bfloat16 *)dst)[idx] = h;
}

extern "C" __global__ void vox_f32_to_f16(uint16_t *dst,
                                          const float *src,
                                          int n) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) return;
    __half h = __float2half_rn(src[idx]);
    (( __half *)dst)[idx] = h;
}

extern "C" __global__ void vox_apply_rope_f32(float *x,
                                              const float *freqs,
                                              int seq,
                                              int n_heads,
                                              int head_dim) {
    int half_dim = head_dim / 2;
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = seq * n_heads * half_dim;
    if (idx >= total) return;

    int d = idx % half_dim;
    int t = idx / half_dim;
    int h = t % n_heads;
    int s = t / n_heads;

    const float *f = freqs + (size_t)s * (size_t)half_dim * 2 + (size_t)d * 2;
    float c = f[0];
    float si = f[1];

    int row_stride = n_heads * head_dim;
    float *row = x + (size_t)s * (size_t)row_stride + (size_t)h * (size_t)head_dim;
    int i0 = d * 2;
    int i1 = i0 + 1;
    float a = row[i0];
    float b = row[i1];
    row[i0] = a * c - b * si;
    row[i1] = a * si + b * c;
}

extern "C" __global__ void vox_downsample4_concat_f32(float *dst,
                                                      const float *src,
                                                      int start,
                                                      int enc_len,
                                                      int dim) {
    int ds_len = enc_len / 4;
    int ds_dim = dim * 4;
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = ds_len * ds_dim;
    if (idx >= total) return;

    int j = idx % ds_dim;
    int i = idx / ds_dim;
    int seg = j / dim;  /* 0..3 */
    int col = j - seg * dim;
    int src_row = start + i * 4 + seg;
    dst[idx] = src[(size_t)src_row * (size_t)dim + (size_t)col];
}

extern "C" __global__ void vox_argmax_f32(int *out_idx,
                                          const float *x,
                                          int n) {
    /* Simple 1-block argmax for vocab-sized arrays.
     * n can be large (e.g. 131072). */
    int tid = (int)threadIdx.x;
    float best = -1.0e30f;
    int best_i = 0;
    for (int i = tid; i < n; i += (int)blockDim.x) {
        float v = x[i];
        if (v > best) { best = v; best_i = i; }
    }

    __shared__ float sh_val[256];
    __shared__ int sh_idx[256];
    sh_val[tid] = best;
    sh_idx[tid] = best_i;
    __syncthreads();

    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float v = sh_val[tid + stride];
            if (v > sh_val[tid]) {
                sh_val[tid] = v;
                sh_idx[tid] = sh_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) out_idx[0] = sh_idx[0];
}

extern "C" __global__ void vox_masked_softmax_causal_inplace_f32(float *scores,
                                                                 int seq_q,
                                                                 int seq_k,
                                                                 int window_size,
                                                                 int q_offset) {
    /* scores layout: [n_heads, seq_q, seq_k] row-major */
    int h = (int)blockIdx.x;
    int i = (int)blockIdx.y;
    if (i >= seq_q) return;

    float *row = scores + ((size_t)h * (size_t)seq_q + (size_t)i) * (size_t)seq_k;

    int global_pos = q_offset + i;
    int k_start = 0;
    if (window_size > 0 && global_pos - window_size + 1 > 0) {
        k_start = global_pos - window_size + 1;
    }
    int k_end = global_pos + 1;
    if (k_end > seq_k) k_end = seq_k;
    if (k_start < 0) k_start = 0;
    if (k_end < k_start) k_end = k_start;

    __shared__ float sh_warp[8]; /* supports up to 8 warps (256 threads) */
    __shared__ float sh_sum;

    float tmax = -1.0e30f;
    for (int j = k_start + (int)threadIdx.x; j < k_end; j += (int)blockDim.x) {
        float v = row[j];
        tmax = (v > tmax) ? v : tmax;
    }
    float vmax = block_reduce_max(tmax, sh_warp);

    float tsum = 0.0f;
    for (int j = k_start + (int)threadIdx.x; j < k_end; j += (int)blockDim.x) {
        tsum += __expf(row[j] - vmax);
    }
    float sum = block_reduce_sum_256(tsum, sh_warp);

    if (threadIdx.x == 0) {
        sh_sum = sum;
    }
    __syncthreads();

    float inv = (sh_sum > 0.0f) ? (1.0f / sh_sum) : 0.0f;
    for (int j = (int)threadIdx.x; j < seq_k; j += (int)blockDim.x) {
        if (j < k_start || j >= k_end) {
            row[j] = 0.0f;
        } else {
            row[j] = __expf(row[j] - vmax) * inv;
        }
    }
}
