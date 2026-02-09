// CUDA kernels compiled to a CUBIN and launched via the CUDA driver API.
// This avoids a libcudart dependency (which has been unreliable under WSL2
// for this project) while still letting us write kernels in CUDA C.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

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

/* v2 attention kernels: same math, different per-thread layout (contiguous 4-dim
 * chunks per lane). These are opt-in via VOX_CUDA_ATTN_V2=1. */

extern "C" __global__ void vox_attn_q4_kv8_fp16_v2(float *out_q,          /* [32*128] */
                                                   const float *q,        /* [32*128] */
                                                   const __half *k_cache,  /* [max_seq*8*128] */
                                                   const __half *v_cache,  /* [max_seq*8*128] */
                                                   int total_seq,
                                                   int window_size,
                                                   float scale) {
    int h = (int)blockIdx.x;     /* 0..31 */
    int lane = (int)threadIdx.x; /* 0..31 */
    if (h >= 32 || lane >= 32) return;

    int kv_h = h >> 2; /* 4 query heads share 1 KV head */

    const float4 qv = ((const float4 *)(q + h * 128))[lane];

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

    const __half2 *k2 = (const __half2 *)k_cache;
    const __half2 *v2 = (const __half2 *)v_cache;
    size_t row_stride = (size_t)(8 * 128) / 2; /* 1024 half -> 512 half2 */
    size_t head_off = (size_t)kv_h * (size_t)128 / 2; /* 128 half -> 64 half2 */
    int off2 = lane * 2;

    for (int j = start; j < end; j++) {
        const __half2 *k_row = k2 + (size_t)j * row_stride + head_off;
        float2 k01 = __half22float2(k_row[off2 + 0]);
        float2 k23 = __half22float2(k_row[off2 + 1]);

        float partial = qv.x * k01.x + qv.y * k01.y + qv.z * k23.x + qv.w * k23.y;
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

        const __half2 *v_row = v2 + (size_t)j * row_stride + head_off;
        float2 v01 = __half22float2(v_row[off2 + 0]);
        float2 v23 = __half22float2(v_row[off2 + 1]);

        if (new_max) {
            out0 = out0 * corr + v01.x;
            out1 = out1 * corr + v01.y;
            out2 = out2 * corr + v23.x;
            out3 = out3 * corr + v23.y;
        } else {
            out0 += w * v01.x;
            out1 += w * v01.y;
            out2 += w * v23.x;
            out3 += w * v23.y;
        }
    }

    float inv_sum = 0.0f;
    if (lane == 0) inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    inv_sum = __shfl_sync(0xffffffff, inv_sum, 0);

    ((float4 *)(out_q + h * 128))[lane] = make_float4(out0 * inv_sum,
                                                      out1 * inv_sum,
                                                      out2 * inv_sum,
                                                      out3 * inv_sum);
}

extern "C" __global__ void vox_attn_q4_kv8_f32_v2(float *out_q,      /* [32*128] */
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

    const float4 qv = ((const float4 *)(q + h * 128))[lane];

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
        const float *k_head = k_cache + ((size_t)j * 8 + (size_t)kv_h) * 128;
        const float4 k4 = ((const float4 *)k_head)[lane];

        float partial = qv.x * k4.x + qv.y * k4.y + qv.z * k4.z + qv.w * k4.w;
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

        const float *v_head = v_cache + ((size_t)j * 8 + (size_t)kv_h) * 128;
        const float4 v4 = ((const float4 *)v_head)[lane];

        if (new_max) {
            out0 = out0 * corr + v4.x;
            out1 = out1 * corr + v4.y;
            out2 = out2 * corr + v4.z;
            out3 = out3 * corr + v4.w;
        } else {
            out0 += w * v4.x;
            out1 += w * v4.y;
            out2 += w * v4.z;
            out3 += w * v4.w;
        }
    }

    float inv_sum = 0.0f;
    if (lane == 0) inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    inv_sum = __shfl_sync(0xffffffff, inv_sum, 0);

    ((float4 *)(out_q + h * 128))[lane] = make_float4(out0 * inv_sum,
                                                      out1 * inv_sum,
                                                      out2 * inv_sum,
                                                      out3 * inv_sum);
}

/* Decoder graph helpers: dynamic KV append + dynamic total_seq (pos on device).
 *
 * These kernels exist primarily to enable CUDA Graph capture for the decoder
 * step by removing host-side pointer arithmetic on `pos`. */

extern "C" __global__ void vox_kv_append_fp16_dyn(__half *k_base, __half *v_base,
                                                  const float *k,
                                                  const float *v,
                                                  const int *p_pos,
                                                  int kv_dim) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= kv_dim) return;
    int pos = *p_pos;
    int dst = pos * kv_dim + idx;
    k_base[dst] = __float2half(k[idx]);
    v_base[dst] = __float2half(v[idx]);
}

extern "C" __global__ void vox_kv_append_f32_dyn(float *k_base, float *v_base,
                                                 const float *k,
                                                 const float *v,
                                                 const int *p_pos,
                                                 int kv_dim) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= kv_dim) return;
    int pos = *p_pos;
    int dst = pos * kv_dim + idx;
    k_base[dst] = k[idx];
    v_base[dst] = v[idx];
}

extern "C" __global__ void vox_attn_q4_kv8_fp16_dyn(float *out_q,          /* [32*128] */
                                                    const float *q,        /* [32*128] */
                                                    const __half *k_cache,  /* [max_seq*8*128] */
                                                    const __half *v_cache,  /* [max_seq*8*128] */
                                                    const int *p_pos,
                                                    int window_size,
                                                    float scale) {
    int h = (int)blockIdx.x;     /* 0..31 */
    int lane = (int)threadIdx.x; /* 0..31 */
    if (h >= 32 || lane >= 32) return;

    int pos = 0;
    if (lane == 0) pos = *p_pos;
    pos = __shfl_sync(0xffffffff, pos, 0);
    int end = pos + 1;

    int kv_h = h >> 2; /* 4 query heads share 1 KV head */

    float qv0 = q[h * 128 + (lane + 0 * 32)];
    float qv1 = q[h * 128 + (lane + 1 * 32)];
    float qv2 = q[h * 128 + (lane + 2 * 32)];
    float qv3 = q[h * 128 + (lane + 3 * 32)];

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
        const __half *k_row = k_cache + ((size_t)j * 8 + (size_t)kv_h) * 128;
        float k0 = __half2float(k_row[lane + 0 * 32]);
        float k1 = __half2float(k_row[lane + 1 * 32]);
        float k2 = __half2float(k_row[lane + 2 * 32]);
        float k3 = __half2float(k_row[lane + 3 * 32]);

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

extern "C" __global__ void vox_attn_q4_kv8_f32_dyn(float *out_q,      /* [32*128] */
                                                   const float *q,    /* [32*128] */
                                                   const float *k_cache, /* [max_seq*8*128] */
                                                   const float *v_cache, /* [max_seq*8*128] */
                                                   const int *p_pos,
                                                   int window_size,
                                                   float scale) {
    int h = (int)blockIdx.x;     /* 0..31 */
    int lane = (int)threadIdx.x; /* 0..31 */
    if (h >= 32 || lane >= 32) return;

    int pos = 0;
    if (lane == 0) pos = *p_pos;
    pos = __shfl_sync(0xffffffff, pos, 0);
    int end = pos + 1;

    int kv_h = h >> 2;

    float qv0 = q[h * 128 + (lane + 0 * 32)];
    float qv1 = q[h * 128 + (lane + 1 * 32)];
    float qv2 = q[h * 128 + (lane + 2 * 32)];
    float qv3 = q[h * 128 + (lane + 3 * 32)];

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

/* v2 dynamic attention kernels: same math/layout as v2, but total_seq is
 * derived from `*p_pos` (device scalar). Used to enable decoder CUDA Graphs. */

extern "C" __global__ void vox_attn_q4_kv8_fp16_dyn_v2(float *out_q,          /* [32*128] */
                                                       const float *q,        /* [32*128] */
                                                       const __half *k_cache,  /* [max_seq*8*128] */
                                                       const __half *v_cache,  /* [max_seq*8*128] */
                                                       const int *p_pos,
                                                       int window_size,
                                                       float scale) {
    int h = (int)blockIdx.x;     /* 0..31 */
    int lane = (int)threadIdx.x; /* 0..31 */
    if (h >= 32 || lane >= 32) return;

    int pos = 0;
    if (lane == 0) pos = *p_pos;
    pos = __shfl_sync(0xffffffff, pos, 0);
    int end = pos + 1;

    int kv_h = h >> 2; /* 4 query heads share 1 KV head */

    const float4 qv = ((const float4 *)(q + h * 128))[lane];

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

    const __half2 *k2 = (const __half2 *)k_cache;
    const __half2 *v2 = (const __half2 *)v_cache;
    size_t row_stride = (size_t)(8 * 128) / 2;              /* 1024 half -> 512 half2 */
    size_t head_off = (size_t)kv_h * (size_t)128 / 2;       /* 128 half -> 64 half2 */
    int off2 = lane * 2;                                    /* 0..62 */

    for (int j = start; j < end; j++) {
        const __half2 *k_row = k2 + (size_t)j * row_stride + head_off;
        float2 k01 = __half22float2(k_row[off2 + 0]);
        float2 k23 = __half22float2(k_row[off2 + 1]);

        float partial = qv.x * k01.x + qv.y * k01.y + qv.z * k23.x + qv.w * k23.y;
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

        const __half2 *v_row = v2 + (size_t)j * row_stride + head_off;
        float2 v01 = __half22float2(v_row[off2 + 0]);
        float2 v23 = __half22float2(v_row[off2 + 1]);

        if (new_max) {
            out0 = out0 * corr + v01.x;
            out1 = out1 * corr + v01.y;
            out2 = out2 * corr + v23.x;
            out3 = out3 * corr + v23.y;
        } else {
            out0 += w * v01.x;
            out1 += w * v01.y;
            out2 += w * v23.x;
            out3 += w * v23.y;
        }
    }

    float inv_sum = 0.0f;
    if (lane == 0) inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    inv_sum = __shfl_sync(0xffffffff, inv_sum, 0);

    ((float4 *)(out_q + h * 128))[lane] = make_float4(out0 * inv_sum,
                                                      out1 * inv_sum,
                                                      out2 * inv_sum,
                                                      out3 * inv_sum);
}

extern "C" __global__ void vox_attn_q4_kv8_f32_dyn_v2(float *out_q,      /* [32*128] */
                                                      const float *q,    /* [32*128] */
                                                      const float *k_cache, /* [max_seq*8*128] */
                                                      const float *v_cache, /* [max_seq*8*128] */
                                                      const int *p_pos,
                                                      int window_size,
                                                      float scale) {
    int h = (int)blockIdx.x;     /* 0..31 */
    int lane = (int)threadIdx.x; /* 0..31 */
    if (h >= 32 || lane >= 32) return;

    int pos = 0;
    if (lane == 0) pos = *p_pos;
    pos = __shfl_sync(0xffffffff, pos, 0);
    int end = pos + 1;

    int kv_h = h >> 2;

    const float4 qv = ((const float4 *)(q + h * 128))[lane];

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
        const float *k_head = k_cache + ((size_t)j * 8 + (size_t)kv_h) * 128;
        const float4 k4 = ((const float4 *)k_head)[lane];

        float partial = qv.x * k4.x + qv.y * k4.y + qv.z * k4.z + qv.w * k4.w;
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

        const float *v_head = v_cache + ((size_t)j * 8 + (size_t)kv_h) * 128;
        const float4 v4 = ((const float4 *)v_head)[lane];

        if (new_max) {
            out0 = out0 * corr + v4.x;
            out1 = out1 * corr + v4.y;
            out2 = out2 * corr + v4.z;
            out3 = out3 * corr + v4.w;
        } else {
            out0 += w * v4.x;
            out1 += w * v4.y;
            out2 += w * v4.z;
            out3 += w * v4.w;
        }
    }

    float inv_sum = 0.0f;
    if (lane == 0) inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    inv_sum = __shfl_sync(0xffffffff, inv_sum, 0);

    ((float4 *)(out_q + h * 128))[lane] = make_float4(out0 * inv_sum,
                                                      out1 * inv_sum,
                                                      out2 * inv_sum,
                                                      out3 * inv_sum);
}

/* v3 decoder attention kernels: chunked reduction + GQA shared-load.
 *
 * Motivation: the v1/v2 kernels launch 1 warp per *query head* and therefore
 * read the same KV head 4x (since GQA: 4 query heads share 1 KV head). v3
 * reduces redundant KV loads by having one block compute 4 query heads for a
 * single KV head, over a fixed set of chunks (so it can be used under CUDA
 * Graph capture when paired with a dyn variant).
 *
 * Structure:
 * - partial kernel: grid=(kv_heads=8, chunks=N), block=(128 threads = 4 warps)
 *   Each warp computes a partial online-softmax state + partial output vector
 *   for one query head over its chunk.
 * - reduce kernel: grid=(query_heads=32), block=(32 threads)
 *   Combine chunk partials using standard max/sumexp rescaling.
 *
 * This is intentionally opt-in and tuned for head_dim=128, n_heads=32, kv_heads=8. */

#define VOX_ATTN_V3_CHUNK 256
#define VOX_ATTN_V3_TILE 4

extern "C" __global__ void vox_attn_q4_kv8_fp16_v3_partial(float *out_part,     /* [32*n_chunks*128] */
                                                           float *max_part,     /* [32*n_chunks] */
                                                           float *sum_part,     /* [32*n_chunks] */
                                                           const float *q,       /* [32*128] */
                                                           const __half *k_cache,/* [max_seq*8*128] */
                                                           const __half *v_cache,/* [max_seq*8*128] */
                                                           int total_seq,
                                                           int window_size,
                                                           float scale,
                                                           int n_chunks) {
    int kv_h = (int)blockIdx.x;       /* 0..7 */
    int chunk = (int)blockIdx.y;      /* 0..n_chunks-1 */
    int tid = (int)threadIdx.x;       /* 0..127 */
    int warp = tid >> 5;              /* 0..3 */
    int lane = tid & 31;              /* 0..31 */
    if (kv_h >= 8 || warp >= 4) return;

    int h = kv_h * 4 + warp;          /* query head 0..31 */

    int end = total_seq;
    int start = 0;
    if (window_size > 0) {
        int s = end - window_size;
        if (s > 0) start = s;
    }
    if (start < 0) start = 0;
    if (end < start) end = start;

    /* Chunks are relative to the active attention window [start,end), not
     * absolute token indices. This keeps the grid shape fixed (n_chunks) while
     * sliding over long sequences. */
    int chunk_start = start + chunk * VOX_ATTN_V3_CHUNK;
    int chunk_end = chunk_start + VOX_ATTN_V3_CHUNK;
    if (chunk_start > end) chunk_start = end;
    if (chunk_end > end) chunk_end = end;

    /* Outputs are stored densely by query head (h). */
    int base_max = h * n_chunks + chunk;
    size_t base_vec = ((size_t)base_max) * (size_t)128;

    if (chunk_start >= chunk_end) {
        /* Empty chunk: write neutral partials. */
        if (lane == 0) {
            max_part[base_max] = -1.0e30f;
            sum_part[base_max] = 0.0f;
        }
        ((float4 *)(out_part + base_vec))[lane] = make_float4(0.f, 0.f, 0.f, 0.f);
        return;
    }

    const float4 qv = ((const float4 *)(q + h * 128))[lane];

    float max_score = -1.0e30f;
    float sum_exp = 0.0f;
    float out0 = 0.0f, out1 = 0.0f, out2 = 0.0f, out3 = 0.0f;

    const __half2 *k2 = (const __half2 *)k_cache;
    const __half2 *v2 = (const __half2 *)v_cache;
    size_t row_stride = (size_t)(8 * 128) / 2;        /* 512 half2 per token */
    size_t head_off = (size_t)kv_h * (size_t)128 / 2; /* 64 half2 per head */

    __shared__ __half2 shK[VOX_ATTN_V3_TILE][64];
    __shared__ __half2 shV[VOX_ATTN_V3_TILE][64];

    int off2 = lane * 2; /* half2 index for this lane's 4 dims (2*lane,2*lane+1) */

    for (int j0 = chunk_start; j0 < chunk_end; j0 += VOX_ATTN_V3_TILE) {
#pragma unroll
        for (int t = 0; t < VOX_ATTN_V3_TILE; t++) {
            int j = j0 + t;
            if (j >= chunk_end) continue;
            const __half2 *k_row = k2 + (size_t)j * row_stride + head_off;
            const __half2 *v_row = v2 + (size_t)j * row_stride + head_off;
            if (tid < 64) {
                shK[t][tid] = k_row[tid];
            } else {
                shV[t][tid - 64] = v_row[tid - 64];
            }
        }
        __syncthreads();

#pragma unroll
        for (int t = 0; t < VOX_ATTN_V3_TILE; t++) {
            int j = j0 + t;
            if (j >= chunk_end) break;

            float2 k01 = __half22float2(shK[t][off2 + 0]);
            float2 k23 = __half22float2(shK[t][off2 + 1]);
            float partial = qv.x * k01.x + qv.y * k01.y + qv.z * k23.x + qv.w * k23.y;
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

            float2 v01 = __half22float2(shV[t][off2 + 0]);
            float2 v23 = __half22float2(shV[t][off2 + 1]);
            if (new_max) {
                out0 = out0 * corr + v01.x;
                out1 = out1 * corr + v01.y;
                out2 = out2 * corr + v23.x;
                out3 = out3 * corr + v23.y;
            } else {
                out0 += w * v01.x;
                out1 += w * v01.y;
                out2 += w * v23.x;
                out3 += w * v23.y;
            }
        }
        __syncthreads();
    }

    if (lane == 0) {
        max_part[base_max] = max_score;
        sum_part[base_max] = sum_exp;
    }
    ((float4 *)(out_part + base_vec))[lane] = make_float4(out0, out1, out2, out3);
}

extern "C" __global__ void vox_attn_q4_kv8_fp16_dyn_v3_partial(float *out_part,      /* [32*n_chunks*128] */
                                                               float *max_part,      /* [32*n_chunks] */
                                                               float *sum_part,      /* [32*n_chunks] */
                                                               const float *q,        /* [32*128] */
                                                               const __half *k_cache, /* [max_seq*8*128] */
                                                               const __half *v_cache, /* [max_seq*8*128] */
                                                               const int *p_pos,
                                                               int window_size,
                                                               float scale,
                                                               int n_chunks) {
    int kv_h = (int)blockIdx.x;
    int chunk = (int)blockIdx.y;
    int tid = (int)threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    if (kv_h >= 8 || warp >= 4) return;

    int h = kv_h * 4 + warp;

    int pos = 0;
    if (lane == 0) pos = *p_pos;
    pos = __shfl_sync(0xffffffff, pos, 0);
    int end = pos + 1;

    int start = 0;
    if (window_size > 0) {
        int s = end - window_size;
        if (s > 0) start = s;
    }
    if (start < 0) start = 0;
    if (end < start) end = start;

    /* Chunks are relative to the active attention window [start,end), not
     * absolute token indices. This keeps the grid shape fixed (n_chunks) while
     * sliding over long sequences. */
    int chunk_start = start + chunk * VOX_ATTN_V3_CHUNK;
    int chunk_end = chunk_start + VOX_ATTN_V3_CHUNK;
    if (chunk_start > end) chunk_start = end;
    if (chunk_end > end) chunk_end = end;

    int base_max = h * n_chunks + chunk;
    size_t base_vec = ((size_t)base_max) * (size_t)128;

    if (chunk_start >= chunk_end) {
        if (lane == 0) {
            max_part[base_max] = -1.0e30f;
            sum_part[base_max] = 0.0f;
        }
        ((float4 *)(out_part + base_vec))[lane] = make_float4(0.f, 0.f, 0.f, 0.f);
        return;
    }

    const float4 qv = ((const float4 *)(q + h * 128))[lane];

    float max_score = -1.0e30f;
    float sum_exp = 0.0f;
    float out0 = 0.0f, out1 = 0.0f, out2 = 0.0f, out3 = 0.0f;

    const __half2 *k2 = (const __half2 *)k_cache;
    const __half2 *v2 = (const __half2 *)v_cache;
    size_t row_stride = (size_t)(8 * 128) / 2;
    size_t head_off = (size_t)kv_h * (size_t)128 / 2;

    __shared__ __half2 shK[VOX_ATTN_V3_TILE][64];
    __shared__ __half2 shV[VOX_ATTN_V3_TILE][64];

    int off2 = lane * 2;

    for (int j0 = chunk_start; j0 < chunk_end; j0 += VOX_ATTN_V3_TILE) {
#pragma unroll
        for (int t = 0; t < VOX_ATTN_V3_TILE; t++) {
            int j = j0 + t;
            if (j >= chunk_end) continue;
            const __half2 *k_row = k2 + (size_t)j * row_stride + head_off;
            const __half2 *v_row = v2 + (size_t)j * row_stride + head_off;
            if (tid < 64) {
                shK[t][tid] = k_row[tid];
            } else {
                shV[t][tid - 64] = v_row[tid - 64];
            }
        }
        __syncthreads();

#pragma unroll
        for (int t = 0; t < VOX_ATTN_V3_TILE; t++) {
            int j = j0 + t;
            if (j >= chunk_end) break;

            float2 k01 = __half22float2(shK[t][off2 + 0]);
            float2 k23 = __half22float2(shK[t][off2 + 1]);
            float partial = qv.x * k01.x + qv.y * k01.y + qv.z * k23.x + qv.w * k23.y;
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

            float2 v01 = __half22float2(shV[t][off2 + 0]);
            float2 v23 = __half22float2(shV[t][off2 + 1]);
            if (new_max) {
                out0 = out0 * corr + v01.x;
                out1 = out1 * corr + v01.y;
                out2 = out2 * corr + v23.x;
                out3 = out3 * corr + v23.y;
            } else {
                out0 += w * v01.x;
                out1 += w * v01.y;
                out2 += w * v23.x;
                out3 += w * v23.y;
            }
        }
        __syncthreads();
    }

    if (lane == 0) {
        max_part[base_max] = max_score;
        sum_part[base_max] = sum_exp;
    }
    ((float4 *)(out_part + base_vec))[lane] = make_float4(out0, out1, out2, out3);
}

/* v4: fuse KV append (for the current token) into the v3 partial kernel.
 *
 * The current token K/V (float, [8*128]) are:
 * - used directly for attention (so we don't have to read them back from the cache)
 * - converted+stored into the FP16 KV cache for future steps.
 *
 * This is safe without cross-block sync because only the chunk containing `pos`
 * ever touches token index `pos` (end=pos+1). */

extern "C" __global__ void vox_attn_q4_kv8_fp16_v4_partial(float *out_part,        /* [32*n_chunks*128] */
                                                          float *max_part,        /* [32*n_chunks] */
                                                          float *sum_part,        /* [32*n_chunks] */
                                                          const float *q,         /* [32*128] */
                                                          __half *k_cache,        /* [max_seq*8*128] */
                                                          __half *v_cache,        /* [max_seq*8*128] */
                                                          const float *k_in,      /* [8*128] */
                                                          const float *v_in,      /* [8*128] */
                                                          int total_seq,
                                                          int window_size,
                                                          float scale,
                                                          int n_chunks) {
    int kv_h = (int)blockIdx.x;  /* 0..7 */
    int chunk = (int)blockIdx.y; /* 0..n_chunks-1 */
    int tid = (int)threadIdx.x;  /* 0..127 */
    int warp = tid >> 5;         /* 0..3 */
    int lane = tid & 31;         /* 0..31 */
    if (kv_h >= 8 || warp >= 4) return;

    int h = kv_h * 4 + warp; /* query head 0..31 */

    int end = total_seq;
    int pos = end - 1;
    if (pos < 0) pos = 0;

    int start = 0;
    if (window_size > 0) {
        int s = end - window_size;
        if (s > 0) start = s;
    }
    if (start < 0) start = 0;
    if (end < start) end = start;

    int chunk_start = start + chunk * VOX_ATTN_V3_CHUNK;
    int chunk_end = chunk_start + VOX_ATTN_V3_CHUNK;
    if (chunk_start > end) chunk_start = end;
    if (chunk_end > end) chunk_end = end;

    int base_max = h * n_chunks + chunk;
    size_t base_vec = ((size_t)base_max) * (size_t)128;

    if (chunk_start >= chunk_end) {
        if (lane == 0) {
            max_part[base_max] = -1.0e30f;
            sum_part[base_max] = 0.0f;
        }
        ((float4 *)(out_part + base_vec))[lane] = make_float4(0.f, 0.f, 0.f, 0.f);
        return;
    }

    const float4 qv = ((const float4 *)(q + h * 128))[lane];

    float max_score = -1.0e30f;
    float sum_exp = 0.0f;
    float out0 = 0.0f, out1 = 0.0f, out2 = 0.0f, out3 = 0.0f;

    const __half2 *k2 = (const __half2 *)k_cache;
    const __half2 *v2 = (const __half2 *)v_cache;
    __half2 *k2w = (__half2 *)k_cache;
    __half2 *v2w = (__half2 *)v_cache;
    size_t row_stride = (size_t)(8 * 128) / 2;
    size_t head_off = (size_t)kv_h * (size_t)128 / 2;

    __shared__ __half2 shK[VOX_ATTN_V3_TILE][64];
    __shared__ __half2 shV[VOX_ATTN_V3_TILE][64];

    int off2 = lane * 2;

    for (int j0 = chunk_start; j0 < chunk_end; j0 += VOX_ATTN_V3_TILE) {
#pragma unroll
        for (int t = 0; t < VOX_ATTN_V3_TILE; t++) {
            int j = j0 + t;
            if (j >= chunk_end) continue;

            if (j == pos) {
                /* Fuse: load current-token K/V from inputs, and write to cache. */
                if (tid < 64) {
                    int i = tid * 2;
                    float f0 = k_in[kv_h * 128 + i + 0];
                    float f1 = k_in[kv_h * 128 + i + 1];
                    __half2 hv = __floats2half2_rn(f0, f1);
                    shK[t][tid] = hv;
                    k2w[(size_t)pos * row_stride + head_off + (size_t)tid] = hv;
                } else {
                    int tv = tid - 64;
                    int i = tv * 2;
                    float f0 = v_in[kv_h * 128 + i + 0];
                    float f1 = v_in[kv_h * 128 + i + 1];
                    __half2 hv = __floats2half2_rn(f0, f1);
                    shV[t][tv] = hv;
                    v2w[(size_t)pos * row_stride + head_off + (size_t)tv] = hv;
                }
            } else {
                const __half2 *k_row = k2 + (size_t)j * row_stride + head_off;
                const __half2 *v_row = v2 + (size_t)j * row_stride + head_off;
                if (tid < 64) {
                    shK[t][tid] = k_row[tid];
                } else {
                    shV[t][tid - 64] = v_row[tid - 64];
                }
            }
        }
        __syncthreads();

#pragma unroll
        for (int t = 0; t < VOX_ATTN_V3_TILE; t++) {
            int j = j0 + t;
            if (j >= chunk_end) break;

            float2 k01 = __half22float2(shK[t][off2 + 0]);
            float2 k23 = __half22float2(shK[t][off2 + 1]);
            float partial = qv.x * k01.x + qv.y * k01.y + qv.z * k23.x + qv.w * k23.y;
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

            float2 v01 = __half22float2(shV[t][off2 + 0]);
            float2 v23 = __half22float2(shV[t][off2 + 1]);
            if (new_max) {
                out0 = out0 * corr + v01.x;
                out1 = out1 * corr + v01.y;
                out2 = out2 * corr + v23.x;
                out3 = out3 * corr + v23.y;
            } else {
                out0 += w * v01.x;
                out1 += w * v01.y;
                out2 += w * v23.x;
                out3 += w * v23.y;
            }
        }
        __syncthreads();
    }

    if (lane == 0) {
        max_part[base_max] = max_score;
        sum_part[base_max] = sum_exp;
    }
    ((float4 *)(out_part + base_vec))[lane] = make_float4(out0, out1, out2, out3);
}

extern "C" __global__ void vox_attn_q4_kv8_fp16_dyn_v4_partial(float *out_part,        /* [32*n_chunks*128] */
                                                              float *max_part,        /* [32*n_chunks] */
                                                              float *sum_part,        /* [32*n_chunks] */
                                                              const float *q,         /* [32*128] */
                                                              __half *k_cache,        /* [max_seq*8*128] */
                                                              __half *v_cache,        /* [max_seq*8*128] */
                                                              const float *k_in,      /* [8*128] */
                                                              const float *v_in,      /* [8*128] */
                                                              const int *p_pos,
                                                              int window_size,
                                                              float scale,
                                                              int n_chunks) {
    int kv_h = (int)blockIdx.x;
    int chunk = (int)blockIdx.y;
    int tid = (int)threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    if (kv_h >= 8 || warp >= 4) return;

    int h = kv_h * 4 + warp;

    int pos = 0;
    if (lane == 0) pos = *p_pos;
    pos = __shfl_sync(0xffffffff, pos, 0);
    int end = pos + 1;

    int start = 0;
    if (window_size > 0) {
        int s = end - window_size;
        if (s > 0) start = s;
    }
    if (start < 0) start = 0;
    if (end < start) end = start;

    int chunk_start = start + chunk * VOX_ATTN_V3_CHUNK;
    int chunk_end = chunk_start + VOX_ATTN_V3_CHUNK;
    if (chunk_start > end) chunk_start = end;
    if (chunk_end > end) chunk_end = end;

    int base_max = h * n_chunks + chunk;
    size_t base_vec = ((size_t)base_max) * (size_t)128;

    if (chunk_start >= chunk_end) {
        if (lane == 0) {
            max_part[base_max] = -1.0e30f;
            sum_part[base_max] = 0.0f;
        }
        ((float4 *)(out_part + base_vec))[lane] = make_float4(0.f, 0.f, 0.f, 0.f);
        return;
    }

    const float4 qv = ((const float4 *)(q + h * 128))[lane];

    float max_score = -1.0e30f;
    float sum_exp = 0.0f;
    float out0 = 0.0f, out1 = 0.0f, out2 = 0.0f, out3 = 0.0f;

    const __half2 *k2 = (const __half2 *)k_cache;
    const __half2 *v2 = (const __half2 *)v_cache;
    __half2 *k2w = (__half2 *)k_cache;
    __half2 *v2w = (__half2 *)v_cache;
    size_t row_stride = (size_t)(8 * 128) / 2;
    size_t head_off = (size_t)kv_h * (size_t)128 / 2;

    __shared__ __half2 shK[VOX_ATTN_V3_TILE][64];
    __shared__ __half2 shV[VOX_ATTN_V3_TILE][64];

    int off2 = lane * 2;

    for (int j0 = chunk_start; j0 < chunk_end; j0 += VOX_ATTN_V3_TILE) {
#pragma unroll
        for (int t = 0; t < VOX_ATTN_V3_TILE; t++) {
            int j = j0 + t;
            if (j >= chunk_end) continue;

            if (j == pos) {
                if (tid < 64) {
                    int i = tid * 2;
                    float f0 = k_in[kv_h * 128 + i + 0];
                    float f1 = k_in[kv_h * 128 + i + 1];
                    __half2 hv = __floats2half2_rn(f0, f1);
                    shK[t][tid] = hv;
                    k2w[(size_t)pos * row_stride + head_off + (size_t)tid] = hv;
                } else {
                    int tv = tid - 64;
                    int i = tv * 2;
                    float f0 = v_in[kv_h * 128 + i + 0];
                    float f1 = v_in[kv_h * 128 + i + 1];
                    __half2 hv = __floats2half2_rn(f0, f1);
                    shV[t][tv] = hv;
                    v2w[(size_t)pos * row_stride + head_off + (size_t)tv] = hv;
                }
            } else {
                const __half2 *k_row = k2 + (size_t)j * row_stride + head_off;
                const __half2 *v_row = v2 + (size_t)j * row_stride + head_off;
                if (tid < 64) {
                    shK[t][tid] = k_row[tid];
                } else {
                    shV[t][tid - 64] = v_row[tid - 64];
                }
            }
        }
        __syncthreads();

#pragma unroll
        for (int t = 0; t < VOX_ATTN_V3_TILE; t++) {
            int j = j0 + t;
            if (j >= chunk_end) break;

            float2 k01 = __half22float2(shK[t][off2 + 0]);
            float2 k23 = __half22float2(shK[t][off2 + 1]);
            float partial = qv.x * k01.x + qv.y * k01.y + qv.z * k23.x + qv.w * k23.y;
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

            float2 v01 = __half22float2(shV[t][off2 + 0]);
            float2 v23 = __half22float2(shV[t][off2 + 1]);
            if (new_max) {
                out0 = out0 * corr + v01.x;
                out1 = out1 * corr + v01.y;
                out2 = out2 * corr + v23.x;
                out3 = out3 * corr + v23.y;
            } else {
                out0 += w * v01.x;
                out1 += w * v01.y;
                out2 += w * v23.x;
                out3 += w * v23.y;
            }
        }
        __syncthreads();
    }

    if (lane == 0) {
        max_part[base_max] = max_score;
        sum_part[base_max] = sum_exp;
    }
    ((float4 *)(out_part + base_vec))[lane] = make_float4(out0, out1, out2, out3);
}

extern "C" __global__ void vox_attn_q4_kv8_fp16_v3_reduce(float *out_q,           /* [32*128] */
                                                          const float *out_part, /* [32*n_chunks*128] */
                                                          const float *max_part, /* [32*n_chunks] */
                                                          const float *sum_part, /* [32*n_chunks] */
                                                          int n_chunks) {
    int h = (int)blockIdx.x;
    int lane = (int)threadIdx.x;
    if (h >= 32 || lane >= 32) return;

    /* Find global max for this head across chunks. */
    float m = -1.0e30f;
    if (lane == 0) {
        for (int c = 0; c < n_chunks; c++) {
            float mc = max_part[h * n_chunks + c];
            if (mc > m) m = mc;
        }
    }
    m = __shfl_sync(0xffffffff, m, 0);

    /* Compute global sumexp (lane 0) and weighted sum vector (all lanes). */
    float4 acc4 = make_float4(0.f, 0.f, 0.f, 0.f);
    float sum = 0.0f;
    for (int c = 0; c < n_chunks; c++) {
        int idx = h * n_chunks + c;
        float mc = max_part[idx];
        float sc = sum_part[idx];
        float scale = __expf(mc - m);

        if (lane == 0) sum += sc * scale;

        size_t base_vec = ((size_t)idx) * (size_t)128;
        float4 v4 = ((const float4 *)(out_part + base_vec))[lane];
        acc4.x += v4.x * scale;
        acc4.y += v4.y * scale;
        acc4.z += v4.z * scale;
        acc4.w += v4.w * scale;
    }

    float inv_sum = 0.0f;
    if (lane == 0) inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    inv_sum = __shfl_sync(0xffffffff, inv_sum, 0);

    ((float4 *)(out_q + h * 128))[lane] = make_float4(acc4.x * inv_sum,
                                                      acc4.y * inv_sum,
                                                      acc4.z * inv_sum,
                                                      acc4.w * inv_sum);
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

extern "C" __global__ void vox_rms_norm_to_bf16(uint16_t *out_bf16,
                                                const float *x,
                                                const float *weight,
                                                int rows,
                                                int hidden,
                                                float eps) {
    int r = (int)blockIdx.x;
    if (r >= rows) return;

    const float *x_row = x + (size_t)r * (size_t)hidden;
    uint16_t *o_row = out_bf16 + (size_t)r * (size_t)hidden;

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
        float v = x_row[i] * inv_rms * weight[i];
        __nv_bfloat16 h = __float2bfloat16_rn(v);
        (( __nv_bfloat16 *)o_row)[i] = h;
    }
}

extern "C" __global__ void vox_rms_norm_to_bf16_ada(uint16_t *out_bf16,
                                                    const float *x,
                                                    const float *weight,
                                                    const float *ada,
                                                    int rows,
                                                    int hidden,
                                                    float eps) {
    int r = (int)blockIdx.x;
    if (r >= rows) return;
    if (!ada) return;

    const float *x_row = x + (size_t)r * (size_t)hidden;
    uint16_t *o_row = out_bf16 + (size_t)r * (size_t)hidden;

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
        float v = x_row[i] * inv_rms * weight[i];
        v *= (1.0f + ada[i]);
        __nv_bfloat16 h = __float2bfloat16_rn(v);
        (( __nv_bfloat16 *)o_row)[i] = h;
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

extern "C" __global__ void vox_mul_1p_rows_inplace_f32(float *x,
                                                       const float *scale,
                                                       int rows,
                                                       int cols) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = rows * cols;
    if (idx >= total) return;
    int c = idx % cols;
    x[idx] *= (1.0f + scale[c]);
}

extern "C" __global__ void vox_silu_inplace_f32(float *x, int n) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) return;
    float v = x[idx];
    x[idx] = v / (1.0f + __expf(-v));
}

extern "C" __global__ void vox_silu_mul_inplace_f32(float *x,
                                                    const float *y,
                                                    int n) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) return;
    float v = x[idx];
    float s = v / (1.0f + __expf(-v));
    x[idx] = s * y[idx];
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

/* ========================================================================
 * Encoder Conv Stem Helpers (CUDA, optional)
 * ======================================================================== */

/* Build im2col for the encoder conv stem conv0:
 * - causal conv1d, k=3, stride=1 (left_pad=2)
 * - input is mel in row-major [length, 128]
 * - output is im2col in row-major [K=128*3, out_len=length]
 *
 * Matches vox_causal_conv1d() indexing:
 * im2col[ic*3 + k, ol] = mel[ol - 2 + k, ic] (0 if OOB)
 */
extern "C" __global__ void vox_im2col_causal_k3_s1_mel_f32(float *dst,
                                                           const float *mel,
                                                           int length) {
    int out_len = length;
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = (128 * 3) * out_len;
    if (idx >= total) return;

    int row = idx / out_len; /* 0..383 */
    int ol = idx - row * out_len;
    int ic = row / 3;
    int k = row - ic * 3;
    int il = ol - 2 + k;

    float v = 0.0f;
    if (il >= 0 && il < length) {
        v = mel[(size_t)il * 128u + (size_t)ic];
    }
    dst[(size_t)row * (size_t)out_len + (size_t)ol] = v;
}

/* Build im2col for the encoder conv stem conv1:
 * - causal conv1d, k=3, stride=2 (left_pad=1)
 * - input is channel-first row-major [channels, length]
 * - output is im2col row-major [K=channels*3, out_len]
 *
 * im2col[ic*3 + k, ol] = in[ic, ol*2 - 1 + k] (0 if OOB)
 */
extern "C" __global__ void vox_im2col_causal_k3_s2_f32(float *dst,
                                                       const float *in,
                                                       int channels,
                                                       int length,
                                                       int out_len) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int K = channels * 3;
    int total = K * out_len;
    if (idx >= total) return;

    int row = idx / out_len; /* 0..K-1 */
    int ol = idx - row * out_len;
    int ic = row / 3;
    int k = row - ic * 3;
    int il = ol * 2 - 1 + k;

    float v = 0.0f;
    if (il >= 0 && il < length) {
        v = in[(size_t)ic * (size_t)length + (size_t)il];
    }
    dst[(size_t)row * (size_t)out_len + (size_t)ol] = v;
}

/* x is channel-first row-major [channels, length]. Applies:
 *   x = GELU(x + bias[channel])
 */
extern "C" __global__ void vox_add_bias_gelu_chfirst_f32(float *x,
                                                         const float *bias,
                                                         int channels,
                                                         int length) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = channels * length;
    if (idx >= total) return;
    int c = idx / length;
    float v = x[idx] + bias[c];
    float x3 = v * v * v;
    float inner = 0.7978845608028654f * (v + 0.044715f * x3);
    x[idx] = 0.5f * v * (1.0f + tanhf(inner));
}

/* Convert channel-first row-major [channels, length] to time-major row-major
 * [length, channels]. This is effectively a transpose. */
extern "C" __global__ void vox_chfirst_to_rowmajor_f32(float *dst,
                                                       const float *src,
                                                       int channels,
                                                       int length) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int total = channels * length;
    if (idx >= total) return;
    int c = idx / length;
    int t = idx - c * length;
    dst[(size_t)t * (size_t)channels + (size_t)c] = src[(size_t)c * (size_t)length + (size_t)t];
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

/* Generate RoPE freqs for a single logical position (used by CUDA Graph mode):
 *   out[d,0]=cos(pos*inv_freq[d]), out[d,1]=sin(pos*inv_freq[d])
 * where inv_freq[d] = 1/pow(theta,2d/head_dim) is precomputed on the host. */
extern "C" __global__ void vox_rope_freqs_1pos_f32(float *out,
                                                   const float *inv_freq,
                                                   const int *pos_dev,
                                                   int half_dim) {
    int d = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (d >= half_dim) return;
    float p = (float)(*pos_dev);
    float angle = p * inv_freq[d];
    float s, c;
    sincosf(angle, &s, &c);
    out[(size_t)d * 2] = c;
    out[(size_t)d * 2 + 1] = s;
}

/* Build the decoder step embedding on-device:
 *   dst[dim] = adapter[adapter_slot, dim] + tok_embed_bf16[token_id, dim] */
extern "C" __global__ void vox_step_embed_from_adapter_f32(float *dst,
                                                           const float *adapter,
                                                           const uint16_t *tok_emb_bf16,
                                                           int token_id,
                                                           int adapter_slot,
                                                           int dim) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= dim) return;
    const float *a = adapter + (size_t)adapter_slot * (size_t)dim;
    const uint16_t *t = tok_emb_bf16 + (size_t)token_id * (size_t)dim;
    float tf = __uint_as_float(((uint32_t)t[idx]) << 16);
    dst[idx] = a[idx] + tf;
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

/* Fused top1-only logits projection (BF16 weights):
 * - Computes argmax over vocab without materializing logits[].
 *
 * Intended for the common path where alternatives are disabled (logits_or_null==NULL).
 * This is not expected to be faster than cuBLAS on all GPUs; keep it opt-in. */

__device__ __forceinline__ float vox_bf16_to_f32(uint16_t h) {
    return __uint_as_float(((uint32_t)h) << 16);
}

/* Map float bits to an unsigned integer that preserves ordering under unsigned
 * comparisons. This allows packing into a u64 and using atomicMax. */
__device__ __forceinline__ uint32_t vox_f32_to_ordered_u32(float f) {
    uint32_t u = __float_as_uint(f);
    uint32_t mask = (u & 0x80000000u) ? 0xffffffffu : 0x80000000u;
    return u ^ mask;
}

extern "C" __global__ void vox_logits_best_init_u64(unsigned long long *best_packed) {
    if (blockIdx.x == 0 && threadIdx.x == 0) best_packed[0] = 0ULL;
}

extern "C" __global__ void vox_logits_best_bf16_top1(unsigned long long *best_packed,
                                                     const uint16_t *x_bf16,
                                                     const uint16_t *tok_bf16,
                                                     int dim,
                                                     int vocab) {
    /* Block = 256 threads (8 warps). Each block reduces a small tile of rows, then
     * issues a single atomicMax on (best_logit, best_idx). */
    const int WARPS = 8;
    const int ROWS_PER_WARP = 4;
    const int ROWS_PER_BLOCK = WARPS * ROWS_PER_WARP; /* 32 */

    int tid = (int)threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int base = (int)blockIdx.x * ROWS_PER_BLOCK;
    if (base >= vocab) return;

    /* Dynamic shared: x_bf16[dim]. */
    extern __shared__ uint16_t shx[];
    for (int i = tid; i < dim; i += (int)blockDim.x) {
        shx[i] = x_bf16[i];
    }
    __syncthreads();

    float warp_best = -1.0e30f;
    int warp_best_idx = 0;

    /* Assign rows interleaved by warp for coalescing. */
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        int row = base + warp + r * WARPS;
        if (row >= vocab) continue;
        const uint16_t *w = tok_bf16 + (size_t)row * (size_t)dim;

        float sum = 0.0f;
        for (int k = lane; k < dim; k += 32) {
            float a = vox_bf16_to_f32(shx[k]);
            float b = vox_bf16_to_f32(w[k]);
            sum += a * b;
        }
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            if (sum > warp_best || (sum == warp_best && row < warp_best_idx)) {
                warp_best = sum;
                warp_best_idx = row;
            }
        }
    }

    __shared__ float sh_best_val[WARPS];
    __shared__ int sh_best_idx[WARPS];
    if (lane == 0) {
        sh_best_val[warp] = warp_best;
        sh_best_idx[warp] = warp_best_idx;
    }
    __syncthreads();

    if (warp == 0 && lane == 0) {
        float best = sh_best_val[0];
        int best_idx = sh_best_idx[0];
        for (int w = 1; w < WARPS; w++) {
            float v = sh_best_val[w];
            int idx = sh_best_idx[w];
            if (v > best || (v == best && idx < best_idx)) {
                best = v;
                best_idx = idx;
            }
        }

        uint32_t ord = vox_f32_to_ordered_u32(best);
        /* Low bits are ~idx so ties prefer smaller idx under atomicMax. */
        unsigned long long packed = ((unsigned long long)ord << 32) | (unsigned long long)(~(uint32_t)best_idx);
        (void)atomicMax(best_packed, packed);
    }
}

extern "C" __global__ void vox_logits_best_unpack_u64(int *out_idx,
                                                     const unsigned long long *best_packed) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long packed = best_packed[0];
        uint32_t low = (uint32_t)(packed & 0xffffffffu);
        uint32_t idx = ~low;
        out_idx[0] = (int)idx;
    }
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
