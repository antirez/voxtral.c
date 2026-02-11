/*
 * voxtral_cuda.cu - CUDA GPU acceleration for Voxtral inference
 *
 * Phase 1: cuBLAS matmul with BF16 weight caching.
 * Phase 2: Causal attention kernel with online softmax.
 * Phase 3: Full GPU pipeline — element-wise kernels, device-pointer matmul,
 *          monolithic decoder/encoder steps (no CPU↔GPU round-trips per layer).
 */

#include "voxtral_cuda.h"
#include "voxtral.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========================================================================
 * Global State
 * ======================================================================== */

static int g_initialized = 0;
static cudaStream_t g_stream = NULL;
static cublasHandle_t g_cublas = NULL;
static size_t g_memory_used = 0;

/* ========================================================================
 * BF16 Weight Cache
 *
 * Maps CPU bf16 pointer -> GPU device bf16 pointer.
 * Linear-probe hash table. Weights are uploaded once and stay on GPU.
 * ======================================================================== */

#define WEIGHT_CACHE_SIZE 1024

typedef struct {
    const uint16_t *cpu_ptr;  /* key (NULL = empty slot) */
    void *gpu_ptr;            /* device pointer (bf16) */
    size_t num_elements;
} weight_cache_entry_t;

static weight_cache_entry_t g_weight_cache[WEIGHT_CACHE_SIZE];

static int weight_cache_find(const uint16_t *cpu_ptr) {
    unsigned long hash = ((unsigned long)cpu_ptr >> 4) % WEIGHT_CACHE_SIZE;
    for (int i = 0; i < WEIGHT_CACHE_SIZE; i++) {
        int idx = (hash + i) % WEIGHT_CACHE_SIZE;
        if (g_weight_cache[idx].cpu_ptr == cpu_ptr) return idx;
        if (g_weight_cache[idx].cpu_ptr == NULL) return -1;
    }
    return -1;
}

static void *weight_cache_get_or_upload(const uint16_t *cpu_ptr, size_t num_elements) {
    int idx = weight_cache_find(cpu_ptr);
    if (idx >= 0) return g_weight_cache[idx].gpu_ptr;

    /* Upload to GPU */
    void *gpu_ptr = NULL;
    size_t bytes = num_elements * sizeof(uint16_t);
    cudaError_t err = cudaMalloc(&gpu_ptr, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA: weight upload failed (%zu bytes): %s\n",
                bytes, cudaGetErrorString(err));
        return NULL;
    }
    err = cudaMemcpyAsync(gpu_ptr, cpu_ptr, bytes, cudaMemcpyHostToDevice, g_stream);
    if (err != cudaSuccess) {
        cudaFree(gpu_ptr);
        return NULL;
    }
    cudaStreamSynchronize(g_stream);
    g_memory_used += bytes;

    /* Insert into cache */
    unsigned long hash = ((unsigned long)cpu_ptr >> 4) % WEIGHT_CACHE_SIZE;
    for (int i = 0; i < WEIGHT_CACHE_SIZE; i++) {
        idx = (hash + i) % WEIGHT_CACHE_SIZE;
        if (g_weight_cache[idx].cpu_ptr == NULL) {
            g_weight_cache[idx].cpu_ptr = cpu_ptr;
            g_weight_cache[idx].gpu_ptr = gpu_ptr;
            g_weight_cache[idx].num_elements = num_elements;
            return gpu_ptr;
        }
    }

    /* Cache full (shouldn't happen with 1024 slots and ~400 weight tensors) */
    fprintf(stderr, "CUDA: weight cache full\n");
    cudaFree(gpu_ptr);
    g_memory_used -= bytes;
    return NULL;
}

/* ========================================================================
 * F32 Weight Cache
 *
 * Maps CPU f32 pointer -> GPU device f32 pointer.
 * Used for norm weights, biases, and ada_scale (small tensors).
 * ======================================================================== */

#define F32_CACHE_SIZE 512

typedef struct {
    const float *cpu_ptr;
    float *gpu_ptr;
    size_t num_elements;
} f32_cache_entry_t;

static f32_cache_entry_t g_f32_cache[F32_CACHE_SIZE];

static float *f32_cache_get_or_upload(const float *cpu_ptr, size_t num_elements) {
    if (!cpu_ptr) return NULL;

    /* Look up */
    unsigned long hash = ((unsigned long)cpu_ptr >> 4) % F32_CACHE_SIZE;
    for (int i = 0; i < F32_CACHE_SIZE; i++) {
        int idx = (hash + i) % F32_CACHE_SIZE;
        if (g_f32_cache[idx].cpu_ptr == cpu_ptr) return g_f32_cache[idx].gpu_ptr;
        if (g_f32_cache[idx].cpu_ptr == NULL) break;
    }

    /* Upload */
    float *gpu_ptr = NULL;
    size_t bytes = num_elements * sizeof(float);
    if (cudaMalloc((void **)&gpu_ptr, bytes) != cudaSuccess) return NULL;
    cudaMemcpyAsync(gpu_ptr, cpu_ptr, bytes, cudaMemcpyHostToDevice, g_stream);
    cudaStreamSynchronize(g_stream);
    g_memory_used += bytes;

    /* Insert */
    for (int i = 0; i < F32_CACHE_SIZE; i++) {
        int idx = (hash + i) % F32_CACHE_SIZE;
        if (g_f32_cache[idx].cpu_ptr == NULL) {
            g_f32_cache[idx].cpu_ptr = cpu_ptr;
            g_f32_cache[idx].gpu_ptr = gpu_ptr;
            g_f32_cache[idx].num_elements = num_elements;
            return gpu_ptr;
        }
    }

    cudaFree(gpu_ptr);
    g_memory_used -= bytes;
    return NULL;
}

/* ========================================================================
 * Activation Buffer Pool
 *
 * Reusable GPU buffers for transient activation/output data.
 * Avoids cudaMalloc/cudaFree per matmul call.
 * ======================================================================== */

#define POOL_SIZE 4

typedef struct {
    void *ptr;
    size_t capacity;  /* bytes */
    int in_use;
} pool_entry_t;

static pool_entry_t g_pool[POOL_SIZE];

static void *pool_alloc(size_t bytes) {
    /* Find a free buffer large enough */
    for (int i = 0; i < POOL_SIZE; i++) {
        if (!g_pool[i].in_use && g_pool[i].capacity >= bytes) {
            g_pool[i].in_use = 1;
            return g_pool[i].ptr;
        }
    }

    /* Find a free slot (grow or create) */
    for (int i = 0; i < POOL_SIZE; i++) {
        if (!g_pool[i].in_use) {
            if (g_pool[i].ptr) {
                g_memory_used -= g_pool[i].capacity;
                cudaFree(g_pool[i].ptr);
            }
            /* Allocate with some headroom to reduce re-allocations */
            size_t alloc_size = bytes < 64 * 1024 * 1024 ? 64 * 1024 * 1024 : bytes;
            cudaError_t err = cudaMalloc(&g_pool[i].ptr, alloc_size);
            if (err != cudaSuccess) {
                /* Try exact size */
                alloc_size = bytes;
                err = cudaMalloc(&g_pool[i].ptr, alloc_size);
                if (err != cudaSuccess) {
                    g_pool[i].ptr = NULL;
                    g_pool[i].capacity = 0;
                    return NULL;
                }
            }
            g_pool[i].capacity = alloc_size;
            g_pool[i].in_use = 1;
            g_memory_used += alloc_size;
            return g_pool[i].ptr;
        }
    }

    /* All slots in use — fallback to direct allocation */
    void *ptr = NULL;
    if (cudaMalloc(&ptr, bytes) != cudaSuccess) return NULL;
    g_memory_used += bytes;
    return ptr;
}

static void pool_free(void *ptr) {
    for (int i = 0; i < POOL_SIZE; i++) {
        if (g_pool[i].ptr == ptr) {
            g_pool[i].in_use = 0;
            return;
        }
    }
    /* Was a direct allocation, actually free it */
    cudaFree(ptr);
}

/* ========================================================================
 * Persistent GPU Buffers (forward declarations for shutdown)
 * ======================================================================== */

/* Decoder: fixed-size single-token buffers */
static struct {
    float *x, *x_norm, *q, *k, *v;
    float *attn_out, *proj_out;
    float *gate, *up, *ffn_out;
    float *rope_freqs, *logits;
    float *base;
    int allocated;
} g_dec_bufs;

/* Encoder: variable-size buffers */
static struct {
    float *x, *x_norm, *q, *k, *v;
    float *attn_out, *proj_out;
    float *gate, *up, *ffn_out;
    float *rope_freqs;
    float *base;
    int capacity;
} g_enc_bufs;

/* ========================================================================
 * Lifecycle
 * ======================================================================== */

extern "C" int vox_cuda_init(void) {
    if (g_initialized) return 1;

    /* Check for CUDA device */
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        fprintf(stderr, "CUDA: no devices found\n");
        return 0;
    }

    /* Check compute capability (need SM 8.0+ for native BF16) */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int sm = prop.major * 10 + prop.minor;
    if (sm < 80) {
        fprintf(stderr, "CUDA: device SM %d.%d < 8.0, BF16 not supported\n",
                prop.major, prop.minor);
        return 0;
    }

    fprintf(stderr, "CUDA: %s (SM %d.%d, %.0f MB)\n",
            prop.name, prop.major, prop.minor,
            prop.totalGlobalMem / (1024.0 * 1024.0));

    /* Create stream and cuBLAS handle */
    err = cudaStreamCreate(&g_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA: stream creation failed: %s\n", cudaGetErrorString(err));
        return 0;
    }

    cublasStatus_t cberr = cublasCreate(&g_cublas);
    if (cberr != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUDA: cuBLAS creation failed: %d\n", cberr);
        cudaStreamDestroy(g_stream);
        g_stream = NULL;
        return 0;
    }
    cublasSetStream(g_cublas, g_stream);

    /* Use TF32 for better performance on Ampere+ (still precise enough) */
    cublasSetMathMode(g_cublas, CUBLAS_TF32_TENSOR_OP_MATH);

    /* Clear caches */
    memset(g_weight_cache, 0, sizeof(g_weight_cache));
    memset(g_f32_cache, 0, sizeof(g_f32_cache));
    memset(g_pool, 0, sizeof(g_pool));
    g_memory_used = 0;

    g_initialized = 1;
    return 1;
}

extern "C" int vox_cuda_available(void) {
    return g_initialized;
}

extern "C" void vox_cuda_shutdown(void) {
    if (!g_initialized) return;

    /* Free bf16 weight cache */
    for (int i = 0; i < WEIGHT_CACHE_SIZE; i++) {
        if (g_weight_cache[i].gpu_ptr) {
            cudaFree(g_weight_cache[i].gpu_ptr);
            g_weight_cache[i].gpu_ptr = NULL;
            g_weight_cache[i].cpu_ptr = NULL;
        }
    }

    /* Free f32 weight cache */
    for (int i = 0; i < F32_CACHE_SIZE; i++) {
        if (g_f32_cache[i].gpu_ptr) {
            cudaFree(g_f32_cache[i].gpu_ptr);
            g_f32_cache[i].gpu_ptr = NULL;
            g_f32_cache[i].cpu_ptr = NULL;
        }
    }

    /* Free persistent GPU buffers */
    if (g_dec_bufs.base) { cudaFree(g_dec_bufs.base); }
    memset(&g_dec_bufs, 0, sizeof(g_dec_bufs));
    if (g_enc_bufs.base) { cudaFree(g_enc_bufs.base); }
    memset(&g_enc_bufs, 0, sizeof(g_enc_bufs));

    /* Free pool buffers */
    for (int i = 0; i < POOL_SIZE; i++) {
        if (g_pool[i].ptr) {
            cudaFree(g_pool[i].ptr);
            g_pool[i].ptr = NULL;
            g_pool[i].capacity = 0;
        }
    }

    if (g_cublas) { cublasDestroy(g_cublas); g_cublas = NULL; }
    if (g_stream) { cudaStreamDestroy(g_stream); g_stream = NULL; }

    g_memory_used = 0;
    g_initialized = 0;
}

/* ========================================================================
 * F32 -> BF16 Conversion Kernel
 * ======================================================================== */

__global__ void kernel_f32_to_bf16(__nv_bfloat16 *out, const float *in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(in[idx]);
    }
}

/* ========================================================================
 * Causal Attention Kernel
 *
 * Grid: (n_heads, seq_q) — one block per (head, query position)
 * Block: (head_dim) threads — one thread per dimension
 *
 * Online softmax: each thread tracks its own output dimension.
 * Dot product (Q·K) is reduced across threads via warp shuffle + shared memory.
 * ======================================================================== */

__global__ void kernel_causal_attention(
    float *out,             /* [seq_q, n_heads * head_dim] */
    const float *Q,         /* [seq_q, n_heads * head_dim] device memory */
    const float *K,         /* [seq_k, n_kv_heads * head_dim] managed memory */
    const float *V,         /* [seq_k, n_kv_heads * head_dim] managed memory */
    int seq_k,
    int n_heads, int n_kv_heads,
    int head_dim,
    float scale,
    int window_size,
    int q_offset)
{
    int h = blockIdx.x;            /* head index */
    int i = blockIdx.y;            /* query position index */
    int tid = threadIdx.x;         /* dimension index [0, head_dim) */

    int heads_per_kv = n_heads / n_kv_heads;
    int kv_h = h / heads_per_kv;
    int q_hidden = n_heads * head_dim;
    int kv_hidden = n_kv_heads * head_dim;

    /* Load Q element for this thread's dimension */
    float q_val = Q[i * q_hidden + h * head_dim + tid];

    /* Compute valid K range (causal + sliding window) */
    int global_pos = q_offset + i;
    int k_start = 0;
    if (window_size > 0 && global_pos - window_size + 1 > 0)
        k_start = global_pos - window_size + 1;
    int k_end = global_pos + 1;
    if (k_end > seq_k) k_end = seq_k;

    /* Shared memory for inter-warp dot product reduction */
    extern __shared__ float s_warp[];
    int n_warps = (blockDim.x + 31) / 32;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    /* Online softmax state (per-thread, each tracks one output dimension) */
    float max_score = -1e30f;
    float sum_exp = 0.0f;
    float o_val = 0.0f;

    for (int j = k_start; j < k_end; j++) {
        /* Dot product: q · k (partial per thread, reduce across block) */
        float k_val = K[(size_t)j * kv_hidden + kv_h * head_dim + tid];
        float partial = q_val * k_val;

        /* Warp-level reduction via shuffle */
        for (int offset = 16; offset > 0; offset >>= 1)
            partial += __shfl_down_sync(0xffffffff, partial, offset);

        /* Inter-warp reduction via shared memory */
        if (lane_id == 0) s_warp[warp_id] = partial;
        __syncthreads();

        float score;
        if (tid == 0) {
            score = 0.0f;
            for (int w = 0; w < n_warps; w++) score += s_warp[w];
            score *= scale;
            s_warp[0] = score;  /* broadcast to all threads */
        }
        __syncthreads();
        score = s_warp[0];

        /* Load V element for this thread's dimension */
        float v_val = V[(size_t)j * kv_hidden + kv_h * head_dim + tid];

        /* Online softmax update */
        if (score > max_score) {
            float correction = expf(max_score - score);
            sum_exp = sum_exp * correction + 1.0f;
            o_val = o_val * correction + v_val;
            max_score = score;
        } else {
            float weight = expf(score - max_score);
            sum_exp += weight;
            o_val += weight * v_val;
        }
    }

    /* Normalize and write output */
    if (sum_exp > 0.0f) o_val /= sum_exp;
    out[i * q_hidden + h * head_dim + tid] = o_val;
}

extern "C" void vox_cuda_causal_attention(
    float *out, const float *Q, const float *K, const float *V,
    int seq_q, int seq_k, int n_heads, int n_kv_heads,
    int head_dim, float scale, int window_size, int q_offset)
{
    if (!g_initialized) return;

    int q_hidden = n_heads * head_dim;
    size_t q_bytes = (size_t)seq_q * q_hidden * sizeof(float);
    size_t out_bytes = q_bytes;

    /* Upload Q, allocate output on device */
    void *d_Q = pool_alloc(q_bytes);
    void *d_out = pool_alloc(out_bytes);
    if (!d_Q || !d_out) {
        if (d_Q) pool_free(d_Q);
        if (d_out) pool_free(d_out);
        return;
    }

    cudaMemcpyAsync(d_Q, Q, q_bytes, cudaMemcpyHostToDevice, g_stream);

    /* K, V are in managed memory — pass directly to kernel */
    dim3 grid(n_heads, seq_q);
    dim3 block(head_dim);
    int n_warps = (head_dim + 31) / 32;
    size_t smem = n_warps * sizeof(float);

    kernel_causal_attention<<<grid, block, smem, g_stream>>>(
        (float *)d_out, (float *)d_Q, K, V,
        seq_k, n_heads, n_kv_heads,
        head_dim, scale, window_size, q_offset);

    /* Download result */
    cudaMemcpyAsync(out, d_out, out_bytes, cudaMemcpyDeviceToHost, g_stream);
    cudaStreamSynchronize(g_stream);

    pool_free(d_Q);
    pool_free(d_out);
}

/* ========================================================================
 * Element-wise CUDA Kernels
 * ======================================================================== */

/* RMS Norm: out[s,d] = (x[s,d] / rms) * weight[d]
 * Grid: (seq_len), Block: (256). Each block handles one sequence position. */
__global__ void kernel_rms_norm(float *out, const float *x, const float *weight,
                                int hidden, float eps) {
    int s = blockIdx.x;
    int tid = threadIdx.x;
    const float *row = x + (size_t)s * hidden;
    float *out_row = out + (size_t)s * hidden;

    extern __shared__ float sdata[];
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden; i += blockDim.x)
        sum_sq += row[i] * row[i];
    sdata[tid] = sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    float rms = rsqrtf(sdata[0] / (float)hidden + eps);
    for (int i = tid; i < hidden; i += blockDim.x)
        out_row[i] = row[i] * rms * weight[i];
}

__global__ void kernel_silu(float *x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        x[idx] = val / (1.0f + expf(-val));
    }
}

__global__ void kernel_gelu(float *x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        float inner = 0.7978845608f * (val + 0.044715f * val * val * val);
        x[idx] = 0.5f * val * (1.0f + tanhf(inner));
    }
}

__global__ void kernel_add_inplace(float *a, const float *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) a[idx] += b[idx];
}

__global__ void kernel_mul_inplace(float *a, const float *b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) a[idx] *= b[idx];
}

/* Apply RoPE: rotate pairs by (cos, sin) freqs.
 * x: [seq, heads * head_dim], freqs: [seq, head_dim/2, 2]
 * Grid: (seq * heads), Block: (head_dim/2) */
__global__ void kernel_apply_rope(float *x, const float *freqs,
                                   int seq, int heads, int head_dim) {
    int sh = blockIdx.x;
    int s = sh / heads;
    int h = sh % heads;
    int pair = threadIdx.x;
    int half = head_dim / 2;

    float *row = x + (size_t)s * heads * head_dim + h * head_dim;
    const float *freq_row = freqs + (size_t)s * half * 2;

    float cos_val = freq_row[pair * 2];
    float sin_val = freq_row[pair * 2 + 1];
    float x0 = row[pair * 2];
    float x1 = row[pair * 2 + 1];
    row[pair * 2]     = x0 * cos_val - x1 * sin_val;
    row[pair * 2 + 1] = x0 * sin_val + x1 * cos_val;
}

/* Ada scale: x[i] *= (1 + ada[i % dim]) */
__global__ void kernel_ada_scale(float *x, const float *ada, int n, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] *= (1.0f + ada[idx % dim]);
}

/* Bias add: data[s, d] += bias[d] */
__global__ void kernel_bias_add(float *data, const float *bias, int n, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] += bias[idx % dim];
}

/* ========================================================================
 * Device-pointer Matrix Multiplication
 *
 * Like vox_cuda_sgemm_bf16 but A and C are already on GPU.
 * No host<->device copies — only the F32->BF16 conversion of A.
 * ======================================================================== */

static void sgemm_bf16_dev(int M, int N, int K,
                            const float *d_A,
                            const uint16_t *B_bf16_cpu,
                            float *d_C) {
    void *d_B = weight_cache_get_or_upload(B_bf16_cpu, (size_t)N * K);
    if (!d_B) return;

    size_t a_bf16_bytes = (size_t)M * K * sizeof(__nv_bfloat16);
    void *d_A_bf16 = pool_alloc(a_bf16_bytes);
    if (!d_A_bf16) return;

    int total = M * K;
    kernel_f32_to_bf16<<<(total + 255) / 256, 256, 0, g_stream>>>(
        (__nv_bfloat16 *)d_A_bf16, d_A, total);

    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(g_cublas,
                 CUBLAS_OP_T, CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 d_B, CUDA_R_16BF, K,
                 d_A_bf16, CUDA_R_16BF, K,
                 &beta,
                 d_C, CUDA_R_32F, N,
                 CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    pool_free(d_A_bf16);
}

/* ========================================================================
 * Persistent GPU Buffer Allocation
 * ======================================================================== */

static int ensure_dec_gpu_bufs(void) {
    if (g_dec_bufs.allocated) return 0;
    int dim = VOX_DEC_DIM;
    int q_dim = VOX_DEC_HEADS * VOX_DEC_HEAD_DIM;
    int kv_dim = VOX_DEC_KV_HEADS * VOX_DEC_HEAD_DIM;
    int hidden = VOX_DEC_HIDDEN;
    int head_dim = VOX_DEC_HEAD_DIM;

    size_t total = (size_t)(dim + dim + q_dim + kv_dim + kv_dim + q_dim + dim +
                   hidden + hidden + dim + head_dim + VOX_VOCAB_SIZE) * sizeof(float);
    float *buf = NULL;
    if (cudaMalloc((void **)&buf, total) != cudaSuccess) return -1;
    g_memory_used += total;

    float *p = buf;
    g_dec_bufs.x = p;          p += dim;
    g_dec_bufs.x_norm = p;     p += dim;
    g_dec_bufs.q = p;          p += q_dim;
    g_dec_bufs.k = p;          p += kv_dim;
    g_dec_bufs.v = p;          p += kv_dim;
    g_dec_bufs.attn_out = p;   p += q_dim;
    g_dec_bufs.proj_out = p;   p += dim;
    g_dec_bufs.gate = p;       p += hidden;
    g_dec_bufs.up = p;         p += hidden;
    g_dec_bufs.ffn_out = p;    p += dim;
    g_dec_bufs.rope_freqs = p; p += head_dim;
    g_dec_bufs.logits = p;

    g_dec_bufs.base = buf;
    g_dec_bufs.allocated = 1;
    return 0;
}

static int ensure_enc_gpu_bufs(int new_len) {
    if (new_len <= g_enc_bufs.capacity) return 0;
    if (g_enc_bufs.base) { cudaFree(g_enc_bufs.base); g_enc_bufs.base = NULL; }

    int dim = VOX_ENC_DIM;
    int qkv_dim = VOX_ENC_HEADS * VOX_ENC_HEAD_DIM;
    int hidden = VOX_ENC_HIDDEN;
    int head_dim = VOX_ENC_HEAD_DIM;

    size_t per_pos = (size_t)(dim + dim + qkv_dim*3 + qkv_dim + dim +
                     hidden + hidden + dim + head_dim) * sizeof(float);
    size_t total = per_pos * new_len;
    float *buf = NULL;
    if (cudaMalloc((void **)&buf, total) != cudaSuccess) return -1;
    g_memory_used += total;

    float *p = buf;
    g_enc_bufs.x = p;          p += (size_t)new_len * dim;
    g_enc_bufs.x_norm = p;     p += (size_t)new_len * dim;
    g_enc_bufs.q = p;          p += (size_t)new_len * qkv_dim;
    g_enc_bufs.k = p;          p += (size_t)new_len * qkv_dim;
    g_enc_bufs.v = p;          p += (size_t)new_len * qkv_dim;
    g_enc_bufs.attn_out = p;   p += (size_t)new_len * qkv_dim;
    g_enc_bufs.proj_out = p;   p += (size_t)new_len * dim;
    g_enc_bufs.gate = p;       p += (size_t)new_len * hidden;
    g_enc_bufs.up = p;         p += (size_t)new_len * hidden;
    g_enc_bufs.ffn_out = p;    p += (size_t)new_len * dim;
    g_enc_bufs.rope_freqs = p;

    g_enc_bufs.base = buf;
    g_enc_bufs.capacity = new_len;
    return 0;
}

/* ========================================================================
 * Matrix Multiplication
 * ======================================================================== */

/*
 * C[M,N] = A_f32[M,K] @ B_bf16[N,K]^T
 *
 * cuBLAS requires both A and B to be the same type for BF16 GEMM.
 * Strategy: convert A from F32 to BF16 on GPU, then run BF16 x BF16 -> F32 GEMM.
 *
 * Row-major layout. Uses cuBLAS column-major convention:
 *   C'[N,M] = B^T[N,K] * A_bf16'[K,M]
 */
extern "C" void vox_cuda_sgemm_bf16(int M, int N, int K,
                                      const float *A,
                                      const uint16_t *B_bf16,
                                      float *C) {
    if (!g_initialized) return;

    /* Look up or upload weights */
    void *d_B = weight_cache_get_or_upload(B_bf16, (size_t)N * K);
    if (!d_B) return;

    /* Allocate buffers: F32 input, BF16 converted input, F32 output */
    size_t a_f32_bytes = (size_t)M * K * sizeof(float);
    size_t a_bf16_bytes = (size_t)M * K * sizeof(__nv_bfloat16);
    size_t c_bytes = (size_t)M * N * sizeof(float);

    void *d_A_f32 = pool_alloc(a_f32_bytes);
    void *d_A_bf16 = pool_alloc(a_bf16_bytes);
    void *d_C = pool_alloc(c_bytes);
    if (!d_A_f32 || !d_A_bf16 || !d_C) {
        if (d_A_f32) pool_free(d_A_f32);
        if (d_A_bf16) pool_free(d_A_bf16);
        if (d_C) pool_free(d_C);
        return;
    }

    /* Upload F32 activations */
    cudaMemcpyAsync(d_A_f32, A, a_f32_bytes, cudaMemcpyHostToDevice, g_stream);

    /* Convert F32 -> BF16 on GPU */
    int total = M * K;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    kernel_f32_to_bf16<<<blocks, threads, 0, g_stream>>>(
        (__nv_bfloat16 *)d_A_bf16, (const float *)d_A_f32, total);

    /* cuBLAS GEMM: C[M,N] = A_bf16[M,K] @ B_bf16[N,K]^T
     * Both inputs BF16, output F32, compute in F32.
     *
     * In column-major: C'[N,M] = B^T[N,K] * A_bf16'[K,M]
     * transa=T: stored B[K,N]col = B[N,K]row, transpose -> [N,K]
     * transb=N: stored A[K,M]col = A[M,K]row, no transpose -> [K,M] */
    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(g_cublas,
                 CUBLAS_OP_T, CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 d_B, CUDA_R_16BF, K,          /* B_bf16[N,K] row-major */
                 d_A_bf16, CUDA_R_16BF, K,     /* A_bf16[M,K] row-major */
                 &beta,
                 d_C, CUDA_R_32F, N,           /* C_f32[M,N] row-major */
                 CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    /* Download result */
    cudaMemcpyAsync(C, d_C, c_bytes, cudaMemcpyDeviceToHost, g_stream);
    cudaStreamSynchronize(g_stream);

    pool_free(d_A_f32);
    pool_free(d_A_bf16);
    pool_free(d_C);
}

extern "C" void vox_cuda_sgemm(int M, int N, int K,
                                 const float *A,
                                 const float *B,
                                 float *C) {
    if (!g_initialized) return;

    size_t a_bytes = (size_t)M * K * sizeof(float);
    size_t b_bytes = (size_t)N * K * sizeof(float);
    size_t c_bytes = (size_t)M * N * sizeof(float);

    void *d_A = pool_alloc(a_bytes);
    void *d_B = pool_alloc(b_bytes);
    void *d_C = pool_alloc(c_bytes);
    if (!d_A || !d_B || !d_C) {
        if (d_A) pool_free(d_A);
        if (d_B) pool_free(d_B);
        if (d_C) pool_free(d_C);
        return;
    }

    cudaMemcpyAsync(d_A, A, a_bytes, cudaMemcpyHostToDevice, g_stream);
    cudaMemcpyAsync(d_B, B, b_bytes, cudaMemcpyHostToDevice, g_stream);

    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(g_cublas,
                 CUBLAS_OP_T, CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 d_B, CUDA_R_32F, K,
                 d_A, CUDA_R_32F, K,
                 &beta,
                 d_C, CUDA_R_32F, N,
                 CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    cudaMemcpyAsync(C, d_C, c_bytes, cudaMemcpyDeviceToHost, g_stream);
    cudaStreamSynchronize(g_stream);

    pool_free(d_A);
    pool_free(d_B);
    pool_free(d_C);
}

/* ========================================================================
 * Weight Warmup
 * ======================================================================== */

extern "C" void vox_cuda_warmup_bf16(const uint16_t *bf16_weights, size_t num_elements) {
    if (!g_initialized || !bf16_weights) return;
    weight_cache_get_or_upload(bf16_weights, num_elements);
}

extern "C" void vox_cuda_warmup_f32(const float *f32_weights, size_t num_elements) {
    if (!g_initialized || !f32_weights) return;
    f32_cache_get_or_upload(f32_weights, num_elements);
}

/* ========================================================================
 * Shared Memory (Unified Memory)
 * ======================================================================== */

extern "C" void *vox_cuda_shared_alloc(size_t size) {
    if (!g_initialized) return calloc(1, size);

    void *ptr = NULL;
    cudaError_t err = cudaMallocManaged(&ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA: managed alloc failed (%zu bytes): %s\n",
                size, cudaGetErrorString(err));
        return calloc(1, size);
    }

    memset(ptr, 0, size);

    g_memory_used += size;
    return ptr;
}

extern "C" void vox_cuda_shared_free(void *ptr) {
    if (!ptr) return;
    if (!g_initialized) { free(ptr); return; }

    /* Try cudaFree — works for both managed and device memory.
     * If ptr was allocated with calloc (fallback), this will fail silently
     * and we'll free with free(). */
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
        /* Not a CUDA allocation — use regular free */
        cudaGetLastError(); /* clear the error */
        free(ptr);
    }
}

extern "C" size_t vox_cuda_memory_used(void) {
    return g_memory_used;
}

/* ========================================================================
 * Monolithic Decoder Step (single token generation)
 *
 * All 26 layers + final norm + logits on GPU with ONE cudaStreamSynchronize.
 * KV writes go device→managed (no page thrashing).
 * Returns argmax token ID, or -1 on failure.
 * ======================================================================== */

extern "C" int vox_cuda_decoder_step(void *ctx_, const float *input_embeds,
                                      const float *rope_freqs, float *logits) {
    vox_ctx_t *ctx = (vox_ctx_t *)ctx_;
    if (!g_initialized) return -1;
    if (ensure_dec_gpu_bufs() != 0) return -1;
    const int dim = VOX_DEC_DIM;
    const int n_heads = VOX_DEC_HEADS;
    const int n_kv_heads = VOX_DEC_KV_HEADS;
    const int head_dim = VOX_DEC_HEAD_DIM;
    const int hidden = VOX_DEC_HIDDEN;
    const int q_dim = n_heads * head_dim;
    const int kv_dim = n_kv_heads * head_dim;
    const int pos = ctx->kv_cache_len;
    const float scale = 1.0f / sqrtf((float)head_dim);

    /* Upload input + rope freqs */
    cudaMemcpyAsync(g_dec_bufs.x, input_embeds, dim * sizeof(float),
                    cudaMemcpyHostToDevice, g_stream);
    cudaMemcpyAsync(g_dec_bufs.rope_freqs, rope_freqs, head_dim * sizeof(float),
                    cudaMemcpyHostToDevice, g_stream);

    for (int layer = 0; layer < VOX_DEC_LAYERS; layer++) {
        vox_dec_layer_t *l = &ctx->decoder.layers[layer];

        /* --- Attention --- */
        float *d_anorm = f32_cache_get_or_upload(l->attention_norm, dim);
        kernel_rms_norm<<<1, 256, 256 * sizeof(float), g_stream>>>(
            g_dec_bufs.x_norm, g_dec_bufs.x, d_anorm, dim, VOX_DEC_NORM_EPS);

        sgemm_bf16_dev(1, q_dim, dim, g_dec_bufs.x_norm, l->wq_weight_bf16, g_dec_bufs.q);
        sgemm_bf16_dev(1, kv_dim, dim, g_dec_bufs.x_norm, l->wk_weight_bf16, g_dec_bufs.k);
        sgemm_bf16_dev(1, kv_dim, dim, g_dec_bufs.x_norm, l->wv_weight_bf16, g_dec_bufs.v);

        kernel_apply_rope<<<n_heads, head_dim / 2, 0, g_stream>>>(
            g_dec_bufs.q, g_dec_bufs.rope_freqs, 1, n_heads, head_dim);
        kernel_apply_rope<<<n_kv_heads, head_dim / 2, 0, g_stream>>>(
            g_dec_bufs.k, g_dec_bufs.rope_freqs, 1, n_kv_heads, head_dim);

        /* Write K,V to managed cache (device→managed, pages stay on GPU) */
        float *ck = ctx->kv_cache_k + ((size_t)layer * ctx->kv_cache_max + pos) * kv_dim;
        float *cv = ctx->kv_cache_v + ((size_t)layer * ctx->kv_cache_max + pos) * kv_dim;
        cudaMemcpyAsync(ck, g_dec_bufs.k, kv_dim * sizeof(float),
                        cudaMemcpyDeviceToDevice, g_stream);
        cudaMemcpyAsync(cv, g_dec_bufs.v, kv_dim * sizeof(float),
                        cudaMemcpyDeviceToDevice, g_stream);

        /* Attention: Q on device, K/V in managed memory */
        int total_seq = pos + 1;
        float *fk = ctx->kv_cache_k + (size_t)layer * ctx->kv_cache_max * kv_dim;
        float *fv = ctx->kv_cache_v + (size_t)layer * ctx->kv_cache_max * kv_dim;

        int n_warps = (head_dim + 31) / 32;
        kernel_causal_attention<<<dim3(n_heads, 1), dim3(head_dim),
                                  n_warps * sizeof(float), g_stream>>>(
            g_dec_bufs.attn_out, g_dec_bufs.q, fk, fv,
            total_seq, n_heads, n_kv_heads, head_dim, scale,
            VOX_DEC_WINDOW, pos);

        sgemm_bf16_dev(1, dim, q_dim, g_dec_bufs.attn_out, l->wo_weight_bf16, g_dec_bufs.proj_out);
        kernel_add_inplace<<<(dim + 255) / 256, 256, 0, g_stream>>>(
            g_dec_bufs.x, g_dec_bufs.proj_out, dim);

        /* --- FFN --- */
        float *d_fnorm = f32_cache_get_or_upload(l->ffn_norm, dim);
        kernel_rms_norm<<<1, 256, 256 * sizeof(float), g_stream>>>(
            g_dec_bufs.x_norm, g_dec_bufs.x, d_fnorm, dim, VOX_DEC_NORM_EPS);

        if (ctx->ada_scale) {
            float *d_ada = f32_cache_get_or_upload(ctx->ada_scale + (size_t)layer * dim, dim);
            kernel_ada_scale<<<(dim + 255) / 256, 256, 0, g_stream>>>(
                g_dec_bufs.x_norm, d_ada, dim, dim);
        }

        sgemm_bf16_dev(1, hidden, dim, g_dec_bufs.x_norm, l->w1_weight_bf16, g_dec_bufs.gate);
        kernel_silu<<<(hidden + 255) / 256, 256, 0, g_stream>>>(g_dec_bufs.gate, hidden);
        sgemm_bf16_dev(1, hidden, dim, g_dec_bufs.x_norm, l->w3_weight_bf16, g_dec_bufs.up);
        kernel_mul_inplace<<<(hidden + 255) / 256, 256, 0, g_stream>>>(
            g_dec_bufs.gate, g_dec_bufs.up, hidden);
        sgemm_bf16_dev(1, dim, hidden, g_dec_bufs.gate, l->w2_weight_bf16, g_dec_bufs.ffn_out);
        kernel_add_inplace<<<(dim + 255) / 256, 256, 0, g_stream>>>(
            g_dec_bufs.x, g_dec_bufs.ffn_out, dim);

    }

    /* Final norm + logits */
    float *d_fnorm = f32_cache_get_or_upload(ctx->decoder.norm, dim);
    kernel_rms_norm<<<1, 256, 256 * sizeof(float), g_stream>>>(
        g_dec_bufs.x, g_dec_bufs.x, d_fnorm, dim, VOX_DEC_NORM_EPS);
    sgemm_bf16_dev(1, VOX_VOCAB_SIZE, dim, g_dec_bufs.x,
                   ctx->decoder.tok_embeddings_bf16, g_dec_bufs.logits);

    /* Download logits — single sync for the entire step */
    cudaMemcpyAsync(logits, g_dec_bufs.logits, VOX_VOCAB_SIZE * sizeof(float),
                    cudaMemcpyDeviceToHost, g_stream);
    cudaStreamSynchronize(g_stream);

    /* Check for CUDA errors (deferred from kernel launches) */
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA decoder step error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    ctx->kv_cache_len = pos + 1;

    /* Argmax */
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < VOX_VOCAB_SIZE; i++) {
        if (logits[i] > best_val) { best_val = logits[i]; best = i; }
    }
    return best;
}

/* ========================================================================
 * Monolithic Encoder Step (incremental, new_len positions)
 *
 * All 32 layers + final norm on GPU with ONE cudaStreamSynchronize.
 * x is modified in-place (input: new positions, output: normed result).
 * Returns 0 on success, -1 on failure.
 * ======================================================================== */

extern "C" int vox_cuda_encoder_step(void *ctx_, float *x, int new_len,
                                      const float *rope_freqs, int cache_len) {
    vox_ctx_t *ctx = (vox_ctx_t *)ctx_;
    if (!g_initialized) return -1;
    if (ensure_enc_gpu_bufs(new_len) != 0) return -1;

    const int dim = VOX_ENC_DIM;
    const int n_heads = VOX_ENC_HEADS;
    const int n_kv_heads = VOX_ENC_KV_HEADS;
    const int head_dim = VOX_ENC_HEAD_DIM;
    const int hidden = VOX_ENC_HIDDEN;
    const int qkv_dim = n_heads * head_dim;
    const float scale = 1.0f / sqrtf((float)head_dim);

    /* Upload x and rope freqs */
    cudaMemcpyAsync(g_enc_bufs.x, x, (size_t)new_len * dim * sizeof(float),
                    cudaMemcpyHostToDevice, g_stream);
    size_t rope_bytes = (size_t)new_len * (head_dim / 2) * 2 * sizeof(float);
    cudaMemcpyAsync(g_enc_bufs.rope_freqs, rope_freqs, rope_bytes,
                    cudaMemcpyHostToDevice, g_stream);

    for (int layer = 0; layer < VOX_ENC_LAYERS; layer++) {
        vox_enc_layer_t *l = &ctx->encoder.layers[layer];

        /* --- Attention --- */
        float *d_anorm = f32_cache_get_or_upload(l->attention_norm, dim);
        kernel_rms_norm<<<new_len, 256, 256 * sizeof(float), g_stream>>>(
            g_enc_bufs.x_norm, g_enc_bufs.x, d_anorm, dim, VOX_ENC_NORM_EPS);

        sgemm_bf16_dev(new_len, qkv_dim, dim, g_enc_bufs.x_norm, l->wq_weight_bf16, g_enc_bufs.q);
        sgemm_bf16_dev(new_len, qkv_dim, dim, g_enc_bufs.x_norm, l->wk_weight_bf16, g_enc_bufs.k);
        sgemm_bf16_dev(new_len, qkv_dim, dim, g_enc_bufs.x_norm, l->wv_weight_bf16, g_enc_bufs.v);

        /* Add biases (wq has bias, wk has NO bias, wv has bias) */
        int q_total = new_len * qkv_dim;
        float *d_wq_bias = f32_cache_get_or_upload(l->wq_bias, qkv_dim);
        float *d_wv_bias = f32_cache_get_or_upload(l->wv_bias, qkv_dim);
        kernel_bias_add<<<(q_total + 255) / 256, 256, 0, g_stream>>>(
            g_enc_bufs.q, d_wq_bias, q_total, qkv_dim);
        kernel_bias_add<<<(q_total + 255) / 256, 256, 0, g_stream>>>(
            g_enc_bufs.v, d_wv_bias, q_total, qkv_dim);

        kernel_apply_rope<<<new_len * n_heads, head_dim / 2, 0, g_stream>>>(
            g_enc_bufs.q, g_enc_bufs.rope_freqs, new_len, n_heads, head_dim);
        kernel_apply_rope<<<new_len * n_heads, head_dim / 2, 0, g_stream>>>(
            g_enc_bufs.k, g_enc_bufs.rope_freqs, new_len, n_heads, head_dim);

        /* Write K,V to managed cache */
        float *ck = ctx->enc_kv_cache_k + ((size_t)layer * ctx->enc_kv_cache_max + cache_len) * qkv_dim;
        float *cv = ctx->enc_kv_cache_v + ((size_t)layer * ctx->enc_kv_cache_max + cache_len) * qkv_dim;
        cudaMemcpyAsync(ck, g_enc_bufs.k, (size_t)new_len * qkv_dim * sizeof(float),
                        cudaMemcpyDeviceToDevice, g_stream);
        cudaMemcpyAsync(cv, g_enc_bufs.v, (size_t)new_len * qkv_dim * sizeof(float),
                        cudaMemcpyDeviceToDevice, g_stream);

        /* Attention */
        int total_kv = cache_len + new_len;
        float *fk = ctx->enc_kv_cache_k + (size_t)layer * ctx->enc_kv_cache_max * qkv_dim;
        float *fv = ctx->enc_kv_cache_v + (size_t)layer * ctx->enc_kv_cache_max * qkv_dim;

        int n_warps = (head_dim + 31) / 32;
        kernel_causal_attention<<<dim3(n_heads, new_len), dim3(head_dim),
                                  n_warps * sizeof(float), g_stream>>>(
            g_enc_bufs.attn_out, g_enc_bufs.q, fk, fv,
            total_kv, n_heads, n_kv_heads, head_dim, scale,
            VOX_ENC_WINDOW, cache_len);

        /* Output projection + bias + residual */
        sgemm_bf16_dev(new_len, dim, qkv_dim, g_enc_bufs.attn_out, l->wo_weight_bf16, g_enc_bufs.proj_out);
        int proj_n = new_len * dim;
        float *d_wo_bias = f32_cache_get_or_upload(l->wo_bias, dim);
        kernel_bias_add<<<(proj_n + 255) / 256, 256, 0, g_stream>>>(
            g_enc_bufs.proj_out, d_wo_bias, proj_n, dim);
        kernel_add_inplace<<<(proj_n + 255) / 256, 256, 0, g_stream>>>(
            g_enc_bufs.x, g_enc_bufs.proj_out, proj_n);

        /* --- FFN --- */
        float *d_fnorm = f32_cache_get_or_upload(l->ffn_norm, dim);
        kernel_rms_norm<<<new_len, 256, 256 * sizeof(float), g_stream>>>(
            g_enc_bufs.x_norm, g_enc_bufs.x, d_fnorm, dim, VOX_ENC_NORM_EPS);

        int gate_n = new_len * hidden;
        sgemm_bf16_dev(new_len, hidden, dim, g_enc_bufs.x_norm, l->w1_weight_bf16, g_enc_bufs.gate);
        kernel_silu<<<(gate_n + 255) / 256, 256, 0, g_stream>>>(g_enc_bufs.gate, gate_n);
        sgemm_bf16_dev(new_len, hidden, dim, g_enc_bufs.x_norm, l->w3_weight_bf16, g_enc_bufs.up);
        kernel_mul_inplace<<<(gate_n + 255) / 256, 256, 0, g_stream>>>(
            g_enc_bufs.gate, g_enc_bufs.up, gate_n);
        sgemm_bf16_dev(new_len, dim, hidden, g_enc_bufs.gate, l->w2_weight_bf16, g_enc_bufs.ffn_out);

        /* Add w2 bias + residual */
        int ffn_n = new_len * dim;
        float *d_w2_bias = f32_cache_get_or_upload(l->w2_bias, dim);
        kernel_bias_add<<<(ffn_n + 255) / 256, 256, 0, g_stream>>>(
            g_enc_bufs.ffn_out, d_w2_bias, ffn_n, dim);
        kernel_add_inplace<<<(ffn_n + 255) / 256, 256, 0, g_stream>>>(
            g_enc_bufs.x, g_enc_bufs.ffn_out, ffn_n);
    }

    /* Final norm */
    float *d_enorm = f32_cache_get_or_upload(ctx->encoder.norm, dim);
    kernel_rms_norm<<<new_len, 256, 256 * sizeof(float), g_stream>>>(
        g_enc_bufs.x, g_enc_bufs.x, d_enorm, dim, VOX_ENC_NORM_EPS);

    /* Download result — single sync for entire step */
    cudaMemcpyAsync(x, g_enc_bufs.x, (size_t)new_len * dim * sizeof(float),
                    cudaMemcpyDeviceToHost, g_stream);
    cudaStreamSynchronize(g_stream);

    ctx->enc_kv_cache_len = cache_len + new_len;
    return 0;
}
