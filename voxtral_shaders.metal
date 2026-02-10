/*
 * voxtral_shaders.metal - Metal compute shaders for Voxtral inference
 *
 * GPU kernels for element-wise ops that avoid CPU round-trips when used
 * between MPS matmul calls. All operate on f32 tensors.
 */

#include <metal_stdlib>
using namespace metal;

/* ========================================================================
 * RMSNorm: out[i] = x[i] * rsqrt(mean(x^2) + eps) * weight[i]
 * One threadgroup per row. x: [seq, hidden], weight: [hidden]
 * ======================================================================== */

kernel void rms_norm(
    device const float *x [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant int &hidden [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]
) {
    threadgroup float shared_sum[256];

    device const float *x_row = x + row * hidden;
    device float *out_row = out + row * hidden;

    float local_sum = 0.0f;
    for (int i = tid; i < hidden; i += threads) {
        float val = x_row[i];
        local_sum += val * val;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms_inv = rsqrt(shared_sum[0] / float(hidden) + eps);

    for (int i = tid; i < hidden; i += threads) {
        out_row[i] = x_row[i] * rms_inv * weight[i];
    }
}

/* ========================================================================
 * SiLU: x = x / (1 + exp(-x))
 * ======================================================================== */

kernel void silu(
    device float *x [[buffer(0)]],
    constant int &n [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < uint(n)) {
        float val = x[gid];
        x[gid] = val / (1.0f + exp(-val));
    }
}

/* ========================================================================
 * GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
 * ======================================================================== */

kernel void gelu(
    device float *x [[buffer(0)]],
    constant int &n [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < uint(n)) {
        float val = x[gid];
        float x3 = val * val * val;
        float inner = 0.7978845608028654f * (val + 0.044715f * x3);
        x[gid] = 0.5f * val * (1.0f + tanh(inner));
    }
}

/* ========================================================================
 * Element-wise ops
 * ======================================================================== */

kernel void add_inplace(
    device float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    constant int &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < uint(n)) a[gid] += b[gid];
}

kernel void mul_inplace(
    device float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    constant int &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < uint(n)) a[gid] *= b[gid];
}

/* x[i] *= (1 + scale[i]) — adaptive RMS norm conditioning */
kernel void ada_scale_mul(
    device float *x [[buffer(0)]],
    device const float *scale [[buffer(1)]],
    constant int &n [[buffer(2)]],
    constant int &stride [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < uint(n)) x[gid] *= (1.0f + scale[gid % stride]);
}

/* ========================================================================
 * Argmax over a float array. Returns index of max value.
 * One threadgroup, result written to out[0].
 * ======================================================================== */

kernel void argmax_f32(
    device const float *data [[buffer(0)]],
    device int *out [[buffer(1)]],
    constant int &n [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]
) {
    threadgroup float shared_val[256];
    threadgroup int shared_idx[256];

    float best_val = -INFINITY;
    int best_idx = 0;
    for (int i = tid; i < n; i += threads) {
        float v = data[i];
        if (v > best_val) { best_val = v; best_idx = i; }
    }
    shared_val[tid] = best_val;
    shared_idx[tid] = best_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_val[tid + stride] > shared_val[tid]) {
                shared_val[tid] = shared_val[tid + stride];
                shared_idx[tid] = shared_idx[tid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) out[0] = shared_idx[0];
}

/* ========================================================================
 * Causal masked softmax for attention scores.
 * scores: [n_heads, seq_q, seq_k] (contiguous per head)
 * One threadgroup per (query_position, head) pair.
 *
 * Applies:
 *   - Causal mask: query at q_offset+qi attends to keys 0..q_offset+qi
 *   - Sliding window: keys below max(0, q_pos - window + 1) are masked
 *   - Softmax normalization (numerically stable)
 * ======================================================================== */

kernel void causal_softmax(
    device float *scores [[buffer(0)]],
    constant int &seq_q [[buffer(1)]],
    constant int &seq_k [[buffer(2)]],
    constant int &window_size [[buffer(3)]],
    constant int &q_offset [[buffer(4)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    int qi = group_id % seq_q;
    int head = group_id / seq_q;

    device float *row = scores + ((long)head * seq_q + qi) * seq_k;

    int q_pos = q_offset + qi;
    int valid_end = min(q_pos, seq_k - 1);
    int valid_start = (window_size > 0) ? max(0, q_pos - window_size + 1) : 0;

    threadgroup float shared[256];

    /* Phase 1: apply mask, find row max */
    float local_max = -INFINITY;
    for (int j = tid; j < seq_k; j += tg_size) {
        float val = (j >= valid_start && j <= valid_end) ? row[j] : -INFINITY;
        row[j] = val;
        local_max = fmax(local_max, val);
    }
    shared[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] = fmax(shared[tid], shared[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared[0];

    /* Phase 2: exp(x - max) and sum */
    float local_sum = 0.0f;
    for (int j = tid; j < seq_k; j += tg_size) {
        float val = exp(row[j] - row_max);
        row[j] = val;
        local_sum += val;
    }
    shared[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tg_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_sum = 1.0f / (shared[0] + 1e-10f);

    /* Phase 3: normalize */
    for (int j = tid; j < seq_k; j += tg_size) {
        row[j] *= inv_sum;
    }
}

/* ========================================================================
 * RoPE: apply rotary position embedding in-place.
 * data: [n_heads * head_dim], freqs: [head_dim/2 * 2] = (cos,sin) pairs.
 * One thread per (head, half_dim_index) pair.
 * ======================================================================== */

kernel void rope_apply(
    device float *data [[buffer(0)]],
    device const float *freqs [[buffer(1)]],
    constant int &n_heads [[buffer(2)]],
    constant int &head_dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    int half_dim = head_dim / 2;
    int total = n_heads * half_dim;
    if ((int)gid >= total) return;

    int head = (int)gid / half_dim;
    int i = (int)gid % half_dim;

    float cos_val = freqs[i * 2];
    float sin_val = freqs[i * 2 + 1];

    int base = head * head_dim;
    float x0 = data[base + i * 2];
    float x1 = data[base + i * 2 + 1];

    data[base + i * 2]     = x0 * cos_val - x1 * sin_val;
    data[base + i * 2 + 1] = x0 * sin_val + x1 * cos_val;
}

/* ========================================================================
 * KV cache copy: write kv_dim floats to a position in the cache.
 * cache: large buffer, data written at float_offset + gid.
 * ======================================================================== */

kernel void kv_cache_copy(
    device float *cache [[buffer(0)]],
    device const float *data [[buffer(1)]],
    constant int &float_offset [[buffer(2)]],
    constant int &kv_dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((int)gid < kv_dim) {
        cache[float_offset + gid] = data[gid];
    }
}

/* ========================================================================
 * Single-token decoder attention (seq_q=1).
 * One threadgroup per query head, 128 threads cooperate on dot products.
 * K/V read from the KV cache buffer at a per-layer offset.
 * Uses online softmax (single pass) with SIMD group reductions.
 * 128 threads = 4 SIMD groups of 32. simd_sum for fast dot product.
 * ======================================================================== */

kernel void decoder_attention(
    device const float *Q [[buffer(0)]],
    device const float *K_cache [[buffer(1)]],
    device const float *V_cache [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant int &n_heads [[buffer(4)]],
    constant int &n_kv_heads [[buffer(5)]],
    constant int &head_dim [[buffer(6)]],
    constant int &kv_dim [[buffer(7)]],
    constant int &seq_k [[buffer(8)]],
    constant float &scale [[buffer(9)]],
    constant int &window_size [[buffer(10)]],
    constant int &q_pos [[buffer(11)]],
    uint head_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    if ((int)head_idx >= n_heads) return;

    int gqa_ratio = n_heads / n_kv_heads;
    int kv_head = (int)head_idx / gqa_ratio;

    device const float *q_h = Q + head_idx * head_dim;
    device float *out_h = out + head_idx * head_dim;

    int valid_end = min(q_pos, seq_k - 1);
    int valid_start = (window_size > 0) ? max(0, q_pos - window_size + 1) : 0;

    /* 128 threads = 4 SIMD groups of 32 */
    threadgroup float shared_simd[4];

    /* Each thread loads one Q element (head_dim=128) */
    float q_val = (int)tid < head_dim ? q_h[tid] : 0.0f;

    /* Online softmax: single pass over keys */
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float acc = 0.0f;

    for (int j = valid_start; j <= valid_end; j++) {
        device const float *k_j = K_cache + j * kv_dim + kv_head * head_dim;

        /* Cooperative dot product using SIMD reductions */
        float partial = (int)tid < head_dim ? q_val * k_j[tid] : 0.0f;
        float simd_partial = simd_sum(partial);

        /* Cross-SIMD reduction: 4 groups → 1 value */
        if (simd_lid == 0) shared_simd[simd_gid] = simd_partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float score;
        if (tid == 0) {
            shared_simd[0] = (shared_simd[0] + shared_simd[1] +
                              shared_simd[2] + shared_simd[3]) * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        score = shared_simd[0];

        /* Online softmax update */
        float old_max = running_max;
        running_max = fmax(running_max, score);
        float correction = exp(old_max - running_max);
        running_sum = running_sum * correction + exp(score - running_max);
        acc = acc * correction;

        /* Accumulate weighted V */
        if ((int)tid < head_dim) {
            device const float *v_j = V_cache + j * kv_dim + kv_head * head_dim;
            acc += exp(score - running_max) * v_j[tid];
        }
    }

    /* Normalize and write output */
    if ((int)tid < head_dim) {
        out_h[tid] = acc / (running_sum + 1e-10f);
    }
}

/* ========================================================================
 * Q-tiled batched attention: one threadgroup per (head, query_block).
 * Processes ATTN_BQ queries per threadgroup, amortizing K/V memory reads.
 * Supports head_dim=64 (64 threads, 2 SIMD groups) and head_dim=128
 * (128 threads, 4 SIMD groups). Used for both encoder and decoder prefill.
 * Q/K/V layout: [seq, n_heads * head_dim] packed (head-interleaved).
 * Uses online softmax, cooperative SIMD dot products.
 *
 * Grid: n_heads * ceil(seq_q / ATTN_BQ) threadgroups.
 * group_idx = h * n_q_blocks + qb.
 * ======================================================================== */

#define ATTN_BQ 8

kernel void encoder_attention(
    device const float *Q [[buffer(0)]],
    device const float *K [[buffer(1)]],
    device const float *V [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant int &n_heads [[buffer(4)]],
    constant int &n_kv_heads [[buffer(5)]],
    constant int &head_dim [[buffer(6)]],
    constant int &seq_q [[buffer(7)]],
    constant int &seq_k [[buffer(8)]],
    constant float &scale [[buffer(9)]],
    constant int &window_size [[buffer(10)]],
    constant int &q_offset [[buffer(11)]],
    uint group_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    int n_q_blocks = (seq_q + ATTN_BQ - 1) / ATTN_BQ;
    int h = (int)group_idx / n_q_blocks;
    int qb = (int)group_idx % n_q_blocks;
    int qi_start = qb * ATTN_BQ;
    if (h >= n_heads) return;

    int gqa_ratio = n_heads / n_kv_heads;
    int kv_h = h / gqa_ratio;
    int stride_q = n_heads * head_dim;
    int stride_kv = n_kv_heads * head_dim;
    int n_simd_groups = (int)tg_size / 32;

    /* Load BQ query values (one head_dim element per thread, BQ queries) */
    float q_vals[ATTN_BQ];
    for (int b = 0; b < ATTN_BQ; b++) {
        int qi = qi_start + b;
        q_vals[b] = (qi < seq_q && (int)tid < head_dim)
            ? Q[(long)qi * stride_q + h * head_dim + tid] : 0.0f;
    }

    /* Per-query online softmax state */
    float rmax[ATTN_BQ], rsum[ATTN_BQ], acc[ATTN_BQ];
    for (int b = 0; b < ATTN_BQ; b++) {
        rmax[b] = -INFINITY;
        rsum[b] = 0.0f;
        acc[b] = 0.0f;
    }

    /* Shared memory for cross-SIMD dot product reduction */
    threadgroup float tg_simd[4 * ATTN_BQ];
    threadgroup float tg_scores[ATTN_BQ];

    /* Compute loop range: union of all BQ queries' valid key ranges */
    int last_qi = min(qi_start + ATTN_BQ - 1, seq_q - 1);
    int first_pos = q_offset + qi_start;
    int last_pos = q_offset + last_qi;
    int loop_start = (window_size > 0) ? max(0, first_pos - window_size + 1) : 0;
    int loop_end = min(last_pos, seq_k - 1);

    for (int j = loop_start; j <= loop_end; j++) {
        device const float *k_j = K + (long)j * stride_kv + kv_h * head_dim;
        float k_val = (int)tid < head_dim ? k_j[tid] : 0.0f;

        /* BQ dot products via simd_sum + cross-SIMD store */
        for (int b = 0; b < ATTN_BQ; b++) {
            float simd_dot = simd_sum(q_vals[b] * k_val);
            if (simd_lid == 0) tg_simd[simd_gid * ATTN_BQ + b] = simd_dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* Cross-SIMD reduction: first BQ threads each reduce one score */
        if ((int)tid < ATTN_BQ) {
            float sum = 0;
            for (int g = 0; g < n_simd_groups; g++)
                sum += tg_simd[g * ATTN_BQ + (int)tid];
            tg_scores[(int)tid] = sum * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* Load V once for this key position */
        device const float *v_j = V + (long)j * stride_kv + kv_h * head_dim;
        float v_val = (int)tid < head_dim ? v_j[tid] : 0.0f;

        /* Update BQ online softmax + accumulate weighted V */
        for (int b = 0; b < ATTN_BQ; b++) {
            int qi = qi_start + b;
            if (qi >= seq_q) continue;
            int q_pos = q_offset + qi;
            int vs = (window_size > 0) ? max(0, q_pos - window_size + 1) : 0;
            if (j < vs || j > q_pos) continue;

            float score = tg_scores[b];
            float old_max = rmax[b];
            rmax[b] = fmax(rmax[b], score);
            float corr = exp(old_max - rmax[b]);
            rsum[b] = rsum[b] * corr + exp(score - rmax[b]);
            acc[b] = acc[b] * corr + exp(score - rmax[b]) * v_val;
        }
    }

    /* Write BQ outputs */
    if ((int)tid < head_dim) {
        for (int b = 0; b < ATTN_BQ; b++) {
            int qi = qi_start + b;
            if (qi < seq_q) {
                device float *out_row = out + (long)qi * stride_q + h * head_dim;
                out_row[tid] = acc[b] / (rsum[b] + 1e-10f);
            }
        }
    }
}

/* ========================================================================
 * Bias add: data[s * dim + j] += bias[j] for each row s.
 * data: [seq_len, dim], bias: [dim].
 * ======================================================================== */

kernel void bias_add(
    device float *data [[buffer(0)]],
    device const float *bias [[buffer(1)]],
    constant int &dim [[buffer(2)]],
    constant int &total [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((int)gid < total) {
        data[gid] += bias[gid % dim];
    }
}

/* ========================================================================
 * Batched RoPE: apply rotary embeddings to [seq_len, n_heads, head_dim].
 * freqs: [seq_len, head_dim/2, 2] = per-position (cos, sin) pairs.
 * One thread per (position, head, half_dim_index) triple.
 * ======================================================================== */

kernel void batched_rope_apply(
    device float *data [[buffer(0)]],
    device const float *freqs [[buffer(1)]],
    constant int &n_heads [[buffer(2)]],
    constant int &head_dim [[buffer(3)]],
    constant int &seq_len [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    int half_dim = head_dim / 2;
    int per_pos = n_heads * half_dim;
    int total = seq_len * per_pos;
    if ((int)gid >= total) return;

    int pos = (int)gid / per_pos;
    int rem = (int)gid % per_pos;
    int head = rem / half_dim;
    int i = rem % half_dim;

    float cos_val = freqs[(pos * half_dim + i) * 2];
    float sin_val = freqs[(pos * half_dim + i) * 2 + 1];

    int base = (pos * n_heads + head) * head_dim;
    float x0 = data[base + i * 2];
    float x1 = data[base + i * 2 + 1];

    data[base + i * 2]     = x0 * cos_val - x1 * sin_val;
    data[base + i * 2 + 1] = x0 * sin_val + x1 * cos_val;
}

/* ========================================================================
 * Batched KV cache copy: write [seq_len, kv_dim] to cache at offset.
 * cache: large buffer, data copied to cache[cache_offset + gid].
 * ======================================================================== */

kernel void batched_kv_cache_copy(
    device float *cache [[buffer(0)]],
    device const float *data [[buffer(1)]],
    constant int &cache_offset [[buffer(2)]],
    constant int &total [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((int)gid < total) {
        cache[cache_offset + gid] = data[gid];
    }
}

/* ========================================================================
 * Deinterleave: copy one column slice from [M, total_cols] to [M, chunk_cols].
 * src layout: row i -> [col_0..col_{total_cols-1}]
 * dst layout: row i -> [col_offset..col_offset+chunk_cols-1] extracted contiguously.
 * total threads = M * chunk_cols.
 * ======================================================================== */

kernel void deinterleave(
    device const float *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant int &src_stride [[buffer(2)]],    /* total cols per src row */
    constant int &chunk_cols [[buffer(3)]],    /* cols to copy per row */
    constant int &col_offset [[buffer(4)]],    /* start column in src row */
    constant int &total [[buffer(5)]],         /* M * chunk_cols */
    uint gid [[thread_position_in_grid]]
) {
    if ((int)gid >= total) return;
    int row = (int)gid / chunk_cols;
    int col = (int)gid % chunk_cols;
    dst[gid] = src[row * src_stride + col_offset + col];
}

/* ========================================================================
 * Fused SiLU + multiply for merged w1+w3 output.
 * Data layout: [M, hidden*2] where each row is [gate(hidden), up(hidden)].
 * gate = silu(gate), gate *= up.  In-place.
 * total threads = M * hidden.
 * ======================================================================== */

kernel void silu_mul_merged(
    device float *data [[buffer(0)]],
    constant int &hidden [[buffer(1)]],     /* 5120 */
    constant int &total [[buffer(2)]],      /* M * hidden */
    uint gid [[thread_position_in_grid]]
) {
    if ((int)gid >= total) return;
    int row = (int)gid / hidden;
    int col = (int)gid % hidden;
    int idx_gate = row * hidden * 2 + col;
    int idx_up = idx_gate + hidden;
    float g = data[idx_gate];
    g = g / (1.0f + exp(-g));  /* silu */
    data[idx_gate] = g * data[idx_up];
}

/* ========================================================================
 * INT8 GEMV: y[N] = x[K] @ W[K,N] where W is INT8 with per-tensor scale.
 *
 * Optimized for single-token decode (M=1, memory-bound).
 * Weight layout: W[K,N] row-major (K rows, N cols), stored as INT8.
 * Dequantization: float_val = int8_val * scale
 *
 * Strategy:
 * - Each threadgroup computes TILE_N=4 consecutive output elements.
 * - 256 threads per threadgroup, each handles K/256 elements.
 * - Use simd_sum for fast parallel reduction across K dimension.
 * - Load 4 bytes (4 INT8) per thread per iteration = 1KB per threadgroup.
 *
 * Grid: (N/TILE_N) threadgroups. Each outputs 4 floats.
 * ======================================================================== */

#define GEMV_TILE_N 4
#define GEMV_THREADS 512

kernel void int8_gemv(
    device const float *x [[buffer(0)]],        /* Input: [K] */
    device const char *W [[buffer(1)]],         /* Weights: [N, K] INT8 row-major (PyTorch layout) */
    device float *y [[buffer(2)]],              /* Output: [N] */
    constant float &scale [[buffer(3)]],        /* Dequant scale */
    constant int &K [[buffer(4)]],
    constant int &N [[buffer(5)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    int n_base = (int)group_id * GEMV_TILE_N;
    if (n_base >= N) return;

    /* Partial sums for TILE_N outputs */
    float acc[GEMV_TILE_N] = {0.0f, 0.0f, 0.0f, 0.0f};

    /* Vectorized: process 4 K elements at a time using char4 loads */
    int K4 = K / 4;
    device const char4 *W4 = (device const char4 *)W;
    device const float4 *x4 = (device const float4 *)x;

    for (int k4 = (int)tid; k4 < K4; k4 += GEMV_THREADS) {
        float4 xv = x4[k4];

        /* Load TILE_N weight vectors: W is [N, K] row-major */
        for (int t = 0; t < GEMV_TILE_N; t++) {
            int n = n_base + t;
            if (n < N) {
                char4 wv = W4[n * K4 + k4];
                acc[t] += xv.x * float(wv.x) + xv.y * float(wv.y) +
                          xv.z * float(wv.z) + xv.w * float(wv.w);
            }
        }
    }

    /* Handle remainder (K not divisible by 4) */
    int k_rem_start = K4 * 4;
    for (int k = k_rem_start + (int)tid; k < K; k += GEMV_THREADS) {
        float x_val = x[k];
        for (int t = 0; t < GEMV_TILE_N; t++) {
            int n = n_base + t;
            if (n < N) {
                char w_int8 = W[n * K + k];
                acc[t] += x_val * float(w_int8);
            }
        }
    }

    /* SIMD reduction: sum across threads in same SIMD group */
    for (int t = 0; t < GEMV_TILE_N; t++) {
        acc[t] = simd_sum(acc[t]);
    }

    /* Cross-SIMD reduction using threadgroup memory */
    threadgroup float shared[16 * GEMV_TILE_N];  /* 16 SIMD groups (512 threads / 32) */
    if (simd_lid == 0) {
        for (int t = 0; t < GEMV_TILE_N; t++) {
            shared[simd_gid * GEMV_TILE_N + t] = acc[t];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    /* First TILE_N threads finalize and write output */
    if (tid < GEMV_TILE_N) {
        int n = n_base + (int)tid;
        if (n < N) {
            float sum = 0.0f;
            int n_simd_groups = GEMV_THREADS / 32;
            for (int g = 0; g < n_simd_groups; g++) {
                sum += shared[g * GEMV_TILE_N + (int)tid];
            }
            y[n] = sum * scale;
        }
    }
}

/* ========================================================================
 * INT8 GEMM: C[M,N] = A[M,K] @ B[K,N] where B is INT8 with per-tensor scale.
 *
 * Tiled GEMM for batched decode / encoder prefill (M > 1).
 * Uses threadgroup shared memory for A tile (F32) and B tile (dequant'd F32).
 *
 * Tile sizes: TILE_M=8, TILE_N=32, TILE_K=32.
 * Threadgroup: 8x32 = 256 threads. Each thread computes one C element.
 *
 * Grid: (ceil(N/TILE_N), ceil(M/TILE_M)) threadgroups.
 * ======================================================================== */

#define TILE_M 8
#define TILE_N 32
#define TILE_K 32

kernel void int8_gemm(
    device const float *A [[buffer(0)]],        /* Input: [M, K] */
    device const char *B [[buffer(1)]],         /* Weights: [N, K] INT8 row-major (PyTorch layout) */
    device float *C [[buffer(2)]],              /* Output: [M, N] */
    constant float &scale [[buffer(3)]],        /* Dequant scale */
    constant int &M [[buffer(4)]],
    constant int &K [[buffer(5)]],
    constant int &N [[buffer(6)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint2 local_id [[thread_position_in_threadgroup]]
) {
    int m_base = (int)group_id.y * TILE_M;
    int n_base = (int)group_id.x * TILE_N;

    int local_m = (int)local_id.y;  /* 0..TILE_M-1 */
    int local_n = (int)local_id.x;  /* 0..TILE_N-1 */

    int global_m = m_base + local_m;
    int global_n = n_base + local_n;

    /* Shared memory tiles */
    threadgroup float A_tile[TILE_M][TILE_K];
    threadgroup float B_tile[TILE_K][TILE_N];

    float acc = 0.0f;

    /* Loop over K in tiles */
    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        /* Cooperative load: A tile (8x32 threads load 8x32 tile) */
        /* Each thread loads one element */
        int a_k = k0 + local_n;  /* local_n = 0..31, but TILE_K=32 */
        if (local_n < TILE_K && global_m < M && a_k < K) {
            A_tile[local_m][local_n] = A[global_m * K + a_k];
        } else if (local_n < TILE_K) {
            A_tile[local_m][local_n] = 0.0f;
        }

        /* Cooperative load + dequant: B tile using char4 vectorized loads
         * B is [N, K] row-major (PyTorch layout), we want W^T for C = A @ W^T
         * Access: B[n, k] at memory B[n * K + k]
         * 256 threads each load 4 consecutive k-values (char4) for one n-row. */
        int flat_id = local_m * TILE_N + local_n; /* 0..255 */
        int load_n = flat_id / 8;         /* 0..31: which n row */
        int k_chunk = flat_id % 8;        /* 0..7: which char4 chunk */
        int load_k_base = k_chunk * 4;    /* 0, 4, 8, ..., 28 */

        int b_n = n_base + load_n;
        int b_k_base = k0 + load_k_base;

        if (b_n < N && b_k_base + 3 < K) {
            /* Fast path: load 4 consecutive k-values as char4 */
            device const char4 *ptr = (device const char4 *)(B + b_n * K + b_k_base);
            char4 data = *ptr;
            B_tile[load_k_base + 0][load_n] = float(data.x);
            B_tile[load_k_base + 1][load_n] = float(data.y);
            B_tile[load_k_base + 2][load_n] = float(data.z);
            B_tile[load_k_base + 3][load_n] = float(data.w);
        } else {
            /* Boundary handling: load individual bytes */
            for (int dk = 0; dk < 4; dk++) {
                int b_k = b_k_base + dk;
                if (b_n < N && b_k < K) {
                    B_tile[load_k_base + dk][load_n] = float(B[b_n * K + b_k]);
                } else {
                    B_tile[load_k_base + dk][load_n] = 0.0f;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* Compute partial products */
        if (global_m < M && global_n < N) {
            for (int kk = 0; kk < TILE_K; kk++) {
                acc += A_tile[local_m][kk] * B_tile[kk][local_n];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* Write output */
    if (global_m < M && global_n < N) {
        C[global_m * N + global_n] = acc * scale;
    }
}

/* ========================================================================
 * Fused INT8 GEMM for decoder QKV (batched prefill).
 *
 * Computes:
 *   Q[M, Nq] = A[M, K] @ Wq[Nq, K]^T
 *   K[M, Nk] = A[M, K] @ Wk[Nk, K]^T
 *   V[M, Nv] = A[M, K] @ Wv[Nv, K]^T
 *
 * Weight layout: W*[N, K] INT8 row-major (PyTorch layout).
 *
 * Tile sizes match int8_gemm: TILE_M=8, TILE_N=32, TILE_K=32.
 * Grid: (ceil((Nq+Nk+Nv)/TILE_N), ceil(M/TILE_M)).
 * ======================================================================== */

kernel void int8_gemm_qkv(
    device const float *A [[buffer(0)]],        /* Input: [M, K] */
    device const char *Wq [[buffer(1)]],        /* Weights: [Nq, K] INT8 */
    device const char *Wk [[buffer(2)]],        /* Weights: [Nk, K] INT8 */
    device const char *Wv [[buffer(3)]],        /* Weights: [Nv, K] INT8 */
    device float *Q [[buffer(4)]],              /* Output: [M, Nq] */
    device float *Kout [[buffer(5)]],           /* Output: [M, Nk] */
    device float *Vout [[buffer(6)]],           /* Output: [M, Nv] */
    constant float &scale_q [[buffer(7)]],      /* Scalar scale for Q */
    constant float &scale_k [[buffer(8)]],      /* Scalar scale for K */
    constant float &scale_v [[buffer(9)]],      /* Scalar scale for V */
    constant int &M [[buffer(10)]],
    constant int &Kdim [[buffer(11)]],
    constant int &Nq [[buffer(12)]],
    constant int &Nk [[buffer(13)]],
    constant int &Nv [[buffer(14)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint2 local_id [[thread_position_in_threadgroup]]
) {
    int m_base = (int)group_id.y * TILE_M;
    int n_base = (int)group_id.x * TILE_N;
    int Ntotal = Nq + Nk + Nv;

    /* Choose the source weights once per tile when the tile does not cross
     * the Q/K/V boundaries. This avoids per-element branching in the inner
     * load loop. */
    int n_offset = 0;
    int Nout = Nq;
    device const char *W = Wq;
    device float *Out = Q;
    float scalar_scale = scale_q;
    int mixed_tile = 0;
    int n_end = n_base + TILE_N - 1;
    if (n_end < Nq) {
        /* Q */
    } else if (n_base >= Nq && n_end < Nq + Nk) {
        /* K */
        n_offset = Nq;
        Nout = Nk;
        W = Wk;
        Out = Kout;
        scalar_scale = scale_k;
    } else if (n_base >= Nq + Nk) {
        /* V */
        n_offset = Nq + Nk;
        Nout = Nv;
        W = Wv;
        Out = Vout;
        scalar_scale = scale_v;
    } else {
        mixed_tile = 1;
    }

    int local_m = (int)local_id.y;
    int local_n = (int)local_id.x;

    int global_m = m_base + local_m;
    int global_n = n_base + local_n;

    threadgroup float A_tile[TILE_M][TILE_K];
    threadgroup float B_tile[TILE_K][TILE_N];

    float acc = 0.0f;

    for (int k0 = 0; k0 < Kdim; k0 += TILE_K) {
        int a_k = k0 + local_n;
        if (local_n < TILE_K && global_m < M && a_k < Kdim) {
            A_tile[local_m][local_n] = A[global_m * Kdim + a_k];
        } else if (local_n < TILE_K) {
            A_tile[local_m][local_n] = 0.0f;
        }

        /* Cooperative B tile load with char4 vectorized loads for non-mixed tiles.
         * TILE_K=32 / 4 = 8 char4 chunks per row, 256 threads (TILE_M * TILE_N)
         * cover all 32 rows × 8 chunks = 256 loads exactly. */
        int flat_id = local_m * TILE_N + local_n;
        int load_n_idx = flat_id / 8;      /* 0..31: which n row */
        int k_chunk = flat_id % 8;         /* 0..7: which char4 chunk */
        int load_k_base = k_chunk * 4;     /* 0, 4, 8, ..., 28 */

        int b_n = n_base + load_n_idx;
        int b_k_base = k0 + load_k_base;

        if (!mixed_tile && b_n < Ntotal && b_k_base + 3 < Kdim) {
            /* Fast path: use char4 vectorized loads */
            int row = b_n - n_offset;
            device const char4 *ptr = (device const char4 *)(W + row * Kdim + b_k_base);
            char4 data = *ptr;
            B_tile[load_k_base + 0][load_n_idx] = float(data.x);
            B_tile[load_k_base + 1][load_n_idx] = float(data.y);
            B_tile[load_k_base + 2][load_n_idx] = float(data.z);
            B_tile[load_k_base + 3][load_n_idx] = float(data.w);
        } else {
            /* Slow path: mixed tile or boundary conditions */
            int tile_elems = TILE_K * TILE_N;
            int tg_elems = TILE_M * TILE_N;
            for (int idx = flat_id; idx < tile_elems; idx += tg_elems) {
                int load_k = idx / TILE_N;
                int load_n = idx % TILE_N;
                int b_k = k0 + load_k;
                int bn = n_base + load_n;

                if (b_k < Kdim && bn < Ntotal) {
                    char w_int8 = 0;
                    if (!mixed_tile) {
                        int row = bn - n_offset;
                        w_int8 = W[row * Kdim + b_k];
                    } else {
                        device const char *Wm = Wq;
                        int row = bn;
                        if (bn >= Nq) {
                            if (bn < Nq + Nk) {
                                Wm = Wk;
                                row = bn - Nq;
                            } else {
                                Wm = Wv;
                                row = bn - (Nq + Nk);
                            }
                        }
                        w_int8 = Wm[row * Kdim + b_k];
                    }
                    B_tile[load_k][load_n] = float(w_int8);
                } else {
                    B_tile[load_k][load_n] = 0.0f;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (global_m < M && global_n < Ntotal) {
            for (int kk = 0; kk < TILE_K; kk++) {
                acc += A_tile[local_m][kk] * B_tile[kk][local_n];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (global_m < M && global_n < Ntotal) {
        if (!mixed_tile) {
            int col = global_n - n_offset;
            Out[global_m * Nout + col] = acc * scalar_scale;
        } else {
            if (global_n < Nq) {
                Q[global_m * Nq + global_n] = acc * scale_q;
            } else if (global_n < Nq + Nk) {
                int n_k = global_n - Nq;
                Kout[global_m * Nk + n_k] = acc * scale_k;
            } else {
                int n_v = global_n - (Nq + Nk);
                Vout[global_m * Nv + n_v] = acc * scale_v;
            }
        }
    }
}

/* ========================================================================
 * Fused INT8 GEMM for decoder FFN W1+W3 (batched prefill).
 *
 * Computes merged output:
 *   C[M, 2*hidden] = A[M, K] @ [W1; W3]^T
 *
 * Data layout in C matches silu_mul_merged kernel:
 *   row i: [gate(hidden), up(hidden)]
 * ======================================================================== */

kernel void int8_gemm_w1w3(
    device const float *A [[buffer(0)]],         /* Input: [M, K] */
    device const char *W1 [[buffer(1)]],         /* Weights: [hidden, K] INT8 */
    device const char *W3 [[buffer(2)]],         /* Weights: [hidden, K] INT8 */
    device float *C [[buffer(3)]],               /* Output: [M, 2*hidden] */
    constant float &scale_1 [[buffer(4)]],       /* Scalar scale for W1 */
    constant float &scale_3 [[buffer(5)]],       /* Scalar scale for W3 */
    constant int &M [[buffer(6)]],
    constant int &Kdim [[buffer(7)]],
    constant int &hidden [[buffer(8)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint2 local_id [[thread_position_in_threadgroup]]
) {
    int m_base = (int)group_id.y * TILE_M;
    int n_base = (int)group_id.x * TILE_N;
    int Ntotal = hidden * 2;

    /* Select weights once per tile when the tile does not cross the W1/W3 boundary. */
    int n_offset = 0;
    device const char *W = W1;
    float scalar_scale = scale_1;
    int mixed_tile = 0;
    int n_end = n_base + TILE_N - 1;
    if (n_end < hidden) {
        /* W1 */
    } else if (n_base >= hidden) {
        /* W3 */
        n_offset = hidden;
        W = W3;
        scalar_scale = scale_3;
    } else {
        mixed_tile = 1;
    }

    int local_m = (int)local_id.y;
    int local_n = (int)local_id.x;

    int global_m = m_base + local_m;
    int global_n = n_base + local_n;

    threadgroup float A_tile[TILE_M][TILE_K];
    threadgroup float B_tile[TILE_K][TILE_N];

    float acc = 0.0f;

    for (int k0 = 0; k0 < Kdim; k0 += TILE_K) {
        int a_k = k0 + local_n;
        if (local_n < TILE_K && global_m < M && a_k < Kdim) {
            A_tile[local_m][local_n] = A[global_m * Kdim + a_k];
        } else if (local_n < TILE_K) {
            A_tile[local_m][local_n] = 0.0f;
        }

        /* Cooperative load using char4 vectorized loads (non-mixed case) */
        int flat_id = local_m * TILE_N + local_n;
        int load_n_idx = flat_id / 8;      /* 0..31: which n row */
        int k_chunk = flat_id % 8;         /* 0..7: which char4 chunk */
        int load_k_base = k_chunk * 4;     /* 0, 4, 8, ..., 28 */

        int b_n = n_base + load_n_idx;
        int b_k_base = k0 + load_k_base;

        if (!mixed_tile && b_n < Ntotal && b_k_base + 3 < Kdim) {
            /* Fast path: non-mixed tile, load 4 consecutive k-values as char4 */
            int row = b_n - n_offset;
            device const char4 *ptr = (device const char4 *)(W + row * Kdim + b_k_base);
            char4 data = *ptr;
            B_tile[load_k_base + 0][load_n_idx] = float(data.x);
            B_tile[load_k_base + 1][load_n_idx] = float(data.y);
            B_tile[load_k_base + 2][load_n_idx] = float(data.z);
            B_tile[load_k_base + 3][load_n_idx] = float(data.w);
        } else {
            /* Slow path: boundary or mixed tile */
            for (int dk = 0; dk < 4; dk++) {
                int b_k = b_k_base + dk;
                if (b_k < Kdim && b_n < Ntotal) {
                    char w_int8 = 0;
                    if (!mixed_tile) {
                        int row = b_n - n_offset;
                        w_int8 = W[row * Kdim + b_k];
                    } else {
                        device const char *Wm = (b_n < hidden) ? W1 : W3;
                        int row = (b_n < hidden) ? b_n : (b_n - hidden);
                        w_int8 = Wm[row * Kdim + b_k];
                    }
                    B_tile[load_k_base + dk][load_n_idx] = float(w_int8);
                } else {
                    B_tile[load_k_base + dk][load_n_idx] = 0.0f;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (global_m < M && global_n < Ntotal) {
            for (int kk = 0; kk < TILE_K; kk++) {
                acc += A_tile[local_m][kk] * B_tile[kk][local_n];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (global_m < M && global_n < Ntotal) {
        if (!mixed_tile) {
            int col = global_n - n_offset;
            C[global_m * Ntotal + global_n] = acc * scalar_scale;
        } else {
            if (global_n < hidden) {
                C[global_m * Ntotal + global_n] = acc * scale_1;
            } else {
                int n3 = global_n - hidden;
                C[global_m * Ntotal + global_n] = acc * scale_3;
            }
        }
    }
}
