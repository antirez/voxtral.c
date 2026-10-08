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

kernel void kv_cache_copy_f16(
    device half *cache [[buffer(0)]],
    device const float *data [[buffer(1)]],
    constant int &elem_offset [[buffer(2)]],
    constant int &kv_dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((int)gid < kv_dim) {
        cache[elem_offset + gid] = half(data[gid]);
    }
}

/* ========================================================================
 * Single-token decoder attention (seq_q=1).
 * One threadgroup per query head, 32 threads (one SIMD group).
 * Each lane handles 4 dims (head_dim=128), avoiding cross-SIMD barriers.
 * K/V read from the KV cache buffer at a per-layer offset.
 * Uses online softmax (single pass) with SIMD reductions.
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
    uint tid [[thread_position_in_threadgroup]]
) {
    if ((int)head_idx >= n_heads) return;

    int gqa_ratio = n_heads / n_kv_heads;
    int kv_head = (int)head_idx / gqa_ratio;

    device const float *q_h = Q + head_idx * head_dim;
    device float *out_h = out + head_idx * head_dim;

    int valid_end = min(q_pos, seq_k - 1);
    int valid_start = (window_size > 0) ? max(0, q_pos - window_size + 1) : 0;

    /* Decoder uses head_dim=128. One lane computes 4 dimensions. */
    int d0 = (int)tid;
    int d1 = d0 + 32;
    int d2 = d1 + 32;
    int d3 = d2 + 32;

    float q0 = (d0 < head_dim) ? q_h[d0] : 0.0f;
    float q1 = (d1 < head_dim) ? q_h[d1] : 0.0f;
    float q2 = (d2 < head_dim) ? q_h[d2] : 0.0f;
    float q3 = (d3 < head_dim) ? q_h[d3] : 0.0f;

    /* Online softmax: single pass over keys */
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    for (int j = valid_start; j <= valid_end; j++) {
        device const float *k_j = K_cache + j * kv_dim + kv_head * head_dim;

        /* One SIMD-group dot product (32 lanes x 4 dims/lane). */
        float partial = q0 * k_j[d0] + q1 * k_j[d1] + q2 * k_j[d2] + q3 * k_j[d3];
        float score = simd_sum(partial) * scale;

        /* Online softmax update */
        float old_max = running_max;
        running_max = fmax(running_max, score);
        float correction = exp(old_max - running_max);
        float weight = exp(score - running_max);
        running_sum = running_sum * correction + weight;
        acc0 *= correction;
        acc1 *= correction;
        acc2 *= correction;
        acc3 *= correction;

        /* Accumulate weighted V */
        device const float *v_j = V_cache + j * kv_dim + kv_head * head_dim;
        acc0 += weight * v_j[d0];
        acc1 += weight * v_j[d1];
        acc2 += weight * v_j[d2];
        acc3 += weight * v_j[d3];
    }

    /* Normalize and write output */
    float inv = 1.0f / (running_sum + 1e-10f);
    out_h[d0] = acc0 * inv;
    out_h[d1] = acc1 * inv;
    out_h[d2] = acc2 * inv;
    out_h[d3] = acc3 * inv;
}

kernel void decoder_attention_f16(
    device const float *Q [[buffer(0)]],
    device const half *K_cache [[buffer(1)]],
    device const half *V_cache [[buffer(2)]],
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
    uint tid [[thread_position_in_threadgroup]]
) {
    if ((int)head_idx >= n_heads) return;

    int gqa_ratio = n_heads / n_kv_heads;
    int kv_head = (int)head_idx / gqa_ratio;

    device const float *q_h = Q + head_idx * head_dim;
    device float *out_h = out + head_idx * head_dim;

    int valid_end = min(q_pos, seq_k - 1);
    int valid_start = (window_size > 0) ? max(0, q_pos - window_size + 1) : 0;

    int d0 = (int)tid;
    int d1 = d0 + 32;
    int d2 = d1 + 32;
    int d3 = d2 + 32;

    float q0 = (d0 < head_dim) ? q_h[d0] : 0.0f;
    float q1 = (d1 < head_dim) ? q_h[d1] : 0.0f;
    float q2 = (d2 < head_dim) ? q_h[d2] : 0.0f;
    float q3 = (d3 < head_dim) ? q_h[d3] : 0.0f;

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    for (int j = valid_start; j <= valid_end; j++) {
        device const half *k_j = K_cache + j * kv_dim + kv_head * head_dim;
        float partial = q0 * float(k_j[d0]) + q1 * float(k_j[d1]) +
                        q2 * float(k_j[d2]) + q3 * float(k_j[d3]);
        float score = simd_sum(partial) * scale;

        float old_max = running_max;
        running_max = fmax(running_max, score);
        float correction = exp(old_max - running_max);
        float weight = exp(score - running_max);
        running_sum = running_sum * correction + weight;
        acc0 *= correction;
        acc1 *= correction;
        acc2 *= correction;
        acc3 *= correction;

        device const half *v_j = V_cache + j * kv_dim + kv_head * head_dim;
        acc0 += weight * float(v_j[d0]);
        acc1 += weight * float(v_j[d1]);
        acc2 += weight * float(v_j[d2]);
        acc3 += weight * float(v_j[d3]);
    }

    float inv = 1.0f / (running_sum + 1e-10f);
    out_h[d0] = acc0 * inv;
    out_h[d1] = acc1 * inv;
    out_h[d2] = acc2 * inv;
    out_h[d3] = acc3 * inv;
}

/* ========================================================================
 * Split-KV single-token attention (flash-decoding).
 *
 * decoder_attention_f16 gives one threadgroup per query head, so a decode step
 * with a long KV cache runs 32 threadgroups that each walk the whole cache
 * serially, limiting GPU parallelism on long audio.
 *
 * These two kernels split the key range into n_splits chunks. The first pass
 * runs one threadgroup per (head, chunk) and emits a partial softmax for its
 * chunk: the running max, the running sum, and the unnormalised value
 * accumulator. The second pass merges the chunks per head with the usual
 * online-softmax rescaling. Results are identical to the serial kernel up to
 * floating point association.
 * ======================================================================== */

kernel void decoder_attention_f16_split(
    device const float *Q [[buffer(0)]],
    device const half *K_cache [[buffer(1)]],
    device const half *V_cache [[buffer(2)]],
    device float *part_acc [[buffer(3)]],   /* [n_heads][n_splits][head_dim] */
    device float *part_max [[buffer(4)]],   /* [n_heads][n_splits] */
    device float *part_sum [[buffer(5)]],   /* [n_heads][n_splits] */
    constant int &n_heads [[buffer(6)]],
    constant int &n_kv_heads [[buffer(7)]],
    constant int &head_dim [[buffer(8)]],
    constant int &kv_dim [[buffer(9)]],
    constant int &seq_k [[buffer(10)]],
    constant float &scale [[buffer(11)]],
    constant int &window_size [[buffer(12)]],
    constant int &q_pos [[buffer(13)]],
    constant int &n_splits [[buffer(14)]],
    uint group_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    int head_idx = (int)group_idx / n_splits;
    int split = (int)group_idx % n_splits;
    if (head_idx >= n_heads) return;

    int gqa_ratio = n_heads / n_kv_heads;
    int kv_head = head_idx / gqa_ratio;

    device const float *q_h = Q + head_idx * head_dim;

    int valid_end = min(q_pos, seq_k - 1);
    int valid_start = (window_size > 0) ? max(0, q_pos - window_size + 1) : 0;

    /* Even split of the valid range across chunks. */
    int total = valid_end - valid_start + 1;
    int per = (total + n_splits - 1) / n_splits;
    int j0 = valid_start + split * per;
    int j1 = min(j0 + per - 1, valid_end);

    int out_base = (head_idx * n_splits + split) * head_dim;

    if (total <= 0 || j0 > valid_end) {
        if (tid == 0) {
            part_max[head_idx * n_splits + split] = -INFINITY;
            part_sum[head_idx * n_splits + split] = 0.0f;
        }
        for (int d = (int)tid; d < head_dim; d += 32) part_acc[out_base + d] = 0.0f;
        return;
    }

    int d0 = (int)tid;
    int d1 = d0 + 32;
    int d2 = d1 + 32;
    int d3 = d2 + 32;

    float q0 = (d0 < head_dim) ? q_h[d0] : 0.0f;
    float q1 = (d1 < head_dim) ? q_h[d1] : 0.0f;
    float q2 = (d2 < head_dim) ? q_h[d2] : 0.0f;
    float q3 = (d3 < head_dim) ? q_h[d3] : 0.0f;

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    for (int j = j0; j <= j1; j++) {
        device const half *k_j = K_cache + j * kv_dim + kv_head * head_dim;
        float partial = q0 * float(k_j[d0]) + q1 * float(k_j[d1]) +
                        q2 * float(k_j[d2]) + q3 * float(k_j[d3]);
        float score = simd_sum(partial) * scale;

        float old_max = running_max;
        running_max = fmax(running_max, score);
        float correction = exp(old_max - running_max);
        float weight = exp(score - running_max);
        running_sum = running_sum * correction + weight;
        acc0 *= correction; acc1 *= correction;
        acc2 *= correction; acc3 *= correction;

        device const half *v_j = V_cache + j * kv_dim + kv_head * head_dim;
        acc0 += weight * float(v_j[d0]);
        acc1 += weight * float(v_j[d1]);
        acc2 += weight * float(v_j[d2]);
        acc3 += weight * float(v_j[d3]);
    }

    if (tid == 0) {
        part_max[head_idx * n_splits + split] = running_max;
        part_sum[head_idx * n_splits + split] = running_sum;
    }
    if (d0 < head_dim) part_acc[out_base + d0] = acc0;
    if (d1 < head_dim) part_acc[out_base + d1] = acc1;
    if (d2 < head_dim) part_acc[out_base + d2] = acc2;
    if (d3 < head_dim) part_acc[out_base + d3] = acc3;
}

kernel void decoder_attention_combine(
    device const float *part_acc [[buffer(0)]],
    device const float *part_max [[buffer(1)]],
    device const float *part_sum [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant int &n_heads [[buffer(4)]],
    constant int &head_dim [[buffer(5)]],
    constant int &n_splits [[buffer(6)]],
    uint head_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if ((int)head_idx >= n_heads) return;

    float gmax = -INFINITY;
    for (int s = 0; s < n_splits; s++)
        gmax = fmax(gmax, part_max[head_idx * n_splits + s]);

    float gsum = 0.0f;
    for (int s = 0; s < n_splits; s++) {
        float m = part_max[head_idx * n_splits + s];
        if (isinf(m) && m < 0.0f) continue;
        gsum += part_sum[head_idx * n_splits + s] * exp(m - gmax);
    }
    float inv = 1.0f / (gsum + 1e-10f);

    for (int d = (int)tid; d < head_dim; d += 32) {
        float acc = 0.0f;
        for (int s = 0; s < n_splits; s++) {
            float m = part_max[head_idx * n_splits + s];
            if (isinf(m) && m < 0.0f) continue;
            acc += part_acc[(head_idx * n_splits + s) * head_dim + d] * exp(m - gmax);
        }
        out[head_idx * head_dim + d] = acc * inv;
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

kernel void encoder_attention_kv_f16(
    device const float *Q [[buffer(0)]],
    device const half *K [[buffer(1)]],
    device const half *V [[buffer(2)]],
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

    float q_vals[ATTN_BQ];
    for (int b = 0; b < ATTN_BQ; b++) {
        int qi = qi_start + b;
        q_vals[b] = (qi < seq_q && (int)tid < head_dim)
            ? Q[(long)qi * stride_q + h * head_dim + tid] : 0.0f;
    }

    float rmax[ATTN_BQ], rsum[ATTN_BQ], acc[ATTN_BQ];
    for (int b = 0; b < ATTN_BQ; b++) {
        rmax[b] = -INFINITY;
        rsum[b] = 0.0f;
        acc[b] = 0.0f;
    }

    threadgroup float tg_simd[4 * ATTN_BQ];
    threadgroup float tg_scores[ATTN_BQ];

    int last_qi = min(qi_start + ATTN_BQ - 1, seq_q - 1);
    int first_pos = q_offset + qi_start;
    int last_pos = q_offset + last_qi;
    int loop_start = (window_size > 0) ? max(0, first_pos - window_size + 1) : 0;
    int loop_end = min(last_pos, seq_k - 1);

    for (int j = loop_start; j <= loop_end; j++) {
        device const half *k_j = K + (long)j * stride_kv + kv_h * head_dim;
        float k_val = (int)tid < head_dim ? float(k_j[tid]) : 0.0f;

        for (int b = 0; b < ATTN_BQ; b++) {
            float simd_dot = simd_sum(q_vals[b] * k_val);
            if (simd_lid == 0) tg_simd[simd_gid * ATTN_BQ + b] = simd_dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if ((int)tid < ATTN_BQ) {
            float sum = 0;
            for (int g = 0; g < n_simd_groups; g++)
                sum += tg_simd[g * ATTN_BQ + (int)tid];
            tg_scores[(int)tid] = sum * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device const half *v_j = V + (long)j * stride_kv + kv_h * head_dim;
        float v_val = (int)tid < head_dim ? float(v_j[tid]) : 0.0f;

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

kernel void encoder_attention_qstrided(
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
    constant int &q_stride [[buffer(12)]],
    constant int &q_col_offset [[buffer(13)]],
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
    int stride_q_out = n_heads * head_dim;
    int stride_kv = n_kv_heads * head_dim;
    int n_simd_groups = (int)tg_size / 32;

    float q_vals[ATTN_BQ];
    for (int b = 0; b < ATTN_BQ; b++) {
        int qi = qi_start + b;
        q_vals[b] = (qi < seq_q && (int)tid < head_dim)
            ? Q[(long)qi * q_stride + q_col_offset + h * head_dim + tid] : 0.0f;
    }

    float rmax[ATTN_BQ], rsum[ATTN_BQ], acc[ATTN_BQ];
    for (int b = 0; b < ATTN_BQ; b++) {
        rmax[b] = -INFINITY;
        rsum[b] = 0.0f;
        acc[b] = 0.0f;
    }

    threadgroup float tg_simd[4 * ATTN_BQ];
    threadgroup float tg_scores[ATTN_BQ];

    int last_qi = min(qi_start + ATTN_BQ - 1, seq_q - 1);
    int first_pos = q_offset + qi_start;
    int last_pos = q_offset + last_qi;
    int loop_start = (window_size > 0) ? max(0, first_pos - window_size + 1) : 0;
    int loop_end = min(last_pos, seq_k - 1);

    for (int j = loop_start; j <= loop_end; j++) {
        device const float *k_j = K + (long)j * stride_kv + kv_h * head_dim;
        float k_val = (int)tid < head_dim ? k_j[tid] : 0.0f;

        for (int b = 0; b < ATTN_BQ; b++) {
            float simd_dot = simd_sum(q_vals[b] * k_val);
            if (simd_lid == 0) tg_simd[simd_gid * ATTN_BQ + b] = simd_dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if ((int)tid < ATTN_BQ) {
            float sum = 0;
            for (int g = 0; g < n_simd_groups; g++)
                sum += tg_simd[g * ATTN_BQ + (int)tid];
            tg_scores[(int)tid] = sum * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device const float *v_j = V + (long)j * stride_kv + kv_h * head_dim;
        float v_val = (int)tid < head_dim ? v_j[tid] : 0.0f;

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

    if ((int)tid < head_dim) {
        for (int b = 0; b < ATTN_BQ; b++) {
            int qi = qi_start + b;
            if (qi < seq_q) {
                device float *out_row = out + (long)qi * stride_q_out + h * head_dim;
                out_row[tid] = acc[b] / (rsum[b] + 1e-10f);
            }
        }
    }
}

kernel void encoder_attention_kv_f16_qstrided(
    device const float *Q [[buffer(0)]],
    device const half *K [[buffer(1)]],
    device const half *V [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant int &n_heads [[buffer(4)]],
    constant int &n_kv_heads [[buffer(5)]],
    constant int &head_dim [[buffer(6)]],
    constant int &seq_q [[buffer(7)]],
    constant int &seq_k [[buffer(8)]],
    constant float &scale [[buffer(9)]],
    constant int &window_size [[buffer(10)]],
    constant int &q_offset [[buffer(11)]],
    constant int &q_stride [[buffer(12)]],
    constant int &q_col_offset [[buffer(13)]],
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
    int stride_q_out = n_heads * head_dim;
    int stride_kv = n_kv_heads * head_dim;
    int n_simd_groups = (int)tg_size / 32;

    float q_vals[ATTN_BQ];
    for (int b = 0; b < ATTN_BQ; b++) {
        int qi = qi_start + b;
        q_vals[b] = (qi < seq_q && (int)tid < head_dim)
            ? Q[(long)qi * q_stride + q_col_offset + h * head_dim + tid] : 0.0f;
    }

    float rmax[ATTN_BQ], rsum[ATTN_BQ], acc[ATTN_BQ];
    for (int b = 0; b < ATTN_BQ; b++) {
        rmax[b] = -INFINITY;
        rsum[b] = 0.0f;
        acc[b] = 0.0f;
    }

    threadgroup float tg_simd[4 * ATTN_BQ];
    threadgroup float tg_scores[ATTN_BQ];

    int last_qi = min(qi_start + ATTN_BQ - 1, seq_q - 1);
    int first_pos = q_offset + qi_start;
    int last_pos = q_offset + last_qi;
    int loop_start = (window_size > 0) ? max(0, first_pos - window_size + 1) : 0;
    int loop_end = min(last_pos, seq_k - 1);

    for (int j = loop_start; j <= loop_end; j++) {
        device const half *k_j = K + (long)j * stride_kv + kv_h * head_dim;
        float k_val = (int)tid < head_dim ? float(k_j[tid]) : 0.0f;

        for (int b = 0; b < ATTN_BQ; b++) {
            float simd_dot = simd_sum(q_vals[b] * k_val);
            if (simd_lid == 0) tg_simd[simd_gid * ATTN_BQ + b] = simd_dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if ((int)tid < ATTN_BQ) {
            float sum = 0;
            for (int g = 0; g < n_simd_groups; g++)
                sum += tg_simd[g * ATTN_BQ + (int)tid];
            tg_scores[(int)tid] = sum * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device const half *v_j = V + (long)j * stride_kv + kv_h * head_dim;
        float v_val = (int)tid < head_dim ? float(v_j[tid]) : 0.0f;

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

    if ((int)tid < head_dim) {
        for (int b = 0; b < ATTN_BQ; b++) {
            int qi = qi_start + b;
            if (qi < seq_q) {
                device float *out_row = out + (long)qi * stride_q_out + h * head_dim;
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

kernel void bias_add_strided(
    device float *data [[buffer(0)]],
    device const float *bias [[buffer(1)]],
    constant int &row_stride [[buffer(2)]],
    constant int &chunk_cols [[buffer(3)]],
    constant int &col_offset [[buffer(4)]],
    constant int &total [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((int)gid >= total) return;
    int row = (int)gid / chunk_cols;
    int col = (int)gid % chunk_cols;
    data[row * row_stride + col_offset + col] += bias[col];
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

kernel void batched_rope_apply_strided(
    device float *data [[buffer(0)]],
    device const float *freqs [[buffer(1)]],
    constant int &n_heads [[buffer(2)]],
    constant int &head_dim [[buffer(3)]],
    constant int &seq_len [[buffer(4)]],
    constant int &row_stride [[buffer(5)]],
    constant int &col_offset [[buffer(6)]],
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

    int base = pos * row_stride + col_offset + head * head_dim;
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

kernel void batched_kv_cache_copy_f16(
    device half *cache [[buffer(0)]],
    device const float *data [[buffer(1)]],
    constant int &cache_offset [[buffer(2)]],
    constant int &total [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((int)gid < total) {
        cache[cache_offset + gid] = half(data[gid]);
    }
}

kernel void batched_kv_cache_copy_strided(
    device float *cache [[buffer(0)]],
    device const float *data [[buffer(1)]],
    constant int &cache_offset [[buffer(2)]],
    constant int &src_stride [[buffer(3)]],
    constant int &src_col_offset [[buffer(4)]],
    constant int &chunk_cols [[buffer(5)]],
    constant int &total [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((int)gid >= total) return;
    int row = (int)gid / chunk_cols;
    int col = (int)gid % chunk_cols;
    cache[cache_offset + gid] = data[row * src_stride + src_col_offset + col];
}

kernel void batched_kv_cache_copy_strided_f16(
    device half *cache [[buffer(0)]],
    device const float *data [[buffer(1)]],
    constant int &cache_offset [[buffer(2)]],
    constant int &src_stride [[buffer(3)]],
    constant int &src_col_offset [[buffer(4)]],
    constant int &chunk_cols [[buffer(5)]],
    constant int &total [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    if ((int)gid >= total) return;
    int row = (int)gid / chunk_cols;
    int col = (int)gid % chunk_cols;
    cache[cache_offset + gid] = half(data[row * src_stride + src_col_offset + col]);
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
 * Decoder FFN gate kernel (M=1):
 * out[h] = silu(dot(x, w1[h])) * dot(x, w3[h])
 * w1 and w3 are stored merged in rows [0:hidden) and [hidden:2*hidden).
 * ======================================================================== */

kernel void decoder_ffn_gate(
    device const float *x [[buffer(0)]],
    device const half *w_merged [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant int &dim [[buffer(3)]],
    constant int &hidden [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if ((int)row >= hidden) return;

    device const half *w1 = w_merged + (long)row * dim;
    device const half *w3 = w_merged + (long)(row + hidden) * dim;

    float sum1 = 0.0f;
    float sum3 = 0.0f;
    int i = (int)tid * 4;
    for (; i + 3 < dim; i += (int)tg_size * 4) {
        float4 xv = float4(x[i], x[i + 1], x[i + 2], x[i + 3]);
        half4 w1v = *((device const half4 *)(w1 + i));
        half4 w3v = *((device const half4 *)(w3 + i));
        sum1 += dot(xv, float4(w1v));
        sum3 += dot(xv, float4(w3v));
    }
    for (; i < dim; i += (int)tg_size) {
        float xv = x[i];
        sum1 += xv * float(w1[i]);
        sum3 += xv * float(w3[i]);
    }

    float s1 = simd_sum(sum1);
    float s3 = simd_sum(sum3);

    const int max_simd_groups = 8; /* 256 threads / 32 lanes */
    threadgroup float tg_s1[max_simd_groups];
    threadgroup float tg_s3[max_simd_groups];
    if (simd_lid == 0) {
        tg_s1[simd_gid] = s1;
        tg_s3[simd_gid] = s3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        int n_groups = ((int)tg_size + 31) / 32;
        float d1 = 0.0f;
        float d3 = 0.0f;
        for (int g = 0; g < n_groups; g++) {
            d1 += tg_s1[g];
            d3 += tg_s3[g];
        }
        float gate = d1 / (1.0f + exp(-d1));
        out[row] = gate * d3;
    }
}

/* ========================================================================
 * Decoder FFN W2 + residual add kernel (M=1):
 * x[d] += dot(gate[0:hidden], w2[d, 0:hidden])
 * ======================================================================== */

kernel void decoder_w2_residual(
    device float *x [[buffer(0)]],
    device const float *gate [[buffer(1)]],
    device const half *w2 [[buffer(2)]],
    constant int &hidden [[buffer(3)]],
    constant int &dim [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if ((int)row >= dim) return;

    device const half *w = w2 + (long)row * hidden;

    float sum = 0.0f;
    int i = (int)tid * 4;
    for (; i + 3 < hidden; i += (int)tg_size * 4) {
        float4 gv = float4(gate[i], gate[i + 1], gate[i + 2], gate[i + 3]);
        half4 wv = *((device const half4 *)(w + i));
        sum += dot(gv, float4(wv));
    }
    for (; i < hidden; i += (int)tg_size) {
        sum += gate[i] * float(w[i]);
    }

    float s = simd_sum(sum);

    const int max_simd_groups = 8; /* 256 threads / 32 lanes */
    threadgroup float tg_s[max_simd_groups];
    if (simd_lid == 0) tg_s[simd_gid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        int n_groups = ((int)tg_size + 31) / 32;
        float dot = 0.0f;
        for (int g = 0; g < n_groups; g++) dot += tg_s[g];
        x[row] += dot;
    }
}

/* ========================================================================
 * Decoder WO + residual add kernel (M=1):
 * x[d] += dot(attn[0:q_dim], wo[d, 0:q_dim])
 * ======================================================================== */

kernel void decoder_wo_residual(
    device float *x [[buffer(0)]],
    device const float *attn [[buffer(1)]],
    device const half *wo [[buffer(2)]],
    constant int &q_dim [[buffer(3)]],
    constant int &dim [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    if ((int)row >= dim) return;

    device const half *w = wo + (long)row * q_dim;

    float sum = 0.0f;
    int i = (int)tid * 4;
    for (; i + 3 < q_dim; i += (int)tg_size * 4) {
        float4 av = float4(attn[i], attn[i + 1], attn[i + 2], attn[i + 3]);
        half4 wv = *((device const half4 *)(w + i));
        sum += dot(av, float4(wv));
    }
    for (; i < q_dim; i += (int)tg_size) {
        sum += attn[i] * float(w[i]);
    }

    float s = simd_sum(sum);

    const int max_simd_groups = 8; /* 256 threads / 32 lanes */
    threadgroup float tg_s[max_simd_groups];
    if (simd_lid == 0) tg_s[simd_gid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        int n_groups = ((int)tg_size + 31) / 32;
        float dot = 0.0f;
        for (int g = 0; g < n_groups; g++) dot += tg_s[g];
        x[row] += dot;
    }
}
