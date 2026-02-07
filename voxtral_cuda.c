#include "voxtral_cuda.h"

#ifdef USE_CUDA

#include <cuda.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "voxtral_cuda_kernels_cubin.h"
#include "voxtral.h"
#include "voxtral_kernels.h"

static cublasHandle_t g_handle;
static cublasLtHandle_t g_lt_handle;
static CUcontext g_ctx;
static CUstream g_stream;
static CUdevice g_dev;
static int g_init = 0;
static int g_available = 0;
static char g_device_name[256] = "unavailable";

static CUmodule g_mod = 0;
static CUfunction g_fn_attn = 0;
static CUfunction g_fn_attn_fp16 = 0;
static CUfunction g_fn_attn_f32 = 0;
static CUfunction g_fn_attn_dyn_fp16 = 0;
static CUfunction g_fn_attn_dyn_f32 = 0;
static CUfunction g_fn_attn_fp16_v2 = 0;
static CUfunction g_fn_attn_f32_v2 = 0;
static CUfunction g_fn_attn_dyn_fp16_v2 = 0;
static CUfunction g_fn_attn_dyn_f32_v2 = 0;
static CUfunction g_fn_kv_append_dyn_fp16 = 0;
static CUfunction g_fn_kv_append_dyn_f32 = 0;
static CUfunction g_fn_causal_attn = 0;
static CUfunction g_fn_pack_heads = 0;
static CUfunction g_fn_unpack_heads = 0;
static CUfunction g_fn_expand_kv_heads = 0;
static CUfunction g_fn_softmax = 0;
static CUfunction g_fn_rms_norm = 0;
static CUfunction g_fn_rms_norm_to_bf16 = 0;
static CUfunction g_fn_add_bias = 0;
static CUfunction g_fn_add_inplace = 0;
static CUfunction g_fn_mul_inplace = 0;
static CUfunction g_fn_mul_1p_inplace = 0;
static CUfunction g_fn_mul_1p_rows_inplace = 0;
static CUfunction g_fn_silu = 0;
static CUfunction g_fn_gelu = 0;
static CUfunction g_fn_f32_to_bf16 = 0;
static CUfunction g_fn_f32_to_f16 = 0;
static CUfunction g_fn_apply_rope = 0;
static CUfunction g_fn_downsample4 = 0;
static CUfunction g_fn_argmax = 0;

static CUdeviceptr g_dA = 0;
static CUdeviceptr g_dB = 0;
static CUdeviceptr g_dC = 0;
static CUdeviceptr g_dC2 = 0;
static CUdeviceptr g_dA_bf16 = 0;
static size_t g_cap_a = 0;
static size_t g_cap_b = 0;
static size_t g_cap_c = 0;
static size_t g_cap_c2 = 0;
static size_t g_cap_a_bf16 = 0;

/* cuBLASLt workspace + algo cache (used primarily for M=1 matmuls). */
static CUdeviceptr g_lt_workspace = 0;
static size_t g_lt_workspace_cap = 0;

typedef struct {
    int M, K, N;
    cublasLtMatmulAlgo_t algo;
    cublasLtMatmulDesc_t op;
    cublasLtMatrixLayout_t a;
    cublasLtMatrixLayout_t b;
    cublasLtMatrixLayout_t c;
    size_t workspace_bytes;
    int valid;
} lt_algo_entry_t;

static lt_algo_entry_t g_lt_algos[32];
static int g_lt_algos_len = 0;

/* Decoder attention: device-side KV cache and work buffers */
static CUdeviceptr g_k_cache = 0;
static CUdeviceptr g_v_cache = 0;
static int g_kv_max_seq = 0;
static int g_kv_dim = 0;
static size_t g_kv_elem_bytes = 0;

static int kv_cache_use_fp16(void) {
    /* Default on: FP16 KV cache cuts VRAM in half and materially reduces
     * weight-cache thrash on 12 GiB cards (WSL2 tends to have < 12 GiB free). */
    static int cached = -1;
    if (cached != -1) return cached;
    const char *env = getenv("VOX_CUDA_KV_FP16");
    if (!env || !env[0]) { cached = 1; return cached; }
    cached = (env[0] != '0');
    return cached;
}

static int attn_v2_enabled(void) {
    /* Opt-in: the v2 attention kernels use a different per-thread layout with
     * vectorized loads/stores. Keep it behind an env gate until it has broader
     * coverage across cards/drivers. */
    static int cached = -1;
    if (cached != -1) return cached;
    const char *disable = getenv("VOX_DISABLE_CUDA_ATTN_V2");
    if (disable && disable[0] && disable[0] != '0') { cached = 0; return cached; }
    const char *env = getenv("VOX_CUDA_ATTN_V2");
    cached = (env && env[0] && env[0] != '0');
    return cached;
}

static uint16_t f32_to_f16bits(float x) {
#if defined(__FLT16_MANT_DIG__)
    _Float16 h = (_Float16)x;
    uint16_t bits;
    memcpy(&bits, &h, sizeof(bits));
    return bits;
#else
    /* Fallback (should be rare in our supported toolchains). */
    union { float f; uint32_t u; } v;
    v.f = x;
    uint32_t sign = (v.u >> 16) & 0x8000u;
    uint32_t exp = (v.u >> 23) & 0xFFu;
    uint32_t mant = v.u & 0x7FFFFFu;

    if (exp == 0xFFu) {
        /* Inf/NaN */
        if (mant) return (uint16_t)(sign | 0x7E00u);
        return (uint16_t)(sign | 0x7C00u);
    }

    int32_t e = (int32_t)exp - 127;
    if (e > 15) {
        return (uint16_t)(sign | 0x7C00u); /* overflow -> inf */
    } else if (e >= -14) {
        /* Normal half */
        uint32_t he = (uint32_t)(e + 15);
        uint32_t hm = mant >> 13;
        uint32_t round = mant & 0x1FFFu;
        /* round-to-nearest-even */
        if (round > 0x1000u || (round == 0x1000u && (hm & 1u))) {
            hm++;
            if (hm == 0x400u) { hm = 0; he++; }
            if (he >= 31u) return (uint16_t)(sign | 0x7C00u);
        }
        return (uint16_t)(sign | (he << 10) | hm);
    } else if (e >= -24) {
        /* Subnormal half */
        uint32_t m = mant | 0x800000u;
        uint32_t shift = (uint32_t)(-14 - e);
        uint32_t hm = m >> (13u + shift);
        uint32_t round_mask = (1u << (13u + shift)) - 1u;
        uint32_t round = m & round_mask;
        uint32_t halfway = 1u << (12u + shift);
        if (round > halfway || (round == halfway && (hm & 1u))) hm++;
        return (uint16_t)(sign | hm);
    } else {
        return (uint16_t)sign; /* underflow -> signed zero */
    }
#endif
}

static CUdeviceptr g_dQ = 0;
static CUdeviceptr g_dAttn = 0;
static size_t g_cap_q = 0;
static size_t g_cap_attn = 0;

static CUdeviceptr g_dQ_attn = 0;
static CUdeviceptr g_dK_attn = 0;
static CUdeviceptr g_dV_attn = 0;
static CUdeviceptr g_dOut_attn = 0;
static size_t g_cap_q_attn = 0;
static size_t g_cap_k_attn = 0;
static size_t g_cap_v_attn = 0;
static size_t g_cap_out_attn = 0;

/* Large-attention (encoder) work buffers */
static CUdeviceptr g_dQp_attn = 0;
static CUdeviceptr g_dKp_attn = 0;
static CUdeviceptr g_dVp_attn = 0;
static CUdeviceptr g_dKfull_attn = 0;
static CUdeviceptr g_dVfull_attn = 0;
static CUdeviceptr g_dScores_attn = 0;
static CUdeviceptr g_dOutPacked_attn = 0;
static size_t g_cap_qp_attn = 0;
static size_t g_cap_kp_attn = 0;
static size_t g_cap_vp_attn = 0;
static size_t g_cap_kfull_attn = 0;
static size_t g_cap_vfull_attn = 0;
static size_t g_cap_scores_attn = 0;
static size_t g_cap_outpacked_attn = 0;

/* Full encoder/adapter forward buffers (keep intermediates on-device). */
static CUdeviceptr g_enc_x = 0;
static CUdeviceptr g_enc_x_norm = 0;
static CUdeviceptr g_enc_x_bf16 = 0;
static CUdeviceptr g_enc_q = 0;
static CUdeviceptr g_enc_k = 0;
static CUdeviceptr g_enc_v = 0;
static CUdeviceptr g_enc_attn = 0;
static CUdeviceptr g_enc_attn_bf16 = 0;
static CUdeviceptr g_enc_proj = 0;
static CUdeviceptr g_enc_gate = 0;
static CUdeviceptr g_enc_up = 0;
static CUdeviceptr g_enc_gate_bf16 = 0;
static CUdeviceptr g_enc_ffn = 0;
static CUdeviceptr g_enc_rope_freqs = 0;
static CUdeviceptr g_enc_ds = 0;
static CUdeviceptr g_enc_ds_bf16 = 0;
static CUdeviceptr g_enc_mid = 0;
static CUdeviceptr g_enc_mid_bf16 = 0;
static CUdeviceptr g_enc_adapter = 0;

static size_t g_cap_enc_x = 0;
static size_t g_cap_enc_x_norm = 0;
static size_t g_cap_enc_x_bf16 = 0;
static size_t g_cap_enc_q = 0;
static size_t g_cap_enc_k = 0;
static size_t g_cap_enc_v = 0;
static size_t g_cap_enc_attn = 0;
static size_t g_cap_enc_attn_bf16 = 0;
static size_t g_cap_enc_proj = 0;
static size_t g_cap_enc_gate = 0;
static size_t g_cap_enc_up = 0;
static size_t g_cap_enc_gate_bf16 = 0;
static size_t g_cap_enc_ffn = 0;
static size_t g_cap_enc_rope = 0;
static size_t g_cap_enc_ds = 0;
static size_t g_cap_enc_ds_bf16 = 0;
static size_t g_cap_enc_mid = 0;
static size_t g_cap_enc_mid_bf16 = 0;
static size_t g_cap_enc_adapter = 0;

/* Full decoder step buffers (keep intermediates on-device). */
static CUdeviceptr g_dec_x = 0;
static CUdeviceptr g_dec_x_norm = 0;
static CUdeviceptr g_dec_x_bf16 = 0;
static CUdeviceptr g_dec_q = 0;
static CUdeviceptr g_dec_k = 0;
static CUdeviceptr g_dec_v = 0;
static CUdeviceptr g_dec_attn = 0;
static CUdeviceptr g_dec_attn_bf16 = 0;
static CUdeviceptr g_dec_proj = 0;
static CUdeviceptr g_dec_gate = 0;
static CUdeviceptr g_dec_up = 0;
static CUdeviceptr g_dec_gate_bf16 = 0;
static CUdeviceptr g_dec_ffn = 0;
static CUdeviceptr g_dec_rope_freqs = 0;
static CUdeviceptr g_dec_logits = 0;
static CUdeviceptr g_dec_best = 0;

static size_t g_cap_dec_x = 0;
static size_t g_cap_dec_x_norm = 0;
static size_t g_cap_dec_x_bf16 = 0;
static size_t g_cap_dec_q = 0;
static size_t g_cap_dec_k = 0;
static size_t g_cap_dec_v = 0;
static size_t g_cap_dec_attn = 0;
static size_t g_cap_dec_attn_bf16 = 0;
static size_t g_cap_dec_proj = 0;
static size_t g_cap_dec_gate = 0;
static size_t g_cap_dec_up = 0;
static size_t g_cap_dec_gate_bf16 = 0;
static size_t g_cap_dec_ffn = 0;
static size_t g_cap_dec_rope = 0;
static size_t g_cap_dec_logits = 0;
static size_t g_cap_dec_best = 0;

static const float **g_batched_A = NULL;
static const float **g_batched_B = NULL;
static float **g_batched_C = NULL;
static int g_batched_cap = 0;

/* CUDA Graph for decoder single-token step (opt-in via VOX_CUDA_GRAPHS=1). */
static CUgraph g_dec_graph = 0;
static CUgraphExec g_dec_graph_exec = 0;
static int g_dec_graph_ready = 0;
static CUdeviceptr g_dec_pos_dev = 0; /* device-side scalar int */
static int g_dec_graph_kv_fp16 = -1;

static int ensure_buffer(CUdeviceptr *buf, size_t *cap, size_t needed_bytes) {
    if (*cap >= needed_bytes) return 1;
    if (*buf) cuMemFree(*buf);
    *buf = 0;
    *cap = 0;
    if (cuMemAlloc(buf, needed_bytes) != CUDA_SUCCESS) return 0;
    *cap = needed_bytes;
    return 1;
}

typedef struct {
    const uint16_t *host;
    CUdeviceptr dev;
    size_t bytes;
    uint64_t use_tick;
} bf16_cache_entry_t;

typedef struct {
    const float *host;
    CUdeviceptr dev;
    size_t bytes;
} f32_cache_entry_t;

static bf16_cache_entry_t *g_bf16_cache = NULL;
static int g_bf16_cache_cap = 0;
static int g_bf16_cache_len = 0;
static size_t g_bf16_cache_bytes = 0;
static size_t g_bf16_cache_limit = 0;
static uint64_t g_bf16_tick = 1;
static uint64_t g_bf16_hits = 0;
static uint64_t g_bf16_misses = 0;
static uint64_t g_bf16_upload_bytes = 0;
static uint64_t g_bf16_evictions = 0;

static f32_cache_entry_t *g_f32_cache = NULL;
static int g_f32_cache_cap = 0;
static int g_f32_cache_len = 0;

static uint16_t f32_to_bf16bits(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    /* Round-to-nearest-even: add 0x7FFF + lsb before truncation. */
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    return (uint16_t)(u >> 16);
}

static uint16_t *g_host_a_bf16 = NULL;
static size_t g_host_a_bf16_cap = 0;
static uint16_t *host_a_bf16_get(size_t n) {
    if (n > g_host_a_bf16_cap) {
        size_t new_cap = g_host_a_bf16_cap ? g_host_a_bf16_cap : 4096;
        while (new_cap < n) new_cap *= 2;
        uint16_t *tmp = (uint16_t *)realloc(g_host_a_bf16, new_cap * sizeof(uint16_t));
        if (!tmp) return NULL;
        g_host_a_bf16 = tmp;
        g_host_a_bf16_cap = new_cap;
    }
    return g_host_a_bf16;
}

/* voxtral.c global verbosity flag */
extern int vox_verbose;

/* Filled in by Makefile for CUDA builds (e.g. -DVOX_CUDA_ARCH=sm_86). */
#ifndef VOX_CUDA_ARCH
#define VOX_CUDA_ARCH unknown
#endif
#define VOX_STR1(x) #x
#define VOX_STR(x) VOX_STR1(x)
#define VOX_CUDA_ARCH_STR VOX_STR(VOX_CUDA_ARCH)

static void bf16_cache_init_limit(void) {
    if (g_bf16_cache_limit) return;

    size_t free_b = 0, total_b = 0;
    if (cuMemGetInfo(&free_b, &total_b) == CUDA_SUCCESS && total_b > (size_t)1024 * 1024 * 1024) {
        const char *lim_env = getenv("VOX_CUDA_BF16_CACHE_GIB");
        if (lim_env && lim_env[0]) {
            double gib = strtod(lim_env, NULL);
            if (gib > 0.0) {
                g_bf16_cache_limit = (size_t)(gib * 1024.0 * 1024.0 * 1024.0);
                return;
            }
        }

        /* Use *free* VRAM (not total) and reserve enough for KV cache + work buffers.
         * WSL2 frequently reports ~12 GiB total but materially less free. */
        int max_seq = g_kv_max_seq > 0 ? g_kv_max_seq : 10240;
        int kv_dim = 1024; /* 8 kv heads * 128 head dim */
        size_t kv_elem = kv_cache_use_fp16() ? sizeof(uint16_t) : sizeof(float);
        size_t kv_bytes = (size_t)2 * (size_t)VOX_DEC_LAYERS * (size_t)max_seq * (size_t)kv_dim * kv_elem;
        size_t extra = (size_t)512 * 1024 * 1024; /* fragmentation + other buffers */
        size_t reserve = kv_bytes + extra;

        size_t cap = (free_b > reserve) ? (free_b - reserve) : (free_b * 8 / 10);
        /* Avoid trying to consume essentially all VRAM; keep a safety margin. */
        size_t max_frac = (total_b * 9) / 10; /* 90% of total */
        if (cap > max_frac) cap = max_frac;
        g_bf16_cache_limit = cap;
    } else {
        /* Fallback: 8 GiB. */
        g_bf16_cache_limit = (size_t)8 * 1024 * 1024 * 1024ULL;
    }
}

static void bf16_cache_evict_one(void) {
    if (g_bf16_cache_len <= 0) return;
    g_bf16_evictions++;
    int lru = 0;
    for (int i = 1; i < g_bf16_cache_len; i++) {
        if (g_bf16_cache[i].use_tick < g_bf16_cache[lru].use_tick) lru = i;
    }

    if (g_bf16_cache[lru].dev) cuMemFree(g_bf16_cache[lru].dev);
    if (g_bf16_cache[lru].bytes <= g_bf16_cache_bytes) g_bf16_cache_bytes -= g_bf16_cache[lru].bytes;

    g_bf16_cache[lru] = g_bf16_cache[g_bf16_cache_len - 1];
    g_bf16_cache_len--;
}

static CUdeviceptr bf16_cache_get(const uint16_t *host, size_t bytes) {
    if (!host || bytes == 0) return 0;
    if (!vox_cuda_available()) return 0;

    bf16_cache_init_limit();

    /* Fast path: linear scan is fine (O(#weights) ~ few hundred). */
    for (int i = 0; i < g_bf16_cache_len; i++) {
        if (g_bf16_cache[i].host == host && g_bf16_cache[i].bytes == bytes) {
            g_bf16_cache[i].use_tick = g_bf16_tick++;
            g_bf16_hits++;
            return g_bf16_cache[i].dev;
        }
    }
    g_bf16_misses++;

    /* Grow table. */
    if (g_bf16_cache_len == g_bf16_cache_cap) {
        int new_cap = g_bf16_cache_cap ? g_bf16_cache_cap * 2 : 256;
        bf16_cache_entry_t *tmp = (bf16_cache_entry_t *)realloc(g_bf16_cache, (size_t)new_cap * sizeof(*tmp));
        if (!tmp) return 0;
        g_bf16_cache = tmp;
        g_bf16_cache_cap = new_cap;
    }

    /* Ensure space under cache limit. */
    if (bytes <= g_bf16_cache_limit) {
        while (g_bf16_cache_bytes + bytes > g_bf16_cache_limit && g_bf16_cache_len > 0) {
            bf16_cache_evict_one();
        }
    }

    CUdeviceptr dev = 0;
    if (cuMemAlloc(&dev, bytes) != CUDA_SUCCESS) {
        /* Under memory pressure (WSL2, big KV cache), evict until alloc succeeds. */
        while (g_bf16_cache_len > 0) {
            bf16_cache_evict_one();
            if (cuMemAlloc(&dev, bytes) == CUDA_SUCCESS) break;
        }
        if (!dev) return 0;
    }
    if (cuMemcpyHtoDAsync(dev, host, bytes, g_stream) != CUDA_SUCCESS) {
        cuMemFree(dev);
        return 0;
    }
    g_bf16_upload_bytes += bytes;

    g_bf16_cache[g_bf16_cache_len++] = (bf16_cache_entry_t){
        .host = host,
        .dev = dev,
        .bytes = bytes,
        .use_tick = g_bf16_tick++,
    };
    g_bf16_cache_bytes += bytes;
    return dev;
}

static CUdeviceptr f32_cache_get(const float *host, size_t bytes) {
    if (!host || bytes == 0) return 0;
    if (!vox_cuda_available()) return 0;

    /* Fast path: tiny table. */
    for (int i = 0; i < g_f32_cache_len; i++) {
        if (g_f32_cache[i].host == host && g_f32_cache[i].bytes == bytes) {
            return g_f32_cache[i].dev;
        }
    }

    if (g_f32_cache_len == g_f32_cache_cap) {
        int new_cap = g_f32_cache_cap ? g_f32_cache_cap * 2 : 128;
        f32_cache_entry_t *tmp = (f32_cache_entry_t *)realloc(g_f32_cache, (size_t)new_cap * sizeof(*tmp));
        if (!tmp) return 0;
        g_f32_cache = tmp;
        g_f32_cache_cap = new_cap;
    }

    CUdeviceptr dev = 0;
    if (cuMemAlloc(&dev, bytes) != CUDA_SUCCESS) return 0;
    if (cuMemcpyHtoDAsync(dev, host, bytes, g_stream) != CUDA_SUCCESS) {
        cuMemFree(dev);
        return 0;
    }
    g_f32_cache[g_f32_cache_len++] = (f32_cache_entry_t){
        .host = host,
        .dev = dev,
        .bytes = bytes,
    };
    return dev;
}

static void log_cu_error(const char *what, CUresult r) {
    if (vox_verbose < 2) return;
    const char *s = NULL;
    (void)cuGetErrorString(r, &s);
    fprintf(stderr, "[cuda] %s: %d (%s)\n", what, (int)r, s ? s : "unknown");
}

static int cuda_load_kernel_module(void) {
    if (g_mod && g_fn_attn_f32 && g_fn_attn_fp16 &&
        g_fn_pack_heads && g_fn_unpack_heads && g_fn_expand_kv_heads && g_fn_softmax &&
        g_fn_rms_norm && g_fn_add_bias && g_fn_add_inplace && g_fn_mul_inplace &&
        g_fn_mul_1p_inplace && g_fn_silu && g_fn_gelu &&
        g_fn_f32_to_bf16 && g_fn_f32_to_f16 &&
        g_fn_apply_rope && g_fn_downsample4 && g_fn_argmax) {
        /* Optional fusions (best-effort). */
        if (g_mod) {
            if (!g_fn_rms_norm_to_bf16)
                (void)cuModuleGetFunction(&g_fn_rms_norm_to_bf16, g_mod, "vox_rms_norm_to_bf16");
            if (!g_fn_mul_1p_rows_inplace)
                (void)cuModuleGetFunction(&g_fn_mul_1p_rows_inplace, g_mod, "vox_mul_1p_rows_inplace_f32");
        }
        /* Optional kernels used for CUDA Graph capture (best-effort). */
        if (g_mod) {
            if (!g_fn_kv_append_dyn_fp16)
                (void)cuModuleGetFunction(&g_fn_kv_append_dyn_fp16, g_mod, "vox_kv_append_fp16_dyn");
            if (!g_fn_kv_append_dyn_f32)
                (void)cuModuleGetFunction(&g_fn_kv_append_dyn_f32, g_mod, "vox_kv_append_f32_dyn");
            if (!g_fn_attn_dyn_fp16)
                (void)cuModuleGetFunction(&g_fn_attn_dyn_fp16, g_mod, "vox_attn_q4_kv8_fp16_dyn");
            if (!g_fn_attn_dyn_f32)
                (void)cuModuleGetFunction(&g_fn_attn_dyn_f32, g_mod, "vox_attn_q4_kv8_f32_dyn");
        }
        /* Optional v2 attention kernels (best-effort). */
        if (g_mod) {
            if (!g_fn_attn_f32_v2)
                (void)cuModuleGetFunction(&g_fn_attn_f32_v2, g_mod, "vox_attn_q4_kv8_f32_v2");
            if (!g_fn_attn_fp16_v2)
                (void)cuModuleGetFunction(&g_fn_attn_fp16_v2, g_mod, "vox_attn_q4_kv8_fp16_v2");
            if (!g_fn_attn_dyn_fp16_v2)
                (void)cuModuleGetFunction(&g_fn_attn_dyn_fp16_v2, g_mod, "vox_attn_q4_kv8_fp16_dyn_v2");
            if (!g_fn_attn_dyn_f32_v2)
                (void)cuModuleGetFunction(&g_fn_attn_dyn_f32_v2, g_mod, "vox_attn_q4_kv8_f32_dyn_v2");
        }
        return 1;
    }
    if (!vox_cuda_available()) return 0;

    /* Ensure current context for module load */
    (void)cuCtxSetCurrent(g_ctx);

    /* xxd -i yields `unsigned char voxtral_cuda_kernels_cubin[]` + `_len` */
    CUresult r = cuModuleLoadDataEx(&g_mod, (const void *)voxtral_cuda_kernels_cubin, 0, NULL, NULL);
    if (r != CUDA_SUCCESS) { log_cu_error("cuModuleLoadDataEx(CUBIN)", r); return 0; }

    r = cuModuleGetFunction(&g_fn_attn_f32, g_mod, "vox_attn_q4_kv8_f32");
    if (r != CUDA_SUCCESS) { log_cu_error("cuModuleGetFunction(vox_attn_q4_kv8_f32)", r); return 0; }

    r = cuModuleGetFunction(&g_fn_attn_fp16, g_mod, "vox_attn_q4_kv8_fp16");
    if (r != CUDA_SUCCESS) { log_cu_error("cuModuleGetFunction(vox_attn_q4_kv8_fp16)", r); return 0; }

    r = cuModuleGetFunction(&g_fn_pack_heads, g_mod, "vox_pack_heads_f32");
    if (r != CUDA_SUCCESS) { log_cu_error("cuModuleGetFunction(vox_pack_heads_f32)", r); return 0; }

    r = cuModuleGetFunction(&g_fn_unpack_heads, g_mod, "vox_unpack_heads_f32");
    if (r != CUDA_SUCCESS) { log_cu_error("cuModuleGetFunction(vox_unpack_heads_f32)", r); return 0; }

    r = cuModuleGetFunction(&g_fn_expand_kv_heads, g_mod, "vox_expand_kv_heads_f32");
    if (r != CUDA_SUCCESS) { log_cu_error("cuModuleGetFunction(vox_expand_kv_heads_f32)", r); return 0; }

    r = cuModuleGetFunction(&g_fn_softmax, g_mod, "vox_masked_softmax_causal_inplace_f32");
    if (r != CUDA_SUCCESS) { log_cu_error("cuModuleGetFunction(vox_masked_softmax_causal_inplace_f32)", r); return 0; }

    r = cuModuleGetFunction(&g_fn_rms_norm, g_mod, "vox_rms_norm_f32");
    if (r != CUDA_SUCCESS) { log_cu_error("cuModuleGetFunction(vox_rms_norm_f32)", r); return 0; }
    (void)cuModuleGetFunction(&g_fn_rms_norm_to_bf16, g_mod, "vox_rms_norm_to_bf16");
    r = cuModuleGetFunction(&g_fn_add_bias, g_mod, "vox_add_bias_f32");
    if (r != CUDA_SUCCESS) { log_cu_error("cuModuleGetFunction(vox_add_bias_f32)", r); return 0; }
    r = cuModuleGetFunction(&g_fn_add_inplace, g_mod, "vox_add_inplace_f32");
    if (r != CUDA_SUCCESS) { log_cu_error("cuModuleGetFunction(vox_add_inplace_f32)", r); return 0; }
    r = cuModuleGetFunction(&g_fn_mul_inplace, g_mod, "vox_mul_inplace_f32");
    if (r != CUDA_SUCCESS) { log_cu_error("cuModuleGetFunction(vox_mul_inplace_f32)", r); return 0; }
    r = cuModuleGetFunction(&g_fn_mul_1p_inplace, g_mod, "vox_mul_1p_inplace_f32");
    if (r != CUDA_SUCCESS) { log_cu_error("cuModuleGetFunction(vox_mul_1p_inplace_f32)", r); return 0; }
    (void)cuModuleGetFunction(&g_fn_mul_1p_rows_inplace, g_mod, "vox_mul_1p_rows_inplace_f32");
    r = cuModuleGetFunction(&g_fn_silu, g_mod, "vox_silu_inplace_f32");
    if (r != CUDA_SUCCESS) { log_cu_error("cuModuleGetFunction(vox_silu_inplace_f32)", r); return 0; }
    r = cuModuleGetFunction(&g_fn_gelu, g_mod, "vox_gelu_inplace_f32");
    if (r != CUDA_SUCCESS) { log_cu_error("cuModuleGetFunction(vox_gelu_inplace_f32)", r); return 0; }
    r = cuModuleGetFunction(&g_fn_f32_to_bf16, g_mod, "vox_f32_to_bf16");
    if (r != CUDA_SUCCESS) { log_cu_error("cuModuleGetFunction(vox_f32_to_bf16)", r); return 0; }
    r = cuModuleGetFunction(&g_fn_f32_to_f16, g_mod, "vox_f32_to_f16");
    if (r != CUDA_SUCCESS) { log_cu_error("cuModuleGetFunction(vox_f32_to_f16)", r); return 0; }
    r = cuModuleGetFunction(&g_fn_apply_rope, g_mod, "vox_apply_rope_f32");
    if (r != CUDA_SUCCESS) { log_cu_error("cuModuleGetFunction(vox_apply_rope_f32)", r); return 0; }
    r = cuModuleGetFunction(&g_fn_downsample4, g_mod, "vox_downsample4_concat_f32");
    if (r != CUDA_SUCCESS) { log_cu_error("cuModuleGetFunction(vox_downsample4_concat_f32)", r); return 0; }
    r = cuModuleGetFunction(&g_fn_argmax, g_mod, "vox_argmax_f32");
    if (r != CUDA_SUCCESS) { log_cu_error("cuModuleGetFunction(vox_argmax_f32)", r); return 0; }

    /* Optional legacy kernel (kept for now; not used in the fast path). */
    (void)cuModuleGetFunction(&g_fn_causal_attn, g_mod, "vox_causal_attn_f32");

    /* Optional kernels used for CUDA Graph capture (best-effort). */
    (void)cuModuleGetFunction(&g_fn_kv_append_dyn_fp16, g_mod, "vox_kv_append_fp16_dyn");
    (void)cuModuleGetFunction(&g_fn_kv_append_dyn_f32, g_mod, "vox_kv_append_f32_dyn");
    (void)cuModuleGetFunction(&g_fn_attn_dyn_fp16, g_mod, "vox_attn_q4_kv8_fp16_dyn");
    (void)cuModuleGetFunction(&g_fn_attn_dyn_f32, g_mod, "vox_attn_q4_kv8_f32_dyn");

    /* Optional v2 attention kernels (best-effort). */
    (void)cuModuleGetFunction(&g_fn_attn_f32_v2, g_mod, "vox_attn_q4_kv8_f32_v2");
    (void)cuModuleGetFunction(&g_fn_attn_fp16_v2, g_mod, "vox_attn_q4_kv8_fp16_v2");
    (void)cuModuleGetFunction(&g_fn_attn_dyn_fp16_v2, g_mod, "vox_attn_q4_kv8_fp16_dyn_v2");
    (void)cuModuleGetFunction(&g_fn_attn_dyn_f32_v2, g_mod, "vox_attn_q4_kv8_f32_dyn_v2");
    return 1;
}

static int ensure_kv_cache(int max_seq, int kv_dim) {
    if (max_seq <= 0 || kv_dim <= 0) return 0;
    if (!vox_cuda_available()) return 0;
    if (!cuda_load_kernel_module()) return 0;

    size_t elem_bytes = kv_cache_use_fp16() ? sizeof(uint16_t) : sizeof(float);
    if (g_k_cache && g_v_cache && g_kv_max_seq >= max_seq && g_kv_dim == kv_dim && g_kv_elem_bytes == elem_bytes) return 1;

    /* Reallocate (simple; grows rarely in practice). */
    if (g_k_cache) cuMemFree(g_k_cache);
    if (g_v_cache) cuMemFree(g_v_cache);
    g_k_cache = g_v_cache = 0;
    g_kv_max_seq = 0;
    g_kv_dim = 0;
    g_kv_elem_bytes = 0;

    size_t elems = (size_t)VOX_DEC_LAYERS * (size_t)max_seq * (size_t)kv_dim;
    size_t bytes = elems * elem_bytes;
    CUresult r;
    r = cuMemAlloc(&g_k_cache, bytes);
    if (r != CUDA_SUCCESS) { log_cu_error("cuMemAlloc(k_cache)", r); return 0; }
    r = cuMemAlloc(&g_v_cache, bytes);
    if (r != CUDA_SUCCESS) { log_cu_error("cuMemAlloc(v_cache)", r); return 0; }

    /* Zero to avoid reading garbage if something mis-sizes. */
    r = cuMemsetD8Async(g_k_cache, 0, bytes, g_stream);
    if (r != CUDA_SUCCESS) { log_cu_error("cuMemsetD8Async(k_cache)", r); return 0; }
    r = cuMemsetD8Async(g_v_cache, 0, bytes, g_stream);
    if (r != CUDA_SUCCESS) { log_cu_error("cuMemsetD8Async(v_cache)", r); return 0; }

    g_kv_max_seq = max_seq;
    g_kv_dim = kv_dim;
    g_kv_elem_bytes = elem_bytes;
    r = cuStreamSynchronize(g_stream);
    if (r != CUDA_SUCCESS) { log_cu_error("cuStreamSynchronize(kv_cache_init)", r); return 0; }
    return 1;
}

static int ensure_attn_workbufs(size_t q_bytes, size_t out_bytes) {
    if (!ensure_buffer(&g_dQ, &g_cap_q, q_bytes)) return 0;
    if (!ensure_buffer(&g_dAttn, &g_cap_attn, out_bytes)) return 0;
    return 1;
}

static int ensure_batched_ptr_arrays(int n_heads) {
    if (n_heads <= 0) return 0;
    if (g_batched_cap >= n_heads && g_batched_A && g_batched_B && g_batched_C) return 1;

    int new_cap = g_batched_cap ? g_batched_cap : 32;
    while (new_cap < n_heads) new_cap *= 2;

    const float **A = (const float **)realloc((void *)g_batched_A, (size_t)new_cap * sizeof(*A));
    const float **B = (const float **)realloc((void *)g_batched_B, (size_t)new_cap * sizeof(*B));
    float **C = (float **)realloc((void *)g_batched_C, (size_t)new_cap * sizeof(*C));
    if (!A || !B || !C) {
        free((void *)A);
        free((void *)B);
        free((void *)C);
        g_batched_A = g_batched_B = NULL;
        g_batched_C = NULL;
        g_batched_cap = 0;
        return 0;
    }

    g_batched_A = A;
    g_batched_B = B;
    g_batched_C = C;
    g_batched_cap = new_cap;
    return 1;
}

static int launch_rms_norm(CUdeviceptr out,
                           CUdeviceptr x,
                           CUdeviceptr weight,
                           int rows,
                           int hidden,
                           float eps) {
    if (!out || !x || !weight) return 0;
    if (rows <= 0 || hidden <= 0) return 0;
    if (!cuda_load_kernel_module()) return 0;

    int threads = 256;
    void *params[] = { &out, &x, &weight, &rows, &hidden, &eps };
    CUresult r = cuLaunchKernel(g_fn_rms_norm,
                                rows, 1, 1,
                                threads, 1, 1,
                                0, g_stream,
                                params, NULL);
    if (r != CUDA_SUCCESS) { log_cu_error("cuLaunchKernel(rms_norm)", r); return 0; }
    return 1;
}

static int launch_rms_norm_to_bf16(CUdeviceptr out_bf16,
                                   CUdeviceptr x,
                                   CUdeviceptr weight,
                                   int rows,
                                   int hidden,
                                   float eps) {
    if (!out_bf16 || !x || !weight) return 0;
    if (rows <= 0 || hidden <= 0) return 0;
    if (!cuda_load_kernel_module()) return 0;
    const char *disable = getenv("VOX_DISABLE_CUDA_RMSNORM_BF16_FUSED");
    if (disable && disable[0] && disable[0] != '0') return 0;
    if (!g_fn_rms_norm_to_bf16) return 0;

    int threads = 256;
    void *params[] = { &out_bf16, &x, &weight, &rows, &hidden, &eps };
    CUresult r = cuLaunchKernel(g_fn_rms_norm_to_bf16,
                                rows, 1, 1,
                                threads, 1, 1,
                                0, g_stream,
                                params, NULL);
    if (r != CUDA_SUCCESS) { log_cu_error("cuLaunchKernel(rms_norm_to_bf16)", r); return 0; }
    return 1;
}

static int launch_add_bias(CUdeviceptr x,
                           CUdeviceptr bias,
                           int rows,
                           int cols) {
    if (!x || !bias) return 0;
    if (rows <= 0 || cols <= 0) return 0;
    if (!cuda_load_kernel_module()) return 0;

    int threads = 256;
    int total = rows * cols;
    int blocks = (total + threads - 1) / threads;
    void *params[] = { &x, &bias, &rows, &cols };
    CUresult r = cuLaunchKernel(g_fn_add_bias,
                                blocks, 1, 1,
                                threads, 1, 1,
                                0, g_stream,
                                params, NULL);
    if (r != CUDA_SUCCESS) { log_cu_error("cuLaunchKernel(add_bias)", r); return 0; }
    return 1;
}

static int launch_add_inplace(CUdeviceptr x,
                              CUdeviceptr y,
                              int n) {
    if (!x || !y) return 0;
    if (n <= 0) return 0;
    if (!cuda_load_kernel_module()) return 0;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    void *params[] = { &x, &y, &n };
    CUresult r = cuLaunchKernel(g_fn_add_inplace,
                                blocks, 1, 1,
                                threads, 1, 1,
                                0, g_stream,
                                params, NULL);
    if (r != CUDA_SUCCESS) { log_cu_error("cuLaunchKernel(add_inplace)", r); return 0; }
    return 1;
}

static int launch_mul_inplace(CUdeviceptr x,
                              CUdeviceptr y,
                              int n) {
    if (!x || !y) return 0;
    if (n <= 0) return 0;
    if (!cuda_load_kernel_module()) return 0;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    void *params[] = { &x, &y, &n };
    CUresult r = cuLaunchKernel(g_fn_mul_inplace,
                                blocks, 1, 1,
                                threads, 1, 1,
                                0, g_stream,
                                params, NULL);
    if (r != CUDA_SUCCESS) { log_cu_error("cuLaunchKernel(mul_inplace)", r); return 0; }
    return 1;
}

static int launch_mul_1p_inplace(CUdeviceptr x,
                                 CUdeviceptr scale,
                                 int n) {
    if (!x || !scale) return 0;
    if (n <= 0) return 0;
    if (!cuda_load_kernel_module()) return 0;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    void *params[] = { &x, &scale, &n };
    CUresult r = cuLaunchKernel(g_fn_mul_1p_inplace,
                                blocks, 1, 1,
                                threads, 1, 1,
                                0, g_stream,
                                params, NULL);
    if (r != CUDA_SUCCESS) { log_cu_error("cuLaunchKernel(mul_1p)", r); return 0; }
    return 1;
}

static int launch_mul_1p_rows_inplace(CUdeviceptr x,
                                      CUdeviceptr scale,
                                      int rows,
                                      int cols) {
    if (!x || !scale) return 0;
    if (rows <= 0 || cols <= 0) return 0;
    if (!cuda_load_kernel_module()) return 0;
    if (!g_fn_mul_1p_rows_inplace) return 0;

    int threads = 256;
    int total = rows * cols;
    int blocks = (total + threads - 1) / threads;
    void *params[] = { &x, &scale, &rows, &cols };
    CUresult r = cuLaunchKernel(g_fn_mul_1p_rows_inplace,
                                blocks, 1, 1,
                                threads, 1, 1,
                                0, g_stream,
                                params, NULL);
    if (r != CUDA_SUCCESS) { log_cu_error("cuLaunchKernel(mul_1p_rows)", r); return 0; }
    return 1;
}

static int launch_silu_inplace(CUdeviceptr x, int n) {
    if (!x) return 0;
    if (n <= 0) return 0;
    if (!cuda_load_kernel_module()) return 0;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    void *params[] = { &x, &n };
    CUresult r = cuLaunchKernel(g_fn_silu,
                                blocks, 1, 1,
                                threads, 1, 1,
                                0, g_stream,
                                params, NULL);
    if (r != CUDA_SUCCESS) { log_cu_error("cuLaunchKernel(silu)", r); return 0; }
    return 1;
}

static int launch_gelu_inplace(CUdeviceptr x, int n) {
    if (!x) return 0;
    if (n <= 0) return 0;
    if (!cuda_load_kernel_module()) return 0;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    void *params[] = { &x, &n };
    CUresult r = cuLaunchKernel(g_fn_gelu,
                                blocks, 1, 1,
                                threads, 1, 1,
                                0, g_stream,
                                params, NULL);
    if (r != CUDA_SUCCESS) { log_cu_error("cuLaunchKernel(gelu)", r); return 0; }
    return 1;
}

static int launch_f32_to_bf16(CUdeviceptr dst_u16, CUdeviceptr src_f32, int n) {
    if (!dst_u16 || !src_f32) return 0;
    if (n <= 0) return 0;
    if (!cuda_load_kernel_module()) return 0;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    void *params[] = { &dst_u16, &src_f32, &n };
    CUresult r = cuLaunchKernel(g_fn_f32_to_bf16,
                                blocks, 1, 1,
                                threads, 1, 1,
                                0, g_stream,
                                params, NULL);
    if (r != CUDA_SUCCESS) { log_cu_error("cuLaunchKernel(f32_to_bf16)", r); return 0; }
    return 1;
}

static int launch_f32_to_f16(CUdeviceptr dst_u16, CUdeviceptr src_f32, int n) {
    if (!dst_u16 || !src_f32) return 0;
    if (n <= 0) return 0;
    if (!cuda_load_kernel_module()) return 0;

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    void *params[] = { &dst_u16, &src_f32, &n };
    CUresult r = cuLaunchKernel(g_fn_f32_to_f16,
                                blocks, 1, 1,
                                threads, 1, 1,
                                0, g_stream,
                                params, NULL);
    if (r != CUDA_SUCCESS) { log_cu_error("cuLaunchKernel(f32_to_f16)", r); return 0; }
    return 1;
}

static int launch_apply_rope(CUdeviceptr x,
                             CUdeviceptr freqs,
                             int seq,
                             int n_heads,
                             int head_dim) {
    if (!x || !freqs) return 0;
    if (seq <= 0 || n_heads <= 0 || head_dim <= 0) return 0;
    if (!cuda_load_kernel_module()) return 0;

    int half = head_dim / 2;
    int threads = 256;
    int total = seq * n_heads * half;
    int blocks = (total + threads - 1) / threads;
    void *params[] = { &x, &freqs, &seq, &n_heads, &head_dim };
    CUresult r = cuLaunchKernel(g_fn_apply_rope,
                                blocks, 1, 1,
                                threads, 1, 1,
                                0, g_stream,
                                params, NULL);
    if (r != CUDA_SUCCESS) { log_cu_error("cuLaunchKernel(apply_rope)", r); return 0; }
    return 1;
}

static int launch_downsample4_concat(CUdeviceptr dst,
                                     CUdeviceptr src,
                                     int start,
                                     int enc_len,
                                     int dim) {
    if (!dst || !src) return 0;
    if (enc_len <= 0 || dim <= 0) return 0;
    if (!cuda_load_kernel_module()) return 0;

    int ds_len = enc_len / 4;
    int ds_dim = dim * 4;
    int total = ds_len * ds_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    void *params[] = { &dst, &src, &start, &enc_len, &dim };
    CUresult r = cuLaunchKernel(g_fn_downsample4,
                                blocks, 1, 1,
                                threads, 1, 1,
                                0, g_stream,
                                params, NULL);
    if (r != CUDA_SUCCESS) { log_cu_error("cuLaunchKernel(downsample4)", r); return 0; }
    return 1;
}

static int launch_argmax(CUdeviceptr out_idx,
                         CUdeviceptr x,
                         int n) {
    if (!out_idx || !x) return 0;
    if (n <= 0) return 0;
    if (!cuda_load_kernel_module()) return 0;

    int threads = 256;
    void *params[] = { &out_idx, &x, &n };
    CUresult r = cuLaunchKernel(g_fn_argmax,
                                1, 1, 1,
                                threads, 1, 1,
                                0, g_stream,
                                params, NULL);
    if (r != CUDA_SUCCESS) { log_cu_error("cuLaunchKernel(argmax)", r); return 0; }
    return 1;
}

static int ensure_lt_workspace(size_t needed_bytes) {
    if (needed_bytes == 0) return 1;
    return ensure_buffer(&g_lt_workspace, &g_lt_workspace_cap, needed_bytes);
}

static int lt_get_algo_t_bf16(int M, int K, int N,
                              cublasLtMatmulAlgo_t *out_algo,
                              size_t *out_ws,
                              cublasLtMatmulDesc_t *out_op,
                              cublasLtMatrixLayout_t *out_a,
                              cublasLtMatrixLayout_t *out_b,
                              cublasLtMatrixLayout_t *out_c) {
    if (!out_algo || !out_ws) return 0;
    if (out_op) *out_op = NULL;
    if (out_a) *out_a = NULL;
    if (out_b) *out_b = NULL;
    if (out_c) *out_c = NULL;
    if (!g_lt_handle) return 0;

    for (int i = 0; i < g_lt_algos_len; i++) {
        if (g_lt_algos[i].valid &&
            g_lt_algos[i].M == M &&
            g_lt_algos[i].K == K &&
            g_lt_algos[i].N == N) {
            *out_algo = g_lt_algos[i].algo;
            *out_ws = g_lt_algos[i].workspace_bytes;
            if (out_op) *out_op = g_lt_algos[i].op;
            if (out_a) *out_a = g_lt_algos[i].a;
            if (out_b) *out_b = g_lt_algos[i].b;
            if (out_c) *out_c = g_lt_algos[i].c;
            return 1;
        }
    }

    cublasLtMatmulDesc_t op = NULL;
    cublasLtMatrixLayout_t a = NULL, b = NULL, c = NULL;
    cublasLtMatmulPreference_t pref = NULL;
    cublasLtMatmulHeuristicResult_t heur;
    int returned = 0;

    cublasStatus_t st;
    st = cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (st != CUBLAS_STATUS_SUCCESS) goto fail;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_T;
    (void)cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
    (void)cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));

    /* Row-major layouts. */
    st = cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, M, K, K);
    if (st != CUBLAS_STATUS_SUCCESS) goto fail;
    st = cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, N, K, K);
    if (st != CUBLAS_STATUS_SUCCESS) goto fail;
    st = cublasLtMatrixLayoutCreate(&c, CUDA_R_32F, M, N, N);
    if (st != CUBLAS_STATUS_SUCCESS) goto fail;

    cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    (void)cublasLtMatrixLayoutSetAttribute(a, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    (void)cublasLtMatrixLayoutSetAttribute(b, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    (void)cublasLtMatrixLayoutSetAttribute(c, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));

    st = cublasLtMatmulPreferenceCreate(&pref);
    if (st != CUBLAS_STATUS_SUCCESS) goto fail;

    size_t max_ws = (size_t)32 * 1024 * 1024;
    (void)cublasLtMatmulPreferenceSetAttribute(pref,
                                               CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                               &max_ws, sizeof(max_ws));

    st = cublasLtMatmulAlgoGetHeuristic(g_lt_handle,
                                        op,
                                        a, b,
                                        c, c,
                                        pref,
                                        1, &heur, &returned);
    if (st != CUBLAS_STATUS_SUCCESS || returned <= 0) goto fail;

    *out_algo = heur.algo;
    *out_ws = heur.workspaceSize;

    if (g_lt_algos_len < (int)(sizeof(g_lt_algos) / sizeof(g_lt_algos[0]))) {
        g_lt_algos[g_lt_algos_len++] = (lt_algo_entry_t){
            .M = M, .K = K, .N = N,
            .algo = heur.algo,
            .op = op,
            .a = a,
            .b = b,
            .c = c,
            .workspace_bytes = heur.workspaceSize,
            .valid = 1,
        };
        if (out_op) *out_op = op;
        if (out_a) *out_a = a;
        if (out_b) *out_b = b;
        if (out_c) *out_c = c;
        /* Descriptors are now owned by the cache; do not destroy here. */
        op = NULL;
        a = b = c = NULL;
    }

    cublasLtMatmulPreferenceDestroy(pref);
    return 1;

fail:
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (c) cublasLtMatrixLayoutDestroy(c);
    if (op) cublasLtMatmulDescDestroy(op);
    return 0;
}

static int gemm_t_bf16_bf16_f32(CUdeviceptr dC,
                                CUdeviceptr dA_bf16,
                                CUdeviceptr dB_bf16,
                                int M,
                                int K,
                                int N) {
    if (!dC || !dA_bf16 || !dB_bf16) return 0;
    if (M <= 0 || K <= 0 || N <= 0) return 0;

    const char *no_lt = getenv("VOX_DISABLE_CUBLASLT");
    if (g_lt_handle && M == 1 && (!no_lt || !no_lt[0] || no_lt[0] == '0')) {
        cublasLtMatmulAlgo_t algo;
        size_t ws = 0;
        cublasLtMatmulDesc_t op = NULL;
        cublasLtMatrixLayout_t a = NULL, b = NULL, c = NULL;
        if (lt_get_algo_t_bf16(M, K, N, &algo, &ws, &op, &a, &b, &c) &&
            op && a && b && c &&
            ensure_lt_workspace(ws)) {
            const float alpha = 1.0f;
            const float beta = 0.0f;
            cublasStatus_t st = cublasLtMatmul(g_lt_handle,
                                               op,
                                               &alpha,
                                               (const void *)(uintptr_t)dA_bf16, a,
                                               (const void *)(uintptr_t)dB_bf16, b,
                                               &beta,
                                               (const void *)(uintptr_t)dC, c,
                                               (void *)(uintptr_t)dC, c,
                                               &algo,
                                               (void *)(uintptr_t)g_lt_workspace, ws,
                                               (cudaStream_t)g_stream);
            if (st == CUBLAS_STATUS_SUCCESS) return 1;
            /* Fall through to cuBLAS GEMMEx. */
        }
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t st = cublasGemmEx(
        g_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        (const void *)(uintptr_t)dB_bf16, CUDA_R_16BF, K,
        (const void *)(uintptr_t)dA_bf16, CUDA_R_16BF, K,
        &beta,
        (void *)(uintptr_t)dC, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return st == CUBLAS_STATUS_SUCCESS;
}

static int vox_cuda_causal_attention_dev(CUdeviceptr dOut,
                                         CUdeviceptr dQ,
                                         CUdeviceptr dK,
                                         CUdeviceptr dV,
                                         int seq_q,
                                         int seq_k,
                                         int n_heads,
                                         int n_kv_heads,
                                         int head_dim,
                                         float scale,
                                         int window_size,
                                         int q_offset) {
    if (!dOut || !dQ || !dK || !dV) return 0;
    if (seq_q <= 0 || seq_k <= 0 || n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0) return 0;
    if ((n_heads % n_kv_heads) != 0) return 0;
    if (head_dim > 128) return 0;
    if (!cuda_load_kernel_module()) return 0;

    /* Optional direct sliding-window kernel (avoids O(seq^2) scores matrix).
     * Currently opt-in since the cuBLAS GEMM path is faster on this workload. */
    const char *direct_env = getenv("VOX_CUDA_DIRECT_ATTN");
    if (direct_env && direct_env[0] && direct_env[0] != '0' &&
        g_fn_causal_attn && window_size > 0) {
        int threads = 32;
        void *params[] = { &dOut, &dQ, &dK, &dV,
                           &seq_q, &seq_k, &n_heads, &n_kv_heads,
                           &head_dim, &scale, &window_size, &q_offset };
        CUresult rr = cuLaunchKernel(g_fn_causal_attn,
                                     n_heads, seq_q, 1,
                                     threads, 1, 1,
                                     0, g_stream, params, NULL);
        if (rr == CUDA_SUCCESS) return 1;
        log_cu_error("cuLaunchKernel(causal_attn_direct)", rr);
        /* Fall back to GEMM-based path. */
    }

    int q_hidden = n_heads * head_dim;
    int kv_hidden = n_kv_heads * head_dim;

    size_t bytes_q = (size_t)seq_q * (size_t)q_hidden * sizeof(float);
    size_t bytes_k = (size_t)seq_k * (size_t)kv_hidden * sizeof(float);
    size_t bytes_v = (size_t)seq_k * (size_t)kv_hidden * sizeof(float);
    size_t bytes_kfull = (size_t)seq_k * (size_t)q_hidden * sizeof(float);
    size_t bytes_vfull = (size_t)seq_k * (size_t)q_hidden * sizeof(float);
    size_t bytes_scores = (size_t)n_heads * (size_t)seq_q * (size_t)seq_k * sizeof(float);
    size_t bytes_out = (size_t)seq_q * (size_t)q_hidden * sizeof(float);

    if (!ensure_buffer(&g_dQp_attn, &g_cap_qp_attn, bytes_q)) return 0;
    if (!ensure_buffer(&g_dKp_attn, &g_cap_kp_attn, bytes_k)) return 0;
    if (!ensure_buffer(&g_dVp_attn, &g_cap_vp_attn, bytes_v)) return 0;
    if (!ensure_buffer(&g_dKfull_attn, &g_cap_kfull_attn, bytes_kfull)) return 0;
    if (!ensure_buffer(&g_dVfull_attn, &g_cap_vfull_attn, bytes_vfull)) return 0;
    if (!ensure_buffer(&g_dScores_attn, &g_cap_scores_attn, bytes_scores)) return 0;
    if (!ensure_buffer(&g_dOutPacked_attn, &g_cap_outpacked_attn, bytes_out)) return 0;

    /* Pack to contiguous-per-head layouts for cuBLAS. */
    int threads = 256;
    int total_q = seq_q * n_heads * head_dim;
    int total_kv = seq_k * n_kv_heads * head_dim;
    int blocks_q = (total_q + threads - 1) / threads;
    int blocks_kv = (total_kv + threads - 1) / threads;

    CUresult r;
    void *pack_q_params[] = { &g_dQp_attn, &dQ, &seq_q, &n_heads, &head_dim };
    r = cuLaunchKernel(g_fn_pack_heads, blocks_q, 1, 1, threads, 1, 1, 0, g_stream, pack_q_params, NULL);
    if (r != CUDA_SUCCESS) { log_cu_error("cuLaunchKernel(pack_Q_dev)", r); return 0; }

    void *pack_k_params[] = { &g_dKp_attn, &dK, &seq_k, &n_kv_heads, &head_dim };
    r = cuLaunchKernel(g_fn_pack_heads, blocks_kv, 1, 1, threads, 1, 1, 0, g_stream, pack_k_params, NULL);
    if (r != CUDA_SUCCESS) { log_cu_error("cuLaunchKernel(pack_K_dev)", r); return 0; }

    void *pack_v_params[] = { &g_dVp_attn, &dV, &seq_k, &n_kv_heads, &head_dim };
    r = cuLaunchKernel(g_fn_pack_heads, blocks_kv, 1, 1, threads, 1, 1, 0, g_stream, pack_v_params, NULL);
    if (r != CUDA_SUCCESS) { log_cu_error("cuLaunchKernel(pack_V_dev)", r); return 0; }

    /* Expand KV heads to per-query-head layout for strided-batched GEMMs. */
    int total_kfull = seq_k * n_heads * head_dim;
    int blocks_kfull = (total_kfull + threads - 1) / threads;
    void *expand_k_params[] = { &g_dKfull_attn, &g_dKp_attn, &seq_k, &n_heads, &n_kv_heads, &head_dim };
    r = cuLaunchKernel(g_fn_expand_kv_heads, blocks_kfull, 1, 1, threads, 1, 1, 0, g_stream, expand_k_params, NULL);
    if (r != CUDA_SUCCESS) { log_cu_error("cuLaunchKernel(expand_K_dev)", r); return 0; }

    void *expand_v_params[] = { &g_dVfull_attn, &g_dVp_attn, &seq_k, &n_heads, &n_kv_heads, &head_dim };
    r = cuLaunchKernel(g_fn_expand_kv_heads, blocks_kfull, 1, 1, threads, 1, 1, 0, g_stream, expand_v_params, NULL);
    if (r != CUDA_SUCCESS) { log_cu_error("cuLaunchKernel(expand_V_dev)", r); return 0; }

    /* 1) scores_h = Q_h @ K_h^T  (scaled) */
    const float alpha0 = scale;
    const float beta0 = 0.0f;
    long long strideA0 = (long long)((size_t)seq_k * (size_t)head_dim);
    long long strideB0 = (long long)((size_t)seq_q * (size_t)head_dim);
    long long strideC0 = (long long)((size_t)seq_q * (size_t)seq_k);
    cublasStatus_t st = cublasSgemmStridedBatched(
        g_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        seq_k, seq_q, head_dim,
        &alpha0,
        (const float *)(uintptr_t)g_dKfull_attn, head_dim, strideA0,
        (const float *)(uintptr_t)g_dQp_attn, head_dim, strideB0,
        &beta0,
        (float *)(uintptr_t)g_dScores_attn, seq_k, strideC0,
        n_heads);
    if (st != CUBLAS_STATUS_SUCCESS) return 0;

    /* 2) In-place masked softmax over K dimension. */
    void *softmax_params[] = { &g_dScores_attn, &seq_q, &seq_k, &window_size, &q_offset };
    r = cuLaunchKernel(g_fn_softmax,
                       n_heads, seq_q, 1,
                       threads, 1, 1,
                       0, g_stream, softmax_params, NULL);
    if (r != CUDA_SUCCESS) { log_cu_error("cuLaunchKernel(softmax_dev)", r); return 0; }

    /* 3) out_h = P_h @ V_h  */
    const float alpha1 = 1.0f;
    const float beta1 = 0.0f;
    long long strideA1 = (long long)((size_t)seq_k * (size_t)head_dim);
    long long strideB1 = (long long)((size_t)seq_q * (size_t)seq_k);
    long long strideC1 = (long long)((size_t)seq_q * (size_t)head_dim);
    st = cublasSgemmStridedBatched(
        g_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        head_dim, seq_q, seq_k,
        &alpha1,
        (const float *)(uintptr_t)g_dVfull_attn, head_dim, strideA1,
        (const float *)(uintptr_t)g_dScores_attn, seq_k, strideB1,
        &beta1,
        (float *)(uintptr_t)g_dOutPacked_attn, head_dim, strideC1,
        n_heads);
    if (st != CUBLAS_STATUS_SUCCESS) return 0;

    /* Unpack back to interleaved [seq_q, n_heads*head_dim] layout. */
    void *unpack_params[] = { &dOut, &g_dOutPacked_attn, &seq_q, &n_heads, &head_dim };
    r = cuLaunchKernel(g_fn_unpack_heads, blocks_q, 1, 1, threads, 1, 1, 0, g_stream, unpack_params, NULL);
    if (r != CUDA_SUCCESS) { log_cu_error("cuLaunchKernel(unpack_out_dev)", r); return 0; }
    return 1;
}

static int vox_cuda_decoder_attention_step_dev(CUdeviceptr dAttnOut,
                                               CUdeviceptr dQ,
                                               CUdeviceptr dK,
                                               CUdeviceptr dV,
                                               int layer,
                                               int pos,
                                               int total_seq,
                                               int window_size) {
    if (!dAttnOut || !dQ || !dK || !dV) return 0;
    if (layer < 0 || layer >= VOX_DEC_LAYERS) return 0;
    if (pos < 0 || total_seq <= 0 || pos >= total_seq) return 0;
    if (!cuda_load_kernel_module()) return 0;

    int kv_dim = VOX_DEC_KV_HEADS * VOX_DEC_HEAD_DIM; /* 1024 */
    int max_seq = g_kv_max_seq;
    if (max_seq <= 0) {
        max_seq = window_size + 2048;
        if (max_seq < 10240) max_seq = 10240;
    }
    if (pos >= max_seq) max_seq = pos + 1024;
    if (!ensure_kv_cache(max_seq, kv_dim)) return 0;

    size_t eb = g_kv_elem_bytes ? g_kv_elem_bytes : sizeof(float);
    size_t layer_stride = (size_t)g_kv_max_seq * (size_t)kv_dim * eb;
    size_t elem_off = (size_t)pos * (size_t)kv_dim * eb;
    CUdeviceptr dk = g_k_cache + (size_t)layer * layer_stride + elem_off;
    CUdeviceptr dv = g_v_cache + (size_t)layer * layer_stride + elem_off;

    /* Append K/V to cache with on-device f32->f16 conversion when enabled. */
    if (kv_cache_use_fp16()) {
        if (!launch_f32_to_f16(dk, dK, kv_dim)) return 0;
        if (!launch_f32_to_f16(dv, dV, kv_dim)) return 0;
    } else {
        CUresult r;
        r = cuMemcpyDtoDAsync(dk, dK, (size_t)kv_dim * sizeof(float), g_stream);
        if (r != CUDA_SUCCESS) { log_cu_error("cuMemcpyDtoDAsync(K_row_f32)", r); return 0; }
        r = cuMemcpyDtoDAsync(dv, dV, (size_t)kv_dim * sizeof(float), g_stream);
        if (r != CUDA_SUCCESS) { log_cu_error("cuMemcpyDtoDAsync(V_row_f32)", r); return 0; }
    }

    CUdeviceptr k_base = g_k_cache + (size_t)layer * layer_stride;
    CUdeviceptr v_base = g_v_cache + (size_t)layer * layer_stride;

    /* Launch attention kernel:
     * - grid = 32 blocks (query heads)
     * - block = 32 threads (1 warp), each lane owns 4 dims (head_dim=128) */
    float scale = 1.0f / 11.313708498984761f; /* 1/sqrt(128) */
    void *params[] = { &dAttnOut, &dQ, &k_base, &v_base, &total_seq, &window_size, &scale };
    CUresult r;
    int use_v2 = attn_v2_enabled();
    if (kv_cache_use_fp16()) {
        g_fn_attn = (use_v2 && g_fn_attn_fp16_v2) ? g_fn_attn_fp16_v2 : g_fn_attn_fp16;
    } else {
        g_fn_attn = (use_v2 && g_fn_attn_f32_v2) ? g_fn_attn_f32_v2 : g_fn_attn_f32;
    }
    r = cuLaunchKernel(g_fn_attn,
                       VOX_DEC_HEADS, 1, 1,
                       32, 1, 1,
                       0, g_stream, params, NULL);
    if (r != CUDA_SUCCESS) { log_cu_error("cuLaunchKernel(dec_attn_dev)", r); return 0; }
    return 1;
}

static void vox_cuda_init(void) {
    if (g_init) return;
    g_init = 1;

    if (cuInit(0) != CUDA_SUCCESS) return;

    int device_count = 0;
    if (cuDeviceGetCount(&device_count) != CUDA_SUCCESS || device_count <= 0) return;
    if (cuDeviceGet(&g_dev, 0) != CUDA_SUCCESS) return;

    char name[256] = {0};
    if (cuDeviceGetName(name, (int)sizeof(name), g_dev) == CUDA_SUCCESS) {
        strncpy(g_device_name, name, sizeof(g_device_name) - 1);
        g_device_name[sizeof(g_device_name) - 1] = '\0';
    }

    /* Use the primary context (plays nicer with WSL2 drivers). */
    if (cuDevicePrimaryCtxRetain(&g_ctx, g_dev) != CUDA_SUCCESS) return;
    if (cuCtxSetCurrent(g_ctx) != CUDA_SUCCESS) return;

    if (cuStreamCreate(&g_stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) return;
    if (cublasCreate(&g_handle) != CUBLAS_STATUS_SUCCESS) return;
    if (cublasSetStream(g_handle, (cudaStream_t)g_stream) != CUBLAS_STATUS_SUCCESS) return;

    /* Best-effort: enable tensor op math. */
    (void)cublasSetMathMode(g_handle, CUBLAS_TENSOR_OP_MATH);

    g_lt_handle = NULL;
    (void)cublasLtCreate(&g_lt_handle);

    g_available = 1;
}

int vox_cuda_available(void) {
    vox_cuda_init();
    return g_available;
}

const char *vox_cuda_device_name(void) {
    vox_cuda_init();
    return g_device_name;
}

void vox_cuda_shutdown(void) {
    if (!g_init) return;

    /* Decoder CUDA Graph resources (must be destroyed before freeing buffers they reference). */
    if (g_dec_graph_exec) cuGraphExecDestroy(g_dec_graph_exec);
    if (g_dec_graph) cuGraphDestroy(g_dec_graph);
    g_dec_graph_exec = 0;
    g_dec_graph = 0;
    g_dec_graph_ready = 0;
    g_dec_graph_kv_fp16 = -1;
    if (g_dec_pos_dev) cuMemFree(g_dec_pos_dev);
    g_dec_pos_dev = 0;

    if (g_dA) cuMemFree(g_dA);
    if (g_dB) cuMemFree(g_dB);
    if (g_dC) cuMemFree(g_dC);
    if (g_dC2) cuMemFree(g_dC2);
    if (g_dA_bf16) cuMemFree(g_dA_bf16);
    g_dA = g_dB = g_dC = 0;
    g_dC2 = 0;
    g_dA_bf16 = 0;
    g_cap_a = g_cap_b = g_cap_c = 0;
    g_cap_c2 = 0;
    g_cap_a_bf16 = 0;

    if (g_lt_workspace) cuMemFree(g_lt_workspace);
    g_lt_workspace = 0;
    g_lt_workspace_cap = 0;
    for (int i = 0; i < g_lt_algos_len; i++) {
        if (g_lt_algos[i].op) cublasLtMatmulDescDestroy(g_lt_algos[i].op);
        if (g_lt_algos[i].a) cublasLtMatrixLayoutDestroy(g_lt_algos[i].a);
        if (g_lt_algos[i].b) cublasLtMatrixLayoutDestroy(g_lt_algos[i].b);
        if (g_lt_algos[i].c) cublasLtMatrixLayoutDestroy(g_lt_algos[i].c);
        g_lt_algos[i].op = NULL;
        g_lt_algos[i].a = NULL;
        g_lt_algos[i].b = NULL;
        g_lt_algos[i].c = NULL;
        g_lt_algos[i].valid = 0;
    }
    g_lt_algos_len = 0;

    if (g_dQ) cuMemFree(g_dQ);
    if (g_dAttn) cuMemFree(g_dAttn);
    g_dQ = g_dAttn = 0;
    g_cap_q = g_cap_attn = 0;

    if (g_dQ_attn) cuMemFree(g_dQ_attn);
    if (g_dK_attn) cuMemFree(g_dK_attn);
    if (g_dV_attn) cuMemFree(g_dV_attn);
    if (g_dOut_attn) cuMemFree(g_dOut_attn);
    g_dQ_attn = g_dK_attn = g_dV_attn = g_dOut_attn = 0;
    g_cap_q_attn = g_cap_k_attn = g_cap_v_attn = g_cap_out_attn = 0;

    if (g_dQp_attn) cuMemFree(g_dQp_attn);
    if (g_dKp_attn) cuMemFree(g_dKp_attn);
    if (g_dVp_attn) cuMemFree(g_dVp_attn);
    if (g_dKfull_attn) cuMemFree(g_dKfull_attn);
    if (g_dVfull_attn) cuMemFree(g_dVfull_attn);
    if (g_dScores_attn) cuMemFree(g_dScores_attn);
    if (g_dOutPacked_attn) cuMemFree(g_dOutPacked_attn);
    g_dQp_attn = g_dKp_attn = g_dVp_attn = 0;
    g_dKfull_attn = g_dVfull_attn = 0;
    g_dScores_attn = g_dOutPacked_attn = 0;
    g_cap_qp_attn = g_cap_kp_attn = g_cap_vp_attn = 0;
    g_cap_kfull_attn = g_cap_vfull_attn = 0;
    g_cap_scores_attn = g_cap_outpacked_attn = 0;

    if (g_enc_x) cuMemFree(g_enc_x);
    if (g_enc_x_norm) cuMemFree(g_enc_x_norm);
    if (g_enc_x_bf16) cuMemFree(g_enc_x_bf16);
    if (g_enc_q) cuMemFree(g_enc_q);
    if (g_enc_k) cuMemFree(g_enc_k);
    if (g_enc_v) cuMemFree(g_enc_v);
    if (g_enc_attn) cuMemFree(g_enc_attn);
    if (g_enc_attn_bf16) cuMemFree(g_enc_attn_bf16);
    if (g_enc_proj) cuMemFree(g_enc_proj);
    if (g_enc_gate) cuMemFree(g_enc_gate);
    if (g_enc_up) cuMemFree(g_enc_up);
    if (g_enc_gate_bf16) cuMemFree(g_enc_gate_bf16);
    if (g_enc_ffn) cuMemFree(g_enc_ffn);
    if (g_enc_rope_freqs) cuMemFree(g_enc_rope_freqs);
    if (g_enc_ds) cuMemFree(g_enc_ds);
    if (g_enc_ds_bf16) cuMemFree(g_enc_ds_bf16);
    if (g_enc_mid) cuMemFree(g_enc_mid);
    if (g_enc_mid_bf16) cuMemFree(g_enc_mid_bf16);
    if (g_enc_adapter) cuMemFree(g_enc_adapter);
    g_enc_x = g_enc_x_norm = g_enc_q = g_enc_k = g_enc_v = 0;
    g_enc_x_bf16 = g_enc_attn = g_enc_attn_bf16 = 0;
    g_enc_proj = g_enc_gate = g_enc_up = 0;
    g_enc_gate_bf16 = g_enc_ffn = g_enc_rope_freqs = 0;
    g_enc_ds = g_enc_ds_bf16 = g_enc_mid = g_enc_mid_bf16 = 0;
    g_enc_adapter = 0;
    g_cap_enc_x = g_cap_enc_x_norm = g_cap_enc_x_bf16 = 0;
    g_cap_enc_q = g_cap_enc_k = g_cap_enc_v = 0;
    g_cap_enc_attn = g_cap_enc_attn_bf16 = 0;
    g_cap_enc_proj = g_cap_enc_gate = g_cap_enc_up = 0;
    g_cap_enc_gate_bf16 = g_cap_enc_ffn = 0;
    g_cap_enc_rope = 0;
    g_cap_enc_ds = g_cap_enc_ds_bf16 = 0;
    g_cap_enc_mid = g_cap_enc_mid_bf16 = 0;
    g_cap_enc_adapter = 0;

    if (g_dec_x) cuMemFree(g_dec_x);
    if (g_dec_x_norm) cuMemFree(g_dec_x_norm);
    if (g_dec_x_bf16) cuMemFree(g_dec_x_bf16);
    if (g_dec_q) cuMemFree(g_dec_q);
    if (g_dec_k) cuMemFree(g_dec_k);
    if (g_dec_v) cuMemFree(g_dec_v);
    if (g_dec_attn) cuMemFree(g_dec_attn);
    if (g_dec_attn_bf16) cuMemFree(g_dec_attn_bf16);
    if (g_dec_proj) cuMemFree(g_dec_proj);
    if (g_dec_gate) cuMemFree(g_dec_gate);
    if (g_dec_up) cuMemFree(g_dec_up);
    if (g_dec_gate_bf16) cuMemFree(g_dec_gate_bf16);
    if (g_dec_ffn) cuMemFree(g_dec_ffn);
    if (g_dec_rope_freqs) cuMemFree(g_dec_rope_freqs);
    if (g_dec_logits) cuMemFree(g_dec_logits);
    if (g_dec_best) cuMemFree(g_dec_best);
    g_dec_x = g_dec_x_norm = g_dec_x_bf16 = 0;
    g_dec_q = g_dec_k = g_dec_v = 0;
    g_dec_attn = g_dec_attn_bf16 = 0;
    g_dec_proj = g_dec_gate = g_dec_up = 0;
    g_dec_gate_bf16 = g_dec_ffn = 0;
    g_dec_rope_freqs = g_dec_logits = g_dec_best = 0;
    g_cap_dec_x = g_cap_dec_x_norm = g_cap_dec_x_bf16 = 0;
    g_cap_dec_q = g_cap_dec_k = g_cap_dec_v = 0;
    g_cap_dec_attn = g_cap_dec_attn_bf16 = 0;
    g_cap_dec_proj = g_cap_dec_gate = g_cap_dec_up = 0;
    g_cap_dec_gate_bf16 = g_cap_dec_ffn = 0;
    g_cap_dec_rope = 0;
    g_cap_dec_logits = 0;
    g_cap_dec_best = 0;

    if (g_k_cache) cuMemFree(g_k_cache);
    if (g_v_cache) cuMemFree(g_v_cache);
    g_k_cache = g_v_cache = 0;
    g_kv_max_seq = 0;
    g_kv_dim = 0;
    g_kv_elem_bytes = 0;

    if (g_mod) cuModuleUnload(g_mod);
    g_mod = 0;
    g_fn_attn = 0;
    g_fn_attn_fp16 = 0;
    g_fn_attn_f32 = 0;
    g_fn_attn_dyn_fp16 = 0;
    g_fn_attn_dyn_f32 = 0;
    g_fn_attn_fp16_v2 = 0;
    g_fn_attn_f32_v2 = 0;
    g_fn_attn_dyn_fp16_v2 = 0;
    g_fn_attn_dyn_f32_v2 = 0;
    g_fn_kv_append_dyn_fp16 = 0;
    g_fn_kv_append_dyn_f32 = 0;
    g_fn_causal_attn = 0;
    g_fn_pack_heads = 0;
    g_fn_unpack_heads = 0;
    g_fn_expand_kv_heads = 0;
    g_fn_softmax = 0;
    g_fn_rms_norm = 0;
    g_fn_rms_norm_to_bf16 = 0;
    g_fn_add_bias = 0;
    g_fn_add_inplace = 0;
    g_fn_mul_inplace = 0;
    g_fn_mul_1p_inplace = 0;
    g_fn_mul_1p_rows_inplace = 0;
    g_fn_silu = 0;
    g_fn_gelu = 0;
    g_fn_f32_to_bf16 = 0;
    g_fn_f32_to_f16 = 0;
    g_fn_apply_rope = 0;
    g_fn_downsample4 = 0;
    g_fn_argmax = 0;

    free((void *)g_batched_A);
    free((void *)g_batched_B);
    free((void *)g_batched_C);
    g_batched_A = g_batched_B = NULL;
    g_batched_C = NULL;
    g_batched_cap = 0;

    if (getenv("VOX_PRINT_TIMINGS")) {
        fprintf(stderr,
                "[cuda] bf16_cache: hits=%llu misses=%llu evictions=%llu entries=%d bytes=%.2f GiB limit=%.2f GiB uploaded=%.2f GiB\n",
                (unsigned long long)g_bf16_hits,
                (unsigned long long)g_bf16_misses,
                (unsigned long long)g_bf16_evictions,
                g_bf16_cache_len,
                (double)g_bf16_cache_bytes / (1024.0 * 1024.0 * 1024.0),
                (double)g_bf16_cache_limit / (1024.0 * 1024.0 * 1024.0),
                (double)g_bf16_upload_bytes / (1024.0 * 1024.0 * 1024.0));
    }
    g_bf16_hits = g_bf16_misses = 0;
    g_bf16_upload_bytes = 0;
    g_bf16_evictions = 0;

    for (int i = 0; i < g_bf16_cache_len; i++) {
        if (g_bf16_cache[i].dev) cuMemFree(g_bf16_cache[i].dev);
    }
    free(g_bf16_cache);
    g_bf16_cache = NULL;
    g_bf16_cache_cap = 0;
    g_bf16_cache_len = 0;
    g_bf16_cache_bytes = 0;
    g_bf16_cache_limit = 0;

    for (int i = 0; i < g_f32_cache_len; i++) {
        if (g_f32_cache[i].dev) cuMemFree(g_f32_cache[i].dev);
    }
    free(g_f32_cache);
    g_f32_cache = NULL;
    g_f32_cache_cap = 0;
    g_f32_cache_len = 0;

    free(g_host_a_bf16);
    g_host_a_bf16 = NULL;
    g_host_a_bf16_cap = 0;

    if (g_handle) cublasDestroy(g_handle);
    if (g_lt_handle) cublasLtDestroy(g_lt_handle);
    if (g_stream) cuStreamDestroy(g_stream);
    g_handle = NULL;
    g_lt_handle = NULL;
    g_stream = 0;

    if (g_ctx) cuDevicePrimaryCtxRelease(g_dev);
    g_ctx = NULL;
    g_dev = 0;

    g_init = 0;
    g_available = 0;
    strncpy(g_device_name, "unavailable", sizeof(g_device_name) - 1);
    g_device_name[sizeof(g_device_name) - 1] = '\0';
}

void vox_cuda_kv_cache_reset(void) {
    if (!vox_cuda_available()) return;
    (void)cuCtxSetCurrent(g_ctx);
    /* Just zero the caches; simpler than tracking per-context state. */
    if (g_k_cache && g_v_cache && g_kv_max_seq > 0 && g_kv_dim > 0) {
        size_t elems = (size_t)VOX_DEC_LAYERS * (size_t)g_kv_max_seq * (size_t)g_kv_dim;
        size_t bytes = elems * (g_kv_elem_bytes ? g_kv_elem_bytes : sizeof(float));
        (void)cuMemsetD8Async(g_k_cache, 0, bytes, g_stream);
        (void)cuMemsetD8Async(g_v_cache, 0, bytes, g_stream);
        (void)cuStreamSynchronize(g_stream);
    }
}

void vox_cuda_kv_cache_compact(int discard, int keep, int kv_dim, int max_seq) {
    if (!vox_cuda_available()) return;
    if (discard <= 0 || keep <= 0) return;
    if (!ensure_kv_cache(max_seq, kv_dim)) return;

    (void)cuCtxSetCurrent(g_ctx);
    size_t eb = g_kv_elem_bytes ? g_kv_elem_bytes : sizeof(float);
    size_t keep_bytes = (size_t)keep * (size_t)kv_dim * eb;
    size_t layer_stride = (size_t)max_seq * (size_t)kv_dim * eb;
    size_t src_off = (size_t)discard * (size_t)kv_dim * eb;

    for (int l = 0; l < VOX_DEC_LAYERS; l++) {
        CUdeviceptr k_dst = g_k_cache + (size_t)l * layer_stride;
        CUdeviceptr k_src = k_dst + src_off;
        CUdeviceptr v_dst = g_v_cache + (size_t)l * layer_stride;
        CUdeviceptr v_src = v_dst + src_off;
        (void)cuMemcpyDtoDAsync(k_dst, k_src, keep_bytes, g_stream);
        (void)cuMemcpyDtoDAsync(v_dst, v_src, keep_bytes, g_stream);
    }
    (void)cuStreamSynchronize(g_stream);
}

void vox_cuda_kv_cache_append_block(int layer, int start_pos, int seq_len,
                                    int kv_dim, int window_size,
                                    const float *k, const float *v) {
    if (!vox_cuda_available()) return;
    if (!k || !v) return;
    if (layer < 0 || layer >= VOX_DEC_LAYERS) return;
    if (start_pos < 0 || seq_len <= 0) return;
    if (kv_dim <= 0) return;

    (void)cuCtxSetCurrent(g_ctx);

    int max_seq = g_kv_max_seq;
    if (max_seq <= 0) {
        max_seq = window_size + 2048;
        if (max_seq < 10240) max_seq = 10240;
    }
    if (!ensure_kv_cache(max_seq, kv_dim)) return;

    size_t eb = g_kv_elem_bytes ? g_kv_elem_bytes : sizeof(float);
    size_t layer_stride = (size_t)g_kv_max_seq * (size_t)kv_dim * eb;
    size_t off = (size_t)start_pos * (size_t)kv_dim * eb;
    size_t bytes = (size_t)seq_len * (size_t)kv_dim * eb;

    CUdeviceptr dk = g_k_cache + (size_t)layer * layer_stride + off;
    CUdeviceptr dv = g_v_cache + (size_t)layer * layer_stride + off;

    if (kv_cache_use_fp16()) {
        size_t n = (size_t)seq_len * (size_t)kv_dim;
        uint16_t *hk = (uint16_t *)malloc(n * sizeof(uint16_t));
        uint16_t *hv = (uint16_t *)malloc(n * sizeof(uint16_t));
        if (!hk || !hv) { free(hk); free(hv); return; }
        for (size_t i = 0; i < n; i++) {
            hk[i] = f32_to_f16bits(k[i]);
            hv[i] = f32_to_f16bits(v[i]);
        }
        (void)cuMemcpyHtoDAsync(dk, hk, bytes, g_stream);
        (void)cuMemcpyHtoDAsync(dv, hv, bytes, g_stream);
        /* The source buffers are temporary; ensure the copies complete before freeing. */
        (void)cuStreamSynchronize(g_stream);
        free(hk);
        free(hv);
    } else {
        (void)cuMemcpyHtoDAsync(dk, k, bytes, g_stream);
        (void)cuMemcpyHtoDAsync(dv, v, bytes, g_stream);
    }
}

static int vox_cuda_kv_cache_append_block_dev(int layer, int start_pos, int seq_len,
                                              int kv_dim, int window_size,
                                              CUdeviceptr dK_f32, CUdeviceptr dV_f32) {
    if (!vox_cuda_available()) return 0;
    if (!dK_f32 || !dV_f32) return 0;
    if (layer < 0 || layer >= VOX_DEC_LAYERS) return 0;
    if (start_pos < 0 || seq_len <= 0) return 0;
    if (kv_dim <= 0) return 0;

    (void)cuCtxSetCurrent(g_ctx);

    int max_seq = g_kv_max_seq;
    if (max_seq <= 0) {
        max_seq = window_size + 2048;
        if (max_seq < 10240) max_seq = 10240;
    }
    if (!ensure_kv_cache(max_seq, kv_dim)) return 0;

    size_t eb = g_kv_elem_bytes ? g_kv_elem_bytes : sizeof(float);
    size_t layer_stride = (size_t)g_kv_max_seq * (size_t)kv_dim * eb;
    size_t off = (size_t)start_pos * (size_t)kv_dim * eb;
    CUdeviceptr dk = g_k_cache + (size_t)layer * layer_stride + off;
    CUdeviceptr dv = g_v_cache + (size_t)layer * layer_stride + off;

    if (kv_cache_use_fp16()) {
        int n = seq_len * kv_dim;
        if (!launch_f32_to_f16(dk, dK_f32, n)) return 0;
        if (!launch_f32_to_f16(dv, dV_f32, n)) return 0;
        return 1;
    }

    size_t bytes = (size_t)seq_len * (size_t)kv_dim * sizeof(float);
    CUresult r;
    r = cuMemcpyDtoDAsync(dk, dK_f32, bytes, g_stream);
    if (r != CUDA_SUCCESS) { log_cu_error("cuMemcpyDtoDAsync(k_block)", r); return 0; }
    r = cuMemcpyDtoDAsync(dv, dV_f32, bytes, g_stream);
    if (r != CUDA_SUCCESS) { log_cu_error("cuMemcpyDtoDAsync(v_block)", r); return 0; }
    return 1;
}

int vox_cuda_attention_step(float *attn_out,
                            const float *q,
                            const float *k,
                            const float *v,
                            int layer,
                            int pos,
                            int total_seq,
                            int window_size) {
    if (!vox_cuda_available()) return 0;
    const char *disable = getenv("VOX_DISABLE_CUDA_DECODER_ATTN");
    if (disable && disable[0] && disable[0] != '0') return 0;
    if (!attn_out || !q || !k || !v) return 0;
    if (layer < 0 || layer >= VOX_DEC_LAYERS) return 0;
    if (pos < 0 || total_seq <= 0 || pos >= total_seq) return 0;

    /* Ensure our primary context is current on this thread. */
    (void)cuCtxSetCurrent(g_ctx);

    int kv_dim = VOX_DEC_KV_HEADS * VOX_DEC_HEAD_DIM; /* 1024 */
    int max_seq = g_kv_max_seq;
    if (max_seq <= 0) {
        /* Conservative initial sizing: enough for the sliding window plus headroom. */
        max_seq = window_size + 2048;
        if (max_seq < 10240) max_seq = 10240;
    }

    if (!ensure_kv_cache(max_seq, kv_dim)) return 0;
    static int logged = 0;
    if (!logged && vox_verbose >= 1) {
        int want_v2 = attn_v2_enabled();
        int have_v2 = 0;
        if (want_v2) {
            have_v2 = kv_cache_use_fp16() ? (g_fn_attn_fp16_v2 != 0) : (g_fn_attn_f32_v2 != 0);
        }
        fprintf(stderr, "[cuda] decoder attention enabled (cubin, arch=%s, kv_cache=%s, attn=%s)\n",
                VOX_CUDA_ARCH_STR,
                kv_cache_use_fp16() ? "fp16" : "fp32",
                (want_v2 && have_v2) ? "v2" : "v1");
        logged = 1;
    }
    if (!ensure_attn_workbufs((size_t)VOX_DEC_HEADS * VOX_DEC_HEAD_DIM * sizeof(float),
                              (size_t)VOX_DEC_HEADS * VOX_DEC_HEAD_DIM * sizeof(float))) return 0;

    /* Copy Q to device */
    size_t q_bytes = (size_t)VOX_DEC_HEADS * VOX_DEC_HEAD_DIM * sizeof(float);
    CUresult r;
    r = cuMemcpyHtoDAsync(g_dQ, q, q_bytes, g_stream);
    if (r != CUDA_SUCCESS) { log_cu_error("cuMemcpyHtoDAsync(Q)", r); return 0; }

    /* Append K/V to device cache. */
    size_t eb = g_kv_elem_bytes ? g_kv_elem_bytes : sizeof(float);
    size_t layer_stride = (size_t)g_kv_max_seq * (size_t)kv_dim * eb;
    size_t elem_off = (size_t)pos * (size_t)kv_dim * eb;
    CUdeviceptr dk = g_k_cache + (size_t)layer * layer_stride + elem_off;
    CUdeviceptr dv = g_v_cache + (size_t)layer * layer_stride + elem_off;

    size_t kv_bytes = (size_t)kv_dim * eb;
    if (kv_cache_use_fp16()) {
        uint16_t hk[VOX_DEC_KV_HEADS * VOX_DEC_HEAD_DIM];
        uint16_t hv[VOX_DEC_KV_HEADS * VOX_DEC_HEAD_DIM];
        for (int i = 0; i < kv_dim; i++) {
            hk[i] = f32_to_f16bits(k[i]);
            hv[i] = f32_to_f16bits(v[i]);
        }
        r = cuMemcpyHtoDAsync(dk, hk, kv_bytes, g_stream);
        if (r != CUDA_SUCCESS) { log_cu_error("cuMemcpyHtoDAsync(K_row_fp16)", r); return 0; }
        r = cuMemcpyHtoDAsync(dv, hv, kv_bytes, g_stream);
        if (r != CUDA_SUCCESS) { log_cu_error("cuMemcpyHtoDAsync(V_row_fp16)", r); return 0; }
    } else {
        r = cuMemcpyHtoDAsync(dk, k, kv_bytes, g_stream);
        if (r != CUDA_SUCCESS) { log_cu_error("cuMemcpyHtoDAsync(K_row)", r); return 0; }
        r = cuMemcpyHtoDAsync(dv, v, kv_bytes, g_stream);
        if (r != CUDA_SUCCESS) { log_cu_error("cuMemcpyHtoDAsync(V_row)", r); return 0; }
    }

    CUdeviceptr k_base = g_k_cache + (size_t)layer * layer_stride;
    CUdeviceptr v_base = g_v_cache + (size_t)layer * layer_stride;

    /* Launch: grid=8 blocks (kv heads), block=128 threads */
    float scale = 1.0f / 11.313708498984761f; /* 1/sqrt(128) */

    /* The kernel expects k_cache/v_cache pointers to the base of this layer's cache. */
    void *params[] = { &g_dAttn, &g_dQ, &k_base, &v_base, &total_seq, &window_size, &scale };

    int use_v2 = attn_v2_enabled();
    if (kv_cache_use_fp16()) {
        g_fn_attn = (use_v2 && g_fn_attn_fp16_v2) ? g_fn_attn_fp16_v2 : g_fn_attn_fp16;
    } else {
        g_fn_attn = (use_v2 && g_fn_attn_f32_v2) ? g_fn_attn_f32_v2 : g_fn_attn_f32;
    }
    r = cuLaunchKernel(g_fn_attn,
                                VOX_DEC_HEADS, 1, 1,
                                32, 1, 1,
                                0,
                                g_stream,
                                params,
                                NULL);
    if (r != CUDA_SUCCESS) { log_cu_error("cuLaunchKernel(attn)", r); return 0; }

    /* Copy back */
    size_t out_bytes = (size_t)VOX_DEC_HEADS * VOX_DEC_HEAD_DIM * sizeof(float);
    r = cuMemcpyDtoHAsync(attn_out, g_dAttn, out_bytes, g_stream);
    if (r != CUDA_SUCCESS) { log_cu_error("cuMemcpyDtoHAsync(attn_out)", r); return 0; }
    r = cuStreamSynchronize(g_stream);
    if (r != CUDA_SUCCESS) { log_cu_error("cuStreamSynchronize(attn)", r); return 0; }
    return 1;
}

static int vox_cuda_gemm_rowmajor(float *C, const float *A, const float *B,
                                  int M, int K, int N, int b_is_transposed) {
    if (!vox_cuda_available()) return 0;

    /* Ensure our primary context is current on this thread. */
    (void)cuCtxSetCurrent(g_ctx);

    size_t bytes_a = (size_t)M * K * sizeof(float);
    size_t bytes_b = b_is_transposed ? (size_t)N * K * sizeof(float)
                                     : (size_t)K * N * sizeof(float);
    size_t bytes_c = (size_t)M * N * sizeof(float);

    if (!ensure_buffer(&g_dA, &g_cap_a, bytes_a) ||
        !ensure_buffer(&g_dB, &g_cap_b, bytes_b) ||
        !ensure_buffer(&g_dC, &g_cap_c, bytes_c)) {
        return 0;
    }

    if (cuMemcpyHtoDAsync(g_dA, A, bytes_a, g_stream) != CUDA_SUCCESS ||
        cuMemcpyHtoDAsync(g_dB, B, bytes_b, g_stream) != CUDA_SUCCESS) {
        return 0;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t status;

    if (!b_is_transposed) {
        status = cublasSgemm(g_handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             (const float *)(uintptr_t)g_dB, N,
                             (const float *)(uintptr_t)g_dA, K,
                             &beta,
                             (float *)(uintptr_t)g_dC, N);
    } else {
        status = cublasSgemm(g_handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             (const float *)(uintptr_t)g_dB, K,
                             (const float *)(uintptr_t)g_dA, K,
                             &beta,
                             (float *)(uintptr_t)g_dC, N);
    }

    if (status != CUBLAS_STATUS_SUCCESS) return 0;

    if (cuMemcpyDtoHAsync(C, g_dC, bytes_c, g_stream) != CUDA_SUCCESS) {
        return 0;
    }

    return (cuStreamSynchronize(g_stream) == CUDA_SUCCESS);
}

int vox_cuda_matmul(float *C, const float *A, const float *B, int M, int K, int N) {
    return vox_cuda_gemm_rowmajor(C, A, B, M, K, N, 0);
}

int vox_cuda_matmul_t(float *C, const float *A, const float *B, int M, int K, int N) {
    return vox_cuda_gemm_rowmajor(C, A, B, M, K, N, 1);
}

int vox_cuda_matmul_t_bf16(float *C, const float *A, const uint16_t *B_bf16, int M, int K, int N) {
    if (!vox_cuda_available()) return 0;
    /* Escape hatch for debugging/regressions. */
    const char *disable = getenv("VOX_DISABLE_CUDA_BF16");
    if (disable && disable[0] && disable[0] != '0') return 0;
    const char *a_bf16_env = getenv("VOX_CUDA_A_BF16");
    int use_a_bf16 = (!a_bf16_env || !a_bf16_env[0] || a_bf16_env[0] != '0');

    (void)cuCtxSetCurrent(g_ctx);

    size_t bytes_a = (size_t)M * K * sizeof(float);
    size_t bytes_b = (size_t)N * K * sizeof(uint16_t);
    size_t bytes_c = (size_t)M * N * sizeof(float);

    CUdeviceptr dB = bf16_cache_get(B_bf16, bytes_b);
    if (!dB) return 0;

    CUdeviceptr dA = 0;
    int a_is_bf16 = 0;
    size_t a_elems = (size_t)M * (size_t)K;
    if (use_a_bf16 && a_elems > 0 && a_elems <= (size_t)8 * 1024 * 1024) {
        /* Convert activations to BF16 so GEMM can use tensor cores.
         * This is particularly important for the streaming decoder (M=1), but
         * also helps encoder chunks (moderate M). */
        uint16_t *ha = host_a_bf16_get(a_elems);
        if (!ha) return 0;
        for (size_t i = 0; i < a_elems; i++) ha[i] = f32_to_bf16bits(A[i]);

        size_t bytes_a16 = a_elems * sizeof(uint16_t);
        if (!ensure_buffer(&g_dA_bf16, &g_cap_a_bf16, bytes_a16) ||
            !ensure_buffer(&g_dC, &g_cap_c, bytes_c)) {
            return 0;
        }
        if (cuMemcpyHtoDAsync(g_dA_bf16, ha, bytes_a16, g_stream) != CUDA_SUCCESS) return 0;
        dA = g_dA_bf16;
        a_is_bf16 = 1;
    } else {
        if (!ensure_buffer(&g_dA, &g_cap_a, bytes_a) ||
            !ensure_buffer(&g_dC, &g_cap_c, bytes_c)) {
            return 0;
        }
        if (cuMemcpyHtoDAsync(g_dA, A, bytes_a, g_stream) != CUDA_SUCCESS) {
            return 0;
        }
        dA = g_dA;
        a_is_bf16 = 0;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    /* Row-major: C[M,N] = A[M,K] @ B[N,K]^T
     * Use the same row-major trick as SGEMM path:
     * treat B as column-major (KxN) and A as column-major (KxM),
     * compute Ccol(N,M) = op(B) * A, where op(B)=T => (N,K). */
    cublasStatus_t status = cublasGemmEx(
        g_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        (const void *)(uintptr_t)dB, CUDA_R_16BF, K,
        (const void *)(uintptr_t)dA, a_is_bf16 ? CUDA_R_16BF : CUDA_R_32F, K,
        &beta,
        (void *)(uintptr_t)g_dC, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    if (status != CUBLAS_STATUS_SUCCESS) return 0;

    if (cuMemcpyDtoHAsync(C, g_dC, bytes_c, g_stream) != CUDA_SUCCESS) {
        return 0;
    }
    return (cuStreamSynchronize(g_stream) == CUDA_SUCCESS);
}

int vox_cuda_linear_bf16(float *y, const float *x, const uint16_t *W_bf16, const float *b,
                         int seq_len, int in_dim, int out_dim) {
    if (!vox_cuda_matmul_t_bf16(y, x, W_bf16, seq_len, in_dim, out_dim)) return 0;
    if (b) {
        for (int s = 0; s < seq_len; s++) {
            float *row = y + (size_t)s * out_dim;
            for (int o = 0; o < out_dim; o++) row[o] += b[o];
        }
    }
    return 1;
}

int vox_cuda_linear2_bf16(float *y0, float *y1,
                          const float *x,
                          const uint16_t *W0_bf16,
                          const uint16_t *W1_bf16,
                          int in_dim,
                          int out_dim) {
    if (!vox_cuda_available()) return 0;
    const char *disable = getenv("VOX_DISABLE_CUDA_BF16");
    if (disable && disable[0] && disable[0] != '0') return 0;
    if (!y0 || !y1 || !x || !W0_bf16 || !W1_bf16) return 0;
    if (in_dim <= 0 || out_dim <= 0) return 0;

    (void)cuCtxSetCurrent(g_ctx);

    size_t bytes_a = (size_t)in_dim * sizeof(float);
    size_t bytes_w = (size_t)out_dim * (size_t)in_dim * sizeof(uint16_t);
    size_t bytes_y = (size_t)out_dim * sizeof(float);

    CUdeviceptr dW0 = bf16_cache_get(W0_bf16, bytes_w);
    CUdeviceptr dW1 = bf16_cache_get(W1_bf16, bytes_w);
    if (!dW0 || !dW1) return 0;

    if (!ensure_buffer(&g_dA, &g_cap_a, bytes_a) ||
        !ensure_buffer(&g_dC, &g_cap_c, bytes_y) ||
        !ensure_buffer(&g_dC2, &g_cap_c2, bytes_y)) {
        return 0;
    }

    if (cuMemcpyHtoDAsync(g_dA, x, bytes_a, g_stream) != CUDA_SUCCESS) return 0;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    /* Row-major trick: y[1,out_dim] = x[1,in_dim] @ W[out_dim,in_dim]^T
     * Implement as GEMM in column-major (see vox_cuda_matmul_t_bf16). */
    cublasStatus_t st;
    st = cublasGemmEx(
        g_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        out_dim, 1, in_dim,
        &alpha,
        (const void *)(uintptr_t)dW0, CUDA_R_16BF, in_dim,
        (const void *)(uintptr_t)g_dA, CUDA_R_32F, in_dim,
        &beta,
        (void *)(uintptr_t)g_dC, CUDA_R_32F, out_dim,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st != CUBLAS_STATUS_SUCCESS) return 0;

    st = cublasGemmEx(
        g_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        out_dim, 1, in_dim,
        &alpha,
        (const void *)(uintptr_t)dW1, CUDA_R_16BF, in_dim,
        (const void *)(uintptr_t)g_dA, CUDA_R_32F, in_dim,
        &beta,
        (void *)(uintptr_t)g_dC2, CUDA_R_32F, out_dim,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st != CUBLAS_STATUS_SUCCESS) return 0;

    if (cuMemcpyDtoHAsync(y0, g_dC, bytes_y, g_stream) != CUDA_SUCCESS) return 0;
    if (cuMemcpyDtoHAsync(y1, g_dC2, bytes_y, g_stream) != CUDA_SUCCESS) return 0;
    return (cuStreamSynchronize(g_stream) == CUDA_SUCCESS);
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
    if (!vox_cuda_available()) return 0;
    if (!out || !Q || !K || !V) return 0;
    if (seq_q <= 0 || seq_k <= 0 || n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0) return 0;
    if ((n_heads % n_kv_heads) != 0) return 0;
    if (head_dim > 128) return 0;
    if (!cuda_load_kernel_module()) return 0;

    (void)cuCtxSetCurrent(g_ctx);

    int q_hidden = n_heads * head_dim;
    int kv_hidden = n_kv_heads * head_dim;

    size_t bytes_q = (size_t)seq_q * (size_t)q_hidden * sizeof(float);
    size_t bytes_k = (size_t)seq_k * (size_t)kv_hidden * sizeof(float);
    size_t bytes_v = (size_t)seq_k * (size_t)kv_hidden * sizeof(float);
    size_t bytes_kfull = (size_t)seq_k * (size_t)q_hidden * sizeof(float);
    size_t bytes_vfull = (size_t)seq_k * (size_t)q_hidden * sizeof(float);
    size_t bytes_scores = (size_t)n_heads * (size_t)seq_q * (size_t)seq_k * sizeof(float);
    size_t bytes_out = (size_t)seq_q * (size_t)q_hidden * sizeof(float);

    CUresult r;
    /* Always allocate only the minimal buffers first so the direct-window kernel
     * can avoid materializing the full scores matrix (O(seq^2) memory). */
    if (!ensure_buffer(&g_dQ_attn, &g_cap_q_attn, bytes_q)) return 0;
    if (!ensure_buffer(&g_dK_attn, &g_cap_k_attn, bytes_k)) return 0;
    if (!ensure_buffer(&g_dV_attn, &g_cap_v_attn, bytes_v)) return 0;
    if (!ensure_buffer(&g_dOut_attn, &g_cap_out_attn, bytes_out)) return 0;

    /* Upload interleaved Q/K/V as produced by CPU code. */
    r = cuMemcpyHtoDAsync(g_dQ_attn, Q, bytes_q, g_stream); if (r != CUDA_SUCCESS) { log_cu_error("HtoD(Q)", r); return 0; }
    r = cuMemcpyHtoDAsync(g_dK_attn, K, bytes_k, g_stream); if (r != CUDA_SUCCESS) { log_cu_error("HtoD(K)", r); return 0; }
    r = cuMemcpyHtoDAsync(g_dV_attn, V, bytes_v, g_stream); if (r != CUDA_SUCCESS) { log_cu_error("HtoD(V)", r); return 0; }

    /* Optional direct sliding-window kernel (avoids O(seq^2) scores matrix).
     * Currently opt-in since the cuBLAS GEMM path is faster on this workload. */
    const char *direct_env = getenv("VOX_CUDA_DIRECT_ATTN");
    if (direct_env && direct_env[0] && direct_env[0] != '0' &&
        g_fn_causal_attn && window_size > 0) {
        int threads = 32;
        void *params[] = { &g_dOut_attn, &g_dQ_attn, &g_dK_attn, &g_dV_attn,
                           &seq_q, &seq_k, &n_heads, &n_kv_heads, &head_dim,
                           &scale, &window_size, &q_offset };
        r = cuLaunchKernel(g_fn_causal_attn,
                           n_heads, seq_q, 1,
                           threads, 1, 1,
                           0, g_stream, params, NULL);
        if (r == CUDA_SUCCESS) {
            r = cuMemcpyDtoHAsync(out, g_dOut_attn, bytes_out, g_stream);
            if (r != CUDA_SUCCESS) { log_cu_error("DtoH(out_direct)", r); return 0; }
            r = cuStreamSynchronize(g_stream);
            if (r != CUDA_SUCCESS) { log_cu_error("sync(causal_attn_direct)", r); return 0; }
            return 1;
        }
        log_cu_error("cuLaunchKernel(causal_attn_direct)", r);
    }

    /* GEMM-based path: allocate larger work buffers. */
    if (!ensure_buffer(&g_dQp_attn, &g_cap_qp_attn, bytes_q)) return 0;
    if (!ensure_buffer(&g_dKp_attn, &g_cap_kp_attn, bytes_k)) return 0;
    if (!ensure_buffer(&g_dVp_attn, &g_cap_vp_attn, bytes_v)) return 0;
    if (!ensure_buffer(&g_dKfull_attn, &g_cap_kfull_attn, bytes_kfull)) return 0;
    if (!ensure_buffer(&g_dVfull_attn, &g_cap_vfull_attn, bytes_vfull)) return 0;
    if (!ensure_buffer(&g_dScores_attn, &g_cap_scores_attn, bytes_scores)) return 0;
    if (!ensure_buffer(&g_dOutPacked_attn, &g_cap_outpacked_attn, bytes_out)) return 0;

    /* Pack to contiguous-per-head layouts for cuBLAS. */
    int threads = 256;
    int total_q = seq_q * n_heads * head_dim;
    int total_kv = seq_k * n_kv_heads * head_dim;
    int blocks_q = (total_q + threads - 1) / threads;
    int blocks_kv = (total_kv + threads - 1) / threads;

    void *pack_q_params[] = { &g_dQp_attn, &g_dQ_attn, &seq_q, &n_heads, &head_dim };
    r = cuLaunchKernel(g_fn_pack_heads, blocks_q, 1, 1, threads, 1, 1, 0, g_stream, pack_q_params, NULL);
    if (r != CUDA_SUCCESS) { log_cu_error("cuLaunchKernel(pack_Q)", r); return 0; }

    void *pack_k_params[] = { &g_dKp_attn, &g_dK_attn, &seq_k, &n_kv_heads, &head_dim };
    r = cuLaunchKernel(g_fn_pack_heads, blocks_kv, 1, 1, threads, 1, 1, 0, g_stream, pack_k_params, NULL);
    if (r != CUDA_SUCCESS) { log_cu_error("cuLaunchKernel(pack_K)", r); return 0; }

    void *pack_v_params[] = { &g_dVp_attn, &g_dV_attn, &seq_k, &n_kv_heads, &head_dim };
    r = cuLaunchKernel(g_fn_pack_heads, blocks_kv, 1, 1, threads, 1, 1, 0, g_stream, pack_v_params, NULL);
    if (r != CUDA_SUCCESS) { log_cu_error("cuLaunchKernel(pack_V)", r); return 0; }

    /* Expand KV heads to per-query-head layout for strided-batched GEMMs. */
    int total_kfull = seq_k * n_heads * head_dim;
    int blocks_kfull = (total_kfull + threads - 1) / threads;
    void *expand_k_params[] = { &g_dKfull_attn, &g_dKp_attn, &seq_k, &n_heads, &n_kv_heads, &head_dim };
    r = cuLaunchKernel(g_fn_expand_kv_heads, blocks_kfull, 1, 1, threads, 1, 1, 0, g_stream, expand_k_params, NULL);
    if (r != CUDA_SUCCESS) { log_cu_error("cuLaunchKernel(expand_K)", r); return 0; }

    void *expand_v_params[] = { &g_dVfull_attn, &g_dVp_attn, &seq_k, &n_heads, &n_kv_heads, &head_dim };
    r = cuLaunchKernel(g_fn_expand_kv_heads, blocks_kfull, 1, 1, threads, 1, 1, 0, g_stream, expand_v_params, NULL);
    if (r != CUDA_SUCCESS) { log_cu_error("cuLaunchKernel(expand_V)", r); return 0; }

    /* 1) scores_h = Q_h @ K_h^T  (scaled) */
    const float alpha0 = scale;
    const float beta0 = 0.0f;
    long long strideA0 = (long long)((size_t)seq_k * (size_t)head_dim);
    long long strideB0 = (long long)((size_t)seq_q * (size_t)head_dim);
    long long strideC0 = (long long)((size_t)seq_q * (size_t)seq_k);
    cublasStatus_t st = cublasSgemmStridedBatched(
        g_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        seq_k, seq_q, head_dim,
        &alpha0,
        (const float *)(uintptr_t)g_dKfull_attn, head_dim, strideA0,
        (const float *)(uintptr_t)g_dQp_attn, head_dim, strideB0,
        &beta0,
        (float *)(uintptr_t)g_dScores_attn, seq_k, strideC0,
        n_heads);
    if (st != CUBLAS_STATUS_SUCCESS) return 0;

    /* 2) In-place masked softmax over K dimension. */
    void *softmax_params[] = { &g_dScores_attn, &seq_q, &seq_k, &window_size, &q_offset };
    r = cuLaunchKernel(g_fn_softmax,
                       n_heads, seq_q, 1,
                       threads, 1, 1,
                       0, g_stream, softmax_params, NULL);
    if (r != CUDA_SUCCESS) { log_cu_error("cuLaunchKernel(softmax)", r); return 0; }

    /* 3) out_h = P_h @ V_h  */
    const float alpha1 = 1.0f;
    const float beta1 = 0.0f;
    long long strideA1 = (long long)((size_t)seq_k * (size_t)head_dim);
    long long strideB1 = (long long)((size_t)seq_q * (size_t)seq_k);
    long long strideC1 = (long long)((size_t)seq_q * (size_t)head_dim);
    /* Row-major trick:
     * out_rm[seq_q,head_dim] == out_cm[head_dim,seq_q]
     * out_cm = V_cm(head_dim,seq_k) * P_cm(seq_k,seq_q) */
    st = cublasSgemmStridedBatched(
        g_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        head_dim, seq_q, seq_k,
        &alpha1,
        (const float *)(uintptr_t)g_dVfull_attn, head_dim, strideA1,
        (const float *)(uintptr_t)g_dScores_attn, seq_k, strideB1,
        &beta1,
        (float *)(uintptr_t)g_dOutPacked_attn, head_dim, strideC1,
        n_heads);
    if (st != CUBLAS_STATUS_SUCCESS) return 0;

    /* Unpack back to interleaved [seq_q, n_heads*head_dim] layout expected by CPU. */
    void *unpack_params[] = { &g_dOut_attn, &g_dOutPacked_attn, &seq_q, &n_heads, &head_dim };
    r = cuLaunchKernel(g_fn_unpack_heads, blocks_q, 1, 1, threads, 1, 1, 0, g_stream, unpack_params, NULL);
    if (r != CUDA_SUCCESS) { log_cu_error("cuLaunchKernel(unpack_out)", r); return 0; }

    r = cuMemcpyDtoHAsync(out, g_dOut_attn, bytes_out, g_stream);
    if (r != CUDA_SUCCESS) { log_cu_error("DtoH(out)", r); return 0; }
    r = cuStreamSynchronize(g_stream);
    if (r != CUDA_SUCCESS) { log_cu_error("sync(causal_attn)", r); return 0; }
    return 1;
}

static int causal_conv1d_out_len(int length, int kernel_size, int stride) {
    int padding_total = kernel_size - stride;
    float n_frames = ((float)length - (float)kernel_size + (float)padding_total) / (float)stride + 1.0f;
    int out_len = (int)ceilf(n_frames);
    return out_len < 0 ? 0 : out_len;
}

int vox_cuda_encode_adapter(float **out, int *out_tokens,
                            vox_ctx_t *ctx,
                            const float *mel,
                            int mel_frames,
                            int overlap_mel) {
    if (!out || !out_tokens) return 0;
    *out = NULL;
    *out_tokens = 0;

    if (!vox_cuda_available()) return 0;
    const char *disable = getenv("VOX_DISABLE_CUDA_ENCODER_FULL");
    if (disable && disable[0] && disable[0] != '0') return 0;
    if (!ctx || !mel || mel_frames <= 0) return 0;
    if (!cuda_load_kernel_module()) return 0;

    vox_encoder_t *enc = &ctx->encoder;

    /* ---- CPU conv stem ----
     * It's small relative to the transformer and avoids extra kernels. */
    int dim = VOX_ENC_DIM;
    float *conv_in = (float *)malloc((size_t)VOX_MEL_BINS * (size_t)mel_frames * sizeof(float));
    if (!conv_in) return 0;
    for (int f = 0; f < mel_frames; f++) {
        for (int m = 0; m < VOX_MEL_BINS; m++) {
            conv_in[(size_t)m * (size_t)mel_frames + (size_t)f] = mel[(size_t)f * VOX_MEL_BINS + (size_t)m];
        }
    }

    int conv0_out_len = causal_conv1d_out_len(mel_frames, 3, 1);
    float *conv0_out = (float *)malloc((size_t)dim * (size_t)conv0_out_len * sizeof(float));
    if (!conv0_out) { free(conv_in); return 0; }
    vox_causal_conv1d(conv0_out, conv_in, enc->conv0_weight, enc->conv0_bias,
                      VOX_MEL_BINS, dim, mel_frames, 3, 1);
    vox_gelu(conv0_out, dim * conv0_out_len);
    free(conv_in);

    int conv1_out_len = causal_conv1d_out_len(conv0_out_len, 3, 2);
    float *conv1_out = (float *)malloc((size_t)dim * (size_t)conv1_out_len * sizeof(float));
    if (!conv1_out) { free(conv0_out); return 0; }
    vox_causal_conv1d(conv1_out, conv0_out, enc->conv1_weight, enc->conv1_bias,
                      dim, dim, conv0_out_len, 3, 2);
    vox_gelu(conv1_out, dim * conv1_out_len);
    free(conv0_out);

    int seq_len = conv1_out_len;
    float *x_host = (float *)malloc((size_t)seq_len * (size_t)dim * sizeof(float));
    if (!x_host) { free(conv1_out); return 0; }

    for (int s = 0; s < seq_len; s++) {
        for (int d = 0; d < dim; d++) {
            x_host[(size_t)s * (size_t)dim + (size_t)d] = conv1_out[(size_t)d * (size_t)seq_len + (size_t)s];
        }
    }
    free(conv1_out);

    int overlap_enc = overlap_mel / 2;
    if (overlap_enc < 0) overlap_enc = 0;
    if (overlap_enc > seq_len) overlap_enc = seq_len;
    int new_enc_len = seq_len - overlap_enc;
    new_enc_len = (new_enc_len / 4) * 4;
    int ds_len = new_enc_len / 4;
    if (ds_len <= 0) {
        free(x_host);
        *out_tokens = 0;
        *out = NULL;
        return 1;
    }

    /* RoPE frequencies (CPU) */
    int head_dim = VOX_ENC_HEAD_DIM;
    int half_dim = head_dim / 2;
    int rope_cols = half_dim * 2;
    int *positions = (int *)malloc((size_t)seq_len * sizeof(int));
    float *rope_host = (float *)malloc((size_t)seq_len * (size_t)rope_cols * sizeof(float));
    if (!positions || !rope_host) { free(positions); free(rope_host); free(x_host); return 0; }
    for (int i = 0; i < seq_len; i++) positions[i] = i;
    vox_compute_rope_freqs(rope_host, positions, seq_len, head_dim, VOX_ROPE_THETA);
    free(positions);

    (void)cuCtxSetCurrent(g_ctx);

    /* Upload x + rope freqs */
    size_t bytes_x = (size_t)seq_len * (size_t)dim * sizeof(float);
    if (!ensure_buffer(&g_enc_x, &g_cap_enc_x, bytes_x) ||
        !ensure_buffer(&g_enc_x_norm, &g_cap_enc_x_norm, bytes_x) ||
        !ensure_buffer(&g_enc_x_bf16, &g_cap_enc_x_bf16, (size_t)seq_len * (size_t)dim * sizeof(uint16_t))) {
        free(rope_host);
        free(x_host);
        return 0;
    }

    CUresult r;
    r = cuMemcpyHtoDAsync(g_enc_x, x_host, bytes_x, g_stream);
    free(x_host);
    if (r != CUDA_SUCCESS) { log_cu_error("HtoD(enc_x)", r); free(rope_host); return 0; }

    size_t bytes_rope = (size_t)seq_len * (size_t)rope_cols * sizeof(float);
    if (!ensure_buffer(&g_enc_rope_freqs, &g_cap_enc_rope, bytes_rope)) { free(rope_host); return 0; }
    r = cuMemcpyHtoDAsync(g_enc_rope_freqs, rope_host, bytes_rope, g_stream);
    free(rope_host);
    if (r != CUDA_SUCCESS) { log_cu_error("HtoD(enc_rope)", r); return 0; }

    /* Ensure working buffers */
    int n_heads = VOX_ENC_HEADS;
    int n_kv_heads = VOX_ENC_KV_HEADS;
    int qkv_dim = n_heads * head_dim; /* 2048 */
    int hidden = VOX_ENC_HIDDEN;      /* 5120 */
    size_t bytes_q = (size_t)seq_len * (size_t)qkv_dim * sizeof(float);
    size_t bytes_dim = bytes_x;
    size_t bytes_gate = (size_t)seq_len * (size_t)hidden * sizeof(float);

    if (!ensure_buffer(&g_enc_q, &g_cap_enc_q, bytes_q) ||
        !ensure_buffer(&g_enc_k, &g_cap_enc_k, bytes_q) ||
        !ensure_buffer(&g_enc_v, &g_cap_enc_v, bytes_q) ||
        !ensure_buffer(&g_enc_attn, &g_cap_enc_attn, bytes_q) ||
        !ensure_buffer(&g_enc_attn_bf16, &g_cap_enc_attn_bf16, (size_t)seq_len * (size_t)qkv_dim * sizeof(uint16_t)) ||
        !ensure_buffer(&g_enc_proj, &g_cap_enc_proj, bytes_dim) ||
        !ensure_buffer(&g_enc_gate, &g_cap_enc_gate, bytes_gate) ||
        !ensure_buffer(&g_enc_up, &g_cap_enc_up, bytes_gate) ||
        !ensure_buffer(&g_enc_gate_bf16, &g_cap_enc_gate_bf16, (size_t)seq_len * (size_t)hidden * sizeof(uint16_t)) ||
        !ensure_buffer(&g_enc_ffn, &g_cap_enc_ffn, bytes_dim)) {
        return 0;
    }

    float attn_scale = 1.0f / sqrtf((float)head_dim);

    /* ---- Transformer layers (GPU) ---- */
    for (int layer = 0; layer < VOX_ENC_LAYERS; layer++) {
        vox_enc_layer_t *l = &enc->layers[layer];

        CUdeviceptr d_attn_norm = f32_cache_get(l->attention_norm, (size_t)dim * sizeof(float));
        CUdeviceptr d_ffn_norm = f32_cache_get(l->ffn_norm, (size_t)dim * sizeof(float));
        CUdeviceptr d_wq_bias = f32_cache_get(l->wq_bias, (size_t)qkv_dim * sizeof(float));
        CUdeviceptr d_wv_bias = f32_cache_get(l->wv_bias, (size_t)qkv_dim * sizeof(float));
        CUdeviceptr d_wo_bias = f32_cache_get(l->wo_bias, (size_t)dim * sizeof(float));
        CUdeviceptr d_w2_bias = f32_cache_get(l->w2_bias, (size_t)dim * sizeof(float));
        if (!d_attn_norm || !d_ffn_norm || !d_wq_bias || !d_wv_bias || !d_wo_bias || !d_w2_bias) return 0;

        /* x_norm_bf16 = rms_norm(x) */
        if (!launch_rms_norm_to_bf16(g_enc_x_bf16, g_enc_x, d_attn_norm, seq_len, dim, VOX_ENC_NORM_EPS)) {
            if (!launch_rms_norm(g_enc_x_norm, g_enc_x, d_attn_norm, seq_len, dim, VOX_ENC_NORM_EPS)) return 0;
            if (!launch_f32_to_bf16(g_enc_x_bf16, g_enc_x_norm, seq_len * dim)) return 0;
        }

        /* Q,K,V projections */
        size_t bytes_wq = (size_t)qkv_dim * (size_t)dim * sizeof(uint16_t);
        CUdeviceptr dWq = bf16_cache_get(l->wq_weight_bf16, bytes_wq);
        CUdeviceptr dWk = bf16_cache_get(l->wk_weight_bf16, bytes_wq);
        CUdeviceptr dWv = bf16_cache_get(l->wv_weight_bf16, bytes_wq);
        if (!dWq || !dWk || !dWv) return 0;

        if (!gemm_t_bf16_bf16_f32(g_enc_q, g_enc_x_bf16, dWq, seq_len, dim, qkv_dim)) return 0;
        if (!gemm_t_bf16_bf16_f32(g_enc_k, g_enc_x_bf16, dWk, seq_len, dim, qkv_dim)) return 0;
        if (!gemm_t_bf16_bf16_f32(g_enc_v, g_enc_x_bf16, dWv, seq_len, dim, qkv_dim)) return 0;

        if (!launch_add_bias(g_enc_q, d_wq_bias, seq_len, qkv_dim)) return 0;
        if (!launch_add_bias(g_enc_v, d_wv_bias, seq_len, qkv_dim)) return 0;

        /* RoPE */
        if (!launch_apply_rope(g_enc_q, g_enc_rope_freqs, seq_len, n_heads, head_dim)) return 0;
        if (!launch_apply_rope(g_enc_k, g_enc_rope_freqs, seq_len, n_kv_heads, head_dim)) return 0;

        /* Attention */
        if (!vox_cuda_causal_attention_dev(g_enc_attn, g_enc_q, g_enc_k, g_enc_v,
                                           seq_len, seq_len, n_heads, n_kv_heads,
                                           head_dim, attn_scale, VOX_ENC_WINDOW, 0)) {
            return 0;
        }

        /* Output projection */
        size_t bytes_wo = (size_t)dim * (size_t)qkv_dim * sizeof(uint16_t);
        CUdeviceptr dWo = bf16_cache_get(l->wo_weight_bf16, bytes_wo);
        if (!dWo) return 0;

        if (!launch_f32_to_bf16(g_enc_attn_bf16, g_enc_attn, seq_len * qkv_dim)) return 0;
        if (!gemm_t_bf16_bf16_f32(g_enc_proj, g_enc_attn_bf16, dWo, seq_len, qkv_dim, dim)) return 0;
        if (!launch_add_bias(g_enc_proj, d_wo_bias, seq_len, dim)) return 0;
        if (!launch_add_inplace(g_enc_x, g_enc_proj, seq_len * dim)) return 0;

        /* FFN */
        if (!launch_rms_norm_to_bf16(g_enc_x_bf16, g_enc_x, d_ffn_norm, seq_len, dim, VOX_ENC_NORM_EPS)) {
            if (!launch_rms_norm(g_enc_x_norm, g_enc_x, d_ffn_norm, seq_len, dim, VOX_ENC_NORM_EPS)) return 0;
            if (!launch_f32_to_bf16(g_enc_x_bf16, g_enc_x_norm, seq_len * dim)) return 0;
        }

        size_t bytes_w1 = (size_t)hidden * (size_t)dim * sizeof(uint16_t);
        CUdeviceptr dW1 = bf16_cache_get(l->w1_weight_bf16, bytes_w1);
        CUdeviceptr dW3 = bf16_cache_get(l->w3_weight_bf16, bytes_w1);
        if (!dW1 || !dW3) return 0;
        if (!gemm_t_bf16_bf16_f32(g_enc_gate, g_enc_x_bf16, dW1, seq_len, dim, hidden)) return 0;
        if (!launch_silu_inplace(g_enc_gate, seq_len * hidden)) return 0;
        if (!gemm_t_bf16_bf16_f32(g_enc_up, g_enc_x_bf16, dW3, seq_len, dim, hidden)) return 0;
        if (!launch_mul_inplace(g_enc_gate, g_enc_up, seq_len * hidden)) return 0;

        size_t bytes_w2 = (size_t)dim * (size_t)hidden * sizeof(uint16_t);
        CUdeviceptr dW2 = bf16_cache_get(l->w2_weight_bf16, bytes_w2);
        if (!dW2) return 0;

        if (!launch_f32_to_bf16(g_enc_gate_bf16, g_enc_gate, seq_len * hidden)) return 0;
        if (!gemm_t_bf16_bf16_f32(g_enc_ffn, g_enc_gate_bf16, dW2, seq_len, hidden, dim)) return 0;
        if (!launch_add_bias(g_enc_ffn, d_w2_bias, seq_len, dim)) return 0;
        if (!launch_add_inplace(g_enc_x, g_enc_ffn, seq_len * dim)) return 0;
    }

    /* Final norm (in-place) */
    CUdeviceptr d_norm = f32_cache_get(enc->norm, (size_t)dim * sizeof(float));
    if (!d_norm) return 0;
    if (!launch_rms_norm(g_enc_x, g_enc_x, d_norm, seq_len, dim, VOX_ENC_NORM_EPS)) return 0;

    /* ---- Adapter (GPU) ---- */
    int ds_dim = dim * 4; /* 5120 */
    size_t bytes_ds = (size_t)ds_len * (size_t)ds_dim * sizeof(float);
    size_t bytes_ds_bf16 = (size_t)ds_len * (size_t)ds_dim * sizeof(uint16_t);
    size_t bytes_mid = (size_t)ds_len * (size_t)VOX_DEC_DIM * sizeof(float);
    size_t bytes_mid_bf16 = (size_t)ds_len * (size_t)VOX_DEC_DIM * sizeof(uint16_t);

    if (!ensure_buffer(&g_enc_ds, &g_cap_enc_ds, bytes_ds) ||
        !ensure_buffer(&g_enc_ds_bf16, &g_cap_enc_ds_bf16, bytes_ds_bf16) ||
        !ensure_buffer(&g_enc_mid, &g_cap_enc_mid, bytes_mid) ||
        !ensure_buffer(&g_enc_mid_bf16, &g_cap_enc_mid_bf16, bytes_mid_bf16) ||
        !ensure_buffer(&g_enc_adapter, &g_cap_enc_adapter, bytes_mid)) {
        return 0;
    }

    if (!launch_downsample4_concat(g_enc_ds, g_enc_x, overlap_enc, new_enc_len, dim)) return 0;

    size_t bytes_w0 = (size_t)VOX_DEC_DIM * (size_t)ds_dim * sizeof(uint16_t);
    CUdeviceptr dW0 = bf16_cache_get(ctx->adapter.linear0_weight_bf16, bytes_w0);
    if (!dW0) return 0;
    if (!launch_f32_to_bf16(g_enc_ds_bf16, g_enc_ds, ds_len * ds_dim)) return 0;
    if (!gemm_t_bf16_bf16_f32(g_enc_mid, g_enc_ds_bf16, dW0, ds_len, ds_dim, VOX_DEC_DIM)) return 0;
    if (!launch_gelu_inplace(g_enc_mid, ds_len * VOX_DEC_DIM)) return 0;

    size_t bytes_w1 = (size_t)VOX_DEC_DIM * (size_t)VOX_DEC_DIM * sizeof(uint16_t);
    CUdeviceptr dW1 = bf16_cache_get(ctx->adapter.linear1_weight_bf16, bytes_w1);
    if (!dW1) return 0;
    if (!launch_f32_to_bf16(g_enc_mid_bf16, g_enc_mid, ds_len * VOX_DEC_DIM)) return 0;
    if (!gemm_t_bf16_bf16_f32(g_enc_adapter, g_enc_mid_bf16, dW1, ds_len, VOX_DEC_DIM, VOX_DEC_DIM)) return 0;

    float *host_out = (float *)malloc(bytes_mid);
    if (!host_out) return 0;
    r = cuMemcpyDtoHAsync(host_out, g_enc_adapter, bytes_mid, g_stream);
    if (r != CUDA_SUCCESS) { log_cu_error("DtoH(adapter_out)", r); free(host_out); return 0; }
    r = cuStreamSynchronize(g_stream);
    if (r != CUDA_SUCCESS) { log_cu_error("sync(encoder_adapter)", r); free(host_out); return 0; }

    *out_tokens = ds_len;
    *out = host_out;
    return 1;
}

static int env_truthy(const char *name) {
    const char *v = getenv(name);
    return v && v[0] && v[0] != '0';
}

static int decoder_graph_wanted(void) {
    if (env_truthy("VOX_DISABLE_CUDA_GRAPHS")) return 0;
    return env_truthy("VOX_CUDA_GRAPHS");
}

static void decoder_graph_destroy(void) {
    if (!vox_cuda_available()) return;
    (void)cuCtxSetCurrent(g_ctx);

    if (g_dec_graph_exec) cuGraphExecDestroy(g_dec_graph_exec);
    if (g_dec_graph) cuGraphDestroy(g_dec_graph);
    g_dec_graph_exec = 0;
    g_dec_graph = 0;
    g_dec_graph_ready = 0;
    g_dec_graph_kv_fp16 = -1;

    if (g_dec_pos_dev) cuMemFree(g_dec_pos_dev);
    g_dec_pos_dev = 0;
}

static int decoder_graph_prepare(vox_ctx_t *ctx) {
    if (!ctx) return 0;
    if (!vox_cuda_available()) return 0;
    if (!cuda_load_kernel_module()) return 0;

    int want_fp16 = kv_cache_use_fp16();
    if (want_fp16) {
        if (!g_fn_kv_append_dyn_fp16 || !g_fn_attn_dyn_fp16) return 0;
    } else {
        if (!g_fn_kv_append_dyn_f32 || !g_fn_attn_dyn_f32) return 0;
    }

    (void)cuCtxSetCurrent(g_ctx);

    int dim = VOX_DEC_DIM;
    int n_heads = VOX_DEC_HEADS;
    int n_kv_heads = VOX_DEC_KV_HEADS;
    int head_dim = VOX_DEC_HEAD_DIM;
    int hidden = VOX_DEC_HIDDEN;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;

    /* Ensure device KV cache is ready and large enough. */
    int want_max_seq = ctx->kv_cache_max > 0 ? ctx->kv_cache_max : (VOX_DEC_WINDOW + 2048);
    if (!ensure_kv_cache(want_max_seq, kv_dim)) return 0;

    /* Ensure decoder work buffers exist (M=1 step). */
    size_t bytes_rope = (size_t)((head_dim / 2) * 2) * sizeof(float);
    if (!ensure_buffer(&g_dec_x, &g_cap_dec_x, (size_t)dim * sizeof(float)) ||
        !ensure_buffer(&g_dec_x_norm, &g_cap_dec_x_norm, (size_t)dim * sizeof(float)) ||
        !ensure_buffer(&g_dec_x_bf16, &g_cap_dec_x_bf16, (size_t)dim * sizeof(uint16_t)) ||
        !ensure_buffer(&g_dec_q, &g_cap_dec_q, (size_t)q_dim * sizeof(float)) ||
        !ensure_buffer(&g_dec_k, &g_cap_dec_k, (size_t)kv_dim * sizeof(float)) ||
        !ensure_buffer(&g_dec_v, &g_cap_dec_v, (size_t)kv_dim * sizeof(float)) ||
        !ensure_buffer(&g_dec_attn, &g_cap_dec_attn, (size_t)q_dim * sizeof(float)) ||
        !ensure_buffer(&g_dec_attn_bf16, &g_cap_dec_attn_bf16, (size_t)q_dim * sizeof(uint16_t)) ||
        !ensure_buffer(&g_dec_proj, &g_cap_dec_proj, (size_t)dim * sizeof(float)) ||
        !ensure_buffer(&g_dec_gate, &g_cap_dec_gate, (size_t)hidden * sizeof(float)) ||
        !ensure_buffer(&g_dec_up, &g_cap_dec_up, (size_t)hidden * sizeof(float)) ||
        !ensure_buffer(&g_dec_gate_bf16, &g_cap_dec_gate_bf16, (size_t)hidden * sizeof(uint16_t)) ||
        !ensure_buffer(&g_dec_ffn, &g_cap_dec_ffn, (size_t)dim * sizeof(float)) ||
        !ensure_buffer(&g_dec_rope_freqs, &g_cap_dec_rope, bytes_rope) ||
        !ensure_buffer(&g_dec_logits, &g_cap_dec_logits, (size_t)VOX_VOCAB_SIZE * sizeof(float)) ||
        !ensure_buffer(&g_dec_best, &g_cap_dec_best, sizeof(int))) {
        return 0;
    }

    if (!g_dec_pos_dev) {
        if (cuMemAlloc(&g_dec_pos_dev, sizeof(int)) != CUDA_SUCCESS) return 0;
    }

    /* Warm up cuBLASLt algo cache + workspace so capture doesn't allocate. */
    if (g_lt_handle) {
        struct {
            int M, K, N;
        } shapes[] = {
            { 1, dim, q_dim },
            { 1, dim, kv_dim },
            { 1, q_dim, dim },
            { 1, dim, hidden },
            { 1, hidden, dim },
            { 1, dim, VOX_VOCAB_SIZE },
        };
        size_t max_ws = 0;
        for (int i = 0; i < (int)(sizeof(shapes) / sizeof(shapes[0])); i++) {
            cublasLtMatmulAlgo_t algo;
            size_t ws = 0;
            cublasLtMatmulDesc_t op = NULL;
            cublasLtMatrixLayout_t a = NULL, b = NULL, c = NULL;
            if (lt_get_algo_t_bf16(shapes[i].M, shapes[i].K, shapes[i].N,
                                   &algo, &ws, &op, &a, &b, &c)) {
                if (ws > max_ws) max_ws = ws;
            }
        }
        if (max_ws > 0 && !ensure_lt_workspace(max_ws)) return 0;
    }

    /* Warm weight caches; graphs require stable device pointers. If we evict
     * while warming, disable graphs (memory pressure => pointers may not stay stable). */
    uint64_t ev_before = g_bf16_evictions;
    vox_decoder_t *dec = &ctx->decoder;
    for (int layer = 0; layer < VOX_DEC_LAYERS; layer++) {
        vox_dec_layer_t *l = &dec->layers[layer];
        size_t bytes_wq = (size_t)q_dim * (size_t)dim * sizeof(uint16_t);
        size_t bytes_wkv = (size_t)kv_dim * (size_t)dim * sizeof(uint16_t);
        size_t bytes_wo = (size_t)dim * (size_t)q_dim * sizeof(uint16_t);
        size_t bytes_w1 = (size_t)hidden * (size_t)dim * sizeof(uint16_t);
        size_t bytes_w2 = (size_t)dim * (size_t)hidden * sizeof(uint16_t);
        if (!bf16_cache_get(l->wq_weight_bf16, bytes_wq)) return 0;
        if (!bf16_cache_get(l->wk_weight_bf16, bytes_wkv)) return 0;
        if (!bf16_cache_get(l->wv_weight_bf16, bytes_wkv)) return 0;
        if (!bf16_cache_get(l->wo_weight_bf16, bytes_wo)) return 0;
        if (!bf16_cache_get(l->w1_weight_bf16, bytes_w1)) return 0;
        if (!bf16_cache_get(l->w3_weight_bf16, bytes_w1)) return 0;
        if (!bf16_cache_get(l->w2_weight_bf16, bytes_w2)) return 0;

        if (!f32_cache_get(l->attention_norm, (size_t)dim * sizeof(float))) return 0;
        if (!f32_cache_get(l->ffn_norm, (size_t)dim * sizeof(float))) return 0;
        if (ctx->ada_scale) {
            const float *ada = ctx->ada_scale + (size_t)layer * (size_t)dim;
            if (!f32_cache_get(ada, (size_t)dim * sizeof(float))) return 0;
        }
    }
    if (!f32_cache_get(dec->norm, (size_t)dim * sizeof(float))) return 0;
    size_t bytes_tok = (size_t)VOX_VOCAB_SIZE * (size_t)dim * sizeof(uint16_t);
    if (!bf16_cache_get(dec->tok_embeddings_bf16, bytes_tok)) return 0;

    if (g_bf16_evictions != ev_before) return 0;
    return 1;
}

static int decoder_graph_capture(vox_ctx_t *ctx) {
    if (!ctx) return 0;
    if (!vox_cuda_available()) return 0;
    if (!cuda_load_kernel_module()) return 0;

    if (g_dec_graph_exec && g_dec_graph_ready) return 1;

    int want_fp16 = kv_cache_use_fp16();
    if (want_fp16) {
        if (!g_fn_kv_append_dyn_fp16 || !g_fn_attn_dyn_fp16) return 0;
    } else {
        if (!g_fn_kv_append_dyn_f32 || !g_fn_attn_dyn_f32) return 0;
    }

    (void)cuCtxSetCurrent(g_ctx);

    int dim = VOX_DEC_DIM;
    int n_heads = VOX_DEC_HEADS;
    int n_kv_heads = VOX_DEC_KV_HEADS;
    int head_dim = VOX_DEC_HEAD_DIM;
    int hidden = VOX_DEC_HIDDEN;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;

    vox_decoder_t *dec = &ctx->decoder;

    CUdeviceptr d_attn_norm[VOX_DEC_LAYERS];
    CUdeviceptr d_ffn_norm[VOX_DEC_LAYERS];
    CUdeviceptr dWq[VOX_DEC_LAYERS];
    CUdeviceptr dWk[VOX_DEC_LAYERS];
    CUdeviceptr dWv[VOX_DEC_LAYERS];
    CUdeviceptr dWo[VOX_DEC_LAYERS];
    CUdeviceptr dW1[VOX_DEC_LAYERS];
    CUdeviceptr dW3[VOX_DEC_LAYERS];
    CUdeviceptr dW2[VOX_DEC_LAYERS];
    CUdeviceptr dAda[VOX_DEC_LAYERS];
    memset(dAda, 0, sizeof(dAda));

    for (int layer = 0; layer < VOX_DEC_LAYERS; layer++) {
        vox_dec_layer_t *l = &dec->layers[layer];

        d_attn_norm[layer] = f32_cache_get(l->attention_norm, (size_t)dim * sizeof(float));
        d_ffn_norm[layer] = f32_cache_get(l->ffn_norm, (size_t)dim * sizeof(float));
        if (!d_attn_norm[layer] || !d_ffn_norm[layer]) return 0;

        size_t bytes_wq = (size_t)q_dim * (size_t)dim * sizeof(uint16_t);
        size_t bytes_wkv = (size_t)kv_dim * (size_t)dim * sizeof(uint16_t);
        size_t bytes_wo = (size_t)dim * (size_t)q_dim * sizeof(uint16_t);
        size_t bytes_w1 = (size_t)hidden * (size_t)dim * sizeof(uint16_t);
        size_t bytes_w2 = (size_t)dim * (size_t)hidden * sizeof(uint16_t);
        dWq[layer] = bf16_cache_get(l->wq_weight_bf16, bytes_wq);
        dWk[layer] = bf16_cache_get(l->wk_weight_bf16, bytes_wkv);
        dWv[layer] = bf16_cache_get(l->wv_weight_bf16, bytes_wkv);
        dWo[layer] = bf16_cache_get(l->wo_weight_bf16, bytes_wo);
        dW1[layer] = bf16_cache_get(l->w1_weight_bf16, bytes_w1);
        dW3[layer] = bf16_cache_get(l->w3_weight_bf16, bytes_w1);
        dW2[layer] = bf16_cache_get(l->w2_weight_bf16, bytes_w2);
        if (!dWq[layer] || !dWk[layer] || !dWv[layer] || !dWo[layer] ||
            !dW1[layer] || !dW3[layer] || !dW2[layer]) return 0;

        if (ctx->ada_scale) {
            const float *ada = ctx->ada_scale + (size_t)layer * (size_t)dim;
            dAda[layer] = f32_cache_get(ada, (size_t)dim * sizeof(float));
            if (!dAda[layer]) return 0;
        }
    }

    CUdeviceptr d_norm = f32_cache_get(dec->norm, (size_t)dim * sizeof(float));
    if (!d_norm) return 0;
    size_t bytes_tok = (size_t)VOX_VOCAB_SIZE * (size_t)dim * sizeof(uint16_t);
    CUdeviceptr dTok = bf16_cache_get(dec->tok_embeddings_bf16, bytes_tok);
    if (!dTok) return 0;

    /* Graph capture uses dynamic `pos` stored on device. */
    int zero = 0;
    if (cuMemcpyHtoDAsync(g_dec_pos_dev, &zero, sizeof(zero), g_stream) != CUDA_SUCCESS) return 0;

    CUresult rr;
    rr = cuStreamBeginCapture(g_stream, CU_STREAM_CAPTURE_MODE_GLOBAL);
    if (rr != CUDA_SUCCESS) { log_cu_error("cuStreamBeginCapture(decoder)", rr); return 0; }

    float attn_scale = 1.0f / sqrtf((float)head_dim);
    int window_size = VOX_DEC_WINDOW;
    int threads = 256;
    int blocks_kv = (kv_dim + threads - 1) / threads;
    int use_v2 = attn_v2_enabled();

    size_t eb = g_kv_elem_bytes ? g_kv_elem_bytes : sizeof(float);
    size_t layer_stride = (size_t)g_kv_max_seq * (size_t)kv_dim * eb;

    for (int layer = 0; layer < VOX_DEC_LAYERS; layer++) {
        CUdeviceptr k_base = g_k_cache + (size_t)layer * layer_stride;
        CUdeviceptr v_base = g_v_cache + (size_t)layer * layer_stride;

        /* Attention norm */
        if (!launch_rms_norm_to_bf16(g_dec_x_bf16, g_dec_x, d_attn_norm[layer], 1, dim, VOX_DEC_NORM_EPS)) {
            if (!launch_rms_norm(g_dec_x_norm, g_dec_x, d_attn_norm[layer], 1, dim, VOX_DEC_NORM_EPS)) goto capture_fail;
            if (!launch_f32_to_bf16(g_dec_x_bf16, g_dec_x_norm, dim)) goto capture_fail;
        }

        /* Q,K,V projections */
        if (!gemm_t_bf16_bf16_f32(g_dec_q, g_dec_x_bf16, dWq[layer], 1, dim, q_dim)) goto capture_fail;
        if (!gemm_t_bf16_bf16_f32(g_dec_k, g_dec_x_bf16, dWk[layer], 1, dim, kv_dim)) goto capture_fail;
        if (!gemm_t_bf16_bf16_f32(g_dec_v, g_dec_x_bf16, dWv[layer], 1, dim, kv_dim)) goto capture_fail;

        /* RoPE */
        if (!launch_apply_rope(g_dec_q, g_dec_rope_freqs, 1, n_heads, head_dim)) goto capture_fail;
        if (!launch_apply_rope(g_dec_k, g_dec_rope_freqs, 1, n_kv_heads, head_dim)) goto capture_fail;

        /* Append KV at dynamic pos, then attention reads total_seq = pos+1. */
        if (want_fp16) {
            void *kv_params[] = { &k_base, &v_base, &g_dec_k, &g_dec_v, &g_dec_pos_dev, &kv_dim };
            rr = cuLaunchKernel(g_fn_kv_append_dyn_fp16,
                                blocks_kv, 1, 1,
                                threads, 1, 1,
                                0, g_stream, kv_params, NULL);
        } else {
            void *kv_params[] = { &k_base, &v_base, &g_dec_k, &g_dec_v, &g_dec_pos_dev, &kv_dim };
            rr = cuLaunchKernel(g_fn_kv_append_dyn_f32,
                                blocks_kv, 1, 1,
                                threads, 1, 1,
                                0, g_stream, kv_params, NULL);
        }
        if (rr != CUDA_SUCCESS) { log_cu_error("cuLaunchKernel(kv_append_dyn)", rr); goto capture_fail; }

        void *attn_params[] = { &g_dec_attn, &g_dec_q, &k_base, &v_base, &g_dec_pos_dev, &window_size, &attn_scale };
        CUfunction fn_attn = 0;
        if (want_fp16) {
            fn_attn = (use_v2 && g_fn_attn_dyn_fp16_v2) ? g_fn_attn_dyn_fp16_v2 : g_fn_attn_dyn_fp16;
        } else {
            fn_attn = (use_v2 && g_fn_attn_dyn_f32_v2) ? g_fn_attn_dyn_f32_v2 : g_fn_attn_dyn_f32;
        }
        rr = cuLaunchKernel(fn_attn,
                            VOX_DEC_HEADS, 1, 1,
                            32, 1, 1,
                            0, g_stream, attn_params, NULL);
        if (rr != CUDA_SUCCESS) { log_cu_error("cuLaunchKernel(attn_dyn)", rr); goto capture_fail; }

        /* Output projection + residual */
        if (!launch_f32_to_bf16(g_dec_attn_bf16, g_dec_attn, q_dim)) goto capture_fail;
        if (!gemm_t_bf16_bf16_f32(g_dec_proj, g_dec_attn_bf16, dWo[layer], 1, q_dim, dim)) goto capture_fail;
        if (!launch_add_inplace(g_dec_x, g_dec_proj, dim)) goto capture_fail;

        /* FFN */
        if (!launch_rms_norm(g_dec_x_norm, g_dec_x, d_ffn_norm[layer], 1, dim, VOX_DEC_NORM_EPS)) goto capture_fail;
        if (dAda[layer]) {
            if (!launch_mul_1p_inplace(g_dec_x_norm, dAda[layer], dim)) goto capture_fail;
        }
        if (!launch_f32_to_bf16(g_dec_x_bf16, g_dec_x_norm, dim)) goto capture_fail;

        if (!gemm_t_bf16_bf16_f32(g_dec_gate, g_dec_x_bf16, dW1[layer], 1, dim, hidden)) goto capture_fail;
        if (!launch_silu_inplace(g_dec_gate, hidden)) goto capture_fail;
        if (!gemm_t_bf16_bf16_f32(g_dec_up, g_dec_x_bf16, dW3[layer], 1, dim, hidden)) goto capture_fail;
        if (!launch_mul_inplace(g_dec_gate, g_dec_up, hidden)) goto capture_fail;

        if (!launch_f32_to_bf16(g_dec_gate_bf16, g_dec_gate, hidden)) goto capture_fail;
        if (!gemm_t_bf16_bf16_f32(g_dec_ffn, g_dec_gate_bf16, dW2[layer], 1, hidden, dim)) goto capture_fail;
        if (!launch_add_inplace(g_dec_x, g_dec_ffn, dim)) goto capture_fail;
    }

    /* Final norm + logits + argmax */
    if (!launch_rms_norm(g_dec_x, g_dec_x, d_norm, 1, dim, VOX_DEC_NORM_EPS)) goto capture_fail;
    if (!launch_f32_to_bf16(g_dec_x_bf16, g_dec_x, dim)) goto capture_fail;
    if (!gemm_t_bf16_bf16_f32(g_dec_logits, g_dec_x_bf16, dTok, 1, dim, VOX_VOCAB_SIZE)) goto capture_fail;
    if (!launch_argmax(g_dec_best, g_dec_logits, VOX_VOCAB_SIZE)) goto capture_fail;

    rr = cuStreamEndCapture(g_stream, &g_dec_graph);
    if (rr != CUDA_SUCCESS) { log_cu_error("cuStreamEndCapture(decoder)", rr); goto capture_fail_destroy; }

    rr = cuGraphInstantiate(&g_dec_graph_exec, g_dec_graph, 0);
    if (rr != CUDA_SUCCESS) { log_cu_error("cuGraphInstantiate(decoder)", rr); goto capture_fail_destroy; }

    (void)cuGraphDestroy(g_dec_graph);
    g_dec_graph = 0;
    g_dec_graph_ready = 1;
    g_dec_graph_kv_fp16 = want_fp16;
    if (vox_verbose >= 1) {
        int have_v2 = want_fp16 ? (g_fn_attn_dyn_fp16_v2 != 0) : (g_fn_attn_dyn_f32_v2 != 0);
        fprintf(stderr, "[cuda] decoder graph captured (kv_cache=%s, attn=%s)\n",
                want_fp16 ? "fp16" : "fp32",
                (use_v2 && have_v2) ? "v2" : "v1");
    }
    return 1;

capture_fail:
    (void)cuStreamEndCapture(g_stream, &g_dec_graph);
capture_fail_destroy:
    decoder_graph_destroy();
    return 0;
}

static int vox_cuda_decoder_forward_full_graph(int *out_token,
                                               float *logits_or_null,
                                               vox_ctx_t *ctx,
                                               const float *input_embeds) {
    if (!out_token || !ctx || !input_embeds) return 0;
    if (!decoder_graph_wanted()) return 0;
    if (!vox_cuda_available()) return 0;
    if (!cuda_load_kernel_module()) return 0;

    int want_fp16 = kv_cache_use_fp16();
    if (g_dec_graph_ready && g_dec_graph_kv_fp16 != -1 && g_dec_graph_kv_fp16 != want_fp16) {
        decoder_graph_destroy();
    }

    if (!g_dec_graph_ready) {
        if (!decoder_graph_prepare(ctx)) return 0;
        if (!decoder_graph_capture(ctx)) return 0;
    }
    if (!g_dec_graph_exec) return 0;

    (void)cuCtxSetCurrent(g_ctx);

    int dim = VOX_DEC_DIM;
    int head_dim = VOX_DEC_HEAD_DIM;

    int pos = ctx->kv_cache_len;
    int total_seq = pos + 1;

    /* Upload step embedding + RoPE + pos scalar; then launch the captured graph. */
    CUresult r;
    r = cuMemcpyHtoDAsync(g_dec_x, input_embeds, (size_t)dim * sizeof(float), g_stream);
    if (r != CUDA_SUCCESS) { log_cu_error("HtoD(dec_x_graph)", r); return 0; }

    int logical_pos = ctx->kv_pos_offset + pos;
    int positions[1] = { logical_pos };
    float rope_host[(VOX_DEC_HEAD_DIM / 2) * 2];
    vox_compute_rope_freqs(rope_host, positions, 1, head_dim, VOX_ROPE_THETA);
    r = cuMemcpyHtoDAsync(g_dec_rope_freqs, rope_host, sizeof(rope_host), g_stream);
    if (r != CUDA_SUCCESS) { log_cu_error("HtoD(dec_rope_graph)", r); return 0; }

    r = cuMemcpyHtoDAsync(g_dec_pos_dev, &pos, sizeof(pos), g_stream);
    if (r != CUDA_SUCCESS) { log_cu_error("HtoD(dec_pos_graph)", r); return 0; }

    r = cuGraphLaunch(g_dec_graph_exec, g_stream);
    if (r != CUDA_SUCCESS) { log_cu_error("cuGraphLaunch(decoder)", r); return 0; }

    int best = 2;
    r = cuMemcpyDtoHAsync(&best, g_dec_best, sizeof(best), g_stream);
    if (r != CUDA_SUCCESS) { log_cu_error("DtoH(best_graph)", r); return 0; }

    if (logits_or_null) {
        r = cuMemcpyDtoHAsync(logits_or_null, g_dec_logits, (size_t)VOX_VOCAB_SIZE * sizeof(float), g_stream);
        if (r != CUDA_SUCCESS) { log_cu_error("DtoH(logits_graph)", r); return 0; }
    }

    r = cuStreamSynchronize(g_stream);
    if (r != CUDA_SUCCESS) { log_cu_error("sync(decoder_graph)", r); return 0; }

    ctx->kv_cache_len = total_seq;
    *out_token = best;
    return 1;
}

int vox_cuda_decoder_forward_full(int *out_token,
                                  float *logits_or_null,
                                  vox_ctx_t *ctx,
                                  const float *input_embeds) {
    if (!out_token) return 0;
    *out_token = 2;
    if (!vox_cuda_available()) return 0;
    const char *disable = getenv("VOX_DISABLE_CUDA_DECODER_FULL");
    if (disable && disable[0] && disable[0] != '0') return 0;
    if (!ctx || !input_embeds) return 0;
    if (!cuda_load_kernel_module()) return 0;

    /* Optional CUDA Graph fast path (opt-in). */
    if (vox_cuda_decoder_forward_full_graph(out_token, logits_or_null, ctx, input_embeds)) {
        return 1;
    }

    (void)cuCtxSetCurrent(g_ctx);

    int dim = VOX_DEC_DIM;
    int n_heads = VOX_DEC_HEADS;
    int n_kv_heads = VOX_DEC_KV_HEADS;
    int head_dim = VOX_DEC_HEAD_DIM;
    int hidden = VOX_DEC_HIDDEN;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;

    int pos = ctx->kv_cache_len;
    int total_seq = pos + 1;

    /* Ensure device KV cache is ready (prefill already uploaded blocks). */
    int want_max_seq = ctx->kv_cache_max > 0 ? ctx->kv_cache_max : (VOX_DEC_WINDOW + 2048);
    if (!ensure_kv_cache(want_max_seq, kv_dim)) return 0;

    /* Upload step embedding */
    if (!ensure_buffer(&g_dec_x, &g_cap_dec_x, (size_t)dim * sizeof(float)) ||
        !ensure_buffer(&g_dec_x_norm, &g_cap_dec_x_norm, (size_t)dim * sizeof(float)) ||
        !ensure_buffer(&g_dec_x_bf16, &g_cap_dec_x_bf16, (size_t)dim * sizeof(uint16_t)) ||
        !ensure_buffer(&g_dec_q, &g_cap_dec_q, (size_t)q_dim * sizeof(float)) ||
        !ensure_buffer(&g_dec_k, &g_cap_dec_k, (size_t)kv_dim * sizeof(float)) ||
        !ensure_buffer(&g_dec_v, &g_cap_dec_v, (size_t)kv_dim * sizeof(float)) ||
        !ensure_buffer(&g_dec_attn, &g_cap_dec_attn, (size_t)q_dim * sizeof(float)) ||
        !ensure_buffer(&g_dec_attn_bf16, &g_cap_dec_attn_bf16, (size_t)q_dim * sizeof(uint16_t)) ||
        !ensure_buffer(&g_dec_proj, &g_cap_dec_proj, (size_t)dim * sizeof(float)) ||
        !ensure_buffer(&g_dec_gate, &g_cap_dec_gate, (size_t)hidden * sizeof(float)) ||
        !ensure_buffer(&g_dec_up, &g_cap_dec_up, (size_t)hidden * sizeof(float)) ||
        !ensure_buffer(&g_dec_gate_bf16, &g_cap_dec_gate_bf16, (size_t)hidden * sizeof(uint16_t)) ||
        !ensure_buffer(&g_dec_ffn, &g_cap_dec_ffn, (size_t)dim * sizeof(float))) {
        return 0;
    }

    CUresult r;
    r = cuMemcpyHtoDAsync(g_dec_x, input_embeds, (size_t)dim * sizeof(float), g_stream);
    if (r != CUDA_SUCCESS) { log_cu_error("HtoD(dec_x)", r); return 0; }

    /* RoPE freqs for this position */
    int logical_pos = ctx->kv_pos_offset + pos;
    int positions[1] = { logical_pos };
    float rope_host[(VOX_DEC_HEAD_DIM / 2) * 2];
    vox_compute_rope_freqs(rope_host, positions, 1, head_dim, VOX_ROPE_THETA);
    if (!ensure_buffer(&g_dec_rope_freqs, &g_cap_dec_rope, sizeof(rope_host))) return 0;
    r = cuMemcpyHtoDAsync(g_dec_rope_freqs, rope_host, sizeof(rope_host), g_stream);
    if (r != CUDA_SUCCESS) { log_cu_error("HtoD(dec_rope)", r); return 0; }

    vox_decoder_t *dec = &ctx->decoder;

    for (int layer = 0; layer < VOX_DEC_LAYERS; layer++) {
        vox_dec_layer_t *l = &dec->layers[layer];

        CUdeviceptr d_attn_norm = f32_cache_get(l->attention_norm, (size_t)dim * sizeof(float));
        CUdeviceptr d_ffn_norm = f32_cache_get(l->ffn_norm, (size_t)dim * sizeof(float));
        if (!d_attn_norm || !d_ffn_norm) return 0;

        /* Attention norm */
        if (!launch_rms_norm_to_bf16(g_dec_x_bf16, g_dec_x, d_attn_norm, 1, dim, VOX_DEC_NORM_EPS)) {
            if (!launch_rms_norm(g_dec_x_norm, g_dec_x, d_attn_norm, 1, dim, VOX_DEC_NORM_EPS)) return 0;
            if (!launch_f32_to_bf16(g_dec_x_bf16, g_dec_x_norm, dim)) return 0;
        }

        /* Q,K,V projections */
        size_t bytes_wq = (size_t)q_dim * (size_t)dim * sizeof(uint16_t);
        size_t bytes_wkv = (size_t)kv_dim * (size_t)dim * sizeof(uint16_t);
        CUdeviceptr dWq = bf16_cache_get(l->wq_weight_bf16, bytes_wq);
        CUdeviceptr dWk = bf16_cache_get(l->wk_weight_bf16, bytes_wkv);
        CUdeviceptr dWv = bf16_cache_get(l->wv_weight_bf16, bytes_wkv);
        if (!dWq || !dWk || !dWv) return 0;

        if (!gemm_t_bf16_bf16_f32(g_dec_q, g_dec_x_bf16, dWq, 1, dim, q_dim)) return 0;
        if (!gemm_t_bf16_bf16_f32(g_dec_k, g_dec_x_bf16, dWk, 1, dim, kv_dim)) return 0;
        if (!gemm_t_bf16_bf16_f32(g_dec_v, g_dec_x_bf16, dWv, 1, dim, kv_dim)) return 0;

        /* RoPE */
        if (!launch_apply_rope(g_dec_q, g_dec_rope_freqs, 1, n_heads, head_dim)) return 0;
        if (!launch_apply_rope(g_dec_k, g_dec_rope_freqs, 1, n_kv_heads, head_dim)) return 0;

        /* Attention */
        if (!vox_cuda_decoder_attention_step_dev(g_dec_attn, g_dec_q, g_dec_k, g_dec_v,
                                                 layer, pos, total_seq, VOX_DEC_WINDOW)) {
            return 0;
        }

        /* Output projection */
        size_t bytes_wo = (size_t)dim * (size_t)q_dim * sizeof(uint16_t);
        CUdeviceptr dWo = bf16_cache_get(l->wo_weight_bf16, bytes_wo);
        if (!dWo) return 0;
        if (!launch_f32_to_bf16(g_dec_attn_bf16, g_dec_attn, q_dim)) return 0;
        if (!gemm_t_bf16_bf16_f32(g_dec_proj, g_dec_attn_bf16, dWo, 1, q_dim, dim)) return 0;
        if (!launch_add_inplace(g_dec_x, g_dec_proj, dim)) return 0;

        /* FFN */
        if (!launch_rms_norm(g_dec_x_norm, g_dec_x, d_ffn_norm, 1, dim, VOX_DEC_NORM_EPS)) return 0;
        if (ctx->ada_scale) {
            const float *ada = ctx->ada_scale + (size_t)layer * (size_t)dim;
            CUdeviceptr d_ada = f32_cache_get(ada, (size_t)dim * sizeof(float));
            if (!d_ada) return 0;
            if (!launch_mul_1p_inplace(g_dec_x_norm, d_ada, dim)) return 0;
        }

        if (!launch_f32_to_bf16(g_dec_x_bf16, g_dec_x_norm, dim)) return 0;

        size_t bytes_w1 = (size_t)hidden * (size_t)dim * sizeof(uint16_t);
        CUdeviceptr dW1 = bf16_cache_get(l->w1_weight_bf16, bytes_w1);
        CUdeviceptr dW3 = bf16_cache_get(l->w3_weight_bf16, bytes_w1);
        if (!dW1 || !dW3) return 0;
        if (!gemm_t_bf16_bf16_f32(g_dec_gate, g_dec_x_bf16, dW1, 1, dim, hidden)) return 0;
        if (!launch_silu_inplace(g_dec_gate, hidden)) return 0;
        if (!gemm_t_bf16_bf16_f32(g_dec_up, g_dec_x_bf16, dW3, 1, dim, hidden)) return 0;
        if (!launch_mul_inplace(g_dec_gate, g_dec_up, hidden)) return 0;

        size_t bytes_w2 = (size_t)dim * (size_t)hidden * sizeof(uint16_t);
        CUdeviceptr dW2 = bf16_cache_get(l->w2_weight_bf16, bytes_w2);
        if (!dW2) return 0;
        if (!launch_f32_to_bf16(g_dec_gate_bf16, g_dec_gate, hidden)) return 0;
        if (!gemm_t_bf16_bf16_f32(g_dec_ffn, g_dec_gate_bf16, dW2, 1, hidden, dim)) return 0;
        if (!launch_add_inplace(g_dec_x, g_dec_ffn, dim)) return 0;
    }

    ctx->kv_cache_len = pos + 1;

    /* Final norm */
    CUdeviceptr d_norm = f32_cache_get(dec->norm, (size_t)dim * sizeof(float));
    if (!d_norm) return 0;
    if (!launch_rms_norm(g_dec_x, g_dec_x, d_norm, 1, dim, VOX_DEC_NORM_EPS)) return 0;

    /* Logits projection */
    if (!ensure_buffer(&g_dec_logits, &g_cap_dec_logits, (size_t)VOX_VOCAB_SIZE * sizeof(float))) return 0;
    if (!ensure_buffer(&g_dec_best, &g_cap_dec_best, sizeof(int))) return 0;

    if (!launch_f32_to_bf16(g_dec_x_bf16, g_dec_x, dim)) return 0;
    size_t bytes_emb = (size_t)VOX_VOCAB_SIZE * (size_t)dim * sizeof(uint16_t);
    CUdeviceptr dTok = bf16_cache_get(dec->tok_embeddings_bf16, bytes_emb);
    if (!dTok) return 0;
    if (!gemm_t_bf16_bf16_f32(g_dec_logits, g_dec_x_bf16, dTok, 1, dim, VOX_VOCAB_SIZE)) return 0;

    if (!launch_argmax(g_dec_best, g_dec_logits, VOX_VOCAB_SIZE)) return 0;

    int best = 2;
    r = cuMemcpyDtoHAsync(&best, g_dec_best, sizeof(best), g_stream);
    if (r != CUDA_SUCCESS) { log_cu_error("DtoH(best)", r); return 0; }

    if (logits_or_null) {
        r = cuMemcpyDtoHAsync(logits_or_null, g_dec_logits, (size_t)VOX_VOCAB_SIZE * sizeof(float), g_stream);
        if (r != CUDA_SUCCESS) { log_cu_error("DtoH(logits)", r); return 0; }
    }

    r = cuStreamSynchronize(g_stream);
    if (r != CUDA_SUCCESS) { log_cu_error("sync(decoder_full)", r); return 0; }

    *out_token = best;
    return 1;
}

int vox_cuda_decoder_prefill_full(vox_ctx_t *ctx,
                                  const float *input_embeds,
                                  int seq_len,
                                  const float *rope_freqs) {
    if (!vox_cuda_available()) return 0;
    const char *disable = getenv("VOX_DISABLE_CUDA_PREFILL");
    if (disable && disable[0] && disable[0] != '0') return 0;
    if (!ctx || !input_embeds || !rope_freqs) return 0;
    if (seq_len <= 0) return 0;
    if (!ctx->kv_cache_k || !ctx->kv_cache_v) return 0;
    if (!cuda_load_kernel_module()) return 0;

    /* Ensure our primary context is current on this thread. */
    (void)cuCtxSetCurrent(g_ctx);

    int dim = VOX_DEC_DIM;
    int n_heads = VOX_DEC_HEADS;
    int n_kv_heads = VOX_DEC_KV_HEADS;
    int head_dim = VOX_DEC_HEAD_DIM;
    int hidden = VOX_DEC_HIDDEN;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;

    int start_pos = ctx->kv_cache_len;
    if (start_pos != 0) {
        /* Keep first implementation simple: we only support prefill from an
         * empty cache (the streaming path resets ctx->kv_cache_len=0). */
        return 0;
    }

    int want_max_seq = ctx->kv_cache_max > 0 ? ctx->kv_cache_max : (VOX_DEC_WINDOW + 2048);
    if (!ensure_kv_cache(want_max_seq, kv_dim)) return 0;

    /* Resize work buffers for seq_len. */
    size_t bytes_x = (size_t)seq_len * (size_t)dim * sizeof(float);
    size_t bytes_x_bf16 = (size_t)seq_len * (size_t)dim * sizeof(uint16_t);
    size_t bytes_q = (size_t)seq_len * (size_t)q_dim * sizeof(float);
    size_t bytes_kv = (size_t)seq_len * (size_t)kv_dim * sizeof(float);
    size_t bytes_attn = bytes_q;
    size_t bytes_attn_bf16 = (size_t)seq_len * (size_t)q_dim * sizeof(uint16_t);
    size_t bytes_gate = (size_t)seq_len * (size_t)hidden * sizeof(float);
    size_t bytes_gate_bf16 = (size_t)seq_len * (size_t)hidden * sizeof(uint16_t);
    size_t bytes_rope = (size_t)seq_len * (size_t)((head_dim / 2) * 2) * sizeof(float);

    if (!ensure_buffer(&g_dec_x, &g_cap_dec_x, bytes_x) ||
        !ensure_buffer(&g_dec_x_norm, &g_cap_dec_x_norm, bytes_x) ||
        !ensure_buffer(&g_dec_x_bf16, &g_cap_dec_x_bf16, bytes_x_bf16) ||
        !ensure_buffer(&g_dec_q, &g_cap_dec_q, bytes_q) ||
        !ensure_buffer(&g_dec_k, &g_cap_dec_k, bytes_kv) ||
        !ensure_buffer(&g_dec_v, &g_cap_dec_v, bytes_kv) ||
        !ensure_buffer(&g_dec_attn, &g_cap_dec_attn, bytes_attn) ||
        !ensure_buffer(&g_dec_attn_bf16, &g_cap_dec_attn_bf16, bytes_attn_bf16) ||
        !ensure_buffer(&g_dec_proj, &g_cap_dec_proj, bytes_x) ||
        !ensure_buffer(&g_dec_gate, &g_cap_dec_gate, bytes_gate) ||
        !ensure_buffer(&g_dec_up, &g_cap_dec_up, bytes_gate) ||
        !ensure_buffer(&g_dec_gate_bf16, &g_cap_dec_gate_bf16, bytes_gate_bf16) ||
        !ensure_buffer(&g_dec_ffn, &g_cap_dec_ffn, bytes_x) ||
        !ensure_buffer(&g_dec_rope_freqs, &g_cap_dec_rope, bytes_rope)) {
        return 0;
    }

    CUresult r;
    r = cuMemcpyHtoDAsync(g_dec_x, input_embeds, bytes_x, g_stream);
    if (r != CUDA_SUCCESS) { log_cu_error("HtoD(dec_prefill_x)", r); return 0; }

    r = cuMemcpyHtoDAsync(g_dec_rope_freqs, rope_freqs, bytes_rope, g_stream);
    if (r != CUDA_SUCCESS) { log_cu_error("HtoD(dec_prefill_rope)", r); return 0; }

    static int logged = 0;
    if (!logged && vox_verbose >= 1) {
        fprintf(stderr, "[cuda] decoder prefill enabled (seq_len=%d)\n", seq_len);
        logged = 1;
    }

    float attn_scale = 1.0f / sqrtf((float)head_dim);
    vox_decoder_t *dec = &ctx->decoder;

    for (int layer = 0; layer < VOX_DEC_LAYERS; layer++) {
        vox_dec_layer_t *l = &dec->layers[layer];

        CUdeviceptr d_attn_norm = f32_cache_get(l->attention_norm, (size_t)dim * sizeof(float));
        CUdeviceptr d_ffn_norm = f32_cache_get(l->ffn_norm, (size_t)dim * sizeof(float));
        if (!d_attn_norm || !d_ffn_norm) return 0;

        /* ---- Self-attention ---- */
        if (!launch_rms_norm_to_bf16(g_dec_x_bf16, g_dec_x, d_attn_norm, seq_len, dim, VOX_DEC_NORM_EPS)) {
            if (!launch_rms_norm(g_dec_x_norm, g_dec_x, d_attn_norm, seq_len, dim, VOX_DEC_NORM_EPS)) return 0;
            if (!launch_f32_to_bf16(g_dec_x_bf16, g_dec_x_norm, seq_len * dim)) return 0;
        }

        /* Q, K, V projections (no bias in decoder, bf16 weights) */
        size_t bytes_wq = (size_t)q_dim * (size_t)dim * sizeof(uint16_t);
        size_t bytes_wkv = (size_t)kv_dim * (size_t)dim * sizeof(uint16_t);
        CUdeviceptr dWq = bf16_cache_get(l->wq_weight_bf16, bytes_wq);
        CUdeviceptr dWk = bf16_cache_get(l->wk_weight_bf16, bytes_wkv);
        CUdeviceptr dWv = bf16_cache_get(l->wv_weight_bf16, bytes_wkv);
        if (!dWq || !dWk || !dWv) return 0;

        if (!gemm_t_bf16_bf16_f32(g_dec_q, g_dec_x_bf16, dWq, seq_len, dim, q_dim)) return 0;
        if (!gemm_t_bf16_bf16_f32(g_dec_k, g_dec_x_bf16, dWk, seq_len, dim, kv_dim)) return 0;
        if (!gemm_t_bf16_bf16_f32(g_dec_v, g_dec_x_bf16, dWv, seq_len, dim, kv_dim)) return 0;

        /* Apply RoPE */
        if (!launch_apply_rope(g_dec_q, g_dec_rope_freqs, seq_len, n_heads, head_dim)) return 0;
        if (!launch_apply_rope(g_dec_k, g_dec_rope_freqs, seq_len, n_kv_heads, head_dim)) return 0;

        /* Store K, V in device KV cache for the upcoming single-token decode loop. */
        if (!vox_cuda_kv_cache_append_block_dev(layer, start_pos, seq_len, kv_dim, VOX_DEC_WINDOW,
                                                g_dec_k, g_dec_v)) {
            return 0;
        }

        /* Keep host KV cache in sync (for CPU fallback and for compactions). */
        size_t host_stride = (size_t)ctx->kv_cache_max * (size_t)kv_dim;
        float *hk = ctx->kv_cache_k + (size_t)layer * host_stride + (size_t)start_pos * (size_t)kv_dim;
        float *hv = ctx->kv_cache_v + (size_t)layer * host_stride + (size_t)start_pos * (size_t)kv_dim;
        size_t hv_bytes = (size_t)seq_len * (size_t)kv_dim * sizeof(float);
        r = cuMemcpyDtoHAsync(hk, g_dec_k, hv_bytes, g_stream);
        if (r != CUDA_SUCCESS) { log_cu_error("DtoH(dec_prefill_k)", r); return 0; }
        r = cuMemcpyDtoHAsync(hv, g_dec_v, hv_bytes, g_stream);
        if (r != CUDA_SUCCESS) { log_cu_error("DtoH(dec_prefill_v)", r); return 0; }

        /* Causal attention over the full cached sequence (here: start_pos=0) */
        int total_seq = start_pos + seq_len;
        if (!vox_cuda_causal_attention_dev(g_dec_attn, g_dec_q, g_dec_k, g_dec_v,
                                           seq_len, total_seq, n_heads, n_kv_heads,
                                           head_dim, attn_scale, VOX_DEC_WINDOW, start_pos)) {
            return 0;
        }

        /* Output projection + residual */
        size_t bytes_wo = (size_t)dim * (size_t)q_dim * sizeof(uint16_t);
        CUdeviceptr dWo = bf16_cache_get(l->wo_weight_bf16, bytes_wo);
        if (!dWo) return 0;

        if (!launch_f32_to_bf16(g_dec_attn_bf16, g_dec_attn, seq_len * q_dim)) return 0;
        if (!gemm_t_bf16_bf16_f32(g_dec_proj, g_dec_attn_bf16, dWo, seq_len, q_dim, dim)) return 0;
        if (!launch_add_inplace(g_dec_x, g_dec_proj, seq_len * dim)) return 0;

        /* ---- FFN ---- */
        if (!launch_rms_norm(g_dec_x_norm, g_dec_x, d_ffn_norm, seq_len, dim, VOX_DEC_NORM_EPS)) return 0;

        if (ctx->ada_scale) {
            const float *ada = ctx->ada_scale + (size_t)layer * (size_t)dim;
            CUdeviceptr d_ada = f32_cache_get(ada, (size_t)dim * sizeof(float));
            if (!d_ada) return 0;
            if (!launch_mul_1p_rows_inplace(g_dec_x_norm, d_ada, seq_len, dim)) {
                /* Fallback: per-row kernel launch. */
                for (int s = 0; s < seq_len; s++) {
                    CUdeviceptr row = g_dec_x_norm + (size_t)s * (size_t)dim * sizeof(float);
                    if (!launch_mul_1p_inplace(row, d_ada, dim)) return 0;
                }
            }
        }

        if (!launch_f32_to_bf16(g_dec_x_bf16, g_dec_x_norm, seq_len * dim)) return 0;

        size_t bytes_w1 = (size_t)hidden * (size_t)dim * sizeof(uint16_t);
        CUdeviceptr dW1 = bf16_cache_get(l->w1_weight_bf16, bytes_w1);
        CUdeviceptr dW3 = bf16_cache_get(l->w3_weight_bf16, bytes_w1);
        if (!dW1 || !dW3) return 0;

        if (!gemm_t_bf16_bf16_f32(g_dec_gate, g_dec_x_bf16, dW1, seq_len, dim, hidden)) return 0;
        if (!launch_silu_inplace(g_dec_gate, seq_len * hidden)) return 0;
        if (!gemm_t_bf16_bf16_f32(g_dec_up, g_dec_x_bf16, dW3, seq_len, dim, hidden)) return 0;
        if (!launch_mul_inplace(g_dec_gate, g_dec_up, seq_len * hidden)) return 0;

        size_t bytes_w2 = (size_t)dim * (size_t)hidden * sizeof(uint16_t);
        CUdeviceptr dW2 = bf16_cache_get(l->w2_weight_bf16, bytes_w2);
        if (!dW2) return 0;

        if (!launch_f32_to_bf16(g_dec_gate_bf16, g_dec_gate, seq_len * hidden)) return 0;
        if (!gemm_t_bf16_bf16_f32(g_dec_ffn, g_dec_gate_bf16, dW2, seq_len, hidden, dim)) return 0;
        if (!launch_add_inplace(g_dec_x, g_dec_ffn, seq_len * dim)) return 0;
    }

    r = cuStreamSynchronize(g_stream);
    if (r != CUDA_SUCCESS) { log_cu_error("sync(dec_prefill)", r); return 0; }

    ctx->kv_cache_len = start_pos + seq_len;
    return 1;
}

#else

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
int vox_cuda_attention_step(float *attn_out,
                            const float *q,
                            const float *k,
                            const float *v,
                            int layer,
                            int pos,
                            int total_seq,
                            int window_size) {
    (void)attn_out; (void)q; (void)k; (void)v; (void)layer; (void)pos; (void)total_seq; (void)window_size;
    return 0;
}
void vox_cuda_kv_cache_compact(int discard, int keep, int kv_dim, int max_seq) {
    (void)discard; (void)keep; (void)kv_dim; (void)max_seq;
}
void vox_cuda_kv_cache_reset(void) {}
void vox_cuda_kv_cache_append_block(int layer, int start_pos, int seq_len,
                                    int kv_dim, int window_size,
                                    const float *k, const float *v) {
    (void)layer; (void)start_pos; (void)seq_len; (void)kv_dim; (void)window_size; (void)k; (void)v;
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
void vox_cuda_shutdown(void) {}

#endif
