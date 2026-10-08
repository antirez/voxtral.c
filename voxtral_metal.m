/*
 * voxtral_metal.m - Metal GPU acceleration for Voxtral inference
 *
 * MPS-accelerated matrix multiplication with bf16->f16 weight caching
 * and activation buffer pooling. Ported from flux-2-4b.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "voxtral_metal.h"
#include "voxtral_shaders_source.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <mach/mach_time.h>

extern int vox_verbose;

/* ========================================================================
 * Global Metal State
 * ======================================================================== */

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static int g_initialized = 0;

/* Compute shader pipelines */
static id<MTLLibrary> g_shader_library = nil;
static id<MTLComputePipelineState> g_rms_norm_pipeline = nil;
static id<MTLComputePipelineState> g_silu_pipeline = nil;
static id<MTLComputePipelineState> g_gelu_pipeline = nil;
static id<MTLComputePipelineState> g_add_inplace_pipeline = nil;
static id<MTLComputePipelineState> g_mul_inplace_pipeline = nil;
static id<MTLComputePipelineState> g_causal_softmax_pipeline = nil;
static id<MTLComputePipelineState> g_ada_scale_mul_pipeline = nil;
static id<MTLComputePipelineState> g_argmax_pipeline = nil;
static int g_shaders_initialized = 0;

/* Persistent GPU x buffer for cross-layer decoder fusion */
static id<MTLBuffer> g_dec_x = nil;

/* New kernels for monolithic decoder step */
static id<MTLComputePipelineState> g_rope_apply_pipeline = nil;
static id<MTLComputePipelineState> g_kv_cache_copy_pipeline = nil;
static id<MTLComputePipelineState> g_kv_cache_copy_f16_pipeline = nil;
static id<MTLComputePipelineState> g_decoder_attention_pipeline = nil;
static id<MTLComputePipelineState> g_decoder_attention_f16_pipeline = nil;
static id<MTLComputePipelineState> g_encoder_attention_pipeline = nil;
static id<MTLComputePipelineState> g_encoder_attention_kv_f16_pipeline = nil;
static id<MTLComputePipelineState> g_bias_add_pipeline = nil;
static id<MTLComputePipelineState> g_bias_add_strided_pipeline = nil;
static id<MTLComputePipelineState> g_batched_rope_apply_pipeline = nil;
static id<MTLComputePipelineState> g_batched_rope_apply_strided_pipeline = nil;
static id<MTLComputePipelineState> g_batched_kv_cache_copy_pipeline = nil;
static id<MTLComputePipelineState> g_batched_kv_cache_copy_f16_pipeline = nil;
static id<MTLComputePipelineState> g_batched_kv_cache_copy_strided_pipeline = nil;
static id<MTLComputePipelineState> g_batched_kv_cache_copy_strided_f16_pipeline = nil;
static id<MTLComputePipelineState> g_encoder_attention_qstrided_pipeline = nil;
static id<MTLComputePipelineState> g_encoder_attention_kv_f16_qstrided_pipeline = nil;
static id<MTLComputePipelineState> g_deinterleave_pipeline = nil;
static id<MTLComputePipelineState> g_silu_mul_merged_pipeline = nil;
static id<MTLComputePipelineState> g_decoder_ffn_gate_pipeline = nil;
static id<MTLComputePipelineState> g_decoder_w2_residual_pipeline = nil;
static id<MTLComputePipelineState> g_decoder_wo_residual_pipeline = nil;

/* GPU-shared memory tracking (zero-copy between CPU and GPU) */
#define SHARED_ALLOC_MAX 8
static struct { void *ptr; id<MTLBuffer> buf; } g_shared_allocs[SHARED_ALLOC_MAX];
static int g_shared_count = 0;

/* ========================================================================
 * BF16 -> F16 Conversion
 * MPS only supports mixed f32/f16 matmul, not f32/bf16.
 * We convert bf16 weights to f16 once and cache the result.
 * ======================================================================== */

static inline uint16_t bf16_to_f16(uint16_t bf16) {
    uint32_t sign = (bf16 >> 15) & 0x1;
    int32_t exp = (bf16 >> 7) & 0xFF;
    uint32_t mant = bf16 & 0x7F;

    if (exp == 0) return (uint16_t)(sign << 15);
    if (exp == 0xFF) return (uint16_t)((sign << 15) | 0x7C00 | (mant ? 0x200 : 0));

    int32_t new_exp = exp - 127 + 15;
    if (new_exp <= 0) return (uint16_t)(sign << 15);
    if (new_exp >= 31) return (uint16_t)((sign << 15) | 0x7C00);

    uint32_t new_mant = mant << 3;
    return (uint16_t)((sign << 15) | (new_exp << 10) | new_mant);
}

/* ========================================================================
 * F16 Weight Cache (bf16 converted to f16, cached by CPU pointer)
 * ======================================================================== */

#define F16_WEIGHT_CACHE_SIZE 512

typedef struct {
    const void *cpu_ptr;
    id<MTLBuffer> gpu_buffer;
    size_t num_elements;
} f16_cache_entry_t;

static f16_cache_entry_t g_f16_cache[F16_WEIGHT_CACHE_SIZE];
static int g_f16_cache_count = 0;
static pthread_mutex_t g_f16_cache_mutex = PTHREAD_MUTEX_INITIALIZER;

static id<MTLBuffer> get_cached_bf16_as_f16_buffer(const uint16_t *weights, size_t num_elements) {
    pthread_mutex_lock(&g_f16_cache_mutex);

    for (int i = 0; i < g_f16_cache_count; i++) {
        if (g_f16_cache[i].cpu_ptr == weights) {
            id<MTLBuffer> buf = g_f16_cache[i].gpu_buffer;
            pthread_mutex_unlock(&g_f16_cache_mutex);
            return buf;
        }
    }

    /* Convert bf16 -> f16 */
    uint16_t *f16_data = (uint16_t *)malloc(num_elements * sizeof(uint16_t));
    if (!f16_data) {
        pthread_mutex_unlock(&g_f16_cache_mutex);
        return nil;
    }
    for (size_t i = 0; i < num_elements; i++) {
        f16_data[i] = bf16_to_f16(weights[i]);
    }

    size_t size = num_elements * sizeof(uint16_t);
    id<MTLBuffer> buf = [g_device newBufferWithBytes:f16_data
                                              length:size
                                             options:MTLResourceStorageModeShared];
    free(f16_data);

    if (buf && g_f16_cache_count < F16_WEIGHT_CACHE_SIZE) {
        g_f16_cache[g_f16_cache_count].cpu_ptr = weights;
        g_f16_cache[g_f16_cache_count].gpu_buffer = buf;
        g_f16_cache[g_f16_cache_count].num_elements = num_elements;
        g_f16_cache_count++;
    }

    pthread_mutex_unlock(&g_f16_cache_mutex);
    return buf;
}

static void clear_f16_cache(void) {
    pthread_mutex_lock(&g_f16_cache_mutex);
    for (int i = 0; i < g_f16_cache_count; i++) {
        g_f16_cache[i].gpu_buffer = nil;
        g_f16_cache[i].cpu_ptr = NULL;
    }
    g_f16_cache_count = 0;
    pthread_mutex_unlock(&g_f16_cache_mutex);
}

/* ========================================================================
 * Merged F16 Weight Cache (concatenate two weight matrices for fused matmul)
 * ======================================================================== */

#define MERGED_CACHE_SIZE 256

typedef struct {
    const void *key1, *key2;
    id<MTLBuffer> buffer;
} merged_cache_entry_t;

static merged_cache_entry_t g_merged_cache[MERGED_CACHE_SIZE];
static int g_merged_count = 0;

/* Concatenate two bf16 weight matrices into a single f16 GPU buffer.
 * Result is [a_rows + b_rows, cols] where a is [a_rows, cols] and b is [b_rows, cols].
 * Cached by the pair of source CPU pointers. */
static id<MTLBuffer> get_merged_f16_2(const uint16_t *bf16_a, size_t a_elems,
                                        const uint16_t *bf16_b, size_t b_elems) {
    for (int i = 0; i < g_merged_count; i++) {
        if (g_merged_cache[i].key1 == bf16_a && g_merged_cache[i].key2 == bf16_b)
            return g_merged_cache[i].buffer;
    }

    id<MTLBuffer> buf_a = get_cached_bf16_as_f16_buffer(bf16_a, a_elems);
    id<MTLBuffer> buf_b = get_cached_bf16_as_f16_buffer(bf16_b, b_elems);
    if (!buf_a || !buf_b) return nil;

    size_t total = (a_elems + b_elems) * sizeof(uint16_t);
    id<MTLBuffer> merged = [g_device newBufferWithLength:total
                                                 options:MTLResourceStorageModeShared];
    if (!merged) return nil;
    memcpy([merged contents], [buf_a contents], a_elems * sizeof(uint16_t));
    memcpy((uint8_t *)[merged contents] + a_elems * sizeof(uint16_t),
           [buf_b contents], b_elems * sizeof(uint16_t));

    if (g_merged_count < MERGED_CACHE_SIZE) {
        g_merged_cache[g_merged_count].key1 = bf16_a;
        g_merged_cache[g_merged_count].key2 = bf16_b;
        g_merged_cache[g_merged_count].buffer = merged;
        g_merged_count++;
    }
    return merged;
}

static id<MTLBuffer> get_merged_f16_3(const uint16_t *bf16_a, size_t a_elems,
                                        const uint16_t *bf16_b, size_t b_elems,
                                        const uint16_t *bf16_c, size_t c_elems) {
    for (int i = 0; i < g_merged_count; i++) {
        if (g_merged_cache[i].key1 == bf16_a && g_merged_cache[i].key2 == bf16_b)
            return g_merged_cache[i].buffer;
    }

    id<MTLBuffer> buf_a = get_cached_bf16_as_f16_buffer(bf16_a, a_elems);
    id<MTLBuffer> buf_b = get_cached_bf16_as_f16_buffer(bf16_b, b_elems);
    id<MTLBuffer> buf_c = get_cached_bf16_as_f16_buffer(bf16_c, c_elems);
    if (!buf_a || !buf_b || !buf_c) return nil;

    size_t total = (a_elems + b_elems + c_elems) * sizeof(uint16_t);
    id<MTLBuffer> merged = [g_device newBufferWithLength:total
                                                 options:MTLResourceStorageModeShared];
    if (!merged) return nil;
    memcpy([merged contents], [buf_a contents], a_elems * sizeof(uint16_t));
    memcpy((uint8_t *)[merged contents] + a_elems * sizeof(uint16_t),
           [buf_b contents], b_elems * sizeof(uint16_t));
    memcpy((uint8_t *)[merged contents] + (a_elems + b_elems) * sizeof(uint16_t),
           [buf_c contents], c_elems * sizeof(uint16_t));

    if (g_merged_count < MERGED_CACHE_SIZE) {
        g_merged_cache[g_merged_count].key1 = bf16_a;
        g_merged_cache[g_merged_count].key2 = bf16_b;
        g_merged_cache[g_merged_count].buffer = merged;
        g_merged_count++;
    }
    return merged;
}

static void clear_merged_cache(void) {
    for (int i = 0; i < g_merged_count; i++) {
        g_merged_cache[i].buffer = nil;
        g_merged_cache[i].key1 = NULL;
        g_merged_cache[i].key2 = NULL;
    }
    g_merged_count = 0;
}

/* ========================================================================
 * F32 Weight Cache (for bias and norm weight buffers)
 * ======================================================================== */

#define WEIGHT_CACHE_SIZE 512

typedef struct {
    const void *cpu_ptr;
    id<MTLBuffer> gpu_buffer;
    size_t size;
} weight_cache_entry_t;

static weight_cache_entry_t g_weight_cache[WEIGHT_CACHE_SIZE];
static int g_weight_cache_count = 0;
static pthread_mutex_t g_cache_mutex = PTHREAD_MUTEX_INITIALIZER;

static id<MTLBuffer> get_cached_weight_buffer(const float *weights, size_t size) {
    pthread_mutex_lock(&g_cache_mutex);

    for (int i = 0; i < g_weight_cache_count; i++) {
        if (g_weight_cache[i].cpu_ptr == weights && g_weight_cache[i].size == size) {
            id<MTLBuffer> buf = g_weight_cache[i].gpu_buffer;
            pthread_mutex_unlock(&g_cache_mutex);
            return buf;
        }
    }

    if (g_weight_cache_count >= WEIGHT_CACHE_SIZE) {
        pthread_mutex_unlock(&g_cache_mutex);
        return [g_device newBufferWithBytes:weights
                                     length:size
                                    options:MTLResourceStorageModeShared];
    }

    id<MTLBuffer> buf = [g_device newBufferWithBytes:weights
                                              length:size
                                             options:MTLResourceStorageModeShared];
    if (buf) {
        g_weight_cache[g_weight_cache_count].cpu_ptr = weights;
        g_weight_cache[g_weight_cache_count].gpu_buffer = buf;
        g_weight_cache[g_weight_cache_count].size = size;
        g_weight_cache_count++;
    }

    pthread_mutex_unlock(&g_cache_mutex);
    return buf;
}

/* Drop the cached GPU copy of a host buffer whose contents have changed.
 * The cache is keyed by host pointer, so a buffer rewritten in place would
 * otherwise keep serving the values captured on first use. */
void vox_metal_invalidate_weight(const void *cpu_ptr, size_t size) {
    if (!g_device || !cpu_ptr) return;
    pthread_mutex_lock(&g_cache_mutex);
    for (int i = 0; i < g_weight_cache_count; i++) {
        if (g_weight_cache[i].cpu_ptr != cpu_ptr) continue;
        if (size && g_weight_cache[i].size != size) continue;
        g_weight_cache[i] = g_weight_cache[g_weight_cache_count - 1];
        g_weight_cache[g_weight_cache_count - 1].gpu_buffer = nil;
        g_weight_cache[g_weight_cache_count - 1].cpu_ptr = NULL;
        g_weight_cache_count--;
        i--;
    }
    pthread_mutex_unlock(&g_cache_mutex);
}

static void clear_weight_cache(void) {
    pthread_mutex_lock(&g_cache_mutex);
    for (int i = 0; i < g_weight_cache_count; i++) {
        g_weight_cache[i].gpu_buffer = nil;
        g_weight_cache[i].cpu_ptr = NULL;
    }
    g_weight_cache_count = 0;
    pthread_mutex_unlock(&g_cache_mutex);
}

/* ========================================================================
 * Activation Buffer Pool
 * ======================================================================== */

#define ACTIVATION_POOL_SIZE 64

typedef struct {
    id<MTLBuffer> buffer;
    size_t size;
    int in_use;
} pool_buffer_t;

static pool_buffer_t g_activation_pool[ACTIVATION_POOL_SIZE];
static int g_pool_count = 0;
static pthread_mutex_t g_pool_mutex = PTHREAD_MUTEX_INITIALIZER;

static id<MTLBuffer> pool_get_buffer(size_t size) {
    pthread_mutex_lock(&g_pool_mutex);

    for (int i = 0; i < g_pool_count; i++) {
        if (!g_activation_pool[i].in_use && g_activation_pool[i].size >= size) {
            g_activation_pool[i].in_use = 1;
            id<MTLBuffer> buf = g_activation_pool[i].buffer;
            pthread_mutex_unlock(&g_pool_mutex);
            return buf;
        }
    }

    if (g_pool_count < ACTIVATION_POOL_SIZE) {
        size_t alloc_size = size;
        if (alloc_size < 1024 * 1024) {
            alloc_size = ((alloc_size + 65535) / 65536) * 65536;
        } else {
            alloc_size = ((alloc_size + 1048575) / 1048576) * 1048576;
        }

        id<MTLBuffer> buf = [g_device newBufferWithLength:alloc_size
                                                  options:MTLResourceStorageModeShared];
        if (buf) {
            g_activation_pool[g_pool_count].buffer = buf;
            g_activation_pool[g_pool_count].size = alloc_size;
            g_activation_pool[g_pool_count].in_use = 1;
            g_pool_count++;
            pthread_mutex_unlock(&g_pool_mutex);
            return buf;
        }
    }

    pthread_mutex_unlock(&g_pool_mutex);
    return [g_device newBufferWithLength:size options:MTLResourceStorageModeShared];
}

static void pool_release_buffer(id<MTLBuffer> buffer) {
    if (!buffer) return;
    pthread_mutex_lock(&g_pool_mutex);
    for (int i = 0; i < g_pool_count; i++) {
        if (g_activation_pool[i].buffer == buffer) {
            g_activation_pool[i].in_use = 0;
            break;
        }
    }
    pthread_mutex_unlock(&g_pool_mutex);
}

static void clear_activation_pool(void) {
    pthread_mutex_lock(&g_pool_mutex);
    for (int i = 0; i < g_pool_count; i++) {
        g_activation_pool[i].buffer = nil;
        g_activation_pool[i].in_use = 0;
        g_activation_pool[i].size = 0;
    }
    g_pool_count = 0;
    pthread_mutex_unlock(&g_pool_mutex);
}

/* ========================================================================
 * MPS Matmul Operator Cache
 * Reuse MPSMatrixMultiplication objects across calls with same shape/config.
 * ======================================================================== */

static NSMutableDictionary *g_matmul_op_cache = nil;
static pthread_mutex_t g_matmul_op_mutex = PTHREAD_MUTEX_INITIALIZER;

static MPSMatrixMultiplication *get_cached_matmul_op(BOOL transposeLeft, BOOL transposeRight,
                                                      int resultRows, int resultColumns,
                                                      int interiorColumns,
                                                      double alpha, double beta) {
    pthread_mutex_lock(&g_matmul_op_mutex);
    if (!g_matmul_op_cache) g_matmul_op_cache = [NSMutableDictionary new];

    NSString *key = [NSString stringWithFormat:@"%d:%d:%d:%d:%d:%.9g:%.9g",
                                               (int)transposeLeft, (int)transposeRight,
                                               resultRows, resultColumns, interiorColumns,
                                               alpha, beta];
    MPSMatrixMultiplication *mm = [g_matmul_op_cache objectForKey:key];
    if (!mm) {
        mm = [[MPSMatrixMultiplication alloc]
            initWithDevice:g_device
               transposeLeft:transposeLeft
              transposeRight:transposeRight
                  resultRows:resultRows
               resultColumns:resultColumns
             interiorColumns:interiorColumns
                       alpha:alpha
                        beta:beta];
        if (mm) [g_matmul_op_cache setObject:mm forKey:key];
    }

    pthread_mutex_unlock(&g_matmul_op_mutex);
    return mm;
}

static void clear_matmul_op_cache(void) {
    pthread_mutex_lock(&g_matmul_op_mutex);
    g_matmul_op_cache = nil;
    pthread_mutex_unlock(&g_matmul_op_mutex);
}

/* ========================================================================
 * Shader Compilation
 * ======================================================================== */

static int init_shaders(void) {
    if (g_shaders_initialized) return 1;
    if (!g_initialized) return 0;

    @autoreleasepool {
        NSError *error = nil;

        NSString *shaderSource = [[NSString alloc]
            initWithBytes:voxtral_shaders_metal
                   length:voxtral_shaders_metal_len
                 encoding:NSUTF8StringEncoding];

        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        options.mathMode = MTLMathModeFast;

        g_shader_library = [g_device newLibraryWithSource:shaderSource
                                                  options:options
                                                    error:&error];
        if (!g_shader_library) {
            fprintf(stderr, "Metal shaders: compilation failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            return 0;
        }

        /* Create pipelines for each kernel */
        id<MTLFunction> func;

        func = [g_shader_library newFunctionWithName:@"rms_norm"];
        if (func) g_rms_norm_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];

        func = [g_shader_library newFunctionWithName:@"silu"];
        if (func) g_silu_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];

        func = [g_shader_library newFunctionWithName:@"gelu"];
        if (func) g_gelu_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];

        func = [g_shader_library newFunctionWithName:@"add_inplace"];
        if (func) g_add_inplace_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];

        func = [g_shader_library newFunctionWithName:@"mul_inplace"];
        if (func) g_mul_inplace_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];

        func = [g_shader_library newFunctionWithName:@"causal_softmax"];
        if (func) g_causal_softmax_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];

        func = [g_shader_library newFunctionWithName:@"ada_scale_mul"];
        if (func) g_ada_scale_mul_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];

        func = [g_shader_library newFunctionWithName:@"argmax_f32"];
        if (func) g_argmax_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];

        func = [g_shader_library newFunctionWithName:@"rope_apply"];
        if (func) g_rope_apply_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];

        func = [g_shader_library newFunctionWithName:@"kv_cache_copy"];
        if (func) g_kv_cache_copy_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];

        func = [g_shader_library newFunctionWithName:@"kv_cache_copy_f16"];
        if (func) g_kv_cache_copy_f16_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];

        func = [g_shader_library newFunctionWithName:@"decoder_attention"];
        if (func) g_decoder_attention_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];

        func = [g_shader_library newFunctionWithName:@"decoder_attention_f16"];
        if (func) g_decoder_attention_f16_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];

        func = [g_shader_library newFunctionWithName:@"encoder_attention"];
        if (func) g_encoder_attention_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];

        func = [g_shader_library newFunctionWithName:@"encoder_attention_kv_f16"];
        if (func) g_encoder_attention_kv_f16_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];

        func = [g_shader_library newFunctionWithName:@"bias_add"];
        if (func) g_bias_add_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];

        func = [g_shader_library newFunctionWithName:@"bias_add_strided"];
        if (func) g_bias_add_strided_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];

        func = [g_shader_library newFunctionWithName:@"batched_rope_apply"];
        if (func) g_batched_rope_apply_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];

        func = [g_shader_library newFunctionWithName:@"batched_rope_apply_strided"];
        if (func) g_batched_rope_apply_strided_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];

        func = [g_shader_library newFunctionWithName:@"batched_kv_cache_copy"];
        if (func) g_batched_kv_cache_copy_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];

        func = [g_shader_library newFunctionWithName:@"batched_kv_cache_copy_f16"];
        if (func) g_batched_kv_cache_copy_f16_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];

        func = [g_shader_library newFunctionWithName:@"batched_kv_cache_copy_strided"];
        if (func) g_batched_kv_cache_copy_strided_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];

        func = [g_shader_library newFunctionWithName:@"batched_kv_cache_copy_strided_f16"];
        if (func) g_batched_kv_cache_copy_strided_f16_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];

        func = [g_shader_library newFunctionWithName:@"encoder_attention_qstrided"];
        if (func) g_encoder_attention_qstrided_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];

        func = [g_shader_library newFunctionWithName:@"encoder_attention_kv_f16_qstrided"];
        if (func) g_encoder_attention_kv_f16_qstrided_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];

        func = [g_shader_library newFunctionWithName:@"deinterleave"];
        if (func) g_deinterleave_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];

        func = [g_shader_library newFunctionWithName:@"silu_mul_merged"];
        if (func) g_silu_mul_merged_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];

        func = [g_shader_library newFunctionWithName:@"decoder_ffn_gate"];
        if (func) g_decoder_ffn_gate_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];

        func = [g_shader_library newFunctionWithName:@"decoder_w2_residual"];
        if (func) g_decoder_w2_residual_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];

        func = [g_shader_library newFunctionWithName:@"decoder_wo_residual"];
        if (func) g_decoder_wo_residual_pipeline = [g_device newComputePipelineStateWithFunction:func error:&error];

        g_shaders_initialized = 1;

        if (vox_verbose >= 2) {
            fprintf(stderr, "Metal: compute shaders compiled (%s%s%s%s%s%s)\n",
                    g_rms_norm_pipeline ? "rms_norm " : "",
                    g_silu_pipeline ? "silu " : "",
                    g_gelu_pipeline ? "gelu " : "",
                    g_add_inplace_pipeline ? "add " : "",
                    g_mul_inplace_pipeline ? "mul " : "",
                    g_causal_softmax_pipeline ? "causal_softmax " : "");
        }
    }

    return 1;
}

/* ========================================================================
 * Metal Initialization
 * ======================================================================== */

int vox_metal_init(void) {
    if (g_initialized) return 1;

    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) return 0;

        if (![g_device supportsFamily:MTLGPUFamilyApple7]) {
            if (![g_device supportsFamily:MTLGPUFamilyApple6]) {
                g_device = nil;
                return 0;
            }
        }

        g_queue = [g_device newCommandQueue];
        if (!g_queue) {
            g_device = nil;
            return 0;
        }

        g_initialized = 1;
        if (vox_verbose >= 2)
            fprintf(stderr, "Metal: GPU acceleration enabled (%s)\n",
                    [[g_device name] UTF8String]);

        init_shaders();
    }

    return 1;
}

int vox_metal_available(void) {
    return g_initialized;
}

void vox_metal_shutdown(void) {
    if (!g_initialized) return;

    @autoreleasepool {
        clear_f16_cache();
        clear_merged_cache();
        clear_weight_cache();
        clear_activation_pool();
        clear_matmul_op_cache();

        g_dec_x = nil;

        g_rms_norm_pipeline = nil;
        g_silu_pipeline = nil;
        g_gelu_pipeline = nil;
        g_add_inplace_pipeline = nil;
        g_mul_inplace_pipeline = nil;
        g_causal_softmax_pipeline = nil;
        g_ada_scale_mul_pipeline = nil;
        g_argmax_pipeline = nil;
        g_rope_apply_pipeline = nil;
        g_kv_cache_copy_pipeline = nil;
        g_kv_cache_copy_f16_pipeline = nil;
        g_decoder_attention_pipeline = nil;
        g_decoder_attention_f16_pipeline = nil;
        g_encoder_attention_pipeline = nil;
        g_encoder_attention_kv_f16_pipeline = nil;
        g_bias_add_pipeline = nil;
        g_bias_add_strided_pipeline = nil;
        g_batched_rope_apply_pipeline = nil;
        g_batched_rope_apply_strided_pipeline = nil;
        g_batched_kv_cache_copy_pipeline = nil;
        g_batched_kv_cache_copy_f16_pipeline = nil;
        g_batched_kv_cache_copy_strided_pipeline = nil;
        g_batched_kv_cache_copy_strided_f16_pipeline = nil;
        g_encoder_attention_qstrided_pipeline = nil;
        g_encoder_attention_kv_f16_qstrided_pipeline = nil;
        g_deinterleave_pipeline = nil;
        g_silu_mul_merged_pipeline = nil;
        g_decoder_ffn_gate_pipeline = nil;
        g_decoder_w2_residual_pipeline = nil;
        g_decoder_wo_residual_pipeline = nil;

        /* Release shared allocs */
        for (int i = 0; i < g_shared_count; i++)
            g_shared_allocs[i].buf = nil;
        g_shared_count = 0;

        g_shader_library = nil;
        g_shaders_initialized = 0;

        g_queue = nil;
        g_device = nil;
        g_initialized = 0;
    }
}

/* ========================================================================
 * MPS Matrix Multiplication (bf16 weights)
 *
 * C[M,N] = A[M,K] @ B_bf16[N,K]^T
 * A is f32, B_bf16 is bf16 (converted to f16 and cached), C is f32.
 * ======================================================================== */

void vox_metal_sgemm_bf16(int M, int N, int K,
                           const float *A,
                           const uint16_t *B_bf16,
                           float *C) {
    if (!g_initialized) return;

    @autoreleasepool {
        size_t sizeA = (size_t)M * K * sizeof(float);
        size_t numB = (size_t)N * K;
        size_t sizeC = (size_t)M * N * sizeof(float);

        /* Get cached f16 weight buffer */
        id<MTLBuffer> bufferB = get_cached_bf16_as_f16_buffer(B_bf16, numB);

        /* Activation buffers from pool */
        id<MTLBuffer> bufferA = pool_get_buffer(sizeA);
        if (bufferA) memcpy([bufferA contents], A, sizeA);

        id<MTLBuffer> bufferC = pool_get_buffer(sizeC);

        if (!bufferA || !bufferB || !bufferC) {
            if (bufferA) pool_release_buffer(bufferA);
            if (bufferC) pool_release_buffer(bufferC);
            return;
        }

        /* Matrix descriptors:
         * A: [M, K] f32, row-major
         * B: [N, K] f16, row-major (MPS transposes it)
         * C: [M, N] f32, row-major */
        MPSMatrixDescriptor *descA = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:K
                            rowBytes:K * sizeof(float)
                            dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor *descB = [MPSMatrixDescriptor
            matrixDescriptorWithRows:N columns:K
                            rowBytes:K * sizeof(uint16_t)
                            dataType:MPSDataTypeFloat16];

        MPSMatrixDescriptor *descC = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:N
                            rowBytes:N * sizeof(float)
                            dataType:MPSDataTypeFloat32];

        MPSMatrix *matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
        MPSMatrix *matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
        MPSMatrix *matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];

        /* C = A @ B^T */
        MPSMatrixMultiplication *matmul =
            get_cached_matmul_op(NO, YES, M, N, K, 1.0, 0.0);
        if (!matmul) {
            pool_release_buffer(bufferA);
            pool_release_buffer(bufferC);
            return;
        }

        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];
        [matmul encodeToCommandBuffer:cmdBuffer
                           leftMatrix:matrixA
                          rightMatrix:matrixB
                         resultMatrix:matrixC];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        memcpy(C, [bufferC contents], sizeC);

        pool_release_buffer(bufferA);
        pool_release_buffer(bufferC);
    }
}

/* ========================================================================
 * MPS Matrix Multiplication (f32 weights)
 *
 * C[M,N] = A[M,K] @ B[N,K]^T
 * ======================================================================== */

void vox_metal_sgemm(int M, int N, int K,
                     const float *A,
                     const float *B,
                     float *C) {
    if (!g_initialized) return;

    @autoreleasepool {
        size_t sizeA = (size_t)M * K * sizeof(float);
        size_t sizeB = (size_t)N * K * sizeof(float);
        size_t sizeC = (size_t)M * N * sizeof(float);

        id<MTLBuffer> bufferB = get_cached_weight_buffer(B, sizeB);

        id<MTLBuffer> bufferA = pool_get_buffer(sizeA);
        if (bufferA) memcpy([bufferA contents], A, sizeA);

        id<MTLBuffer> bufferC = pool_get_buffer(sizeC);

        if (!bufferA || !bufferB || !bufferC) {
            if (bufferA) pool_release_buffer(bufferA);
            if (bufferC) pool_release_buffer(bufferC);
            return;
        }

        MPSMatrixDescriptor *descA = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:K
                            rowBytes:K * sizeof(float)
                            dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor *descB = [MPSMatrixDescriptor
            matrixDescriptorWithRows:N columns:K
                            rowBytes:K * sizeof(float)
                            dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor *descC = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:N
                            rowBytes:N * sizeof(float)
                            dataType:MPSDataTypeFloat32];

        MPSMatrix *matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
        MPSMatrix *matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
        MPSMatrix *matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];

        MPSMatrixMultiplication *matmul =
            get_cached_matmul_op(NO, YES, M, N, K, 1.0, 0.0);
        if (!matmul) {
            pool_release_buffer(bufferA);
            pool_release_buffer(bufferC);
            return;
        }

        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];
        [matmul encodeToCommandBuffer:cmdBuffer
                           leftMatrix:matrixA
                          rightMatrix:matrixB
                         resultMatrix:matrixC];
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        memcpy(C, [bufferC contents], sizeC);

        pool_release_buffer(bufferA);
        pool_release_buffer(bufferC);
    }
}

/* ========================================================================
 * Fused RMSNorm + QKV: norm + 3 matmuls in one command buffer
 * ======================================================================== */

void vox_metal_fused_norm_qkv_bf16(int M, int K,
                                     const float *x,
                                     const float *norm_weight, float eps,
                                     const uint16_t *wq_bf16, int Nq,
                                     const uint16_t *wk_bf16, int Nk,
                                     const uint16_t *wv_bf16, int Nv,
                                     float *q_out, float *k_out, float *v_out) {
    if (!g_initialized || !g_shaders_initialized) return;

    @autoreleasepool {
        size_t sizeX = (size_t)M * K * sizeof(float);
        size_t sizeQ = (size_t)M * Nq * sizeof(float);
        size_t sizeK = (size_t)M * Nk * sizeof(float);
        size_t sizeV = (size_t)M * Nv * sizeof(float);

        /* Cached f16 weight buffers */
        id<MTLBuffer> bufWq = get_cached_bf16_as_f16_buffer(wq_bf16, (size_t)Nq * K);
        id<MTLBuffer> bufWk = get_cached_bf16_as_f16_buffer(wk_bf16, (size_t)Nk * K);
        id<MTLBuffer> bufWv = get_cached_bf16_as_f16_buffer(wv_bf16, (size_t)Nv * K);

        /* Upload x, allocate x_norm on GPU */
        id<MTLBuffer> bufX = pool_get_buffer(sizeX);
        if (bufX) memcpy([bufX contents], x, sizeX);
        id<MTLBuffer> bufXnorm = pool_get_buffer(sizeX);

        /* Norm weight on GPU */
        id<MTLBuffer> bufNorm = get_cached_weight_buffer(norm_weight, K * sizeof(float));

        /* Output buffers */
        id<MTLBuffer> bufQ = pool_get_buffer(sizeQ);
        id<MTLBuffer> bufK = pool_get_buffer(sizeK);
        id<MTLBuffer> bufV = pool_get_buffer(sizeV);

        if (!bufX || !bufXnorm || !bufNorm || !bufWq || !bufWk || !bufWv ||
            !bufQ || !bufK || !bufV) {
            pool_release_buffer(bufX);
            pool_release_buffer(bufXnorm);
            pool_release_buffer(bufQ);
            pool_release_buffer(bufK);
            pool_release_buffer(bufV);
            return;
        }

        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];

        /* Step 1: x_norm = rms_norm(x, norm_weight) */
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_rms_norm_pipeline];
            [enc setBuffer:bufX offset:0 atIndex:0];
            [enc setBuffer:bufNorm offset:0 atIndex:1];
            [enc setBuffer:bufXnorm offset:0 atIndex:2];
            int hidden = K;
            [enc setBytes:&hidden length:sizeof(int) atIndex:3];
            [enc setBytes:&eps length:sizeof(float) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)M, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        /* x_norm descriptor for QKV matmuls */
        MPSMatrixDescriptor *descInput = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:K
                            rowBytes:K * sizeof(float)
                            dataType:MPSDataTypeFloat32];
        MPSMatrix *matInput = [[MPSMatrix alloc] initWithBuffer:bufXnorm descriptor:descInput];

        /* Q = x_norm @ Wq^T */
        {
            MPSMatrixDescriptor *descW = [MPSMatrixDescriptor
                matrixDescriptorWithRows:Nq columns:K
                                rowBytes:K * sizeof(uint16_t)
                                dataType:MPSDataTypeFloat16];
            MPSMatrixDescriptor *descOut = [MPSMatrixDescriptor
                matrixDescriptorWithRows:M columns:Nq
                                rowBytes:Nq * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrix *matW = [[MPSMatrix alloc] initWithBuffer:bufWq descriptor:descW];
            MPSMatrix *matOut = [[MPSMatrix alloc] initWithBuffer:bufQ descriptor:descOut];
            MPSMatrixMultiplication *mm =
                get_cached_matmul_op(NO, YES, M, Nq, K, 1.0, 0.0);
            if (mm)
                [mm encodeToCommandBuffer:cmdBuffer leftMatrix:matInput rightMatrix:matW resultMatrix:matOut];
        }

        /* K = x_norm @ Wk^T */
        {
            MPSMatrixDescriptor *descW = [MPSMatrixDescriptor
                matrixDescriptorWithRows:Nk columns:K
                                rowBytes:K * sizeof(uint16_t)
                                dataType:MPSDataTypeFloat16];
            MPSMatrixDescriptor *descOut = [MPSMatrixDescriptor
                matrixDescriptorWithRows:M columns:Nk
                                rowBytes:Nk * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrix *matW = [[MPSMatrix alloc] initWithBuffer:bufWk descriptor:descW];
            MPSMatrix *matOut = [[MPSMatrix alloc] initWithBuffer:bufK descriptor:descOut];
            MPSMatrixMultiplication *mm =
                get_cached_matmul_op(NO, YES, M, Nk, K, 1.0, 0.0);
            if (mm)
                [mm encodeToCommandBuffer:cmdBuffer leftMatrix:matInput rightMatrix:matW resultMatrix:matOut];
        }

        /* V = x_norm @ Wv^T */
        {
            MPSMatrixDescriptor *descW = [MPSMatrixDescriptor
                matrixDescriptorWithRows:Nv columns:K
                                rowBytes:K * sizeof(uint16_t)
                                dataType:MPSDataTypeFloat16];
            MPSMatrixDescriptor *descOut = [MPSMatrixDescriptor
                matrixDescriptorWithRows:M columns:Nv
                                rowBytes:Nv * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrix *matW = [[MPSMatrix alloc] initWithBuffer:bufWv descriptor:descW];
            MPSMatrix *matOut = [[MPSMatrix alloc] initWithBuffer:bufV descriptor:descOut];
            MPSMatrixMultiplication *mm =
                get_cached_matmul_op(NO, YES, M, Nv, K, 1.0, 0.0);
            if (mm)
                [mm encodeToCommandBuffer:cmdBuffer leftMatrix:matInput rightMatrix:matW resultMatrix:matOut];
        }

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        memcpy(q_out, [bufQ contents], sizeQ);
        memcpy(k_out, [bufK contents], sizeK);
        memcpy(v_out, [bufV contents], sizeV);

        pool_release_buffer(bufX);
        pool_release_buffer(bufXnorm);
        pool_release_buffer(bufQ);
        pool_release_buffer(bufK);
        pool_release_buffer(bufV);
    }
}

/* ========================================================================
 * Fused QKV: 3 matmuls in one command buffer, shared input
 *
 * q[M,Nq] = input[M,K] @ wq[Nq,K]^T
 * k[M,Nk] = input[M,K] @ wk[Nk,K]^T
 * v[M,Nv] = input[M,K] @ wv[Nv,K]^T
 * ======================================================================== */

void vox_metal_fused_qkv_bf16(int M, int K,
                                const float *input,
                                const uint16_t *wq_bf16, int Nq,
                                const uint16_t *wk_bf16, int Nk,
                                const uint16_t *wv_bf16, int Nv,
                                float *q_out, float *k_out, float *v_out) {
    if (!g_initialized) return;

    @autoreleasepool {
        size_t sizeInput = (size_t)M * K * sizeof(float);
        size_t sizeQ = (size_t)M * Nq * sizeof(float);
        size_t sizeK = (size_t)M * Nk * sizeof(float);
        size_t sizeV = (size_t)M * Nv * sizeof(float);

        /* Cached f16 weight buffers */
        id<MTLBuffer> bufWq = get_cached_bf16_as_f16_buffer(wq_bf16, (size_t)Nq * K);
        id<MTLBuffer> bufWk = get_cached_bf16_as_f16_buffer(wk_bf16, (size_t)Nk * K);
        id<MTLBuffer> bufWv = get_cached_bf16_as_f16_buffer(wv_bf16, (size_t)Nv * K);

        /* One input copy */
        id<MTLBuffer> bufInput = pool_get_buffer(sizeInput);
        if (bufInput) memcpy([bufInput contents], input, sizeInput);

        /* Output buffers */
        id<MTLBuffer> bufQ = pool_get_buffer(sizeQ);
        id<MTLBuffer> bufK = pool_get_buffer(sizeK);
        id<MTLBuffer> bufV = pool_get_buffer(sizeV);

        if (!bufInput || !bufWq || !bufWk || !bufWv || !bufQ || !bufK || !bufV) {
            pool_release_buffer(bufInput);
            pool_release_buffer(bufQ);
            pool_release_buffer(bufK);
            pool_release_buffer(bufV);
            return;
        }

        /* Shared input descriptor */
        MPSMatrixDescriptor *descInput = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:K
                            rowBytes:K * sizeof(float)
                            dataType:MPSDataTypeFloat32];
        MPSMatrix *matInput = [[MPSMatrix alloc] initWithBuffer:bufInput descriptor:descInput];

        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];

        /* Q = Input @ Wq^T */
        {
            MPSMatrixDescriptor *descW = [MPSMatrixDescriptor
                matrixDescriptorWithRows:Nq columns:K
                                rowBytes:K * sizeof(uint16_t)
                                dataType:MPSDataTypeFloat16];
            MPSMatrixDescriptor *descOut = [MPSMatrixDescriptor
                matrixDescriptorWithRows:M columns:Nq
                                rowBytes:Nq * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrix *matW = [[MPSMatrix alloc] initWithBuffer:bufWq descriptor:descW];
            MPSMatrix *matOut = [[MPSMatrix alloc] initWithBuffer:bufQ descriptor:descOut];
            MPSMatrixMultiplication *mm =
                get_cached_matmul_op(NO, YES, M, Nq, K, 1.0, 0.0);
            if (!mm) {
                pool_release_buffer(bufInput);
                pool_release_buffer(bufQ);
                pool_release_buffer(bufK);
                pool_release_buffer(bufV);
                return;
            }
            [mm encodeToCommandBuffer:cmdBuffer leftMatrix:matInput rightMatrix:matW resultMatrix:matOut];
        }

        /* K = Input @ Wk^T */
        {
            MPSMatrixDescriptor *descW = [MPSMatrixDescriptor
                matrixDescriptorWithRows:Nk columns:K
                                rowBytes:K * sizeof(uint16_t)
                                dataType:MPSDataTypeFloat16];
            MPSMatrixDescriptor *descOut = [MPSMatrixDescriptor
                matrixDescriptorWithRows:M columns:Nk
                                rowBytes:Nk * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrix *matW = [[MPSMatrix alloc] initWithBuffer:bufWk descriptor:descW];
            MPSMatrix *matOut = [[MPSMatrix alloc] initWithBuffer:bufK descriptor:descOut];
            MPSMatrixMultiplication *mm =
                get_cached_matmul_op(NO, YES, M, Nk, K, 1.0, 0.0);
            if (!mm) {
                pool_release_buffer(bufInput);
                pool_release_buffer(bufQ);
                pool_release_buffer(bufK);
                pool_release_buffer(bufV);
                return;
            }
            [mm encodeToCommandBuffer:cmdBuffer leftMatrix:matInput rightMatrix:matW resultMatrix:matOut];
        }

        /* V = Input @ Wv^T */
        {
            MPSMatrixDescriptor *descW = [MPSMatrixDescriptor
                matrixDescriptorWithRows:Nv columns:K
                                rowBytes:K * sizeof(uint16_t)
                                dataType:MPSDataTypeFloat16];
            MPSMatrixDescriptor *descOut = [MPSMatrixDescriptor
                matrixDescriptorWithRows:M columns:Nv
                                rowBytes:Nv * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrix *matW = [[MPSMatrix alloc] initWithBuffer:bufWv descriptor:descW];
            MPSMatrix *matOut = [[MPSMatrix alloc] initWithBuffer:bufV descriptor:descOut];
            MPSMatrixMultiplication *mm =
                get_cached_matmul_op(NO, YES, M, Nv, K, 1.0, 0.0);
            if (!mm) {
                pool_release_buffer(bufInput);
                pool_release_buffer(bufQ);
                pool_release_buffer(bufK);
                pool_release_buffer(bufV);
                return;
            }
            [mm encodeToCommandBuffer:cmdBuffer leftMatrix:matInput rightMatrix:matW resultMatrix:matOut];
        }

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        memcpy(q_out, [bufQ contents], sizeQ);
        memcpy(k_out, [bufK contents], sizeK);
        memcpy(v_out, [bufV contents], sizeV);

        pool_release_buffer(bufInput);
        pool_release_buffer(bufQ);
        pool_release_buffer(bufK);
        pool_release_buffer(bufV);
    }
}

/* ========================================================================
 * Fused SwiGLU FFN: w1+w3+silu+mul+w2 in one command buffer
 *
 * gate = silu(input @ w1^T)
 * up = input @ w3^T
 * output = (gate * up) @ w2^T
 *
 * All intermediate buffers stay on GPU. Only input is copied in,
 * only output is copied out.
 * ======================================================================== */

void vox_metal_fused_ffn_bf16(int M, int dim, int hidden,
                               const float *input,
                               const uint16_t *w1_bf16,
                               const uint16_t *w3_bf16,
                               const uint16_t *w2_bf16,
                               float *output) {
    if (!g_initialized || !g_shaders_initialized) return;

    @autoreleasepool {
        size_t sizeInput = (size_t)M * dim * sizeof(float);
        size_t sizeHidden = (size_t)M * hidden * sizeof(float);
        size_t sizeOutput = (size_t)M * dim * sizeof(float);

        /* Cached f16 weight buffers */
        id<MTLBuffer> bufW1 = get_cached_bf16_as_f16_buffer(w1_bf16, (size_t)hidden * dim);
        id<MTLBuffer> bufW3 = get_cached_bf16_as_f16_buffer(w3_bf16, (size_t)hidden * dim);
        id<MTLBuffer> bufW2 = get_cached_bf16_as_f16_buffer(w2_bf16, (size_t)dim * hidden);

        /* Activation buffers */
        id<MTLBuffer> bufInput = pool_get_buffer(sizeInput);
        if (bufInput) memcpy([bufInput contents], input, sizeInput);

        id<MTLBuffer> bufGate = pool_get_buffer(sizeHidden);
        id<MTLBuffer> bufUp = pool_get_buffer(sizeHidden);
        id<MTLBuffer> bufOutput = pool_get_buffer(sizeOutput);

        if (!bufInput || !bufW1 || !bufW3 || !bufW2 ||
            !bufGate || !bufUp || !bufOutput) {
            pool_release_buffer(bufInput);
            pool_release_buffer(bufGate);
            pool_release_buffer(bufUp);
            pool_release_buffer(bufOutput);
            return;
        }

        /* Shared input matrix descriptor (for w1 and w3) */
        MPSMatrixDescriptor *descInput = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:dim
                            rowBytes:dim * sizeof(float)
                            dataType:MPSDataTypeFloat32];
        MPSMatrix *matInput = [[MPSMatrix alloc] initWithBuffer:bufInput descriptor:descInput];

        /* Hidden output descriptor (shared shape for gate and up) */
        MPSMatrixDescriptor *descHidden = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:hidden
                            rowBytes:hidden * sizeof(float)
                            dataType:MPSDataTypeFloat32];

        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];

        /* Step 1: gate = input @ w1^T */
        {
            MPSMatrixDescriptor *descW = [MPSMatrixDescriptor
                matrixDescriptorWithRows:hidden columns:dim
                                rowBytes:dim * sizeof(uint16_t)
                                dataType:MPSDataTypeFloat16];
            MPSMatrix *matW = [[MPSMatrix alloc] initWithBuffer:bufW1 descriptor:descW];
            MPSMatrix *matGate = [[MPSMatrix alloc] initWithBuffer:bufGate descriptor:descHidden];
            MPSMatrixMultiplication *mm =
                get_cached_matmul_op(NO, YES, M, hidden, dim, 1.0, 0.0);
            if (!mm) {
                pool_release_buffer(bufInput);
                pool_release_buffer(bufGate);
                pool_release_buffer(bufUp);
                pool_release_buffer(bufOutput);
                return;
            }
            [mm encodeToCommandBuffer:cmdBuffer leftMatrix:matInput rightMatrix:matW resultMatrix:matGate];
        }

        /* Step 2: up = input @ w3^T */
        {
            MPSMatrixDescriptor *descW = [MPSMatrixDescriptor
                matrixDescriptorWithRows:hidden columns:dim
                                rowBytes:dim * sizeof(uint16_t)
                                dataType:MPSDataTypeFloat16];
            MPSMatrix *matW = [[MPSMatrix alloc] initWithBuffer:bufW3 descriptor:descW];
            MPSMatrix *matUp = [[MPSMatrix alloc] initWithBuffer:bufUp descriptor:descHidden];
            MPSMatrixMultiplication *mm =
                get_cached_matmul_op(NO, YES, M, hidden, dim, 1.0, 0.0);
            if (!mm) {
                pool_release_buffer(bufInput);
                pool_release_buffer(bufGate);
                pool_release_buffer(bufUp);
                pool_release_buffer(bufOutput);
                return;
            }
            [mm encodeToCommandBuffer:cmdBuffer leftMatrix:matInput rightMatrix:matW resultMatrix:matUp];
        }

        /* Step 3: silu(gate) - GPU compute shader */
        {
            int n = M * hidden;
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_silu_pipeline];
            [enc setBuffer:bufGate offset:0 atIndex:0];
            [enc setBytes:&n length:sizeof(int) atIndex:1];
            NSUInteger tgSize = MIN((NSUInteger)n, g_silu_pipeline.maxTotalThreadsPerThreadgroup);
            [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
            [enc endEncoding];
        }

        /* Step 4: gate *= up - GPU compute shader */
        {
            int n = M * hidden;
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_mul_inplace_pipeline];
            [enc setBuffer:bufGate offset:0 atIndex:0];
            [enc setBuffer:bufUp offset:0 atIndex:1];
            [enc setBytes:&n length:sizeof(int) atIndex:2];
            NSUInteger tgSize = MIN((NSUInteger)n, g_mul_inplace_pipeline.maxTotalThreadsPerThreadgroup);
            [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
            [enc endEncoding];
        }

        /* Step 5: output = gate @ w2^T */
        {
            MPSMatrixDescriptor *descW = [MPSMatrixDescriptor
                matrixDescriptorWithRows:dim columns:hidden
                                rowBytes:hidden * sizeof(uint16_t)
                                dataType:MPSDataTypeFloat16];
            MPSMatrixDescriptor *descOut = [MPSMatrixDescriptor
                matrixDescriptorWithRows:M columns:dim
                                rowBytes:dim * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrix *matGate = [[MPSMatrix alloc] initWithBuffer:bufGate descriptor:descHidden];
            MPSMatrix *matW = [[MPSMatrix alloc] initWithBuffer:bufW2 descriptor:descW];
            MPSMatrix *matOut = [[MPSMatrix alloc] initWithBuffer:bufOutput descriptor:descOut];
            MPSMatrixMultiplication *mm =
                get_cached_matmul_op(NO, YES, M, dim, hidden, 1.0, 0.0);
            if (!mm) {
                pool_release_buffer(bufInput);
                pool_release_buffer(bufGate);
                pool_release_buffer(bufUp);
                pool_release_buffer(bufOutput);
                return;
            }
            [mm encodeToCommandBuffer:cmdBuffer leftMatrix:matGate rightMatrix:matW resultMatrix:matOut];
        }

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        memcpy(output, [bufOutput contents], sizeOutput);

        pool_release_buffer(bufInput);
        pool_release_buffer(bufGate);
        pool_release_buffer(bufUp);
        pool_release_buffer(bufOutput);
    }
}

/* ========================================================================
 * Fused final RMSNorm + logits matmul + argmax
 * ======================================================================== */

int vox_metal_fused_logits_bf16(int dim, int vocab_size,
                                  const float *x,
                                  const float *norm_weight, float eps,
                                  const uint16_t *tok_emb_bf16,
                                  float *logits_out) {
    if (!g_initialized || !g_shaders_initialized) return 0;

    int result = 0;

    @autoreleasepool {
        size_t sizeDim = (size_t)dim * sizeof(float);
        size_t sizeLogits = (size_t)vocab_size * sizeof(float);

        /* Cached f16 weight buffer for tok_embeddings */
        id<MTLBuffer> bufEmb = get_cached_bf16_as_f16_buffer(tok_emb_bf16,
                                                               (size_t)vocab_size * dim);
        id<MTLBuffer> bufNorm = get_cached_weight_buffer(norm_weight, sizeDim);

        /* Activation buffers */
        id<MTLBuffer> bufX = pool_get_buffer(sizeDim);
        if (bufX) memcpy([bufX contents], x, sizeDim);
        id<MTLBuffer> bufXnorm = pool_get_buffer(sizeDim);
        id<MTLBuffer> bufLogits = pool_get_buffer(sizeLogits);
        id<MTLBuffer> bufArgmax = pool_get_buffer(sizeof(int));

        if (!bufX || !bufXnorm || !bufLogits || !bufArgmax || !bufEmb || !bufNorm) {
            pool_release_buffer(bufX);
            pool_release_buffer(bufXnorm);
            pool_release_buffer(bufLogits);
            pool_release_buffer(bufArgmax);
            return 0;
        }

        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];

        /* Step 1: x_norm = rms_norm(x, norm_weight) */
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_rms_norm_pipeline];
            [enc setBuffer:bufX offset:0 atIndex:0];
            [enc setBuffer:bufNorm offset:0 atIndex:1];
            [enc setBuffer:bufXnorm offset:0 atIndex:2];
            [enc setBytes:&dim length:sizeof(int) atIndex:3];
            [enc setBytes:&eps length:sizeof(float) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        /* Step 2: logits = x_norm @ tok_emb^T */
        {
            MPSMatrixDescriptor *descIn = [MPSMatrixDescriptor
                matrixDescriptorWithRows:1 columns:dim
                                rowBytes:dim * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *descW = [MPSMatrixDescriptor
                matrixDescriptorWithRows:vocab_size columns:dim
                                rowBytes:dim * sizeof(uint16_t)
                                dataType:MPSDataTypeFloat16];
            MPSMatrixDescriptor *descOut = [MPSMatrixDescriptor
                matrixDescriptorWithRows:1 columns:vocab_size
                                rowBytes:vocab_size * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrix *matIn = [[MPSMatrix alloc] initWithBuffer:bufXnorm descriptor:descIn];
            MPSMatrix *matW = [[MPSMatrix alloc] initWithBuffer:bufEmb descriptor:descW];
            MPSMatrix *matOut = [[MPSMatrix alloc] initWithBuffer:bufLogits descriptor:descOut];
            MPSMatrixMultiplication *mm =
                get_cached_matmul_op(NO, YES, 1, vocab_size, dim, 1.0, 0.0);
            if (mm)
                [mm encodeToCommandBuffer:cmdBuffer leftMatrix:matIn rightMatrix:matW resultMatrix:matOut];
        }

        /* Step 3: argmax on GPU */
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_argmax_pipeline];
            [enc setBuffer:bufLogits offset:0 atIndex:0];
            [enc setBuffer:bufArgmax offset:0 atIndex:1];
            [enc setBytes:&vocab_size length:sizeof(int) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        /* Read argmax result (just 4 bytes) */
        result = ((int *)[bufArgmax contents])[0];

        /* Copy logits to CPU only if caller wants them */
        if (logits_out)
            memcpy(logits_out, [bufLogits contents], sizeLogits);

        pool_release_buffer(bufX);
        pool_release_buffer(bufXnorm);
        pool_release_buffer(bufLogits);
        pool_release_buffer(bufArgmax);
    }

    return result;
}

/* ========================================================================
 * Fused wo + residual + RMSNorm + ada_scale + FFN + residual
 *
 * One command buffer for: proj_out = attn_out @ wo^T, x += proj_out,
 * x_norm = rms_norm(x), x_norm *= (1+ada_scale), FFN(x_norm), x += ffn_out
 * ======================================================================== */

void vox_metal_fused_wo_ffn_bf16(int M, int dim, int q_dim, int hidden,
                                   float *x,
                                   const float *attn_out,
                                   const uint16_t *wo_bf16,
                                   const float *ffn_norm, float eps,
                                   const float *ada_scale,
                                   const uint16_t *w1_bf16,
                                   const uint16_t *w3_bf16,
                                   const uint16_t *w2_bf16) {
    if (!g_initialized || !g_shaders_initialized) return;

    @autoreleasepool {
        size_t sizeAttn = (size_t)M * q_dim * sizeof(float);
        size_t sizeDim = (size_t)M * dim * sizeof(float);
        size_t sizeHidden = (size_t)M * hidden * sizeof(float);

        /* Cached f16 weight buffers */
        id<MTLBuffer> bufWo = get_cached_bf16_as_f16_buffer(wo_bf16, (size_t)dim * q_dim);
        id<MTLBuffer> bufW1 = get_cached_bf16_as_f16_buffer(w1_bf16, (size_t)hidden * dim);
        id<MTLBuffer> bufW3 = get_cached_bf16_as_f16_buffer(w3_bf16, (size_t)hidden * dim);
        id<MTLBuffer> bufW2 = get_cached_bf16_as_f16_buffer(w2_bf16, (size_t)dim * hidden);

        /* Activation buffers */
        id<MTLBuffer> bufAttn = pool_get_buffer(sizeAttn);
        if (bufAttn) memcpy([bufAttn contents], attn_out, sizeAttn);

        id<MTLBuffer> bufX = pool_get_buffer(sizeDim);
        if (bufX) memcpy([bufX contents], x, sizeDim);

        id<MTLBuffer> bufProj = pool_get_buffer(sizeDim);  /* wo output */
        id<MTLBuffer> bufXnorm = pool_get_buffer(sizeDim);  /* rms_norm output */
        id<MTLBuffer> bufGate = pool_get_buffer(sizeHidden);
        id<MTLBuffer> bufUp = pool_get_buffer(sizeHidden);
        id<MTLBuffer> bufFfnOut = pool_get_buffer(sizeDim);

        /* Norm weight + ada_scale on GPU */
        id<MTLBuffer> bufNorm = get_cached_weight_buffer(ffn_norm, dim * sizeof(float));
        id<MTLBuffer> bufAda = ada_scale ?
            get_cached_weight_buffer(ada_scale, dim * sizeof(float)) : nil;

        if (!bufAttn || !bufX || !bufProj || !bufXnorm ||
            !bufGate || !bufUp || !bufFfnOut || !bufWo || !bufNorm) {
            pool_release_buffer(bufAttn);
            pool_release_buffer(bufX);
            pool_release_buffer(bufProj);
            pool_release_buffer(bufXnorm);
            pool_release_buffer(bufGate);
            pool_release_buffer(bufUp);
            pool_release_buffer(bufFfnOut);
            return;
        }

        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];

        /* Step 1: proj_out = attn_out @ wo^T */
        {
            MPSMatrixDescriptor *descA = [MPSMatrixDescriptor
                matrixDescriptorWithRows:M columns:q_dim
                                rowBytes:q_dim * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *descW = [MPSMatrixDescriptor
                matrixDescriptorWithRows:dim columns:q_dim
                                rowBytes:q_dim * sizeof(uint16_t)
                                dataType:MPSDataTypeFloat16];
            MPSMatrixDescriptor *descOut = [MPSMatrixDescriptor
                matrixDescriptorWithRows:M columns:dim
                                rowBytes:dim * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufAttn descriptor:descA];
            MPSMatrix *matW = [[MPSMatrix alloc] initWithBuffer:bufWo descriptor:descW];
            MPSMatrix *matOut = [[MPSMatrix alloc] initWithBuffer:bufProj descriptor:descOut];
            MPSMatrixMultiplication *mm =
                get_cached_matmul_op(NO, YES, M, dim, q_dim, 1.0, 0.0);
            if (mm)
                [mm encodeToCommandBuffer:cmdBuffer leftMatrix:matA rightMatrix:matW resultMatrix:matOut];
        }

        /* Step 2: x += proj_out */
        {
            int n = M * dim;
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_add_inplace_pipeline];
            [enc setBuffer:bufX offset:0 atIndex:0];
            [enc setBuffer:bufProj offset:0 atIndex:1];
            [enc setBytes:&n length:sizeof(int) atIndex:2];
            NSUInteger tgSize = MIN((NSUInteger)n, g_add_inplace_pipeline.maxTotalThreadsPerThreadgroup);
            [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
            [enc endEncoding];
        }

        /* Step 3: x_norm = rms_norm(x, ffn_norm) */
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_rms_norm_pipeline];
            [enc setBuffer:bufX offset:0 atIndex:0];
            [enc setBuffer:bufNorm offset:0 atIndex:1];
            [enc setBuffer:bufXnorm offset:0 atIndex:2];
            [enc setBytes:&dim length:sizeof(int) atIndex:3];
            [enc setBytes:&eps length:sizeof(float) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)M, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        /* Step 4: x_norm *= (1 + ada_scale) */
        if (bufAda) {
            int n = M * dim;
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_ada_scale_mul_pipeline];
            [enc setBuffer:bufXnorm offset:0 atIndex:0];
            [enc setBuffer:bufAda offset:0 atIndex:1];
            [enc setBytes:&n length:sizeof(int) atIndex:2];
            [enc setBytes:&dim length:sizeof(int) atIndex:3];
            NSUInteger tgSize = MIN((NSUInteger)n, g_ada_scale_mul_pipeline.maxTotalThreadsPerThreadgroup);
            [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
            [enc endEncoding];
        }

        /* Step 5: gate = x_norm @ w1^T */
        {
            MPSMatrixDescriptor *descIn = [MPSMatrixDescriptor
                matrixDescriptorWithRows:M columns:dim
                                rowBytes:dim * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *descW = [MPSMatrixDescriptor
                matrixDescriptorWithRows:hidden columns:dim
                                rowBytes:dim * sizeof(uint16_t)
                                dataType:MPSDataTypeFloat16];
            MPSMatrixDescriptor *descH = [MPSMatrixDescriptor
                matrixDescriptorWithRows:M columns:hidden
                                rowBytes:hidden * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrix *matIn = [[MPSMatrix alloc] initWithBuffer:bufXnorm descriptor:descIn];
            MPSMatrix *matW = [[MPSMatrix alloc] initWithBuffer:bufW1 descriptor:descW];
            MPSMatrix *matGate = [[MPSMatrix alloc] initWithBuffer:bufGate descriptor:descH];
            MPSMatrixMultiplication *mm =
                get_cached_matmul_op(NO, YES, M, hidden, dim, 1.0, 0.0);
            if (mm)
                [mm encodeToCommandBuffer:cmdBuffer leftMatrix:matIn rightMatrix:matW resultMatrix:matGate];
        }

        /* Step 6: up = x_norm @ w3^T */
        {
            MPSMatrixDescriptor *descIn = [MPSMatrixDescriptor
                matrixDescriptorWithRows:M columns:dim
                                rowBytes:dim * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *descW = [MPSMatrixDescriptor
                matrixDescriptorWithRows:hidden columns:dim
                                rowBytes:dim * sizeof(uint16_t)
                                dataType:MPSDataTypeFloat16];
            MPSMatrixDescriptor *descH = [MPSMatrixDescriptor
                matrixDescriptorWithRows:M columns:hidden
                                rowBytes:hidden * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrix *matIn = [[MPSMatrix alloc] initWithBuffer:bufXnorm descriptor:descIn];
            MPSMatrix *matW = [[MPSMatrix alloc] initWithBuffer:bufW3 descriptor:descW];
            MPSMatrix *matUp = [[MPSMatrix alloc] initWithBuffer:bufUp descriptor:descH];
            MPSMatrixMultiplication *mm =
                get_cached_matmul_op(NO, YES, M, hidden, dim, 1.0, 0.0);
            if (mm)
                [mm encodeToCommandBuffer:cmdBuffer leftMatrix:matIn rightMatrix:matW resultMatrix:matUp];
        }

        /* Step 7: silu(gate) */
        {
            int n = M * hidden;
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_silu_pipeline];
            [enc setBuffer:bufGate offset:0 atIndex:0];
            [enc setBytes:&n length:sizeof(int) atIndex:1];
            NSUInteger tgSize = MIN((NSUInteger)n, g_silu_pipeline.maxTotalThreadsPerThreadgroup);
            [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
            [enc endEncoding];
        }

        /* Step 8: gate *= up */
        {
            int n = M * hidden;
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_mul_inplace_pipeline];
            [enc setBuffer:bufGate offset:0 atIndex:0];
            [enc setBuffer:bufUp offset:0 atIndex:1];
            [enc setBytes:&n length:sizeof(int) atIndex:2];
            NSUInteger tgSize = MIN((NSUInteger)n, g_mul_inplace_pipeline.maxTotalThreadsPerThreadgroup);
            [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
            [enc endEncoding];
        }

        /* Step 9: ffn_out = gate @ w2^T */
        {
            MPSMatrixDescriptor *descH = [MPSMatrixDescriptor
                matrixDescriptorWithRows:M columns:hidden
                                rowBytes:hidden * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *descW = [MPSMatrixDescriptor
                matrixDescriptorWithRows:dim columns:hidden
                                rowBytes:hidden * sizeof(uint16_t)
                                dataType:MPSDataTypeFloat16];
            MPSMatrixDescriptor *descOut = [MPSMatrixDescriptor
                matrixDescriptorWithRows:M columns:dim
                                rowBytes:dim * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrix *matGate = [[MPSMatrix alloc] initWithBuffer:bufGate descriptor:descH];
            MPSMatrix *matW = [[MPSMatrix alloc] initWithBuffer:bufW2 descriptor:descW];
            MPSMatrix *matOut = [[MPSMatrix alloc] initWithBuffer:bufFfnOut descriptor:descOut];
            MPSMatrixMultiplication *mm =
                get_cached_matmul_op(NO, YES, M, dim, hidden, 1.0, 0.0);
            if (mm)
                [mm encodeToCommandBuffer:cmdBuffer leftMatrix:matGate rightMatrix:matW resultMatrix:matOut];
        }

        /* Step 10: x += ffn_out */
        {
            int n = M * dim;
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_add_inplace_pipeline];
            [enc setBuffer:bufX offset:0 atIndex:0];
            [enc setBuffer:bufFfnOut offset:0 atIndex:1];
            [enc setBytes:&n length:sizeof(int) atIndex:2];
            NSUInteger tgSize = MIN((NSUInteger)n, g_add_inplace_pipeline.maxTotalThreadsPerThreadgroup);
            [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
            [enc endEncoding];
        }

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        /* Copy x back to CPU */
        memcpy(x, [bufX contents], sizeDim);

        pool_release_buffer(bufAttn);
        pool_release_buffer(bufX);
        pool_release_buffer(bufProj);
        pool_release_buffer(bufXnorm);
        pool_release_buffer(bufGate);
        pool_release_buffer(bufUp);
        pool_release_buffer(bufFfnOut);
    }
}

/* ========================================================================
 * Persistent-x Decoder Step API
 *
 * Keeps x on GPU across all 26 decoder layers. Fuses wo_ffn[i] +
 * norm_qkv[i+1] into a single command buffer per layer transition.
 * Reduces command buffers from 53 to 27 per token.
 * ======================================================================== */

void vox_metal_decoder_start(const float *x, int dim) {
    if (!g_initialized) return;
    size_t size = (size_t)dim * sizeof(float);
    if (!g_dec_x || [g_dec_x length] < size) {
        g_dec_x = [g_device newBufferWithLength:size
                                        options:MTLResourceStorageModeShared];
    }
    memcpy([g_dec_x contents], x, size);
}

void vox_metal_decoder_end(void) {
    /* Keep g_dec_x allocated for reuse across tokens */
}

/* Helper: encode wo+FFN steps into a command buffer (steps 1-10).
 * Reads attn_out from bufAttn, updates g_dec_x in-place.
 * Returns bufXnorm (output of FFN rms_norm, needed for next step).
 * Caller must pool_release all returned buffers after commit. */
static void encode_wo_ffn_steps(id<MTLCommandBuffer> cmdBuffer,
                                  id<MTLBuffer> bufAttn,
                                  id<MTLBuffer> bufProj,
                                  id<MTLBuffer> bufXnorm,
                                  id<MTLBuffer> bufGate, /* must hold hidden*2 floats */
                                  id<MTLBuffer> bufFfnOut,
                                  int dim, int q_dim, int hidden,
                                  const uint16_t *wo_bf16,
                                  const float *ffn_norm, float eps,
                                  const float *ada_scale,
                                  const uint16_t *w1_bf16,
                                  const uint16_t *w3_bf16,
                                  const uint16_t *w2_bf16) {
    int M = 1;

    id<MTLBuffer> bufWo = get_cached_bf16_as_f16_buffer(wo_bf16, (size_t)dim * q_dim);
    id<MTLBuffer> bufW1W3 = get_merged_f16_2(w1_bf16, (size_t)hidden * dim,
                                               w3_bf16, (size_t)hidden * dim);
    id<MTLBuffer> bufW2 = get_cached_bf16_as_f16_buffer(w2_bf16, (size_t)dim * hidden);
    id<MTLBuffer> bufNorm = get_cached_weight_buffer(ffn_norm, dim * sizeof(float));
    id<MTLBuffer> bufAda = ada_scale ?
        get_cached_weight_buffer(ada_scale, dim * sizeof(float)) : nil;
    int fused_wo_residual = 0;

    /* Step 1: proj = attn_out @ wo^T (or fused x += attn_out @ wo^T) */
    if (g_decoder_wo_residual_pipeline) {
        id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
        [enc setComputePipelineState:g_decoder_wo_residual_pipeline];
        [enc setBuffer:g_dec_x offset:0 atIndex:0];
        [enc setBuffer:bufAttn offset:0 atIndex:1];
        [enc setBuffer:bufWo offset:0 atIndex:2];
        [enc setBytes:&q_dim length:sizeof(int) atIndex:3];
        [enc setBytes:&dim length:sizeof(int) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)dim, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        fused_wo_residual = 1;
    } else {
        MPSMatrixDescriptor *descA = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:q_dim
                            rowBytes:q_dim * sizeof(float)
                            dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descW = [MPSMatrixDescriptor
            matrixDescriptorWithRows:dim columns:q_dim
                            rowBytes:q_dim * sizeof(uint16_t)
                            dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor *descOut = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:dim
                            rowBytes:dim * sizeof(float)
                            dataType:MPSDataTypeFloat32];
        MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufAttn descriptor:descA];
        MPSMatrix *matW = [[MPSMatrix alloc] initWithBuffer:bufWo descriptor:descW];
        MPSMatrix *matOut = [[MPSMatrix alloc] initWithBuffer:bufProj descriptor:descOut];
        MPSMatrixMultiplication *mm =
            get_cached_matmul_op(NO, YES, M, dim, q_dim, 1.0, 0.0);
        if (mm)
            [mm encodeToCommandBuffer:cmdBuffer leftMatrix:matA rightMatrix:matW resultMatrix:matOut];
    }

    /* Steps 2+3+4: [x += proj], x_norm = rms_norm(x), x_norm *= (1+ada_scale) */
    {
        int n = M * dim;
        id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];

        if (!fused_wo_residual) {
            /* x += proj */
            [enc setComputePipelineState:g_add_inplace_pipeline];
            [enc setBuffer:g_dec_x offset:0 atIndex:0];
            [enc setBuffer:bufProj offset:0 atIndex:1];
            [enc setBytes:&n length:sizeof(int) atIndex:2];
            NSUInteger tgSize = MIN((NSUInteger)n, g_add_inplace_pipeline.maxTotalThreadsPerThreadgroup);
            [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        }

        /* x_norm = rms_norm(x, ffn_norm) */
        [enc setComputePipelineState:g_rms_norm_pipeline];
        [enc setBuffer:g_dec_x offset:0 atIndex:0];
        [enc setBuffer:bufNorm offset:0 atIndex:1];
        [enc setBuffer:bufXnorm offset:0 atIndex:2];
        [enc setBytes:&dim length:sizeof(int) atIndex:3];
        [enc setBytes:&eps length:sizeof(float) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)M, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        /* x_norm *= (1 + ada_scale) */
        if (bufAda) {
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            [enc setComputePipelineState:g_ada_scale_mul_pipeline];
            [enc setBuffer:bufXnorm offset:0 atIndex:0];
            [enc setBuffer:bufAda offset:0 atIndex:1];
            [enc setBytes:&n length:sizeof(int) atIndex:2];
            [enc setBytes:&dim length:sizeof(int) atIndex:3];
            NSUInteger tgSize;
            tgSize = MIN((NSUInteger)n, g_ada_scale_mul_pipeline.maxTotalThreadsPerThreadgroup);
            [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
        }

        [enc endEncoding];
    }

    /* Step 5+6+7+8: FFN gate in one kernel (fallback to old path if unavailable). */
    if (g_decoder_ffn_gate_pipeline) {
        id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
        [enc setComputePipelineState:g_decoder_ffn_gate_pipeline];
        [enc setBuffer:bufXnorm offset:0 atIndex:0];
        [enc setBuffer:bufW1W3 offset:0 atIndex:1];
        [enc setBuffer:bufGate offset:0 atIndex:2];
        [enc setBytes:&dim length:sizeof(int) atIndex:3];
        [enc setBytes:&hidden length:sizeof(int) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)hidden, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    } else {
        /* Step 5+6: [gate; up] = x_norm @ [w1; w3]^T (merged matmul) */
        {
            int hidden2 = hidden * 2;
            MPSMatrixDescriptor *descIn = [MPSMatrixDescriptor
                matrixDescriptorWithRows:M columns:dim
                                rowBytes:dim * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *descW = [MPSMatrixDescriptor
                matrixDescriptorWithRows:hidden2 columns:dim
                                rowBytes:dim * sizeof(uint16_t)
                                dataType:MPSDataTypeFloat16];
            MPSMatrixDescriptor *descH = [MPSMatrixDescriptor
                matrixDescriptorWithRows:M columns:hidden2
                                rowBytes:hidden2 * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrix *matIn = [[MPSMatrix alloc] initWithBuffer:bufXnorm descriptor:descIn];
            MPSMatrix *matW = [[MPSMatrix alloc] initWithBuffer:bufW1W3 descriptor:descW];
            MPSMatrix *matGateUp = [[MPSMatrix alloc] initWithBuffer:bufGate descriptor:descH];
            MPSMatrixMultiplication *mm =
                get_cached_matmul_op(NO, YES, M, hidden2, dim, 1.0, 0.0);
            if (mm)
                [mm encodeToCommandBuffer:cmdBuffer leftMatrix:matIn rightMatrix:matW resultMatrix:matGateUp];
        }

        /* Steps 7+8: silu(gate), gate *= up — both in bufGate at different offsets */
        {
            int n = M * hidden;
            size_t up_offset = (size_t)hidden * sizeof(float);
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];

            /* silu on gate portion [0:hidden] */
            [enc setComputePipelineState:g_silu_pipeline];
            [enc setBuffer:bufGate offset:0 atIndex:0];
            [enc setBytes:&n length:sizeof(int) atIndex:1];
            NSUInteger tgSize = MIN((NSUInteger)n, g_silu_pipeline.maxTotalThreadsPerThreadgroup);
            [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];

            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

            /* gate[0:hidden] *= up[hidden:hidden*2] */
            [enc setComputePipelineState:g_mul_inplace_pipeline];
            [enc setBuffer:bufGate offset:0 atIndex:0];
            [enc setBuffer:bufGate offset:up_offset atIndex:1];
            [enc setBytes:&n length:sizeof(int) atIndex:2];
            tgSize = MIN((NSUInteger)n, g_mul_inplace_pipeline.maxTotalThreadsPerThreadgroup);
            [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];

            [enc endEncoding];
        }
    }

    /* Step 9+10: W2 matmul + residual add (fallback to old path if unavailable). */
    if (g_decoder_w2_residual_pipeline) {
        id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
        [enc setComputePipelineState:g_decoder_w2_residual_pipeline];
        [enc setBuffer:g_dec_x offset:0 atIndex:0];
        [enc setBuffer:bufGate offset:0 atIndex:1];
        [enc setBuffer:bufW2 offset:0 atIndex:2];
        [enc setBytes:&hidden length:sizeof(int) atIndex:3];
        [enc setBytes:&dim length:sizeof(int) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)dim, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    } else {
        /* Step 9: ffn_out = gate @ w2^T */
        {
            MPSMatrixDescriptor *descH = [MPSMatrixDescriptor
                matrixDescriptorWithRows:M columns:hidden
                                rowBytes:hidden * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *descW = [MPSMatrixDescriptor
                matrixDescriptorWithRows:dim columns:hidden
                                rowBytes:hidden * sizeof(uint16_t)
                                dataType:MPSDataTypeFloat16];
            MPSMatrixDescriptor *descOut = [MPSMatrixDescriptor
                matrixDescriptorWithRows:M columns:dim
                                rowBytes:dim * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrix *matGate = [[MPSMatrix alloc] initWithBuffer:bufGate descriptor:descH];
            MPSMatrix *matW = [[MPSMatrix alloc] initWithBuffer:bufW2 descriptor:descW];
            MPSMatrix *matOut = [[MPSMatrix alloc] initWithBuffer:bufFfnOut descriptor:descOut];
            MPSMatrixMultiplication *mm =
                get_cached_matmul_op(NO, YES, M, dim, hidden, 1.0, 0.0);
            if (mm)
                [mm encodeToCommandBuffer:cmdBuffer leftMatrix:matGate rightMatrix:matW resultMatrix:matOut];
        }

        /* Step 10: x += ffn_out */
        {
            int n = M * dim;
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_add_inplace_pipeline];
            [enc setBuffer:g_dec_x offset:0 atIndex:0];
            [enc setBuffer:bufFfnOut offset:0 atIndex:1];
            [enc setBytes:&n length:sizeof(int) atIndex:2];
            NSUInteger tgSize = MIN((NSUInteger)n, g_add_inplace_pipeline.maxTotalThreadsPerThreadgroup);
            [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)];
            [enc endEncoding];
        }
    }
}

/* Helper: encode rms_norm + QKV matmuls from g_dec_x into a command buffer.
 * Appends to an already-open command buffer. */
static void encode_norm_qkv_steps(id<MTLCommandBuffer> cmdBuffer,
                                    id<MTLBuffer> bufXnorm,
                                    id<MTLBuffer> bufQKV, /* merged output: Q,K,V contiguous */
                                    int K,
                                    const float *norm_weight, float eps,
                                    const uint16_t *wq_bf16, int Nq,
                                    const uint16_t *wk_bf16, int Nk,
                                    const uint16_t *wv_bf16, int Nv) {
    int M = 1;
    int Nqkv = Nq + Nk + Nv;

    id<MTLBuffer> bufWqkv = get_merged_f16_3(wq_bf16, (size_t)Nq * K,
                                               wk_bf16, (size_t)Nk * K,
                                               wv_bf16, (size_t)Nv * K);
    id<MTLBuffer> bufNorm = get_cached_weight_buffer(norm_weight, K * sizeof(float));

    /* rms_norm(g_dec_x, norm_weight) → bufXnorm */
    {
        id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
        [enc setComputePipelineState:g_rms_norm_pipeline];
        [enc setBuffer:g_dec_x offset:0 atIndex:0];
        [enc setBuffer:bufNorm offset:0 atIndex:1];
        [enc setBuffer:bufXnorm offset:0 atIndex:2];
        int hidden = K;
        [enc setBytes:&hidden length:sizeof(int) atIndex:3];
        [enc setBytes:&eps length:sizeof(float) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)M, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
    }

    /* QKV = x_norm @ [Wq; Wk; Wv]^T — single merged matmul */
    {
        MPSMatrixDescriptor *descInput = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:K
                            rowBytes:K * sizeof(float)
                            dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descW = [MPSMatrixDescriptor
            matrixDescriptorWithRows:Nqkv columns:K
                            rowBytes:K * sizeof(uint16_t)
                            dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor *descOut = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:Nqkv
                            rowBytes:Nqkv * sizeof(float)
                            dataType:MPSDataTypeFloat32];
        MPSMatrix *matInput = [[MPSMatrix alloc] initWithBuffer:bufXnorm descriptor:descInput];
        MPSMatrix *matW = [[MPSMatrix alloc] initWithBuffer:bufWqkv descriptor:descW];
        MPSMatrix *matOut = [[MPSMatrix alloc] initWithBuffer:bufQKV descriptor:descOut];
        MPSMatrixMultiplication *mm =
            get_cached_matmul_op(NO, YES, M, Nqkv, K, 1.0, 0.0);
        if (mm)
            [mm encodeToCommandBuffer:cmdBuffer leftMatrix:matInput rightMatrix:matW resultMatrix:matOut];
    }
    /* Output layout: bufQKV = [Q (Nq), K (Nk), V (Nv)] contiguous.
     * Caller uses buffer offsets for K at Nq*sizeof(float) and V at (Nq+Nk)*sizeof(float). */
}

void vox_metal_decoder_norm_qkv(int K,
                                  const float *norm_weight, float eps,
                                  const uint16_t *wq_bf16, int Nq,
                                  const uint16_t *wk_bf16, int Nk,
                                  const uint16_t *wv_bf16, int Nv,
                                  float *q_out, float *k_out, float *v_out) {
    if (!g_initialized || !g_shaders_initialized || !g_dec_x) return;

    @autoreleasepool {
        size_t sizeDim = (size_t)K * sizeof(float);
        size_t sizeQKV = (size_t)(Nq + Nk + Nv) * sizeof(float);

        id<MTLBuffer> bufXnorm = pool_get_buffer(sizeDim);
        id<MTLBuffer> bufQKV = pool_get_buffer(sizeQKV);

        if (!bufXnorm || !bufQKV) {
            pool_release_buffer(bufXnorm);
            pool_release_buffer(bufQKV);
            return;
        }

        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];
        encode_norm_qkv_steps(cmdBuffer, bufXnorm, bufQKV,
                              K, norm_weight, eps,
                              wq_bf16, Nq, wk_bf16, Nk, wv_bf16, Nv);

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        const char *qkv = (const char *)[bufQKV contents];
        memcpy(q_out, qkv, (size_t)Nq * sizeof(float));
        memcpy(k_out, qkv + (size_t)Nq * sizeof(float), (size_t)Nk * sizeof(float));
        memcpy(v_out, qkv + (size_t)(Nq + Nk) * sizeof(float), (size_t)Nv * sizeof(float));

        pool_release_buffer(bufXnorm);
        pool_release_buffer(bufQKV);
    }
}

void vox_metal_decoder_wo_ffn_next_qkv(int dim, int q_dim, int hidden,
                                         const float *attn_out,
                                         const uint16_t *wo_bf16,
                                         const float *ffn_norm, float eps,
                                         const float *ada_scale,
                                         const uint16_t *w1_bf16,
                                         const uint16_t *w3_bf16,
                                         const uint16_t *w2_bf16,
                                         const float *next_attn_norm,
                                         const uint16_t *next_wq_bf16, int next_Nq,
                                         const uint16_t *next_wk_bf16, int next_Nk,
                                         const uint16_t *next_wv_bf16, int next_Nv,
                                         float *q_out, float *k_out, float *v_out) {
    if (!g_initialized || !g_shaders_initialized || !g_dec_x) return;

    @autoreleasepool {
        size_t sizeAttn = (size_t)q_dim * sizeof(float);
        size_t sizeDim = (size_t)dim * sizeof(float);
        size_t sizeHidden = (size_t)hidden * sizeof(float);
        size_t sizeQKV = (size_t)(next_Nq + next_Nk + next_Nv) * sizeof(float);

        /* Scratch buffers */
        id<MTLBuffer> bufAttn = pool_get_buffer(sizeAttn);
        if (bufAttn) memcpy([bufAttn contents], attn_out, sizeAttn);
        id<MTLBuffer> bufProj = pool_get_buffer(sizeDim);
        id<MTLBuffer> bufXnorm = pool_get_buffer(sizeDim);
        id<MTLBuffer> bufGate = pool_get_buffer(sizeHidden * 2);
        id<MTLBuffer> bufFfnOut = pool_get_buffer(sizeDim);
        id<MTLBuffer> bufQKV = pool_get_buffer(sizeQKV);

        if (!bufAttn || !bufProj || !bufXnorm || !bufGate ||
            !bufFfnOut || !bufQKV) {
            pool_release_buffer(bufAttn);
            pool_release_buffer(bufProj);
            pool_release_buffer(bufXnorm);
            pool_release_buffer(bufGate);
            pool_release_buffer(bufFfnOut);
            pool_release_buffer(bufQKV);
            return;
        }

        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];

        /* Steps 1-10: wo + FFN (updates g_dec_x) */
        encode_wo_ffn_steps(cmdBuffer, bufAttn, bufProj, bufXnorm,
                            bufGate, bufFfnOut,
                            dim, q_dim, hidden,
                            wo_bf16, ffn_norm, eps, ada_scale,
                            w1_bf16, w3_bf16, w2_bf16);

        /* Steps 11-14: norm + QKV for next layer (merged matmul) */
        encode_norm_qkv_steps(cmdBuffer, bufXnorm, bufQKV,
                              dim, next_attn_norm, eps,
                              next_wq_bf16, next_Nq,
                              next_wk_bf16, next_Nk,
                              next_wv_bf16, next_Nv);

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        const char *qkv = (const char *)[bufQKV contents];
        memcpy(q_out, qkv, (size_t)next_Nq * sizeof(float));
        memcpy(k_out, qkv + (size_t)next_Nq * sizeof(float), (size_t)next_Nk * sizeof(float));
        memcpy(v_out, qkv + (size_t)(next_Nq + next_Nk) * sizeof(float), (size_t)next_Nv * sizeof(float));

        pool_release_buffer(bufAttn);
        pool_release_buffer(bufProj);
        pool_release_buffer(bufXnorm);
        pool_release_buffer(bufGate);
        pool_release_buffer(bufFfnOut);
        pool_release_buffer(bufQKV);
    }
}

int vox_metal_decoder_wo_ffn_logits(int dim, int q_dim, int hidden, int vocab_size,
                                      const float *attn_out,
                                      const uint16_t *wo_bf16,
                                      const float *ffn_norm, float eps,
                                      const float *ada_scale,
                                      const uint16_t *w1_bf16,
                                      const uint16_t *w3_bf16,
                                      const uint16_t *w2_bf16,
                                      const float *final_norm,
                                      const uint16_t *tok_emb_bf16,
                                      float *logits_out) {
    if (!g_initialized || !g_shaders_initialized || !g_dec_x) return 0;

    int result = 0;

    @autoreleasepool {
        size_t sizeAttn = (size_t)q_dim * sizeof(float);
        size_t sizeDim = (size_t)dim * sizeof(float);
        size_t sizeHidden = (size_t)hidden * sizeof(float);
        size_t sizeLogits = (size_t)vocab_size * sizeof(float);

        id<MTLBuffer> bufEmb = get_cached_bf16_as_f16_buffer(tok_emb_bf16,
                                                               (size_t)vocab_size * dim);
        id<MTLBuffer> bufFinalNorm = get_cached_weight_buffer(final_norm, sizeDim);

        /* Scratch buffers */
        id<MTLBuffer> bufAttn = pool_get_buffer(sizeAttn);
        if (bufAttn) memcpy([bufAttn contents], attn_out, sizeAttn);
        id<MTLBuffer> bufProj = pool_get_buffer(sizeDim);
        id<MTLBuffer> bufXnorm = pool_get_buffer(sizeDim);
        id<MTLBuffer> bufGate = pool_get_buffer(sizeHidden * 2);
        id<MTLBuffer> bufFfnOut = pool_get_buffer(sizeDim);
        id<MTLBuffer> bufLogits = pool_get_buffer(sizeLogits);
        id<MTLBuffer> bufArgmax = pool_get_buffer(sizeof(int));

        if (!bufAttn || !bufProj || !bufXnorm || !bufGate ||
            !bufFfnOut || !bufLogits || !bufArgmax || !bufEmb || !bufFinalNorm) {
            pool_release_buffer(bufAttn);
            pool_release_buffer(bufProj);
            pool_release_buffer(bufXnorm);
            pool_release_buffer(bufGate);
            pool_release_buffer(bufFfnOut);
            pool_release_buffer(bufLogits);
            pool_release_buffer(bufArgmax);
            return 0;
        }

        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];

        /* Steps 1-10: wo + FFN (updates g_dec_x) */
        encode_wo_ffn_steps(cmdBuffer, bufAttn, bufProj, bufXnorm,
                            bufGate, bufFfnOut,
                            dim, q_dim, hidden,
                            wo_bf16, ffn_norm, eps, ada_scale,
                            w1_bf16, w3_bf16, w2_bf16);

        /* Step 11: final rms_norm(g_dec_x, final_norm) → bufXnorm */
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_rms_norm_pipeline];
            [enc setBuffer:g_dec_x offset:0 atIndex:0];
            [enc setBuffer:bufFinalNorm offset:0 atIndex:1];
            [enc setBuffer:bufXnorm offset:0 atIndex:2];
            [enc setBytes:&dim length:sizeof(int) atIndex:3];
            [enc setBytes:&eps length:sizeof(float) atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        /* Step 12: logits = x_norm @ tok_emb^T */
        {
            MPSMatrixDescriptor *descIn = [MPSMatrixDescriptor
                matrixDescriptorWithRows:1 columns:dim
                                rowBytes:dim * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *descW = [MPSMatrixDescriptor
                matrixDescriptorWithRows:vocab_size columns:dim
                                rowBytes:dim * sizeof(uint16_t)
                                dataType:MPSDataTypeFloat16];
            MPSMatrixDescriptor *descOut = [MPSMatrixDescriptor
                matrixDescriptorWithRows:1 columns:vocab_size
                                rowBytes:vocab_size * sizeof(float)
                                dataType:MPSDataTypeFloat32];
            MPSMatrix *matIn = [[MPSMatrix alloc] initWithBuffer:bufXnorm descriptor:descIn];
            MPSMatrix *matW = [[MPSMatrix alloc] initWithBuffer:bufEmb descriptor:descW];
            MPSMatrix *matOut = [[MPSMatrix alloc] initWithBuffer:bufLogits descriptor:descOut];
            MPSMatrixMultiplication *mm =
                get_cached_matmul_op(NO, YES, 1, vocab_size, dim, 1.0, 0.0);
            if (mm)
                [mm encodeToCommandBuffer:cmdBuffer leftMatrix:matIn rightMatrix:matW resultMatrix:matOut];
        }

        /* Step 13: argmax on GPU */
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_argmax_pipeline];
            [enc setBuffer:bufLogits offset:0 atIndex:0];
            [enc setBuffer:bufArgmax offset:0 atIndex:1];
            [enc setBytes:&vocab_size length:sizeof(int) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        result = ((int *)[bufArgmax contents])[0];

        if (logits_out)
            memcpy(logits_out, [bufLogits contents], sizeLogits);

        pool_release_buffer(bufAttn);
        pool_release_buffer(bufProj);
        pool_release_buffer(bufXnorm);
        pool_release_buffer(bufGate);
        pool_release_buffer(bufFfnOut);
        pool_release_buffer(bufLogits);
        pool_release_buffer(bufArgmax);
    }

    return result;
}

/* ========================================================================
 * GPU Batched Attention
 *
 * All heads processed in one command buffer:
 *   1. QK^T matmul per head (strided views, alpha=scale)
 *   2. Causal masked softmax (compute shader)
 *   3. scores * V matmul per head (strided views)
 *
 * Q:   [seq_q, n_heads * head_dim]
 * K:   [seq_k, n_kv_heads * head_dim]
 * V:   [seq_k, n_kv_heads * head_dim]
 * out: [seq_q, n_heads * head_dim]
 * ======================================================================== */

void vox_metal_batched_attention(float *out,
                                  const float *Q, const float *K, const float *V,
                                  int seq_q, int seq_k,
                                  int n_heads, int n_kv_heads,
                                  int head_dim, float scale,
                                  int window_size, int q_offset) {
    if (!g_initialized || !g_causal_softmax_pipeline) return;

    @autoreleasepool {
        int gqa_ratio = n_heads / n_kv_heads;
        size_t q_total = (size_t)seq_q * n_heads * head_dim;
        size_t k_total = (size_t)seq_k * n_kv_heads * head_dim;
        size_t scores_total = (size_t)n_heads * seq_q * seq_k;
        size_t out_total = q_total;

        /* Copy Q, K, V to GPU */
        id<MTLBuffer> bufQ = pool_get_buffer(q_total * sizeof(float));
        id<MTLBuffer> bufK = pool_get_buffer(k_total * sizeof(float));
        id<MTLBuffer> bufV = pool_get_buffer(k_total * sizeof(float));
        id<MTLBuffer> bufScores = pool_get_buffer(scores_total * sizeof(float));
        id<MTLBuffer> bufOut = pool_get_buffer(out_total * sizeof(float));

        if (!bufQ || !bufK || !bufV || !bufScores || !bufOut) {
            pool_release_buffer(bufQ);
            pool_release_buffer(bufK);
            pool_release_buffer(bufV);
            pool_release_buffer(bufScores);
            pool_release_buffer(bufOut);
            return;
        }

        memcpy([bufQ contents], Q, q_total * sizeof(float));
        memcpy([bufK contents], K, k_total * sizeof(float));
        memcpy([bufV contents], V, k_total * sizeof(float));

        /* Row strides for packed [seq, heads * head_dim] layout */
        size_t q_row_bytes = (size_t)n_heads * head_dim * sizeof(float);
        size_t kv_row_bytes = (size_t)n_kv_heads * head_dim * sizeof(float);
        size_t scores_row_bytes = (size_t)seq_k * sizeof(float);
        size_t out_row_bytes = q_row_bytes;

        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];
        MPSMatrixMultiplication *mm_qk =
            get_cached_matmul_op(NO, YES, seq_q, seq_k, head_dim, (double)scale, 0.0);
        MPSMatrixMultiplication *mm_sv =
            get_cached_matmul_op(NO, NO, seq_q, head_dim, seq_k, 1.0, 0.0);
        if (!mm_qk || !mm_sv) {
            pool_release_buffer(bufQ);
            pool_release_buffer(bufK);
            pool_release_buffer(bufV);
            pool_release_buffer(bufScores);
            pool_release_buffer(bufOut);
            return;
        }

        /* --- Step 1: QK^T per head --- */
        for (int h = 0; h < n_heads; h++) {
            int kv_h = h / gqa_ratio;

            /* Q_h: strided view into packed Q */
            MPSMatrixDescriptor *descQh = [MPSMatrixDescriptor
                matrixDescriptorWithRows:seq_q columns:head_dim
                                rowBytes:q_row_bytes
                                dataType:MPSDataTypeFloat32];
            MPSMatrix *matQh = [[MPSMatrix alloc]
                initWithBuffer:bufQ
                        offset:(size_t)h * head_dim * sizeof(float)
                    descriptor:descQh];

            /* K_h: strided view into packed K */
            MPSMatrixDescriptor *descKh = [MPSMatrixDescriptor
                matrixDescriptorWithRows:seq_k columns:head_dim
                                rowBytes:kv_row_bytes
                                dataType:MPSDataTypeFloat32];
            MPSMatrix *matKh = [[MPSMatrix alloc]
                initWithBuffer:bufK
                        offset:(size_t)kv_h * head_dim * sizeof(float)
                    descriptor:descKh];

            /* scores_h: contiguous [seq_q, seq_k] */
            MPSMatrixDescriptor *descSh = [MPSMatrixDescriptor
                matrixDescriptorWithRows:seq_q columns:seq_k
                                rowBytes:scores_row_bytes
                                dataType:MPSDataTypeFloat32];
            MPSMatrix *matSh = [[MPSMatrix alloc]
                initWithBuffer:bufScores
                        offset:(size_t)h * seq_q * seq_k * sizeof(float)
                    descriptor:descSh];

            /* scores_h = scale * Q_h @ K_h^T */
            [mm_qk encodeToCommandBuffer:cmdBuffer
                           leftMatrix:matQh rightMatrix:matKh resultMatrix:matSh];
        }

        /* --- Step 2: Causal masked softmax --- */
        {
            id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
            [enc setComputePipelineState:g_causal_softmax_pipeline];
            [enc setBuffer:bufScores offset:0 atIndex:0];
            [enc setBytes:&seq_q length:sizeof(int) atIndex:1];
            [enc setBytes:&seq_k length:sizeof(int) atIndex:2];
            [enc setBytes:&window_size length:sizeof(int) atIndex:3];
            [enc setBytes:&q_offset length:sizeof(int) atIndex:4];
            /* 1D grid: one threadgroup per (head * seq_q + qi) */
            NSUInteger total_groups = (NSUInteger)n_heads * (NSUInteger)seq_q;
            [enc dispatchThreadgroups:MTLSizeMake(total_groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
        }

        /* --- Step 3: scores * V per head --- */
        for (int h = 0; h < n_heads; h++) {
            int kv_h = h / gqa_ratio;

            /* scores_h: contiguous [seq_q, seq_k] */
            MPSMatrixDescriptor *descSh = [MPSMatrixDescriptor
                matrixDescriptorWithRows:seq_q columns:seq_k
                                rowBytes:scores_row_bytes
                                dataType:MPSDataTypeFloat32];
            MPSMatrix *matSh = [[MPSMatrix alloc]
                initWithBuffer:bufScores
                        offset:(size_t)h * seq_q * seq_k * sizeof(float)
                    descriptor:descSh];

            /* V_h: strided view into packed V */
            MPSMatrixDescriptor *descVh = [MPSMatrixDescriptor
                matrixDescriptorWithRows:seq_k columns:head_dim
                                rowBytes:kv_row_bytes
                                dataType:MPSDataTypeFloat32];
            MPSMatrix *matVh = [[MPSMatrix alloc]
                initWithBuffer:bufV
                        offset:(size_t)kv_h * head_dim * sizeof(float)
                    descriptor:descVh];

            /* out_h: strided view into packed output */
            MPSMatrixDescriptor *descOh = [MPSMatrixDescriptor
                matrixDescriptorWithRows:seq_q columns:head_dim
                                rowBytes:out_row_bytes
                                dataType:MPSDataTypeFloat32];
            MPSMatrix *matOh = [[MPSMatrix alloc]
                initWithBuffer:bufOut
                        offset:(size_t)h * head_dim * sizeof(float)
                    descriptor:descOh];

            /* out_h = scores_h @ V_h */
            [mm_sv encodeToCommandBuffer:cmdBuffer
                           leftMatrix:matSh rightMatrix:matVh resultMatrix:matOh];
        }

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        memcpy(out, [bufOut contents], out_total * sizeof(float));

        pool_release_buffer(bufQ);
        pool_release_buffer(bufK);
        pool_release_buffer(bufV);
        pool_release_buffer(bufScores);
        pool_release_buffer(bufOut);
    }
}

/* ========================================================================
 * Fused Encoder Attention (single compute dispatch, all heads)
 * Replaces 64 per-head MPS matmul encodes with 1 compute kernel.
 * ======================================================================== */

void vox_metal_encoder_attention(float *out,
                                   const float *Q, const float *K, const float *V,
                                   int seq_q, int seq_k,
                                   int n_heads, int n_kv_heads,
                                   int head_dim, float scale,
                                   int window_size, int q_offset) {
    if (!g_initialized || !g_encoder_attention_pipeline) return;

    @autoreleasepool {
        size_t q_total = (size_t)seq_q * n_heads * head_dim;
        size_t k_total = (size_t)seq_k * n_kv_heads * head_dim;
        size_t out_total = q_total;

        id<MTLBuffer> bufQ = pool_get_buffer(q_total * sizeof(float));
        id<MTLBuffer> bufK = pool_get_buffer(k_total * sizeof(float));
        id<MTLBuffer> bufV = pool_get_buffer(k_total * sizeof(float));
        id<MTLBuffer> bufOut = pool_get_buffer(out_total * sizeof(float));

        if (!bufQ || !bufK || !bufV || !bufOut) {
            pool_release_buffer(bufQ);
            pool_release_buffer(bufK);
            pool_release_buffer(bufV);
            pool_release_buffer(bufOut);
            return;
        }

        memcpy([bufQ contents], Q, q_total * sizeof(float));
        memcpy([bufK contents], K, k_total * sizeof(float));
        memcpy([bufV contents], V, k_total * sizeof(float));

        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
        [enc setComputePipelineState:g_encoder_attention_pipeline];
        [enc setBuffer:bufQ offset:0 atIndex:0];
        [enc setBuffer:bufK offset:0 atIndex:1];
        [enc setBuffer:bufV offset:0 atIndex:2];
        [enc setBuffer:bufOut offset:0 atIndex:3];
        [enc setBytes:&n_heads length:sizeof(int) atIndex:4];
        [enc setBytes:&n_kv_heads length:sizeof(int) atIndex:5];
        [enc setBytes:&head_dim length:sizeof(int) atIndex:6];
        [enc setBytes:&seq_q length:sizeof(int) atIndex:7];
        [enc setBytes:&seq_k length:sizeof(int) atIndex:8];
        [enc setBytes:&scale length:sizeof(float) atIndex:9];
        [enc setBytes:&window_size length:sizeof(int) atIndex:10];
        [enc setBytes:&q_offset length:sizeof(int) atIndex:11];

        /* 1D grid: n_heads * ceil(seq_q/BQ) threadgroups, Q-tiled attention */
        int bq = 8; /* must match ATTN_BQ in shader */
        int n_q_blocks = (seq_q + bq - 1) / bq;
        NSUInteger total_groups = (NSUInteger)n_heads * (NSUInteger)n_q_blocks;
        [enc dispatchThreadgroups:MTLSizeMake(total_groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc endEncoding];

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        memcpy(out, [bufOut contents], out_total * sizeof(float));

        pool_release_buffer(bufQ);
        pool_release_buffer(bufK);
        pool_release_buffer(bufV);
        pool_release_buffer(bufOut);
    }
}

/* ========================================================================
 * GPU-Shared Memory Allocation
 * ======================================================================== */

void *vox_metal_shared_alloc(size_t size) {
    if (!g_initialized || g_shared_count >= SHARED_ALLOC_MAX) return calloc(1, size);
    id<MTLBuffer> buf = [g_device newBufferWithLength:size
                                              options:MTLResourceStorageModeShared];
    if (!buf) return calloc(1, size);
    void *ptr = [buf contents];
    memset(ptr, 0, size);
    g_shared_allocs[g_shared_count].ptr = ptr;
    g_shared_allocs[g_shared_count].buf = buf;
    g_shared_count++;
    return ptr;
}

void vox_metal_shared_free(void *ptr) {
    if (!ptr) return;
    for (int i = 0; i < g_shared_count; i++) {
        if (g_shared_allocs[i].ptr == ptr) {
            g_shared_allocs[i].buf = nil;
            g_shared_allocs[i] = g_shared_allocs[--g_shared_count];
            return;
        }
    }
    free(ptr); /* fallback: not a shared allocation */
}

static id<MTLBuffer> find_shared_buffer(void *ptr) {
    for (int i = 0; i < g_shared_count; i++) {
        if (g_shared_allocs[i].ptr == ptr) return g_shared_allocs[i].buf;
    }
    return nil;
}

/* ========================================================================
 * Monolithic Decoder Step: all 26 layers + logits in ONE command buffer
 * ======================================================================== */

#include "voxtral.h"

int vox_metal_decoder_full_step(void *ctx_ptr, const float *rope_freqs, float *logits_out) {
    if (!g_initialized || !g_shaders_initialized || !g_dec_x) return -1;

    vox_ctx_t *ctx = (vox_ctx_t *)ctx_ptr;
    vox_decoder_t *dec = &ctx->decoder;

    int dim = VOX_DEC_DIM;
    int n_heads = VOX_DEC_HEADS;
    int n_kv_heads = VOX_DEC_KV_HEADS;
    int head_dim = VOX_DEC_HEAD_DIM;
    int hidden = VOX_DEC_HIDDEN;
    int q_dim = n_heads * head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int pos = ctx->kv_cache_len;
    int total_seq = pos + 1;
    float scale = 1.0f / sqrtf((float)head_dim);
    int kv_fp16 = ctx->kv_cache_fp16;
    size_t kv_elem_size = kv_fp16 ? sizeof(uint16_t) : sizeof(float);

    /* Find GPU buffer handles for KV cache (allocated with shared_alloc) */
    void *kv_k_ptr = kv_fp16 ? (void *)ctx->kv_cache_k_f16 : (void *)ctx->kv_cache_k;
    void *kv_v_ptr = kv_fp16 ? (void *)ctx->kv_cache_v_f16 : (void *)ctx->kv_cache_v;
    id<MTLBuffer> gpu_kv_k = find_shared_buffer(kv_k_ptr);
    id<MTLBuffer> gpu_kv_v = find_shared_buffer(kv_v_ptr);
    if (!gpu_kv_k || !gpu_kv_v) return -1;
    if (kv_fp16 && (!g_kv_cache_copy_f16_pipeline || !g_decoder_attention_f16_pipeline))
        return -1;

    int result = 0;

    @autoreleasepool {
        /* Scratch buffers — reused across all 26 layers within this cmd buf */
        id<MTLBuffer> bufXnorm = pool_get_buffer(dim * sizeof(float));
        id<MTLBuffer> bufQKV = pool_get_buffer((q_dim + kv_dim + kv_dim) * sizeof(float));
        id<MTLBuffer> bufAttn = pool_get_buffer(q_dim * sizeof(float));
        id<MTLBuffer> bufProj = pool_get_buffer(dim * sizeof(float));
        id<MTLBuffer> bufGate = pool_get_buffer(hidden * 2 * sizeof(float));
        id<MTLBuffer> bufFfnOut = pool_get_buffer(dim * sizeof(float));
        id<MTLBuffer> bufLogits = pool_get_buffer((size_t)VOX_VOCAB_SIZE * sizeof(float));
        id<MTLBuffer> bufArgmax = pool_get_buffer(sizeof(int));

        /* Upload RoPE frequencies (small: 128 floats = 512 bytes) */
        id<MTLBuffer> bufRope = pool_get_buffer(head_dim * sizeof(float));
        if (bufRope) memcpy([bufRope contents], rope_freqs, head_dim * sizeof(float));

        if (!bufXnorm || !bufQKV || !bufAttn ||
            !bufProj || !bufGate || !bufFfnOut ||
            !bufLogits || !bufArgmax || !bufRope) {
            pool_release_buffer(bufXnorm);
            pool_release_buffer(bufQKV);
            pool_release_buffer(bufAttn);
            pool_release_buffer(bufProj);
            pool_release_buffer(bufGate);
            pool_release_buffer(bufFfnOut);
            pool_release_buffer(bufLogits);
            pool_release_buffer(bufArgmax);
            pool_release_buffer(bufRope);
            return -1;
        }

        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];

        /* ---- 26 decoder layers ---- */
        for (int layer = 0; layer < VOX_DEC_LAYERS; layer++) {
            vox_dec_layer_t *l = &dec->layers[layer];

            /* If not first layer, encode wo+FFN for previous layer first */
            if (layer > 0) {
                vox_dec_layer_t *prev = &dec->layers[layer - 1];
                const float *ada_s = ctx->ada_scale ?
                    ctx->ada_scale + (size_t)(layer - 1) * dim : NULL;

                encode_wo_ffn_steps(cmdBuffer, bufAttn, bufProj, bufXnorm,
                                    bufGate, bufFfnOut,
                                    dim, q_dim, hidden,
                                    prev->wo_weight_bf16,
                                    prev->ffn_norm, VOX_DEC_NORM_EPS, ada_s,
                                    prev->w1_weight_bf16, prev->w3_weight_bf16,
                                    prev->w2_weight_bf16);
            }

            /* RMSNorm + QKV projections (merged into single matmul) */
            encode_norm_qkv_steps(cmdBuffer, bufXnorm, bufQKV,
                                  dim, l->attention_norm, VOX_DEC_NORM_EPS,
                                  l->wq_weight_bf16, q_dim,
                                  l->wk_weight_bf16, kv_dim,
                                  l->wv_weight_bf16, kv_dim);

            /* RoPE + KV cache write + attention in single compute encoder.
             * bufQKV layout: [Q (q_dim), K (kv_dim), V (kv_dim)] */
            {
                int kv_offset = (int)((size_t)layer * ctx->kv_cache_max + pos) * kv_dim;
                size_t layer_kv_offset = (size_t)layer * ctx->kv_cache_max * kv_dim * kv_elem_size;
                int window = VOX_DEC_WINDOW;
                int q_pos_val = ctx->kv_pos_offset + pos;
                size_t off_k = (size_t)q_dim * sizeof(float);
                size_t off_v = (size_t)(q_dim + kv_dim) * sizeof(float);

                id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];

                /* RoPE on Q (at offset 0 in bufQKV) */
                int n_threads_q = n_heads * (head_dim / 2);
                [enc setComputePipelineState:g_rope_apply_pipeline];
                [enc setBuffer:bufQKV offset:0 atIndex:0];
                [enc setBuffer:bufRope offset:0 atIndex:1];
                [enc setBytes:&n_heads length:sizeof(int) atIndex:2];
                [enc setBytes:&head_dim length:sizeof(int) atIndex:3];
                {
                    NSUInteger tg = MIN((NSUInteger)n_threads_q,
                                        g_rope_apply_pipeline.maxTotalThreadsPerThreadgroup);
                    [enc dispatchThreads:MTLSizeMake((NSUInteger)n_threads_q, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                }

                /* RoPE on K (at offset off_k in bufQKV) */
                int n_threads_k = n_kv_heads * (head_dim / 2);
                [enc setBuffer:bufQKV offset:off_k atIndex:0];
                [enc setBytes:&n_kv_heads length:sizeof(int) atIndex:2];
                {
                    NSUInteger tg = MIN((NSUInteger)n_threads_k,
                                        g_rope_apply_pipeline.maxTotalThreadsPerThreadgroup);
                    [enc dispatchThreads:MTLSizeMake((NSUInteger)n_threads_k, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                }

                /* Barrier: RoPE must finish before KV cache write reads K */
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                /* Write K to KV cache (from bufQKV at off_k) */
                id<MTLComputePipelineState> kv_copy_ps = kv_fp16 ?
                    g_kv_cache_copy_f16_pipeline : g_kv_cache_copy_pipeline;
                [enc setComputePipelineState:kv_copy_ps];
                [enc setBuffer:gpu_kv_k offset:0 atIndex:0];
                [enc setBuffer:bufQKV offset:off_k atIndex:1];
                [enc setBytes:&kv_offset length:sizeof(int) atIndex:2];
                [enc setBytes:&kv_dim length:sizeof(int) atIndex:3];
                {
                    NSUInteger tg = MIN((NSUInteger)kv_dim,
                                        kv_copy_ps.maxTotalThreadsPerThreadgroup);
                    [enc dispatchThreads:MTLSizeMake((NSUInteger)kv_dim, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                }

                /* Write V to KV cache (from bufQKV at off_v) */
                [enc setBuffer:gpu_kv_v offset:0 atIndex:0];
                [enc setBuffer:bufQKV offset:off_v atIndex:1];
                [enc dispatchThreads:MTLSizeMake((NSUInteger)kv_dim, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(
                    MIN((NSUInteger)kv_dim,
                        kv_copy_ps.maxTotalThreadsPerThreadgroup), 1, 1)];

                /* Barrier: KV cache must be written before attention reads it */
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

                /* Single-token attention (Q from bufQKV at offset 0) */
                [enc setComputePipelineState:kv_fp16 ?
                    g_decoder_attention_f16_pipeline : g_decoder_attention_pipeline];
                [enc setBuffer:bufQKV offset:0 atIndex:0];
                [enc setBuffer:gpu_kv_k offset:layer_kv_offset atIndex:1];
                [enc setBuffer:gpu_kv_v offset:layer_kv_offset atIndex:2];
                [enc setBuffer:bufAttn offset:0 atIndex:3];
                [enc setBytes:&n_heads length:sizeof(int) atIndex:4];
                [enc setBytes:&n_kv_heads length:sizeof(int) atIndex:5];
                [enc setBytes:&head_dim length:sizeof(int) atIndex:6];
                [enc setBytes:&kv_dim length:sizeof(int) atIndex:7];
                [enc setBytes:&total_seq length:sizeof(int) atIndex:8];
                [enc setBytes:&scale length:sizeof(float) atIndex:9];
                [enc setBytes:&window length:sizeof(int) atIndex:10];
                [enc setBytes:&q_pos_val length:sizeof(int) atIndex:11];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)n_heads, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

                [enc endEncoding];
            }
        }

        /* ---- Final: wo+FFN for last layer + logits + argmax ---- */
        {
            vox_dec_layer_t *last = &dec->layers[VOX_DEC_LAYERS - 1];
            const float *ada_s = ctx->ada_scale ?
                ctx->ada_scale + (size_t)(VOX_DEC_LAYERS - 1) * dim : NULL;

            encode_wo_ffn_steps(cmdBuffer, bufAttn, bufProj, bufXnorm,
                                bufGate, bufFfnOut,
                                dim, q_dim, hidden,
                                last->wo_weight_bf16,
                                last->ffn_norm, VOX_DEC_NORM_EPS, ada_s,
                                last->w1_weight_bf16, last->w3_weight_bf16,
                                last->w2_weight_bf16);

            /* Final RMSNorm */
            id<MTLBuffer> bufFinalNorm = get_cached_weight_buffer(dec->norm,
                                                                    dim * sizeof(float));
            {
                float eps = VOX_DEC_NORM_EPS;
                id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
                [enc setComputePipelineState:g_rms_norm_pipeline];
                [enc setBuffer:g_dec_x offset:0 atIndex:0];
                [enc setBuffer:bufFinalNorm offset:0 atIndex:1];
                [enc setBuffer:bufXnorm offset:0 atIndex:2];
                [enc setBytes:&dim length:sizeof(int) atIndex:3];
                [enc setBytes:&eps length:sizeof(float) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }

            /* Logits = x_norm @ tok_emb^T */
            id<MTLBuffer> bufEmb = get_cached_bf16_as_f16_buffer(
                dec->tok_embeddings_bf16, (size_t)VOX_VOCAB_SIZE * dim);
            {
                int vocab = VOX_VOCAB_SIZE;
                MPSMatrixDescriptor *descIn = [MPSMatrixDescriptor
                    matrixDescriptorWithRows:1 columns:dim
                                    rowBytes:dim * sizeof(float)
                                    dataType:MPSDataTypeFloat32];
                MPSMatrixDescriptor *descW = [MPSMatrixDescriptor
                    matrixDescriptorWithRows:vocab columns:dim
                                    rowBytes:dim * sizeof(uint16_t)
                                    dataType:MPSDataTypeFloat16];
                MPSMatrixDescriptor *descOut = [MPSMatrixDescriptor
                    matrixDescriptorWithRows:1 columns:vocab
                                    rowBytes:vocab * sizeof(float)
                                    dataType:MPSDataTypeFloat32];
                MPSMatrix *matIn = [[MPSMatrix alloc] initWithBuffer:bufXnorm descriptor:descIn];
                MPSMatrix *matW = [[MPSMatrix alloc] initWithBuffer:bufEmb descriptor:descW];
                MPSMatrix *matOut = [[MPSMatrix alloc] initWithBuffer:bufLogits descriptor:descOut];
                MPSMatrixMultiplication *mm =
                    get_cached_matmul_op(NO, YES, 1, vocab, dim, 1.0, 0.0);
                if (mm)
                    [mm encodeToCommandBuffer:cmdBuffer leftMatrix:matIn
                                  rightMatrix:matW resultMatrix:matOut];
            }

            /* Argmax on GPU */
            {
                int vocab = VOX_VOCAB_SIZE;
                id<MTLComputeCommandEncoder> enc = [cmdBuffer computeCommandEncoder];
                [enc setComputePipelineState:g_argmax_pipeline];
                [enc setBuffer:bufLogits offset:0 atIndex:0];
                [enc setBuffer:bufArgmax offset:0 atIndex:1];
                [enc setBytes:&vocab length:sizeof(int) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc endEncoding];
            }
        }

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        result = ((int *)[bufArgmax contents])[0];
        if (logits_out)
            memcpy(logits_out, [bufLogits contents], (size_t)VOX_VOCAB_SIZE * sizeof(float));

        pool_release_buffer(bufXnorm);
        pool_release_buffer(bufQKV);
        pool_release_buffer(bufAttn);
        pool_release_buffer(bufProj);
        pool_release_buffer(bufGate);
        pool_release_buffer(bufFfnOut);
        pool_release_buffer(bufLogits);
        pool_release_buffer(bufArgmax);
        pool_release_buffer(bufRope);
    }

    ctx->kv_cache_len = pos + 1;
    return result;
}

/* ========================================================================
 * Monolithic Encoder Step: all 32 layers + final norm in ONE command buffer
 * ======================================================================== */

int vox_metal_encoder_full_step(void *ctx_ptr, float *x, int new_len,
                                 const float *rope_freqs, int cache_len) {
    if (!g_initialized || !g_shaders_initialized) return -1;

    vox_ctx_t *ctx = (vox_ctx_t *)ctx_ptr;
    vox_encoder_t *enc = &ctx->encoder;

    int dim = VOX_ENC_DIM;          /* 1280 */
    int n_heads = VOX_ENC_HEADS;    /* 32 */
    int n_kv_heads = VOX_ENC_KV_HEADS; /* 32 */
    int head_dim = VOX_ENC_HEAD_DIM;/* 64 */
    int hidden = VOX_ENC_HIDDEN;    /* 5120 */
    int qkv_dim = n_heads * head_dim; /* 2048 */
    int kv_dim = n_kv_heads * head_dim; /* 2048 */
    int M = new_len;
    int total_kv = cache_len + new_len;
    float attn_scale = 1.0f / sqrtf((float)head_dim);
    int window = VOX_ENC_WINDOW;

    /* Find GPU buffer handles for encoder KV cache (allocated with shared_alloc) */
    id<MTLBuffer> gpu_kv_k = find_shared_buffer(ctx->enc_kv_cache_k);
    id<MTLBuffer> gpu_kv_v = find_shared_buffer(ctx->enc_kv_cache_v);
    if (!gpu_kv_k || !gpu_kv_v) return -1;
    if (!g_bias_add_strided_pipeline ||
        !g_batched_rope_apply_strided_pipeline ||
        !g_batched_kv_cache_copy_strided_pipeline ||
        !g_encoder_attention_qstrided_pipeline)
        return -1;

    @autoreleasepool {
        /* Scratch buffers — reused across all 32 layers */
        int qkv_merged = qkv_dim + kv_dim + kv_dim; /* 6144 for merged QKV output */
        int ffn_merged = hidden * 2;                  /* 10240 for merged w1+w3 output */
        id<MTLBuffer> bufX = pool_get_buffer((size_t)M * dim * sizeof(float));
        id<MTLBuffer> bufXnorm = pool_get_buffer((size_t)M * dim * sizeof(float));
        id<MTLBuffer> bufQKV = pool_get_buffer((size_t)M * qkv_merged * sizeof(float));
        id<MTLBuffer> bufAttn = pool_get_buffer((size_t)M * qkv_dim * sizeof(float));
        id<MTLBuffer> bufProj = pool_get_buffer((size_t)M * dim * sizeof(float));
        id<MTLBuffer> bufGate = pool_get_buffer((size_t)M * ffn_merged * sizeof(float));
        id<MTLBuffer> bufFfnOut = pool_get_buffer((size_t)M * dim * sizeof(float));

        /* Upload x and RoPE frequencies */
        if (bufX) memcpy([bufX contents], x, (size_t)M * dim * sizeof(float));
        size_t rope_size = (size_t)M * (head_dim / 2) * 2 * sizeof(float);
        id<MTLBuffer> bufRope = pool_get_buffer(rope_size);
        if (bufRope) memcpy([bufRope contents], rope_freqs, rope_size);

        if (!bufX || !bufXnorm || !bufQKV ||
            !bufAttn || !bufProj || !bufGate || !bufFfnOut || !bufRope) {
            pool_release_buffer(bufX);
            pool_release_buffer(bufXnorm);
            pool_release_buffer(bufQKV);
            pool_release_buffer(bufAttn);
            pool_release_buffer(bufProj);
            pool_release_buffer(bufGate);
            pool_release_buffer(bufFfnOut);
            pool_release_buffer(bufRope);
            return -1;
        }

        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];

        uint64_t enc_t0 = mach_absolute_time();

        /* ---- 32 encoder layers ---- */
        for (int layer = 0; layer < VOX_ENC_LAYERS; layer++) {
            vox_enc_layer_t *l = &enc->layers[layer];

            /* Step 1: rms_norm(x, attention_norm) → x_norm */
            {
                id<MTLBuffer> bufNorm = get_cached_weight_buffer(l->attention_norm,
                                                                   dim * sizeof(float));
                float eps = VOX_ENC_NORM_EPS;
                id<MTLComputeCommandEncoder> enc_cmd = [cmdBuffer computeCommandEncoder];
                [enc_cmd setComputePipelineState:g_rms_norm_pipeline];
                [enc_cmd setBuffer:bufX offset:0 atIndex:0];
                [enc_cmd setBuffer:bufNorm offset:0 atIndex:1];
                [enc_cmd setBuffer:bufXnorm offset:0 atIndex:2];
                [enc_cmd setBytes:&dim length:sizeof(int) atIndex:3];
                [enc_cmd setBytes:&eps length:sizeof(float) atIndex:4];
                [enc_cmd dispatchThreadgroups:MTLSizeMake((NSUInteger)M, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc_cmd endEncoding];
            }

            /* Step 2: Merged QKV projection (packed layout in bufQKV) */
            {
                id<MTLBuffer> bufWqkv = get_merged_f16_3(
                    l->wq_weight_bf16, (size_t)qkv_dim * dim,
                    l->wk_weight_bf16, (size_t)kv_dim * dim,
                    l->wv_weight_bf16, (size_t)kv_dim * dim);

                MPSMatrixDescriptor *descIn = [MPSMatrixDescriptor
                    matrixDescriptorWithRows:M columns:dim
                                    rowBytes:dim * sizeof(float)
                                    dataType:MPSDataTypeFloat32];
                MPSMatrixDescriptor *descW = [MPSMatrixDescriptor
                    matrixDescriptorWithRows:qkv_merged columns:dim
                                    rowBytes:dim * sizeof(uint16_t)
                                    dataType:MPSDataTypeFloat16];
                MPSMatrixDescriptor *descOut = [MPSMatrixDescriptor
                    matrixDescriptorWithRows:M columns:qkv_merged
                                    rowBytes:qkv_merged * sizeof(float)
                                    dataType:MPSDataTypeFloat32];
                MPSMatrix *matIn = [[MPSMatrix alloc] initWithBuffer:bufXnorm descriptor:descIn];
                MPSMatrix *matW = [[MPSMatrix alloc] initWithBuffer:bufWqkv descriptor:descW];
                MPSMatrix *matOut = [[MPSMatrix alloc] initWithBuffer:bufQKV descriptor:descOut];
                MPSMatrixMultiplication *mm =
                    get_cached_matmul_op(NO, YES, M, qkv_merged, dim, 1.0, 0.0);
                if (mm)
                    [mm encodeToCommandBuffer:cmdBuffer leftMatrix:matIn
                                  rightMatrix:matW resultMatrix:matOut];
            }

            /* Step 3: Bias add (Q += wq_bias, V += wv_bias) + RoPE + KV cache write */
            {
                id<MTLBuffer> bufQBias = get_cached_weight_buffer(l->wq_bias,
                                              qkv_dim * sizeof(float));
                id<MTLBuffer> bufVBias = get_cached_weight_buffer(l->wv_bias,
                                              kv_dim * sizeof(float));

                id<MTLComputeCommandEncoder> enc_cmd = [cmdBuffer computeCommandEncoder];
                int q_stride = qkv_merged;
                int q_col_off = 0;
                int k_col_off = qkv_dim;
                int v_col_off = qkv_dim + kv_dim;

                /* Q slice += wq_bias */
                int total_q = M * qkv_dim;
                [enc_cmd setComputePipelineState:g_bias_add_strided_pipeline];
                [enc_cmd setBuffer:bufQKV offset:0 atIndex:0];
                [enc_cmd setBuffer:bufQBias offset:0 atIndex:1];
                [enc_cmd setBytes:&q_stride length:sizeof(int) atIndex:2];
                [enc_cmd setBytes:&qkv_dim length:sizeof(int) atIndex:3];
                [enc_cmd setBytes:&q_col_off length:sizeof(int) atIndex:4];
                [enc_cmd setBytes:&total_q length:sizeof(int) atIndex:5];
                {
                    NSUInteger tg = MIN((NSUInteger)total_q,
                                        g_bias_add_strided_pipeline.maxTotalThreadsPerThreadgroup);
                    [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)total_q, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                }

                /* V slice += wv_bias */
                int total_v = M * kv_dim;
                [enc_cmd setBuffer:bufVBias offset:0 atIndex:1];
                [enc_cmd setBytes:&kv_dim length:sizeof(int) atIndex:3];
                [enc_cmd setBytes:&v_col_off length:sizeof(int) atIndex:4];
                [enc_cmd setBytes:&total_v length:sizeof(int) atIndex:5];
                {
                    NSUInteger tg = MIN((NSUInteger)total_v,
                                        g_bias_add_strided_pipeline.maxTotalThreadsPerThreadgroup);
                    [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)total_v, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                }

                [enc_cmd memoryBarrierWithScope:MTLBarrierScopeBuffers];

                /* Batched RoPE on packed Q slice */
                [enc_cmd setComputePipelineState:g_batched_rope_apply_strided_pipeline];
                [enc_cmd setBuffer:bufRope offset:0 atIndex:1];
                [enc_cmd setBytes:&n_heads length:sizeof(int) atIndex:2];
                [enc_cmd setBytes:&head_dim length:sizeof(int) atIndex:3];
                [enc_cmd setBytes:&M length:sizeof(int) atIndex:4];
                [enc_cmd setBytes:&q_stride length:sizeof(int) atIndex:5];
                [enc_cmd setBytes:&q_col_off length:sizeof(int) atIndex:6];
                {
                    int n_threads = M * n_heads * (head_dim / 2);
                    NSUInteger tg = MIN((NSUInteger)n_threads,
                                        g_batched_rope_apply_strided_pipeline.maxTotalThreadsPerThreadgroup);
                    [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)n_threads, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                }

                /* Batched RoPE on packed K slice */
                [enc_cmd setBytes:&n_kv_heads length:sizeof(int) atIndex:2];
                [enc_cmd setBytes:&k_col_off length:sizeof(int) atIndex:6];
                {
                    int n_threads = M * n_kv_heads * (head_dim / 2);
                    NSUInteger tg = MIN((NSUInteger)n_threads,
                                        g_batched_rope_apply_strided_pipeline.maxTotalThreadsPerThreadgroup);
                    [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)n_threads, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                }

                [enc_cmd memoryBarrierWithScope:MTLBarrierScopeBuffers];

                /* Copy K to KV cache */
                int kv_k_offset = (int)((size_t)layer * ctx->enc_kv_cache_max + cache_len) * kv_dim;
                int kv_total = M * kv_dim;
                [enc_cmd setComputePipelineState:g_batched_kv_cache_copy_strided_pipeline];
                [enc_cmd setBuffer:gpu_kv_k offset:0 atIndex:0];
                [enc_cmd setBuffer:bufQKV offset:0 atIndex:1];
                [enc_cmd setBytes:&kv_k_offset length:sizeof(int) atIndex:2];
                [enc_cmd setBytes:&q_stride length:sizeof(int) atIndex:3];
                [enc_cmd setBytes:&k_col_off length:sizeof(int) atIndex:4];
                [enc_cmd setBytes:&kv_dim length:sizeof(int) atIndex:5];
                [enc_cmd setBytes:&kv_total length:sizeof(int) atIndex:6];
                {
                    NSUInteger tg = MIN((NSUInteger)kv_total,
                                        g_batched_kv_cache_copy_strided_pipeline.maxTotalThreadsPerThreadgroup);
                    [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)kv_total, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                }

                /* Copy V to KV cache */
                [enc_cmd setBuffer:gpu_kv_v offset:0 atIndex:0];
                [enc_cmd setBytes:&v_col_off length:sizeof(int) atIndex:4];
                [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)kv_total, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(
                        MIN((NSUInteger)kv_total,
                            g_batched_kv_cache_copy_strided_pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];

                [enc_cmd memoryBarrierWithScope:MTLBarrierScopeBuffers];

                /* Encoder attention: all heads in one dispatch.
                 * q_offset = cache_len (physical, not logical — KV cache uses
                 * physical indices, so window masking must too). */
                int q_offset_val = cache_len;
                size_t layer_kv_offset = (size_t)layer * ctx->enc_kv_cache_max * kv_dim * sizeof(float);
                [enc_cmd setComputePipelineState:g_encoder_attention_qstrided_pipeline];
                [enc_cmd setBuffer:bufQKV offset:0 atIndex:0];
                [enc_cmd setBuffer:gpu_kv_k offset:layer_kv_offset atIndex:1];
                [enc_cmd setBuffer:gpu_kv_v offset:layer_kv_offset atIndex:2];
                [enc_cmd setBuffer:bufAttn offset:0 atIndex:3];
                [enc_cmd setBytes:&n_heads length:sizeof(int) atIndex:4];
                [enc_cmd setBytes:&n_kv_heads length:sizeof(int) atIndex:5];
                [enc_cmd setBytes:&head_dim length:sizeof(int) atIndex:6];
                [enc_cmd setBytes:&M length:sizeof(int) atIndex:7];
                [enc_cmd setBytes:&total_kv length:sizeof(int) atIndex:8];
                [enc_cmd setBytes:&attn_scale length:sizeof(float) atIndex:9];
                [enc_cmd setBytes:&window length:sizeof(int) atIndex:10];
                [enc_cmd setBytes:&q_offset_val length:sizeof(int) atIndex:11];
                [enc_cmd setBytes:&q_stride length:sizeof(int) atIndex:12];
                [enc_cmd setBytes:&q_col_off length:sizeof(int) atIndex:13];
                {
                    int bq = 8; /* must match ATTN_BQ in shader */
                    int n_q_blocks = (M + bq - 1) / bq;
                    int n_groups = n_heads * n_q_blocks;
                    [enc_cmd dispatchThreadgroups:MTLSizeMake((NSUInteger)n_groups, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
                }

                [enc_cmd endEncoding];
            }

            /* Step 4: wo projection */
            {
                id<MTLBuffer> bufWo = get_cached_bf16_as_f16_buffer(l->wo_weight_bf16,
                                            (size_t)dim * qkv_dim);
                MPSMatrixDescriptor *descA = [MPSMatrixDescriptor
                    matrixDescriptorWithRows:M columns:qkv_dim
                                    rowBytes:qkv_dim * sizeof(float)
                                    dataType:MPSDataTypeFloat32];
                MPSMatrixDescriptor *descW = [MPSMatrixDescriptor
                    matrixDescriptorWithRows:dim columns:qkv_dim
                                    rowBytes:qkv_dim * sizeof(uint16_t)
                                    dataType:MPSDataTypeFloat16];
                MPSMatrixDescriptor *descOut = [MPSMatrixDescriptor
                    matrixDescriptorWithRows:M columns:dim
                                    rowBytes:dim * sizeof(float)
                                    dataType:MPSDataTypeFloat32];
                MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufAttn descriptor:descA];
                MPSMatrix *matW = [[MPSMatrix alloc] initWithBuffer:bufWo descriptor:descW];
                MPSMatrix *matOut = [[MPSMatrix alloc] initWithBuffer:bufProj descriptor:descOut];
                MPSMatrixMultiplication *mm =
                    get_cached_matmul_op(NO, YES, M, dim, qkv_dim, 1.0, 0.0);
                if (mm)
                    [mm encodeToCommandBuffer:cmdBuffer leftMatrix:matA
                                  rightMatrix:matW resultMatrix:matOut];
            }

            /* Step 5: wo bias + residual + FFN norm */
            {
                id<MTLBuffer> bufWoBias = get_cached_weight_buffer(l->wo_bias,
                                                dim * sizeof(float));
                id<MTLBuffer> bufFfnNorm = get_cached_weight_buffer(l->ffn_norm,
                                                dim * sizeof(float));
                int n = M * dim;
                float eps = VOX_ENC_NORM_EPS;

                id<MTLComputeCommandEncoder> enc_cmd = [cmdBuffer computeCommandEncoder];

                /* proj_out += wo_bias */
                [enc_cmd setComputePipelineState:g_bias_add_pipeline];
                [enc_cmd setBuffer:bufProj offset:0 atIndex:0];
                [enc_cmd setBuffer:bufWoBias offset:0 atIndex:1];
                [enc_cmd setBytes:&dim length:sizeof(int) atIndex:2];
                [enc_cmd setBytes:&n length:sizeof(int) atIndex:3];
                {
                    NSUInteger tg = MIN((NSUInteger)n,
                                        g_bias_add_pipeline.maxTotalThreadsPerThreadgroup);
                    [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                }

                [enc_cmd memoryBarrierWithScope:MTLBarrierScopeBuffers];

                /* x += proj_out */
                [enc_cmd setComputePipelineState:g_add_inplace_pipeline];
                [enc_cmd setBuffer:bufX offset:0 atIndex:0];
                [enc_cmd setBuffer:bufProj offset:0 atIndex:1];
                [enc_cmd setBytes:&n length:sizeof(int) atIndex:2];
                {
                    NSUInteger tg = MIN((NSUInteger)n,
                                        g_add_inplace_pipeline.maxTotalThreadsPerThreadgroup);
                    [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                }

                [enc_cmd memoryBarrierWithScope:MTLBarrierScopeBuffers];

                /* x_norm = rms_norm(x, ffn_norm) */
                [enc_cmd setComputePipelineState:g_rms_norm_pipeline];
                [enc_cmd setBuffer:bufX offset:0 atIndex:0];
                [enc_cmd setBuffer:bufFfnNorm offset:0 atIndex:1];
                [enc_cmd setBuffer:bufXnorm offset:0 atIndex:2];
                [enc_cmd setBytes:&dim length:sizeof(int) atIndex:3];
                [enc_cmd setBytes:&eps length:sizeof(float) atIndex:4];
                [enc_cmd dispatchThreadgroups:MTLSizeMake((NSUInteger)M, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                [enc_cmd endEncoding];
            }

            /* Step 6: Merged FFN (1 matmul for w1+w3, fused silu*mul, strided w2) */
            {
                id<MTLBuffer> bufW1W3 = get_merged_f16_2(
                    l->w1_weight_bf16, (size_t)hidden * dim,
                    l->w3_weight_bf16, (size_t)hidden * dim);
                id<MTLBuffer> bufW2 = get_cached_bf16_as_f16_buffer(l->w2_weight_bf16,
                                            (size_t)dim * hidden);
                id<MTLBuffer> bufW2Bias = get_cached_weight_buffer(l->w2_bias,
                                            dim * sizeof(float));

                /* [gate; up] = x_norm @ [w1; w3]^T → bufGate [M, hidden*2] */
                {
                    MPSMatrixDescriptor *descIn = [MPSMatrixDescriptor
                        matrixDescriptorWithRows:M columns:dim
                                        rowBytes:dim * sizeof(float)
                                        dataType:MPSDataTypeFloat32];
                    MPSMatrixDescriptor *descW = [MPSMatrixDescriptor
                        matrixDescriptorWithRows:ffn_merged columns:dim
                                        rowBytes:dim * sizeof(uint16_t)
                                        dataType:MPSDataTypeFloat16];
                    MPSMatrixDescriptor *descOut = [MPSMatrixDescriptor
                        matrixDescriptorWithRows:M columns:ffn_merged
                                        rowBytes:ffn_merged * sizeof(float)
                                        dataType:MPSDataTypeFloat32];
                    MPSMatrix *matIn = [[MPSMatrix alloc] initWithBuffer:bufXnorm descriptor:descIn];
                    MPSMatrix *matW = [[MPSMatrix alloc] initWithBuffer:bufW1W3 descriptor:descW];
                    MPSMatrix *matOut = [[MPSMatrix alloc] initWithBuffer:bufGate descriptor:descOut];
                    MPSMatrixMultiplication *mm =
                        get_cached_matmul_op(NO, YES, M, ffn_merged, dim, 1.0, 0.0);
                    if (mm)
                        [mm encodeToCommandBuffer:cmdBuffer leftMatrix:matIn
                                      rightMatrix:matW resultMatrix:matOut];
                }

                /* Fused silu + mul on interleaved [M, hidden*2] layout */
                {
                    int n_gate = M * hidden;
                    id<MTLComputeCommandEncoder> enc_cmd = [cmdBuffer computeCommandEncoder];
                    [enc_cmd setComputePipelineState:g_silu_mul_merged_pipeline];
                    [enc_cmd setBuffer:bufGate offset:0 atIndex:0];
                    [enc_cmd setBytes:&hidden length:sizeof(int) atIndex:1];
                    [enc_cmd setBytes:&n_gate length:sizeof(int) atIndex:2];
                    NSUInteger tg = MIN((NSUInteger)n_gate,
                                        g_silu_mul_merged_pipeline.maxTotalThreadsPerThreadgroup);
                    [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)n_gate, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                    [enc_cmd endEncoding];
                }

                /* ffn_out = gate @ w2^T (strided read: rowBytes = hidden*2) */
                {
                    MPSMatrixDescriptor *descG = [MPSMatrixDescriptor
                        matrixDescriptorWithRows:M columns:hidden
                                        rowBytes:ffn_merged * sizeof(float)
                                        dataType:MPSDataTypeFloat32];
                    MPSMatrixDescriptor *descW2 = [MPSMatrixDescriptor
                        matrixDescriptorWithRows:dim columns:hidden
                                        rowBytes:hidden * sizeof(uint16_t)
                                        dataType:MPSDataTypeFloat16];
                    MPSMatrixDescriptor *descOut = [MPSMatrixDescriptor
                        matrixDescriptorWithRows:M columns:dim
                                        rowBytes:dim * sizeof(float)
                                        dataType:MPSDataTypeFloat32];
                    MPSMatrix *matG = [[MPSMatrix alloc] initWithBuffer:bufGate descriptor:descG];
                    MPSMatrix *matW = [[MPSMatrix alloc] initWithBuffer:bufW2 descriptor:descW2];
                    MPSMatrix *matOut = [[MPSMatrix alloc] initWithBuffer:bufFfnOut descriptor:descOut];
                    MPSMatrixMultiplication *mm =
                        get_cached_matmul_op(NO, YES, M, dim, hidden, 1.0, 0.0);
                    if (mm)
                        [mm encodeToCommandBuffer:cmdBuffer leftMatrix:matG
                                      rightMatrix:matW resultMatrix:matOut];
                }

                /* ffn_out += w2_bias, x += ffn_out */
                {
                    int n = M * dim;
                    id<MTLComputeCommandEncoder> enc_cmd = [cmdBuffer computeCommandEncoder];

                    /* ffn_out += w2_bias */
                    [enc_cmd setComputePipelineState:g_bias_add_pipeline];
                    [enc_cmd setBuffer:bufFfnOut offset:0 atIndex:0];
                    [enc_cmd setBuffer:bufW2Bias offset:0 atIndex:1];
                    [enc_cmd setBytes:&dim length:sizeof(int) atIndex:2];
                    [enc_cmd setBytes:&n length:sizeof(int) atIndex:3];
                    {
                        NSUInteger tg = MIN((NSUInteger)n,
                                            g_bias_add_pipeline.maxTotalThreadsPerThreadgroup);
                        [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                    }

                    [enc_cmd memoryBarrierWithScope:MTLBarrierScopeBuffers];

                    /* x += ffn_out */
                    [enc_cmd setComputePipelineState:g_add_inplace_pipeline];
                    [enc_cmd setBuffer:bufX offset:0 atIndex:0];
                    [enc_cmd setBuffer:bufFfnOut offset:0 atIndex:1];
                    [enc_cmd setBytes:&n length:sizeof(int) atIndex:2];
                    {
                        NSUInteger tg = MIN((NSUInteger)n,
                                            g_add_inplace_pipeline.maxTotalThreadsPerThreadgroup);
                        [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                    }

                    [enc_cmd endEncoding];
                }
            }
        } /* end 32 layers */

        /* Final norm: rms_norm(x, norm) — write back to bufX */
        {
            id<MTLBuffer> bufNorm = get_cached_weight_buffer(enc->norm,
                                                               dim * sizeof(float));
            float eps = VOX_ENC_NORM_EPS;
            /* rms_norm needs separate input/output, use bufXnorm as scratch */
            id<MTLComputeCommandEncoder> enc_cmd = [cmdBuffer computeCommandEncoder];
            [enc_cmd setComputePipelineState:g_rms_norm_pipeline];
            [enc_cmd setBuffer:bufX offset:0 atIndex:0];
            [enc_cmd setBuffer:bufNorm offset:0 atIndex:1];
            [enc_cmd setBuffer:bufXnorm offset:0 atIndex:2];
            [enc_cmd setBytes:&dim length:sizeof(int) atIndex:3];
            [enc_cmd setBytes:&eps length:sizeof(float) atIndex:4];
            [enc_cmd dispatchThreadgroups:MTLSizeMake((NSUInteger)M, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc_cmd endEncoding];
        }

        uint64_t enc_t1 = mach_absolute_time();
        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];
        uint64_t enc_t2 = mach_absolute_time();

        if (vox_verbose >= 2) {
            mach_timebase_info_data_t info;
            mach_timebase_info(&info);
            double encode_ms = (double)(enc_t1 - enc_t0) * info.numer / info.denom / 1e6;
            double commit_ms = (double)(enc_t2 - enc_t1) * info.numer / info.denom / 1e6;
            fprintf(stderr, "[encoder] MPS encoding: %.1f ms, commit+wait: %.1f ms\n",
                    encode_ms, commit_ms);
        }

        /* Download result (from bufXnorm since final norm wrote there) */
        memcpy(x, [bufXnorm contents], (size_t)M * dim * sizeof(float));

        pool_release_buffer(bufX);
        pool_release_buffer(bufXnorm);
        pool_release_buffer(bufQKV);
        pool_release_buffer(bufAttn);
        pool_release_buffer(bufProj);
        pool_release_buffer(bufGate);
        pool_release_buffer(bufFfnOut);
        pool_release_buffer(bufRope);
    }

    return 0;
}

/* ========================================================================
 * Monolithic Decoder Prefill: all 26 layers in ONE command buffer (M>1)
 * ======================================================================== */

void vox_metal_decoder_prefill_step(void *ctx_ptr, float *x, int seq_len,
                                      const float *rope_freqs) {
    if (!g_initialized || !g_shaders_initialized) return;

    vox_ctx_t *ctx = (vox_ctx_t *)ctx_ptr;
    vox_decoder_t *dec = &ctx->decoder;

    int dim = VOX_DEC_DIM;          /* 3072 */
    int n_heads = VOX_DEC_HEADS;    /* 32 */
    int n_kv_heads = VOX_DEC_KV_HEADS; /* 8 */
    int head_dim = VOX_DEC_HEAD_DIM;/* 128 */
    int hidden = VOX_DEC_HIDDEN;    /* 9216 */
    int q_dim = n_heads * head_dim; /* 4096 */
    int kv_dim = n_kv_heads * head_dim; /* 1024 */
    int M = seq_len;
    int start_pos = ctx->kv_cache_len;
    int total_kv = start_pos + seq_len;
    float attn_scale = 1.0f / sqrtf((float)head_dim);
    int window = VOX_DEC_WINDOW;
    int kv_fp16 = ctx->kv_cache_fp16;
    size_t kv_elem_size = kv_fp16 ? sizeof(uint16_t) : sizeof(float);

    /* Find GPU buffer handles for decoder KV cache */
    void *kv_k_ptr = kv_fp16 ? (void *)ctx->kv_cache_k_f16 : (void *)ctx->kv_cache_k;
    void *kv_v_ptr = kv_fp16 ? (void *)ctx->kv_cache_v_f16 : (void *)ctx->kv_cache_v;
    id<MTLBuffer> gpu_kv_k = find_shared_buffer(kv_k_ptr);
    id<MTLBuffer> gpu_kv_v = find_shared_buffer(kv_v_ptr);
    if (!gpu_kv_k || !gpu_kv_v) return;
    if (!g_batched_rope_apply_strided_pipeline ||
        !g_batched_kv_cache_copy_strided_pipeline ||
        !g_encoder_attention_qstrided_pipeline)
        return;
    if (kv_fp16 && (!g_batched_kv_cache_copy_strided_f16_pipeline ||
                    !g_encoder_attention_kv_f16_qstrided_pipeline))
        return;

    @autoreleasepool {
        /* Scratch buffers */
        int qkv_merged = q_dim + kv_dim + kv_dim;  /* 6144 */
        int ffn_merged = hidden * 2;                 /* 18432 */
        id<MTLBuffer> bufX = pool_get_buffer((size_t)M * dim * sizeof(float));
        id<MTLBuffer> bufXnorm = pool_get_buffer((size_t)M * dim * sizeof(float));
        id<MTLBuffer> bufQKV = pool_get_buffer((size_t)M * qkv_merged * sizeof(float));
        id<MTLBuffer> bufAttn = pool_get_buffer((size_t)M * q_dim * sizeof(float));
        id<MTLBuffer> bufProj = pool_get_buffer((size_t)M * dim * sizeof(float));
        id<MTLBuffer> bufGate = pool_get_buffer((size_t)M * ffn_merged * sizeof(float));
        id<MTLBuffer> bufFfnOut = pool_get_buffer((size_t)M * dim * sizeof(float));

        /* Upload x and RoPE frequencies */
        if (bufX) memcpy([bufX contents], x, (size_t)M * dim * sizeof(float));
        size_t rope_size = (size_t)M * (head_dim / 2) * 2 * sizeof(float);
        id<MTLBuffer> bufRope = pool_get_buffer(rope_size);
        if (bufRope) memcpy([bufRope contents], rope_freqs, rope_size);

        if (!bufX || !bufXnorm || !bufQKV ||
            !bufAttn || !bufProj || !bufGate || !bufFfnOut || !bufRope) {
            pool_release_buffer(bufX);
            pool_release_buffer(bufXnorm);
            pool_release_buffer(bufQKV);
            pool_release_buffer(bufAttn);
            pool_release_buffer(bufProj);
            pool_release_buffer(bufGate);
            pool_release_buffer(bufFfnOut);
            pool_release_buffer(bufRope);
            return;
        }

        id<MTLCommandBuffer> cmdBuffer = [g_queue commandBuffer];

        /* ---- 26 decoder layers ---- */
        for (int layer = 0; layer < VOX_DEC_LAYERS; layer++) {
            vox_dec_layer_t *l = &dec->layers[layer];

            /* Step 1: rms_norm(x, attention_norm) → x_norm */
            {
                id<MTLBuffer> bufNorm = get_cached_weight_buffer(l->attention_norm,
                                                                   dim * sizeof(float));
                float eps = VOX_DEC_NORM_EPS;
                id<MTLComputeCommandEncoder> enc_cmd = [cmdBuffer computeCommandEncoder];
                [enc_cmd setComputePipelineState:g_rms_norm_pipeline];
                [enc_cmd setBuffer:bufX offset:0 atIndex:0];
                [enc_cmd setBuffer:bufNorm offset:0 atIndex:1];
                [enc_cmd setBuffer:bufXnorm offset:0 atIndex:2];
                [enc_cmd setBytes:&dim length:sizeof(int) atIndex:3];
                [enc_cmd setBytes:&eps length:sizeof(float) atIndex:4];
                [enc_cmd dispatchThreadgroups:MTLSizeMake((NSUInteger)M, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [enc_cmd endEncoding];
            }

            /* Step 2: Merged QKV projection (packed layout in bufQKV) */
            {
                id<MTLBuffer> bufWqkv = get_merged_f16_3(
                    l->wq_weight_bf16, (size_t)q_dim * dim,
                    l->wk_weight_bf16, (size_t)kv_dim * dim,
                    l->wv_weight_bf16, (size_t)kv_dim * dim);

                MPSMatrixDescriptor *descIn = [MPSMatrixDescriptor
                    matrixDescriptorWithRows:M columns:dim
                                    rowBytes:dim * sizeof(float)
                                    dataType:MPSDataTypeFloat32];
                MPSMatrixDescriptor *descW = [MPSMatrixDescriptor
                    matrixDescriptorWithRows:qkv_merged columns:dim
                                    rowBytes:dim * sizeof(uint16_t)
                                    dataType:MPSDataTypeFloat16];
                MPSMatrixDescriptor *descOut = [MPSMatrixDescriptor
                    matrixDescriptorWithRows:M columns:qkv_merged
                                    rowBytes:qkv_merged * sizeof(float)
                                    dataType:MPSDataTypeFloat32];
                MPSMatrix *matIn = [[MPSMatrix alloc] initWithBuffer:bufXnorm descriptor:descIn];
                MPSMatrix *matW = [[MPSMatrix alloc] initWithBuffer:bufWqkv descriptor:descW];
                MPSMatrix *matOut = [[MPSMatrix alloc] initWithBuffer:bufQKV descriptor:descOut];
                MPSMatrixMultiplication *mm =
                    get_cached_matmul_op(NO, YES, M, qkv_merged, dim, 1.0, 0.0);
                if (mm)
                    [mm encodeToCommandBuffer:cmdBuffer leftMatrix:matIn
                                  rightMatrix:matW resultMatrix:matOut];
            }

            /* Step 3: RoPE + KV cache write + attention */
            {
                id<MTLComputeCommandEncoder> enc_cmd = [cmdBuffer computeCommandEncoder];
                int q_stride = qkv_merged;
                int q_col_off = 0;
                int k_col_off = q_dim;
                int v_col_off = q_dim + kv_dim;

                /* Batched RoPE on packed Q slice */
                [enc_cmd setComputePipelineState:g_batched_rope_apply_strided_pipeline];
                [enc_cmd setBuffer:bufQKV offset:0 atIndex:0];
                [enc_cmd setBuffer:bufRope offset:0 atIndex:1];
                [enc_cmd setBytes:&n_heads length:sizeof(int) atIndex:2];
                [enc_cmd setBytes:&head_dim length:sizeof(int) atIndex:3];
                [enc_cmd setBytes:&M length:sizeof(int) atIndex:4];
                [enc_cmd setBytes:&q_stride length:sizeof(int) atIndex:5];
                [enc_cmd setBytes:&q_col_off length:sizeof(int) atIndex:6];
                {
                    int n_threads = M * n_heads * (head_dim / 2);
                    NSUInteger tg = MIN((NSUInteger)n_threads,
                                        g_batched_rope_apply_strided_pipeline.maxTotalThreadsPerThreadgroup);
                    [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)n_threads, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                }

                /* Batched RoPE on packed K slice */
                [enc_cmd setBytes:&n_kv_heads length:sizeof(int) atIndex:2];
                [enc_cmd setBytes:&k_col_off length:sizeof(int) atIndex:6];
                {
                    int n_threads = M * n_kv_heads * (head_dim / 2);
                    NSUInteger tg = MIN((NSUInteger)n_threads,
                                        g_batched_rope_apply_strided_pipeline.maxTotalThreadsPerThreadgroup);
                    [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)n_threads, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                }

                [enc_cmd memoryBarrierWithScope:MTLBarrierScopeBuffers];

                /* Copy K to KV cache */
                int kv_offset = (int)((size_t)layer * ctx->kv_cache_max + start_pos) * kv_dim;
                int kv_total = M * kv_dim;
                id<MTLComputePipelineState> kv_copy_ps = kv_fp16 ?
                    g_batched_kv_cache_copy_strided_f16_pipeline :
                    g_batched_kv_cache_copy_strided_pipeline;
                [enc_cmd setComputePipelineState:kv_copy_ps];
                [enc_cmd setBuffer:gpu_kv_k offset:0 atIndex:0];
                [enc_cmd setBuffer:bufQKV offset:0 atIndex:1];
                [enc_cmd setBytes:&kv_offset length:sizeof(int) atIndex:2];
                [enc_cmd setBytes:&q_stride length:sizeof(int) atIndex:3];
                [enc_cmd setBytes:&k_col_off length:sizeof(int) atIndex:4];
                [enc_cmd setBytes:&kv_dim length:sizeof(int) atIndex:5];
                [enc_cmd setBytes:&kv_total length:sizeof(int) atIndex:6];
                {
                    NSUInteger tg = MIN((NSUInteger)kv_total,
                                        kv_copy_ps.maxTotalThreadsPerThreadgroup);
                    [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)kv_total, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                }

                /* Copy V to KV cache */
                [enc_cmd setBuffer:gpu_kv_v offset:0 atIndex:0];
                [enc_cmd setBytes:&v_col_off length:sizeof(int) atIndex:4];
                [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)kv_total, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(
                        MIN((NSUInteger)kv_total,
                            kv_copy_ps.maxTotalThreadsPerThreadgroup), 1, 1)];

                [enc_cmd memoryBarrierWithScope:MTLBarrierScopeBuffers];

                /* Batched attention (using encoder_attention kernel with head_dim=128) */
                int q_offset_val = ctx->kv_pos_offset + start_pos;
                size_t layer_kv_offset = (size_t)layer * ctx->kv_cache_max * kv_dim * kv_elem_size;
                [enc_cmd setComputePipelineState:kv_fp16 ?
                    g_encoder_attention_kv_f16_qstrided_pipeline :
                    g_encoder_attention_qstrided_pipeline];
                [enc_cmd setBuffer:bufQKV offset:0 atIndex:0];
                [enc_cmd setBuffer:gpu_kv_k offset:layer_kv_offset atIndex:1];
                [enc_cmd setBuffer:gpu_kv_v offset:layer_kv_offset atIndex:2];
                [enc_cmd setBuffer:bufAttn offset:0 atIndex:3];
                [enc_cmd setBytes:&n_heads length:sizeof(int) atIndex:4];
                [enc_cmd setBytes:&n_kv_heads length:sizeof(int) atIndex:5];
                [enc_cmd setBytes:&head_dim length:sizeof(int) atIndex:6];
                [enc_cmd setBytes:&M length:sizeof(int) atIndex:7];
                [enc_cmd setBytes:&total_kv length:sizeof(int) atIndex:8];
                [enc_cmd setBytes:&attn_scale length:sizeof(float) atIndex:9];
                [enc_cmd setBytes:&window length:sizeof(int) atIndex:10];
                [enc_cmd setBytes:&q_offset_val length:sizeof(int) atIndex:11];
                [enc_cmd setBytes:&q_stride length:sizeof(int) atIndex:12];
                [enc_cmd setBytes:&q_col_off length:sizeof(int) atIndex:13];
                {
                    int bq = 8; /* must match ATTN_BQ in shader */
                    int n_q_blocks = (M + bq - 1) / bq;
                    int n_groups = n_heads * n_q_blocks;
                    [enc_cmd dispatchThreadgroups:MTLSizeMake((NSUInteger)n_groups, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake((NSUInteger)head_dim, 1, 1)];
                }

                [enc_cmd endEncoding];
            }

            /* Step 4: wo projection */
            {
                id<MTLBuffer> bufWo = get_cached_bf16_as_f16_buffer(l->wo_weight_bf16,
                                            (size_t)dim * q_dim);
                MPSMatrixDescriptor *descA = [MPSMatrixDescriptor
                    matrixDescriptorWithRows:M columns:q_dim
                                    rowBytes:q_dim * sizeof(float)
                                    dataType:MPSDataTypeFloat32];
                MPSMatrixDescriptor *descW = [MPSMatrixDescriptor
                    matrixDescriptorWithRows:dim columns:q_dim
                                    rowBytes:q_dim * sizeof(uint16_t)
                                    dataType:MPSDataTypeFloat16];
                MPSMatrixDescriptor *descOut = [MPSMatrixDescriptor
                    matrixDescriptorWithRows:M columns:dim
                                    rowBytes:dim * sizeof(float)
                                    dataType:MPSDataTypeFloat32];
                MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:bufAttn descriptor:descA];
                MPSMatrix *matW = [[MPSMatrix alloc] initWithBuffer:bufWo descriptor:descW];
                MPSMatrix *matOut = [[MPSMatrix alloc] initWithBuffer:bufProj descriptor:descOut];
                MPSMatrixMultiplication *mm =
                    get_cached_matmul_op(NO, YES, M, dim, q_dim, 1.0, 0.0);
                if (mm)
                    [mm encodeToCommandBuffer:cmdBuffer leftMatrix:matA
                                  rightMatrix:matW resultMatrix:matOut];
            }

            /* Step 5: residual + FFN norm + ada_scale */
            {
                id<MTLBuffer> bufFfnNorm = get_cached_weight_buffer(l->ffn_norm,
                                                dim * sizeof(float));
                id<MTLBuffer> bufAda = ctx->ada_scale ?
                    get_cached_weight_buffer(ctx->ada_scale + (size_t)layer * dim,
                                               dim * sizeof(float)) : nil;
                int n = M * dim;
                float eps = VOX_DEC_NORM_EPS;

                id<MTLComputeCommandEncoder> enc_cmd = [cmdBuffer computeCommandEncoder];

                /* x += proj_out */
                [enc_cmd setComputePipelineState:g_add_inplace_pipeline];
                [enc_cmd setBuffer:bufX offset:0 atIndex:0];
                [enc_cmd setBuffer:bufProj offset:0 atIndex:1];
                [enc_cmd setBytes:&n length:sizeof(int) atIndex:2];
                {
                    NSUInteger tg = MIN((NSUInteger)n,
                                        g_add_inplace_pipeline.maxTotalThreadsPerThreadgroup);
                    [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                }

                [enc_cmd memoryBarrierWithScope:MTLBarrierScopeBuffers];

                /* x_norm = rms_norm(x, ffn_norm) */
                [enc_cmd setComputePipelineState:g_rms_norm_pipeline];
                [enc_cmd setBuffer:bufX offset:0 atIndex:0];
                [enc_cmd setBuffer:bufFfnNorm offset:0 atIndex:1];
                [enc_cmd setBuffer:bufXnorm offset:0 atIndex:2];
                [enc_cmd setBytes:&dim length:sizeof(int) atIndex:3];
                [enc_cmd setBytes:&eps length:sizeof(float) atIndex:4];
                [enc_cmd dispatchThreadgroups:MTLSizeMake((NSUInteger)M, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

                /* x_norm *= (1 + ada_scale) if present */
                if (bufAda) {
                    [enc_cmd memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    [enc_cmd setComputePipelineState:g_ada_scale_mul_pipeline];
                    [enc_cmd setBuffer:bufXnorm offset:0 atIndex:0];
                    [enc_cmd setBuffer:bufAda offset:0 atIndex:1];
                    [enc_cmd setBytes:&n length:sizeof(int) atIndex:2];
                    [enc_cmd setBytes:&dim length:sizeof(int) atIndex:3];
                    {
                        NSUInteger tg = MIN((NSUInteger)n,
                                            g_ada_scale_mul_pipeline.maxTotalThreadsPerThreadgroup);
                        [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                    }
                }

                [enc_cmd endEncoding];
            }

            /* Step 6: Merged FFN (1 matmul for w1+w3, fused silu*mul, strided w2) */
            {
                id<MTLBuffer> bufW1W3 = get_merged_f16_2(
                    l->w1_weight_bf16, (size_t)hidden * dim,
                    l->w3_weight_bf16, (size_t)hidden * dim);
                id<MTLBuffer> bufW2 = get_cached_bf16_as_f16_buffer(l->w2_weight_bf16,
                                            (size_t)dim * hidden);

                /* [gate; up] = x_norm @ [w1; w3]^T → bufGate [M, hidden*2] */
                {
                    MPSMatrixDescriptor *descIn = [MPSMatrixDescriptor
                        matrixDescriptorWithRows:M columns:dim
                                        rowBytes:dim * sizeof(float)
                                        dataType:MPSDataTypeFloat32];
                    MPSMatrixDescriptor *descW = [MPSMatrixDescriptor
                        matrixDescriptorWithRows:ffn_merged columns:dim
                                        rowBytes:dim * sizeof(uint16_t)
                                        dataType:MPSDataTypeFloat16];
                    MPSMatrixDescriptor *descOut = [MPSMatrixDescriptor
                        matrixDescriptorWithRows:M columns:ffn_merged
                                        rowBytes:ffn_merged * sizeof(float)
                                        dataType:MPSDataTypeFloat32];
                    MPSMatrix *matIn = [[MPSMatrix alloc] initWithBuffer:bufXnorm descriptor:descIn];
                    MPSMatrix *matW = [[MPSMatrix alloc] initWithBuffer:bufW1W3 descriptor:descW];
                    MPSMatrix *matOut = [[MPSMatrix alloc] initWithBuffer:bufGate descriptor:descOut];
                    MPSMatrixMultiplication *mm =
                        get_cached_matmul_op(NO, YES, M, ffn_merged, dim, 1.0, 0.0);
                    if (mm)
                        [mm encodeToCommandBuffer:cmdBuffer leftMatrix:matIn
                                      rightMatrix:matW resultMatrix:matOut];
                }

                /* Fused silu + mul on interleaved [M, hidden*2] layout */
                {
                    int n_gate = M * hidden;
                    id<MTLComputeCommandEncoder> enc_cmd = [cmdBuffer computeCommandEncoder];
                    [enc_cmd setComputePipelineState:g_silu_mul_merged_pipeline];
                    [enc_cmd setBuffer:bufGate offset:0 atIndex:0];
                    [enc_cmd setBytes:&hidden length:sizeof(int) atIndex:1];
                    [enc_cmd setBytes:&n_gate length:sizeof(int) atIndex:2];
                    NSUInteger tg = MIN((NSUInteger)n_gate,
                                        g_silu_mul_merged_pipeline.maxTotalThreadsPerThreadgroup);
                    [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)n_gate, 1, 1)
                       threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                    [enc_cmd endEncoding];
                }

                /* ffn_out = gate @ w2^T (strided read: rowBytes = hidden*2) */
                {
                    MPSMatrixDescriptor *descG = [MPSMatrixDescriptor
                        matrixDescriptorWithRows:M columns:hidden
                                        rowBytes:ffn_merged * sizeof(float)
                                        dataType:MPSDataTypeFloat32];
                    MPSMatrixDescriptor *descW2 = [MPSMatrixDescriptor
                        matrixDescriptorWithRows:dim columns:hidden
                                        rowBytes:hidden * sizeof(uint16_t)
                                        dataType:MPSDataTypeFloat16];
                    MPSMatrixDescriptor *descOut = [MPSMatrixDescriptor
                        matrixDescriptorWithRows:M columns:dim
                                        rowBytes:dim * sizeof(float)
                                        dataType:MPSDataTypeFloat32];
                    MPSMatrix *matG = [[MPSMatrix alloc] initWithBuffer:bufGate descriptor:descG];
                    MPSMatrix *matW = [[MPSMatrix alloc] initWithBuffer:bufW2 descriptor:descW2];
                    MPSMatrix *matOut = [[MPSMatrix alloc] initWithBuffer:bufFfnOut descriptor:descOut];
                    MPSMatrixMultiplication *mm =
                        get_cached_matmul_op(NO, YES, M, dim, hidden, 1.0, 0.0);
                    if (mm)
                        [mm encodeToCommandBuffer:cmdBuffer leftMatrix:matG
                                      rightMatrix:matW resultMatrix:matOut];
                }

                /* x += ffn_out */
                {
                    int n = M * dim;
                    id<MTLComputeCommandEncoder> enc_cmd = [cmdBuffer computeCommandEncoder];
                    [enc_cmd setComputePipelineState:g_add_inplace_pipeline];
                    [enc_cmd setBuffer:bufX offset:0 atIndex:0];
                    [enc_cmd setBuffer:bufFfnOut offset:0 atIndex:1];
                    [enc_cmd setBytes:&n length:sizeof(int) atIndex:2];
                    {
                        NSUInteger tg = MIN((NSUInteger)n,
                                            g_add_inplace_pipeline.maxTotalThreadsPerThreadgroup);
                        [enc_cmd dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1)
                           threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
                    }
                    [enc_cmd endEncoding];
                }
            }
        } /* end 26 layers */

        [cmdBuffer commit];
        [cmdBuffer waitUntilCompleted];

        /* Download result */
        memcpy(x, [bufX contents], (size_t)M * dim * sizeof(float));

        pool_release_buffer(bufX);
        pool_release_buffer(bufXnorm);
        pool_release_buffer(bufQKV);
        pool_release_buffer(bufAttn);
        pool_release_buffer(bufProj);
        pool_release_buffer(bufGate);
        pool_release_buffer(bufFfnOut);
        pool_release_buffer(bufRope);
    }

    /* Update KV cache length */
    ctx->kv_cache_len = start_pos + seq_len;
}

/* ========================================================================
 * Utility
 * ======================================================================== */

void vox_metal_warmup_bf16(const uint16_t *bf16_weights, size_t num_elements) {
    if (!g_initialized || !bf16_weights || num_elements == 0) return;
    (void)get_cached_bf16_as_f16_buffer(bf16_weights, num_elements);
}

void vox_metal_warmup_merged_2(const uint16_t *a, size_t a_n,
                                const uint16_t *b, size_t b_n) {
    if (!g_initialized) return;
    (void)get_merged_f16_2(a, a_n, b, b_n);
}

void vox_metal_warmup_merged_3(const uint16_t *a, size_t a_n,
                                const uint16_t *b, size_t b_n,
                                const uint16_t *c, size_t c_n) {
    if (!g_initialized) return;
    (void)get_merged_f16_3(a, a_n, b, b_n, c, c_n);
}

void vox_metal_warmup_decoder_ops(void *ctx_ptr) {
    if (!g_initialized || !g_shaders_initialized) return;
    vox_ctx_t *ctx = (vox_ctx_t *)ctx_ptr;
    vox_decoder_t *dec = &ctx->decoder;
    int dim = VOX_DEC_DIM;
    int q_dim = VOX_DEC_HEADS * VOX_DEC_HEAD_DIM;
    int kv_dim = VOX_DEC_KV_HEADS * VOX_DEC_HEAD_DIM;
    int hidden = VOX_DEC_HIDDEN;

    /* Pre-warm MPS matmul ops (compiled on first creation) */
    (void)get_cached_matmul_op(NO, YES, 1, q_dim + kv_dim + kv_dim, dim, 1.0, 0.0);
    (void)get_cached_matmul_op(NO, YES, 1, dim, q_dim, 1.0, 0.0);
    (void)get_cached_matmul_op(NO, YES, 1, hidden * 2, dim, 1.0, 0.0);
    (void)get_cached_matmul_op(NO, YES, 1, dim, hidden, 1.0, 0.0);
    (void)get_cached_matmul_op(NO, YES, 1, VOX_VOCAB_SIZE, dim, 1.0, 0.0);

    /* Pre-warm f32 weight buffers (norms, ada_scale) */
    for (int i = 0; i < VOX_DEC_LAYERS; i++) {
        vox_dec_layer_t *l = &dec->layers[i];
        (void)get_cached_weight_buffer(l->attention_norm, dim * sizeof(float));
        (void)get_cached_weight_buffer(l->ffn_norm, dim * sizeof(float));
        if (ctx->ada_scale)
            (void)get_cached_weight_buffer(
                ctx->ada_scale + (size_t)i * dim, dim * sizeof(float));
    }
    (void)get_cached_weight_buffer(dec->norm, dim * sizeof(float));

    /* Pre-warm encoder MPS matmul ops for typical M values.
     * Encoder M varies per chunk (50-200). We pre-warm for common ranges
     * plus prefill M=38 to trigger GPU kernel compilation at load time. */
    {
        int edim = VOX_ENC_DIM;
        int eqkv = (VOX_ENC_HEADS + VOX_ENC_KV_HEADS + VOX_ENC_KV_HEADS) * VOX_ENC_HEAD_DIM;
        int ewo_k = VOX_ENC_HEADS * VOX_ENC_HEAD_DIM;
        int ehidden = VOX_ENC_HIDDEN;
        int effn = ehidden * 2;
        /* Encoder M values: 64, 128, 200 cover typical chunk sizes */
        int enc_ms[] = {64, 128, 200};
        for (int i = 0; i < 3; i++) {
            int m = enc_ms[i];
            (void)get_cached_matmul_op(NO, YES, m, eqkv, edim, 1.0, 0.0);
            (void)get_cached_matmul_op(NO, YES, m, edim, ewo_k, 1.0, 0.0);
            (void)get_cached_matmul_op(NO, YES, m, effn, edim, 1.0, 0.0);
            (void)get_cached_matmul_op(NO, YES, m, edim, ehidden, 1.0, 0.0);
        }
        /* Prefill M=38 with decoder dimensions */
        int pM = 38;
        (void)get_cached_matmul_op(NO, YES, pM, q_dim + kv_dim + kv_dim, dim, 1.0, 0.0);
        (void)get_cached_matmul_op(NO, YES, pM, dim, q_dim, 1.0, 0.0);
        (void)get_cached_matmul_op(NO, YES, pM, hidden * 2, dim, 1.0, 0.0);
        (void)get_cached_matmul_op(NO, YES, pM, dim, hidden, 1.0, 0.0);
    }

    /* Encode dummy matmuls to trigger GPU pipeline compilation.
     * Uses layer 0 weights (already cached as f16). */
    @autoreleasepool {
        vox_enc_layer_t *el = &ctx->encoder.layers[0];
        int edim = VOX_ENC_DIM;
        int eqkv_n = (VOX_ENC_HEADS + VOX_ENC_KV_HEADS + VOX_ENC_KV_HEADS) * VOX_ENC_HEAD_DIM;
        int ewo_k = VOX_ENC_HEADS * VOX_ENC_HEAD_DIM;
        int ehidden = VOX_ENC_HIDDEN;
        int effn_n = ehidden * 2;
        int M = 128;

        /* Get encoder weight buffers */
        id<MTLBuffer> wQKV = get_merged_f16_3(
            el->wq_weight_bf16, (size_t)ewo_k * edim,
            el->wk_weight_bf16, (size_t)ewo_k * edim,
            el->wv_weight_bf16, (size_t)ewo_k * edim);
        id<MTLBuffer> wWo = get_cached_bf16_as_f16_buffer(
            el->wo_weight_bf16, (size_t)edim * ewo_k);
        id<MTLBuffer> wFFN = get_merged_f16_2(
            el->w1_weight_bf16, (size_t)ehidden * edim,
            el->w3_weight_bf16, (size_t)ehidden * edim);
        id<MTLBuffer> wW2 = get_cached_bf16_as_f16_buffer(
            el->w2_weight_bf16, (size_t)edim * ehidden);

        if (wQKV && wWo && wFFN && wW2) {
            /* Each matmul has different K (input cols): edim, ewo_k, edim, ehidden.
             * Use max K for shared input buffer, per-op output buffers. */
            int max_k = ehidden > ewo_k ? ehidden : ewo_k; /* 5120 */
            int max_n = effn_n > eqkv_n ? effn_n : eqkv_n; /* 10240 */
            id<MTLBuffer> bufIn  = pool_get_buffer((size_t)M * max_k * sizeof(float));
            id<MTLBuffer> bufOut = pool_get_buffer((size_t)M * max_n * sizeof(float));

            if (bufIn && bufOut) {
                id<MTLCommandBuffer> cmd = [g_queue commandBuffer];

                struct { id<MTLBuffer> w; int N, K; } ops[] = {
                    {wQKV, eqkv_n, edim},
                    {wWo,  edim, ewo_k},
                    {wFFN, effn_n, edim},
                    {wW2,  edim, ehidden},
                };
                for (int i = 0; i < 4; i++) {
                    MPSMatrixDescriptor *dA = [MPSMatrixDescriptor
                        matrixDescriptorWithRows:M columns:ops[i].K
                                        rowBytes:ops[i].K * sizeof(float)
                                        dataType:MPSDataTypeFloat32];
                    MPSMatrixDescriptor *dW = [MPSMatrixDescriptor
                        matrixDescriptorWithRows:ops[i].N columns:ops[i].K
                                        rowBytes:ops[i].K * sizeof(uint16_t)
                                        dataType:MPSDataTypeFloat16];
                    MPSMatrixDescriptor *dC = [MPSMatrixDescriptor
                        matrixDescriptorWithRows:M columns:ops[i].N
                                        rowBytes:ops[i].N * sizeof(float)
                                        dataType:MPSDataTypeFloat32];
                    MPSMatrix *mA = [[MPSMatrix alloc] initWithBuffer:bufIn descriptor:dA];
                    MPSMatrix *mW = [[MPSMatrix alloc] initWithBuffer:ops[i].w descriptor:dW];
                    MPSMatrix *mC = [[MPSMatrix alloc] initWithBuffer:bufOut descriptor:dC];
                    MPSMatrixMultiplication *mm = get_cached_matmul_op(NO, YES, M,
                        ops[i].N, ops[i].K, 1.0, 0.0);
                    if (mm) [mm encodeToCommandBuffer:cmd leftMatrix:mA
                                          rightMatrix:mW resultMatrix:mC];
                }
                [cmd commit];
                [cmd waitUntilCompleted];
            }

            pool_release_buffer(bufIn);
            pool_release_buffer(bufOut);
        }

        /* Also pre-warm with prefill M=38 for decoder dimensions */
        {
            vox_dec_layer_t *dl = &dec->layers[0];
            int pM = 38;
            int pqkv_n = q_dim + kv_dim + kv_dim;
            int pffn_n = hidden * 2;

            id<MTLBuffer> dwQKV = get_merged_f16_3(
                dl->wq_weight_bf16, (size_t)q_dim * dim,
                dl->wk_weight_bf16, (size_t)kv_dim * dim,
                dl->wv_weight_bf16, (size_t)kv_dim * dim);
            id<MTLBuffer> dwWo = get_cached_bf16_as_f16_buffer(
                dl->wo_weight_bf16, (size_t)dim * q_dim);
            id<MTLBuffer> dwFFN = get_merged_f16_2(
                dl->w1_weight_bf16, (size_t)hidden * dim,
                dl->w3_weight_bf16, (size_t)hidden * dim);
            id<MTLBuffer> dwW2 = get_cached_bf16_as_f16_buffer(
                dl->w2_weight_bf16, (size_t)dim * hidden);

            if (dwQKV && dwWo && dwFFN && dwW2) {
                int pmax_k = hidden > q_dim ? hidden : q_dim; /* 9216 */
                int pmax_n = pffn_n > pqkv_n ? pffn_n : pqkv_n; /* 18432 */
                id<MTLBuffer> pIn  = pool_get_buffer((size_t)pM * pmax_k * sizeof(float));
                id<MTLBuffer> pOut = pool_get_buffer((size_t)pM * pmax_n * sizeof(float));

                if (pIn && pOut) {
                    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
                    struct { id<MTLBuffer> w; int N, K; } pops[] = {
                        {dwQKV, pqkv_n, dim},
                        {dwWo,  dim, q_dim},
                        {dwFFN, pffn_n, dim},
                        {dwW2,  dim, hidden},
                    };
                    for (int i = 0; i < 4; i++) {
                        MPSMatrixDescriptor *dA = [MPSMatrixDescriptor
                            matrixDescriptorWithRows:pM columns:pops[i].K
                                            rowBytes:pops[i].K * sizeof(float)
                                            dataType:MPSDataTypeFloat32];
                        MPSMatrixDescriptor *dW = [MPSMatrixDescriptor
                            matrixDescriptorWithRows:pops[i].N columns:pops[i].K
                                            rowBytes:pops[i].K * sizeof(uint16_t)
                                            dataType:MPSDataTypeFloat16];
                        MPSMatrixDescriptor *dC = [MPSMatrixDescriptor
                            matrixDescriptorWithRows:pM columns:pops[i].N
                                            rowBytes:pops[i].N * sizeof(float)
                                            dataType:MPSDataTypeFloat32];
                        MPSMatrix *mA = [[MPSMatrix alloc] initWithBuffer:pIn descriptor:dA];
                        MPSMatrix *mW = [[MPSMatrix alloc] initWithBuffer:pops[i].w descriptor:dW];
                        MPSMatrix *mC = [[MPSMatrix alloc] initWithBuffer:pOut descriptor:dC];
                        MPSMatrixMultiplication *mm = get_cached_matmul_op(NO, YES, pM,
                            pops[i].N, pops[i].K, 1.0, 0.0);
                        if (mm) [mm encodeToCommandBuffer:cmd leftMatrix:mA
                                              rightMatrix:mW resultMatrix:mC];
                    }
                    [cmd commit];
                    [cmd waitUntilCompleted];
                }

                pool_release_buffer(pIn);
                pool_release_buffer(pOut);
            }
        }
    }

    /* Pre-warm encoder f32 weight buffers (norms, biases) */
    for (int i = 0; i < VOX_ENC_LAYERS; i++) {
        vox_enc_layer_t *l = &ctx->encoder.layers[i];
        (void)get_cached_weight_buffer(l->attention_norm, VOX_ENC_DIM * sizeof(float));
        (void)get_cached_weight_buffer(l->ffn_norm, VOX_ENC_DIM * sizeof(float));
    }
    (void)get_cached_weight_buffer(ctx->encoder.norm, VOX_ENC_DIM * sizeof(float));
}

size_t vox_metal_memory_used(void) {
    if (!g_initialized) return 0;
    size_t total = 0;
    pthread_mutex_lock(&g_f16_cache_mutex);
    for (int i = 0; i < g_f16_cache_count; i++)
        total += g_f16_cache[i].num_elements * sizeof(uint16_t);
    pthread_mutex_unlock(&g_f16_cache_mutex);
    pthread_mutex_lock(&g_cache_mutex);
    for (int i = 0; i < g_weight_cache_count; i++)
        total += g_weight_cache[i].size;
    pthread_mutex_unlock(&g_cache_mutex);
    return total;
}
