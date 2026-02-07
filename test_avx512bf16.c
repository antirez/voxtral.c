/*
 * test_avx512bf16.c - Benchmark & correctness test for voxtral_avx512.h
 *
 * Compile: gcc -O3 -march=znver5 -o test_avx512bf16 test_avx512bf16.c -lm
 *    (or:  gcc -O3 -mavx512f -mavx512bf16 -o test_avx512bf16 test_avx512bf16.c -lm)
 *
 * This program:
 *  1. Checks that your CPU supports AVX-512 BF16
 *  2. Runs a reference scalar matmul (bf16→fp32 + fp32 dot product)
 *  3. Runs the AVX-512 BF16 matmul
 *  4. Compares outputs for correctness (max absolute error)
 *  5. Benchmarks both paths
 *
 * Typical voxtral dimensions to test:
 *   Encoder attention:  M=1, K=1280, N=1280  (per-token QKV projection)
 *   Encoder FFN:        M=1, K=1280, N=5120  (up projection)
 *   Decoder attention:  M=1, K=3072, N=3072
 *   Adapter:            M=seq, K=5120, N=3072
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "voxtral_avx512.h"

/* ---- Helpers ---- */

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static float rand_float(uint32_t *state) {
    return (float)(int32_t)xorshift32(state) / (float)INT32_MAX;
}

static uint16_t fp32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

/* ---- Reference scalar implementation ---- */

static void matmul_reference(
    float *C,
    const float *A,
    const uint16_t *B,
    int M, int N, int K)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                float bval = bf16_to_fp32(B[(size_t)j * K + k]);
                sum += A[(size_t)i * K + k] * bval;
            }
            C[(size_t)i * N + j] = sum;
        }
    }
}

/* ---- Test runner ---- */

static int test_shape(int M, int N, int K, int warmup, int iters) {
    printf("Shape: M=%d, N=%d, K=%d\n", M, N, K);

    size_t A_size = (size_t)M * K;
    size_t B_size = (size_t)N * K;
    size_t C_size = (size_t)M * N;

    float *A = (float *)aligned_alloc(64, A_size * sizeof(float));
    uint16_t *B = (uint16_t *)aligned_alloc(64, B_size * sizeof(uint16_t));
    float *C_ref = (float *)aligned_alloc(64, C_size * sizeof(float));
    float *C_avx = (float *)aligned_alloc(64, C_size * sizeof(float));

    if (!A || !B || !C_ref || !C_avx) {
        fprintf(stderr, "  Allocation failed\n");
        return 1;
    }

    /* Fill with random data */
    uint32_t rng = 42;
    for (size_t i = 0; i < A_size; i++)
        A[i] = rand_float(&rng) * 0.1f;
    for (size_t i = 0; i < B_size; i++)
        B[i] = fp32_to_bf16(rand_float(&rng) * 0.1f);

    /* Correctness check */
    matmul_reference(C_ref, A, B, M, N, K);
    matmul_avx512bf16_tiled(C_avx, A, B, M, N, K);

    float max_err = 0.0f;
    float max_rel_err = 0.0f;
    for (size_t i = 0; i < C_size; i++) {
        float err = fabsf(C_ref[i] - C_avx[i]);
        if (err > max_err) max_err = err;
        float denom = fabsf(C_ref[i]);
        if (denom > 1e-6f) {
            float rel = err / denom;
            if (rel > max_rel_err) max_rel_err = rel;
        }
    }

    /*
     * BF16 has ~7 bits of mantissa (vs FP32's 23), so the dot product
     * accumulates rounding errors. The AVX path converts A→BF16 and uses
     * VDPBF16PS, while the reference converts B→FP32 and does FP32 FMA.
     * There IS an expected numerical difference from the BF16 truncation
     * of A. This is acceptable for inference.
     *
     * We use absolute tolerance because relative error is meaningless
     * when output values are near zero. For the input scale used here
     * (±0.1), absolute errors under 0.01 are well within bf16 precision.
     */
    int pass = (max_err < 0.01f);  /* absolute tolerance */
    printf("  Correctness: max_abs_err=%.6e, max_rel_err=%.6e  [%s]\n",
           max_err, max_rel_err, pass ? "PASS" : "FAIL");

    if (!pass) {
        /* Print first few mismatches for debugging */
        int printed = 0;
        for (size_t i = 0; i < C_size && printed < 5; i++) {
            float err = fabsf(C_ref[i] - C_avx[i]);
            if (err > max_err * 0.5f) {
                printf("    [%zu] ref=%.6f avx=%.6f diff=%.6e\n",
                       i, C_ref[i], C_avx[i], err);
                printed++;
            }
        }
    }

    /* Benchmark reference (scalar) */
    for (int w = 0; w < warmup; w++)
        matmul_reference(C_ref, A, B, M, N, K);

    double t0 = now_sec();
    for (int it = 0; it < iters; it++)
        matmul_reference(C_ref, A, B, M, N, K);
    double t_ref = (now_sec() - t0) / iters;

    /* Benchmark AVX-512 BF16 */
    for (int w = 0; w < warmup; w++)
        matmul_avx512bf16_tiled(C_avx, A, B, M, N, K);

    t0 = now_sec();
    for (int it = 0; it < iters; it++)
        matmul_avx512bf16_tiled(C_avx, A, B, M, N, K);
    double t_avx = (now_sec() - t0) / iters;

    double gflops_ref = 2.0 * M * N * K / t_ref / 1e9;
    double gflops_avx = 2.0 * M * N * K / t_avx / 1e9;

    printf("  Reference:   %.3f ms  (%.1f GFLOPS)\n", t_ref * 1e3, gflops_ref);
    printf("  AVX-512 BF16: %.3f ms  (%.1f GFLOPS)\n", t_avx * 1e3, gflops_avx);
    printf("  Speedup:     %.1fx\n\n", t_ref / t_avx);

    free(A);
    free(B);
    free(C_ref);
    free(C_avx);

    return pass ? 0 : 1;
}

int main(void) {
    printf("=== AVX-512 BF16 Matmul Benchmark ===\n\n");

    if (!avx512bf16_available()) {
        fprintf(stderr, "ERROR: This CPU does not support AVX-512 BF16.\n");
        fprintf(stderr, "Required: AMD Zen 4+ or Intel Sapphire Rapids+.\n");
        return 1;
    }
    printf("AVX-512 BF16: detected and available.\n\n");

    int failures = 0;
    int warmup = 2;

    /* Small shape — encoder per-token projection */
    failures += test_shape(1, 1280, 1280, warmup, 100);

    /* Medium — encoder FFN up-projection */
    failures += test_shape(1, 5120, 1280, warmup, 50);

    /* Decoder attention projection */
    failures += test_shape(1, 3072, 3072, warmup, 50);

    /* Adapter: batch of tokens */
    failures += test_shape(64, 3072, 5120, warmup, 5);

    /* Encoder: batch of frames (larger M) */
    failures += test_shape(128, 1280, 1280, warmup, 5);

    if (failures) {
        printf("*** %d test(s) FAILED ***\n", failures);
    } else {
        printf("All tests passed.\n");
    }

    return failures;
}
