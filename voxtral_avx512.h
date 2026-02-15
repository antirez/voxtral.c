/*
 * voxtral_avx512.h - AVX-512 BF16 accelerated kernels for voxtral.c
 *
 * Drop-in replacement for the BLAS bf16→fp32→sgemm matmul path.
 * Targets AMD Zen 4/5 and Intel Sapphire Rapids+ with AVX-512 BF16 support.
 *
 * The key instruction is VDPBF16PS (vdpbf16ps): it takes two 512-bit
 * registers of packed BF16 pairs (16 pairs = 32 BF16 values each),
 * computes pairwise products, sums each pair, and accumulates into
 * 16 FP32 lanes. This fuses BF16→FP32 conversion + multiply + add
 * in a single instruction, eliminating the separate conversion pass
 * that makes the current BLAS path slow.
 *
 * Usage:
 *   #include "voxtral_avx512.h"
 *
 *   // Check at startup:
 *   if (avx512bf16_available()) { ... use avx512 path ... }
 *
 *   // Matrix multiply: C[M×N] = A[M×K] (fp32) × B[N×K] (bf16, row-major)
 *   // B is stored row-major so each row of B is a contiguous BF16 vector.
 *   // This matches how safetensors weight matrices are laid out.
 *   matmul_avx512bf16_tiled(C, A, B, M, N, K);
 *
 * Compile with: gcc -mavx512f -mavx512bf16 -O3
 * Or for Zen 5: gcc -march=znver5 -O3
 *
 * License: MIT (same as voxtral.c)
 */

#ifndef VOXTRAL_AVX512_H
#define VOXTRAL_AVX512_H

#include <immintrin.h>
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * CPU Feature Detection
 * ======================================================================== */

static inline int avx512bf16_available(void) {
    uint32_t eax, ebx, ecx, edx;

    /* Check AVX-512 Foundation (CPUID.7.0:EBX bit 16) */
    __asm__ __volatile__(
        "cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(7), "c"(0)
    );
    if (!(ebx & (1u << 16))) return 0;  /* No AVX-512F */

    /* Check AVX-512 BF16 (CPUID.7.1:EAX bit 5) */
    __asm__ __volatile__(
        "cpuid"
        : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
        : "a"(7), "c"(1)
    );
    return (eax & (1u << 5)) ? 1 : 0;  /* AVX512_BF16 */
}

/* ========================================================================
 * BF16 ↔ FP32 Conversion Utilities
 * ======================================================================== */

/* Scalar BF16 → FP32: just a left shift by 16 bits. */
static inline float bf16_to_fp32(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/* Convert 16 FP32 values to 16 BF16 values in a 256-bit register.
 * Uses VCVTNEPS2BF16 (available with AVX-512 BF16).
 * Returns __m256i for easy use with store intrinsics. */
static inline __m256i fp32x16_to_bf16(__m512 v) {
    return (__m256i)_mm512_cvtneps_pbh(v);
}

/* ========================================================================
 * Core Matmul Kernel: C[M×N] += A[M×K](fp32) × B^T[N×K](bf16)
 *
 * This is the operation voxtral.c needs: the activations (A) are fp32,
 * the weights (B) are bf16 mmap'd from safetensors, and we want C in fp32.
 *
 * B is stored as N rows of K bf16 values (row-major), which means
 * computing C[i][j] = dot(A_row_i, B_row_j) over the K dimension.
 *
 * Strategy:
 *   - Convert A on-the-fly to BF16 in tiles (not the whole matrix at once)
 *   - Use VDPBF16PS to compute the dot product in the K dimension
 *   - Process 16 output columns at a time (16 N-values per ZMM register)
 *   - Tile over K in blocks of 32 (VDPBF16PS processes 2 BF16 pairs per lane)
 *
 * VDPBF16PS semantics (per 32-bit lane i):
 *   dst[i] += src1[2*i] * src2[2*i] + src1[2*i+1] * src2[2*i+1]
 *   where src1/src2 contain interleaved BF16 pairs.
 *
 * So for a dot product over K, we need to arrange data so that
 * adjacent BF16 values in a register correspond to adjacent K indices.
 * ======================================================================== */

/*
 * matmul_avx512bf16_tiled - Cache-friendly tiled matmul.
 *
 * Tiles over N to improve L2 cache reuse of the A row's BF16 conversion.
 * For voxtral's typical shapes (K=1280 or 3072, N=1280..5120),
 * this can help keep B-tiles in L2.
 *
 * Processes 4 output columns (N) simultaneously to amortize
 * the cost of loading A from memory.
 */
static void matmul_avx512bf16_tiled(
    float *restrict C,
    const float *restrict A,
    const uint16_t *restrict B,
    int M, int N, int K)
{
    memset(C, 0, (size_t)M * N * sizeof(float));

    int K_padded = (K + 31) & ~31;

    /* Tile size for N dimension — process 4 B-rows at once */
    #define N_TILE 4

    #pragma omp parallel for schedule(static) if(M > 1)
    for (int i = 0; i < M; i++) {
        const float *a_row = A + (size_t)i * K;

        /* Convert A row to BF16 once */
        uint16_t a_bf16_buf[((16384 + 31) & ~31)];  /* stack buffer */
        uint16_t *a_bf16 = a_bf16_buf;
        if (K_padded > (int)sizeof(a_bf16_buf) / (int)sizeof(uint16_t)) {
            a_bf16 = (uint16_t *)__builtin_alloca(K_padded * sizeof(uint16_t));
        }

        int k = 0;
        for (; k + 15 < K; k += 16) {
            __m512 av = _mm512_loadu_ps(a_row + k);
            __m256i abf = fp32x16_to_bf16(av);
            _mm_storeu_si128((__m128i *)(a_bf16 + k), _mm256_castsi256_si128(abf));
            _mm_storeu_si128((__m128i *)(a_bf16 + k + 8),
                             _mm256_extracti128_si256(abf, 1));
        }
        for (; k < K; k++) {
            uint32_t bits;
            memcpy(&bits, &a_row[k], sizeof(bits));
            a_bf16[k] = (uint16_t)(bits >> 16);
        }
        for (k = K; k < K_padded; k++) {
            a_bf16[k] = 0;
        }

        /* Process N in tiles of N_TILE */
        int j = 0;
        for (; j + N_TILE - 1 < N; j += N_TILE) {
            const uint16_t *b0 = B + (size_t)(j + 0) * K;
            const uint16_t *b1 = B + (size_t)(j + 1) * K;
            const uint16_t *b2 = B + (size_t)(j + 2) * K;
            const uint16_t *b3 = B + (size_t)(j + 3) * K;

            __m512 sum0 = _mm512_setzero_ps();
            __m512 sum1 = _mm512_setzero_ps();
            __m512 sum2 = _mm512_setzero_ps();
            __m512 sum3 = _mm512_setzero_ps();

            for (int kk = 0; kk < K_padded; kk += 32) {
                __m512i av = _mm512_loadu_si512(a_bf16 + kk);
                __m512bh abh = (__m512bh)av;

                sum0 = _mm512_dpbf16_ps(sum0, abh, (__m512bh)_mm512_loadu_si512(b0 + kk));
                sum1 = _mm512_dpbf16_ps(sum1, abh, (__m512bh)_mm512_loadu_si512(b1 + kk));
                sum2 = _mm512_dpbf16_ps(sum2, abh, (__m512bh)_mm512_loadu_si512(b2 + kk));
                sum3 = _mm512_dpbf16_ps(sum3, abh, (__m512bh)_mm512_loadu_si512(b3 + kk));
            }

            C[(size_t)i * N + j + 0] = _mm512_reduce_add_ps(sum0);
            C[(size_t)i * N + j + 1] = _mm512_reduce_add_ps(sum1);
            C[(size_t)i * N + j + 2] = _mm512_reduce_add_ps(sum2);
            C[(size_t)i * N + j + 3] = _mm512_reduce_add_ps(sum3);
        }
        /* Remainder columns */
        for (; j < N; j++) {
            const uint16_t *b_row = B + (size_t)j * K;
            __m512 acc = _mm512_setzero_ps();
            for (int kk = 0; kk < K_padded; kk += 32) {
                __m512i av = _mm512_loadu_si512(a_bf16 + kk);
                __m512i bv = _mm512_loadu_si512(b_row + kk);
                acc = _mm512_dpbf16_ps(acc, (__m512bh)av, (__m512bh)bv);
            }
            C[(size_t)i * N + j] = _mm512_reduce_add_ps(acc);
        }
    }
    #undef N_TILE
}

#ifdef __cplusplus
}
#endif

#endif /* VOXTRAL_AVX512_H */
