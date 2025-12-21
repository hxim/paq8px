#pragma once

#include "../Utils.hpp"
#include "../Simd.hpp"
#include <cmath>
#include <cstddef>

// SIMD horizontal sum functions
#ifdef X64_SIMD_AVAILABLE

#if (defined(__GNUC__) || defined(__clang__)) 
__attribute__((target("sse3")))
#endif
float hsum_ps_sse3(__m128 v);

#if (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx")))
#endif
float hsum256_ps_avx(__m256 v);

#if (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,fma")))
#endif
float dot256_ps_fma3(float const* x1, float const* x2, size_t len, float init);

#if (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
#endif
float sum256_ps(float const* x, size_t len, float init);

#if (defined(__GNUC__) || defined(__clang__)) 
__attribute__((target("avx2,fma")))
#endif
__m256 exp256_ps_fma3(__m256 x);

#endif // X64_SIMD_AVAILABLE

// Non-SIMD vector functions
float SumOfSquares(float *v1, size_t n);
float SumOfProducts(float* v1, float* v2, size_t n);

// Fast non-SIMD approximations
float tanha(float v);
float expa(float x);