#pragma once

#include "../Utils.hpp"
#include "../simd.hpp"
#include <cmath>
#include <numeric>

namespace {

/**
  * Horizontal SIMD register sum functions.
  * Adapted from https://stackoverflow.com/questions/6996764/fastest-way-to-do-horizontal-sse-vector-sum-or-other-reduction
  */

#if (defined(__GNUC__) || defined(__clang__)) && (!defined(__ARM_FEATURE_SIMD32) && !defined(__ARM_NEON))
__attribute__((target("sse3")))
#endif
ALWAYS_INLINE float hsum_ps_sse3(__m128 const v) {
#if !defined(__i386__) && !defined(__x86_64__) && !defined(_M_X64)
  return 0.f;
#else
  __m128 shuf = _mm_movehdup_ps(v); // broadcast elements 3,1 to 2,0
  __m128 sums = _mm_add_ps(v, shuf);
  shuf = _mm_movehl_ps(shuf, sums); // high half -> low half
  sums = _mm_add_ss(sums, shuf);
  return _mm_cvtss_f32(sums);
#endif
}

#if (defined(__GNUC__) || defined(__clang__)) && (!defined(__ARM_FEATURE_SIMD32) && !defined(__ARM_NEON))
__attribute__((target("avx")))
#endif
ALWAYS_INLINE float hsum256_ps_avx(__m256 const v) {
#if !defined(__i386__) && !defined(__x86_64__) && !defined(_M_X64)
  return 0.f;
#else
  __m128 vlow = _mm256_castps256_ps128(v);
  __m128 vhigh = _mm256_extractf128_ps(v, 1); // high 128
  vlow = _mm_add_ps(vlow, vhigh);     // add the low 128
  return hsum_ps_sse3(vlow);         // and inline the sse3 version, which is optimal for AVX
                                     // (no wasted instructions, and all of them are the 4B minimum)
#endif
}

#if (defined(__GNUC__) || defined(__clang__)) && (!defined(__ARM_FEATURE_SIMD32) && !defined(__ARM_NEON))
__attribute__((target("avx2,fma")))
#endif
float dot256_ps_fma3(float const* x1, float const* x2, std::size_t const len, float init = 0.) {
#if !defined(__i386__) && !defined(__x86_64__) && !defined(_M_X64)
  return 0.f;
#else
  static constexpr std::size_t SIMDW = 8, CACHELINE = 64u;
  std::size_t const limit = len & static_cast<std::size_t>(-static_cast<std::ptrdiff_t>(SIMDW));
  std::size_t const limit_x2 = len & static_cast<std::size_t>(-static_cast<std::ptrdiff_t>(SIMDW * 2u));
  std::size_t remainder = len & (SIMDW - 1u), i = SIMDW * 2u;
  __m256 sum0 = _mm256_setzero_ps();
  __m256 sum1 = _mm256_setzero_ps();
  _mm_prefetch((char*)(x1 + (CACHELINE / sizeof(float))), _MM_HINT_NTA);
  _mm_prefetch((char*)(x2 + (CACHELINE / sizeof(float))), _MM_HINT_NTA);
  if (i <= limit_x2) {
    sum0 = _mm256_mul_ps(_mm256_loadu_ps(x1), _mm256_loadu_ps(x2));
    sum1 = _mm256_mul_ps(_mm256_loadu_ps(x1 + SIMDW), _mm256_loadu_ps(x2 + SIMDW));
  }
  for (; i < limit_x2; i += SIMDW * 2u) {
    sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(x1 + i), _mm256_loadu_ps(x2 + i), sum0);
    sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(x1 + i + SIMDW), _mm256_loadu_ps(x2 + i + SIMDW), sum1);
  }
  sum0 = _mm256_add_ps(sum0, sum1);
  if (i < limit)
    sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(x1 + i), _mm256_loadu_ps(x2 + i), sum0);
  for (; remainder > 0; remainder--)
    init += x1[len - remainder] * x2[len - remainder];
  return init + hsum256_ps_avx(sum0);
#endif
}

#if (defined(__GNUC__) || defined(__clang__)) && (!defined(__ARM_FEATURE_SIMD32) && !defined(__ARM_NEON))
__attribute__((target("avx2")))
#endif
float sum256_ps(float const* x, std::size_t const len, float init = 0.) {
#if !defined(__i386__) && !defined(__x86_64__) && !defined(_M_X64)
  return 0.f;
#else
  static constexpr std::size_t SIMDW = 8;
  std::size_t const limit = len & static_cast<std::size_t>(-static_cast<std::ptrdiff_t>(SIMDW)), remainder = len & (SIMDW - 1);
  if (limit > 0) {
    __m256 sum = _mm256_loadu_ps(x);
    for (std::size_t i = SIMDW; i < limit; i += SIMDW)
      sum = _mm256_add_ps(_mm256_loadu_ps(x + i), sum);
    init += hsum256_ps_avx(sum);
  }
  return (!remainder) ? init : std::accumulate(x + limit, x + len, init);
#endif
}

#if defined(__i386__) || defined(__x86_64__) || defined(_M_X64)
# if (defined(__GNUC__) || defined(__clang__)) && (!defined(__ARM_FEATURE_SIMD32) && !defined(__ARM_NEON))
__attribute__((target("avx2,fma")))
# endif
/**
  * Float exp function for 256-bit registers using AVX2.
  * Adapted from https://stackoverflow.com/questions/48863719/fastest-implementation-of-exponential-function-using-avx
  * Computes exp(x) for x in [-87.33654, 88.72283]
  * Maximum relative error: 3.1533e-6
  */
__m256 exp256_ps_fma3(__m256 const x)
{
  __m256 t, f, p, r;
  __m256i i, j;

  __m256 const l2e = _mm256_set1_ps(1.442695041f); /* log2(e) */
  __m256 const l2h = _mm256_set1_ps(-6.93145752e-1f); /* -log(2)_hi */
  __m256 const l2l = _mm256_set1_ps(-1.42860677e-6f); /* -log(2)_lo */
                                                       /* coefficients for core approximation to exp() in [-log(2)/2, log(2)/2] */
  __m256 const c0 = _mm256_set1_ps(0.041944388f);
  __m256 const c1 = _mm256_set1_ps(0.168006673f);
  __m256 const c2 = _mm256_set1_ps(0.499999940f);
  __m256 const c3 = _mm256_set1_ps(0.999956906f);
  __m256 const c4 = _mm256_set1_ps(0.999999642f);

  /* exp(x) = 2^i * e^f; i = rint (log2(e) * x), f = x - log(2) * i */
  t = _mm256_mul_ps(x, l2e);      /* t = log2(e) * x */
  r = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); /* r = rint (t) */

  f = _mm256_fmadd_ps(r, l2h, x); /* x - log(2)_hi * r */
  f = _mm256_fmadd_ps(r, l2l, f); /* f = x - log(2)_hi * r - log(2)_lo * r */
  i = _mm256_cvtps_epi32(t);       /* i = (int)rint(t) */

                                   /* p ~= exp (f), -log(2)/2 <= f <= log(2)/2 */
  p = c0;                          /* c0 */
  p = _mm256_fmadd_ps(p, f, c1);  /* c0*f+c1 */
  p = _mm256_fmadd_ps(p, f, c2);  /* (c0*f+c1)*f+c2 */
  p = _mm256_fmadd_ps(p, f, c3);  /* ((c0*f+c1)*f+c2)*f+c3 */
  p = _mm256_fmadd_ps(p, f, c4);  /* (((c0*f+c1)*f+c2)*f+c3)*f+c4 ~= exp(f) */

                                   /* exp(x) = 2^i * p */
  j = _mm256_slli_epi32(i, 23); /* i << 23 */
  return _mm256_castsi256_ps(_mm256_add_epi32(j, _mm256_castps_si256(p))); /* r = p * 2^i */
}
#endif

// non-simd vector functions

float SumOfSquares(float *v1, size_t n) {
  assert(n > 0);
  float result = 0.0f;
  for (size_t i = 0; i < n; i++) {
    float f = v1[i];
    result += f * f;
  }
  return result;
}

float SumOfProducts(float* v1, float* v2, size_t n) {
  assert(n > 0);
  float result = 0.0f;
  for (size_t i = 0; i < n; i++)
    result += v1[i] * v2[i];
  return result;
}

// fast non-simd approximations

float tanha(float v) {
  const float c1 = 0.03138777F;
  const float c2 = 0.276281267F;
  const float c_log2f = 1.442695022F;
  v *= c_log2f;
  int intPart = (int)v;
  float x = (v - intPart);
  float xx = x * x;
  float v1 = c_log2f + c2 * xx;
  float v2 = x + xx * c1 * x;
  float v3 = (v2 + v1);
  *((int*)&v3) += intPart << 24;
  float v4 = v2 - v1;
  return (v3 + v4) / (v3 - v4);
}

float expa(float x) {
  return 2.0f / (tanha(-x * 0.5f) + 1.0f) - 1.0f;
}

} // end of namespace
