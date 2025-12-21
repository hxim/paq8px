#include "SimdFunctions.hpp"
#include <numeric>
#include <cassert>

#ifdef X64_SIMD_AVAILABLE

#if (defined(__GNUC__) || defined(__clang__)) 
__attribute__((target("sse3")))
#endif
float hsum_ps_sse3(__m128 const v) {
  __m128 shuf = _mm_movehdup_ps(v);
  __m128 sums = _mm_add_ps(v, shuf);
  shuf = _mm_movehl_ps(shuf, sums);
  sums = _mm_add_ss(sums, shuf);
  return _mm_cvtss_f32(sums);
}

#if (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx")))
#endif
float hsum256_ps_avx(__m256 const v) {
  __m128 vlow = _mm256_castps256_ps128(v);
  __m128 vhigh = _mm256_extractf128_ps(v, 1);
  vlow = _mm_add_ps(vlow, vhigh);
  return hsum_ps_sse3(vlow);
}

#if (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2,fma")))
#endif
float dot256_ps_fma3(float const* x1, float const* x2, size_t const len, float init) {
  static constexpr size_t SIMDW = 8, CACHELINE = 64;
  size_t const limit = len & static_cast<size_t>(-static_cast<ptrdiff_t>(SIMDW));
  size_t const limit_x2 = len & static_cast<size_t>(-static_cast<ptrdiff_t>(SIMDW * 2));
  size_t remainder = len & (SIMDW - 1), i = SIMDW * 2;
  __m256 sum0 = _mm256_setzero_ps();
  __m256 sum1 = _mm256_setzero_ps();

  _mm_prefetch((char*)(x1 + (CACHELINE / sizeof(float))), _MM_HINT_NTA);
  _mm_prefetch((char*)(x2 + (CACHELINE / sizeof(float))), _MM_HINT_NTA);

  if (i <= limit_x2) {
    sum0 = _mm256_mul_ps(_mm256_loadu_ps(x1), _mm256_loadu_ps(x2));
    sum1 = _mm256_mul_ps(_mm256_loadu_ps(x1 + SIMDW), _mm256_loadu_ps(x2 + SIMDW));
  }

  for (; i < limit_x2; i += SIMDW * 2) {
    sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(x1 + i), _mm256_loadu_ps(x2 + i), sum0);
    sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(x1 + i + SIMDW), _mm256_loadu_ps(x2 + i + SIMDW), sum1);
  }

  sum0 = _mm256_add_ps(sum0, sum1);

  if (i < limit)
    sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(x1 + i), _mm256_loadu_ps(x2 + i), sum0);

  for (; remainder > 0; remainder--)
    init += x1[len - remainder] * x2[len - remainder];

  return init + hsum256_ps_avx(sum0);
}

#if (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
#endif
float sum256_ps(float const* x, size_t const len, float init) {
  static constexpr size_t SIMDW = 8;
  size_t const limit = len & static_cast<size_t>(-static_cast<ptrdiff_t>(SIMDW));
  size_t const remainder = len & (SIMDW - 1);

  if (limit > 0) {
    __m256 sum = _mm256_loadu_ps(x);
    for (size_t i = SIMDW; i < limit; i += SIMDW)
      sum = _mm256_add_ps(_mm256_loadu_ps(x + i), sum);
    init += hsum256_ps_avx(sum);
  }

  return (!remainder) ? init : std::accumulate(x + limit, x + len, init);
}

#if (defined(__GNUC__) || defined(__clang__)) 
__attribute__((target("avx2,fma")))
#endif
__m256 exp256_ps_fma3(__m256 const x) {
  __m256 t, f, p, r;
  __m256i i, j;

  __m256 const l2e = _mm256_set1_ps(1.442695041f);
  __m256 const l2h = _mm256_set1_ps(-6.93145752e-1f);
  __m256 const l2l = _mm256_set1_ps(-1.42860677e-6f);
  __m256 const c0 = _mm256_set1_ps(0.041944388f);
  __m256 const c1 = _mm256_set1_ps(0.168006673f);
  __m256 const c2 = _mm256_set1_ps(0.499999940f);
  __m256 const c3 = _mm256_set1_ps(0.999956906f);
  __m256 const c4 = _mm256_set1_ps(0.999999642f);

  t = _mm256_mul_ps(x, l2e);
  r = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

  f = _mm256_fmadd_ps(r, l2h, x);
  f = _mm256_fmadd_ps(r, l2l, f);
  i = _mm256_cvtps_epi32(t);

  p = c0;
  p = _mm256_fmadd_ps(p, f, c1);
  p = _mm256_fmadd_ps(p, f, c2);
  p = _mm256_fmadd_ps(p, f, c3);
  p = _mm256_fmadd_ps(p, f, c4);

  j = _mm256_slli_epi32(i, 23);
  return _mm256_castsi256_ps(_mm256_add_epi32(j, _mm256_castps_si256(p)));
}

#endif // X64_SIMD_AVAILABLE

// Non-SIMD vector functions

float SumOfSquares(float* v1, size_t n) {
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

// Fast non-SIMD approximations

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
