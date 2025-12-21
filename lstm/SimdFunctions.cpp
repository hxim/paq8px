#include <numeric>
#include <cassert>
#include <cstring>

#include "SimdFunctions.hpp"

// ============================================================================
// Padé Approximants for Activation Functions
// ============================================================================

// Tanh Padé approximant: tanh(x) ≈ x(27 + x²) / (27 + 9x²)
static float constexpr tanh_pade(float x) {
  float x2 = x * x;
  return (x * (27.0f + x2)) / (27.0f + 9.0f * x2);
}

// Sigmoid via tanh: σ(x) = 0.5 * (1 + tanh(x/2))
static float constexpr sigmoid_pade(float x) {
  float u = 0.5f * x;
  float u2 = u * u;
  float tanh_val = (u * (27.0f + u2)) / (27.0f + 9.0f * u2);
  return 0.5f * (1.0f + tanh_val);
}

// Clipping constants
static constexpr float SIGMOID_CLIP_MIN = -5.95f;
static constexpr float SIGMOID_CLIP_MAX = +5.95f;
static constexpr float TANH_CLIP_MIN = SIGMOID_CLIP_MIN / 2.0f;
static constexpr float TANH_CLIP_MAX = SIGMOID_CLIP_MAX / 2.0f;

static constexpr float SIGMOID_MIN = sigmoid_pade(SIGMOID_CLIP_MIN);
static constexpr float SIGMOID_MAX = sigmoid_pade(SIGMOID_CLIP_MAX);
static constexpr float TANH_MIN = tanh_pade(TANH_CLIP_MIN);
static constexpr float TANH_MAX = tanh_pade(TANH_CLIP_MAX);

static_assert(SIGMOID_MIN > 0.0f);
static_assert(SIGMOID_MAX < 1.0f);
static_assert(TANH_MIN > -1.0f);
static_assert(TANH_MAX < +1.0f);

float tanh_pade_clipped(float x) {
  if (x > TANH_CLIP_MAX) return TANH_MAX;
  if (x < TANH_CLIP_MIN) return TANH_MIN;
  return tanh_pade(x);
}

// Sigmoid via tanh: σ(x) = 0.5 * (1 + tanh(x/2))
static float sigmoid_pade_clipped(float x) {
  if (x > SIGMOID_CLIP_MAX) return SIGMOID_MAX;
  if (x < SIGMOID_CLIP_MIN) return SIGMOID_MIN;
  return sigmoid_pade(x);
}

// ============================================================================
// Exponential Function for Softmax
// ============================================================================

// Helper for bitcasting
static inline float bitcast_u32_to_f32(uint32_t x) {
  float result;
  memcpy(&result, &x, sizeof(float));
  return result;
}

// expf
// Cody–Waite style reduction with split ln2 (hi+lo) and degree-5 polynomial on the reduced range.
// Preconditions: x is finite; FP rounding mode is round-to-nearest-even. (softmax pipeline guarantees these)
// => no need to check isfinite - for any case we'll check outputs on the result later
// Notes: 
//   Max relative error ≈ 3.37e-7
//   Max ULP error ≈ 4 ULP
//   Median ULP ≈ 1, mean ≈ 0.61
//   Monotonic, strictly positive over the whole domain.
// to get hex literals use: printf("%a\n", x);  // %a for hex float
// see also https://chromium.googlesource.com/external/github.com/google/XNNPACK/+/refs/heads/upstream/test_638074745/src/math/f32-sigmoid-sse2-rr2-p5-div.c

static inline float expf_compat(float x) {
  //keep (n + 127) in [1,254] for a valid exponent field
  if (x < -87)
    x = -87; //final result will be 1.64581131e-38

  // Cody–Waite split of ln2
  const float INV_LN2 = 0x1.715476p+0f;   // 1.4426950216f  // properly rounded 1/ln2
  const float LN2_HI = 0x1.62e400p-1f;    // 0.693145752f   // coarsened value of ln2 i.e. with zeroed 7 last fraction bits
  const float LN2_LO = 0x1.7f7d1cp-20f;   // 1.42860677e-6f // ln2-LN2_HI
  //This way LN2_HI + LN2_LO (in exact real arithmetic) is extremely close to ln2, and when computed in float it reproduces the correctly rounded float value of ln⁡2.
  //There are many valid Cody–Waite splits; this pair (hi = 0x1.62e400p-1f, lo = 0x1.7f7d1cp-20f) is a well-tested single-precision choice that balances reduction error for ∣n∣≲126.

  // n = round(x / ln2)
  // t = (float)n
  float z = x * INV_LN2;
  const float MAGIC_BIAS = 12582912;   // 1.5 x 2^23 : trick for rounding
  float t = z + MAGIC_BIAS;
  int   n = (int)t - 12582912;         // subtract the bias in integer domain;
  t -= MAGIC_BIAS;

  // r = x - n*ln2 using split constants (Cody–Waite)
  float r = x - t * LN2_HI;
  r = r - t * LN2_LO;

  // Taylor coefficients
  // exp(r) ≈ 1 + r + c2 r^2 + c3 r^3 + c4 r^4 + c5 r^5
  const float c2 = 0x1.fffe24p-2f;  // ≈ 0.499992907 (≈ 1/2)
  const float c3 = 0x1.5554acp-3f;  // ≈ 0.166665405 (≈ 1/6)
  const float c4 = 0x1.5713a4p-5f;  // ≈ 0.041879482 (≈ 1/24)
  const float c5 = 0x1.12266ap-7f;  // ≈ 0.008366401 (≈ 1/120)

  // Estrin's Scheme
  // It gives shorter dependency chains than Horner, which usually wins on SIMD and GPUs.
  float r2 = r * r;
  float q2 = c2 + c3 * r;
  float q4 = c4 + c5 * r;
  float p = (q4 * r2 + q2) * r2 + (r + 1.0f);

  // construct 2^n as float via exponent bits: (n + 127) << 23
  uint32_t expbits = (uint32_t)(n + 127);
  // note: expbits must be in [1,254] for normalized; we've clamped x earlier.
  uint32_t bits = expbits << 23;
  float two_n = bitcast_u32_to_f32(bits);

  return p * two_n;
}

// ============================================================================
// SIMD Horizontal Sum Functions
// ============================================================================

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

// ============================================================================
// SIMD Dot Product (AVX2)
// ============================================================================

#if (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
#endif
float dot256_ps_avx2(float const* x1, float const* x2, size_t const len, float init) {
  static constexpr size_t SIMDW = 8;
  size_t const limit = len & static_cast<size_t>(-static_cast<ptrdiff_t>(SIMDW));
  size_t remainder = len & (SIMDW - 1);

  __m256 sum = _mm256_setzero_ps();

  for (size_t i = 0; i < limit; i += SIMDW) {
    __m256 a = _mm256_loadu_ps(x1 + i);
    __m256 b = _mm256_loadu_ps(x2 + i);
    __m256 prod = _mm256_mul_ps(a, b);
    sum = _mm256_add_ps(sum, prod);
  }

  init += hsum256_ps_avx(sum);

  for (size_t i = limit; i < len; i++)
    init += x1[i] * x2[i];

  return init;
}

// ============================================================================
// SIMD Sum (AVX2)
// ============================================================================

#if (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
#endif
float sum256_ps_avx2(float const* x, size_t const len, float init) {
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

// ============================================================================
// SIMD Softmax (AVX2)
// ============================================================================

#if (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
#endif
void softmax_avx2(float* logits, float* probs, size_t const len, float const max_logit) {
  static constexpr size_t SIMDW = 4;

  // Constants for exponential
  const __m128 INV_LN2 = _mm_set1_ps(0x1.715476p+0f);
  const __m128 LN2_HI = _mm_set1_ps(0x1.62e400p-1f);
  const __m128 LN2_LO = _mm_set1_ps(0x1.7f7d1cp-20f);
  const __m128 c2 = _mm_set1_ps(0x1.fffe24p-2f);
  const __m128 c3 = _mm_set1_ps(0x1.5554acp-3f);
  const __m128 c4 = _mm_set1_ps(0x1.5713a4p-5f);
  const __m128 c5 = _mm_set1_ps(0x1.12266ap-7f);
  const __m128 one_vec = _mm_set1_ps(1.0f);
  const __m128i MAGIC_BIAS = _mm_set1_epi32(12582912);
  const __m128i c127 = _mm_set1_epi32(127);
  const __m128 cplus87 = _mm_set1_ps(87.0f);
  const __m128 cminus87 = _mm_set1_ps(-87.0f);
  const __m128 maxlogit_vec = _mm_set1_ps(max_logit);

  __m128 expsum_vec1 = _mm_setzero_ps();
  __m128 expsum_vec2 = _mm_setzero_ps();

  size_t const limit = len & static_cast<size_t>(-static_cast<ptrdiff_t>(SIMDW * 2));

  // Compute exp(logit - max_logit) for all elements
  for (size_t i = 0; i < limit; i += SIMDW * 2) {
    // First block of 4
    {
      __m128 logits_vec = _mm_load_ps(&logits[i]);
      logits_vec = _mm_sub_ps(logits_vec, maxlogit_vec);
      logits_vec = _mm_min_ps(logits_vec, cplus87);
      logits_vec = _mm_max_ps(logits_vec, cminus87);

      __m128 z = _mm_mul_ps(logits_vec, INV_LN2);
      const __m128 m = _mm_cvtepi32_ps(MAGIC_BIAS);
      __m128 t = _mm_add_ps(z, m);
      __m128i n = _mm_sub_epi32(_mm_cvttps_epi32(t), MAGIC_BIAS);
      t = _mm_sub_ps(t, m);

      __m128 r = _mm_sub_ps(logits_vec, _mm_mul_ps(t, LN2_HI));
      r = _mm_sub_ps(r, _mm_mul_ps(t, LN2_LO));

      __m128 r2 = _mm_mul_ps(r, r);
      __m128 q2 = _mm_add_ps(c2, _mm_mul_ps(c3, r));
      __m128 q4 = _mm_add_ps(c4, _mm_mul_ps(c5, r));
      __m128 p = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(q4, r2), q2), r2), _mm_add_ps(r, one_vec));

      __m128i expbits = _mm_add_epi32(n, c127);
      __m128i bits = _mm_slli_epi32(expbits, 23);
      __m128 two_n = _mm_castsi128_ps(bits);

      __m128 result_vec = _mm_mul_ps(p, two_n);
      _mm_store_ps(&probs[i], result_vec);
      expsum_vec1 = _mm_add_ps(expsum_vec1, result_vec);
    }

    // Second block of 4
    {
      __m128 logits_vec = _mm_load_ps(&logits[i + SIMDW]);
      logits_vec = _mm_sub_ps(logits_vec, maxlogit_vec);
      logits_vec = _mm_min_ps(logits_vec, cplus87);
      logits_vec = _mm_max_ps(logits_vec, cminus87);

      __m128 z = _mm_mul_ps(logits_vec, INV_LN2);
      const __m128 m = _mm_cvtepi32_ps(MAGIC_BIAS);
      __m128 t = _mm_add_ps(z, m);
      __m128i n = _mm_sub_epi32(_mm_cvttps_epi32(t), MAGIC_BIAS);
      t = _mm_sub_ps(t, m);

      __m128 r = _mm_sub_ps(logits_vec, _mm_mul_ps(t, LN2_HI));
      r = _mm_sub_ps(r, _mm_mul_ps(t, LN2_LO));

      __m128 r2 = _mm_mul_ps(r, r);
      __m128 q2 = _mm_add_ps(c2, _mm_mul_ps(c3, r));
      __m128 q4 = _mm_add_ps(c4, _mm_mul_ps(c5, r));
      __m128 p = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(q4, r2), q2), r2), _mm_add_ps(r, one_vec));

      __m128i expbits = _mm_add_epi32(n, c127);
      __m128i bits = _mm_slli_epi32(expbits, 23);
      __m128 two_n = _mm_castsi128_ps(bits);

      __m128 result_vec = _mm_mul_ps(p, two_n);
      _mm_store_ps(&probs[i + SIMDW], result_vec);
      expsum_vec2 = _mm_add_ps(expsum_vec2, result_vec);
    }
  }

  // Horizontal sum
  __m128 sum128 = _mm_add_ps(expsum_vec1, expsum_vec2);
  sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
  sum128 = _mm_add_ss(sum128, _mm_shuffle_ps(sum128, sum128, 0x55));
  float expsum = _mm_cvtss_f32(sum128);

  // Handle remainder
  for (size_t i = limit; i < len; i++) {
    float val = expf_compat(logits[i] - max_logit);
    probs[i] = val;
    expsum += val;
  }

  // Normalize
  float expsum_reciprocal = 1.0f / expsum;
  __m128 expsum_reciprocal_vec = _mm_set1_ps(expsum_reciprocal);

  for (size_t i = 0; i < limit; i += SIMDW) {
    __m128 softmax_probs_vec = _mm_load_ps(&probs[i]);
    __m128 result_vec = _mm_mul_ps(softmax_probs_vec, expsum_reciprocal_vec);
    _mm_store_ps(&probs[i], result_vec);
  }

  for (size_t i = limit; i < len; i++)
    probs[i] *= expsum_reciprocal;
}

// ============================================================================
// Activation Function Implementations
// ============================================================================

// Tanh
Tanh::Tanh(SIMDType simdType) : simd(simdType) {}

#if (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx")))
#endif
void Tanh::RunSimdAVX(float* f, size_t const len) const {
  static constexpr size_t SIMDW = 8;

  __m256 const c_27 = _mm256_set1_ps(27.0f);
  __m256 const c_9 = _mm256_set1_ps(9.0f);
  __m256 const c_clip_lower = _mm256_set1_ps(TANH_CLIP_MIN);
  __m256 const c_clip_upper = _mm256_set1_ps(TANH_CLIP_MAX);

  size_t const limit = len & static_cast<size_t>(-static_cast<ptrdiff_t>(SIMDW));

  for (size_t i = 0; i < limit; i += SIMDW) {
    __m256 x = _mm256_loadu_ps(f + i);
    x = _mm256_max_ps(x, c_clip_lower);
    x = _mm256_min_ps(x, c_clip_upper);

    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 numer = _mm256_mul_ps(x, _mm256_add_ps(c_27, x2));
    __m256 denom = _mm256_add_ps(c_27, _mm256_mul_ps(c_9, x2));
    __m256 result = _mm256_div_ps(numer, denom);

    _mm256_storeu_ps(f + i, result);
  }

  for (size_t i = limit; i < len; i++) {
    float x = f[i];
    f[i] = tanh_pade_clipped(x);
  }
}

void Tanh::Run(float* f, size_t const len) const {
  if (simd == SIMDType::SIMD_AVX2 || simd == SIMDType::SIMD_AVX512) {
#ifdef X64_SIMD_AVAILABLE
    RunSimdAVX(f, len);
#endif
  }
  else {
    for (size_t i = 0; i < len; i++) {
      float x = f[i];
      f[i] = tanh_pade_clipped(x);
    }
  }
}

// Logistic (Sigmoid)
Logistic::Logistic(SIMDType simdType) : simd(simdType) {}

#if (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx")))
#endif
void Logistic::RunSimdAVX(float* f, size_t const len) const {
  static constexpr size_t SIMDW = 8;

  __m256 const c_half = _mm256_set1_ps(0.5f);
  __m256 const c_27 = _mm256_set1_ps(27.0f);
  __m256 const c_9 = _mm256_set1_ps(9.0f);
  __m256 const c_clip_lower = _mm256_set1_ps(SIGMOID_CLIP_MIN);
  __m256 const c_clip_upper = _mm256_set1_ps(SIGMOID_CLIP_MAX);

  size_t const limit = len & static_cast<size_t>(-static_cast<ptrdiff_t>(SIMDW));

  for (size_t i = 0; i < limit; i += SIMDW) {
    __m256 x = _mm256_loadu_ps(f + i);
    x = _mm256_max_ps(x, c_clip_lower);
    x = _mm256_min_ps(x, c_clip_upper);

    __m256 u = _mm256_mul_ps(x, c_half);
    __m256 u2 = _mm256_mul_ps(u, u);

    __m256 numer = _mm256_mul_ps(u, _mm256_add_ps(c_27, u2));
    __m256 denom = _mm256_add_ps(c_27, _mm256_mul_ps(c_9, u2));
    __m256 tanh_val = _mm256_div_ps(numer, denom);

    // sigmoid = 0.5 * (1 + tanh) = 0.5 + 0.5*tanh
    __m256 result = _mm256_add_ps(c_half, _mm256_mul_ps(c_half, tanh_val));

    _mm256_storeu_ps(f + i, result);
  }

  for (size_t i = limit; i < len; i++) {
    float x = f[i];
    f[i] = sigmoid_pade_clipped(x);
  }
}

void Logistic::Run(float* f, size_t const len) const {
  if (simd == SIMDType::SIMD_AVX2 || simd == SIMDType::SIMD_AVX512) {
#ifdef X64_SIMD_AVAILABLE
    RunSimdAVX(f, len);
#endif
  }
  else {
    for (size_t i = 0; i < len; i++) {
      float x = f[i];
      f[i] = sigmoid_pade_clipped(x);
    }
  }
}

#endif // X64_SIMD_AVAILABLE

// ============================================================================
// Scalar Softmax
// ============================================================================

static float horizontal_sum(float x0, float x1, float x2, float x3, float x4, float x5, float x6, float x7)
{
  // Simulate loading of __m256 as an array of 8 floats
  float sum0 = x0 + x4; // Pair 0
  float sum1 = x1 + x5; // Pair 1
  float sum2 = x2 + x6; // Pair 2
  float sum3 = x3 + x7; // Pair 3

  // Combine pairs
  sum0 = sum0 + sum2;
  sum1 = sum1 + sum3;

  // Final horizontal sum
  sum0 = sum0 + sum1;

  return sum0;
}

void softmax_scalar(float* logits, float* probs, size_t const len, float const max_logit) {
  float expsum[8] = { 0.0f };

  // Compute exp(logit - max_logit) in blocks of 8
  size_t i = 0;
  for (; i + 8 <= len; i += 8) {
    for (size_t j = 0; j < 8; j++) {
      float val = expf_compat(logits[i + j] - max_logit);
      probs[i + j] = val;
      expsum[j] += val;
    }
  }

  float expsum_total = horizontal_sum(expsum[0], expsum[1], expsum[2], expsum[3], expsum[4], expsum[5], expsum[6], expsum[7]);

  // Handle remainder
  for (; i < len; i++) {
    float val = expf_compat(logits[i] - max_logit);
    probs[i] = val;
    expsum_total += val;
  }

  float expsum_reciprocal = 1.0f / expsum_total;

  for (i = 0; i < len; i++)
    probs[i] *= expsum_reciprocal;
}

// ============================================================================
// Non-SIMD Vector Functions
// ============================================================================

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
