#include "LMS_AVX.hpp"

#ifdef X64_SIMD_AVAILABLE

#include <immintrin.h> // AVX/AVX2

#if (defined(__GNUC__) || defined(__clang__))
#define AVX_TARGET __attribute__((target("avx")))
#else
#define AVX_TARGET
#endif

// Static helper functions

AVX_TARGET
static float horizontal_sum(__m256 sum_vec) {
  __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
  __m128 sum_low = _mm256_castps256_ps128(sum_vec);
  __m128 sum128 = _mm_add_ps(sum_low, sum_high);
  sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
  sum128 = _mm_add_ss(sum128, _mm_shuffle_ps(sum128, sum128, 0x55));
  float sum = _mm_cvtss_f32(sum128);
  return sum;
}

// Member implementations

AVX_TARGET
float LMS_AVX::predict(const int sample) {
  // Shift other-channel history (the 'd' component)
  memmove(&buffer[s + 1], &buffer[s], (d - 1) * sizeof(float));
  buffer[s] = static_cast<float>(sample);

  // Compute weighted prediction
  __m256 sum = _mm256_setzero_ps();
  const size_t len = s + d;

  for (size_t i = 0; i < len; i += 8) {
    __m256 w = _mm256_load_ps(&weights[i]);
    __m256 b = _mm256_load_ps(&buffer[i]);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(w, b));
  }

  float total = horizontal_sum(sum);
  prediction = total;

  return total;
}

AVX_TARGET
void LMS_AVX::update(const int sample) {
  const float error = static_cast<float>(sample) - prediction;
  const float complement = 1.0f - rho;

  const __m256 v_error = _mm256_set1_ps(error);
  const __m256 v_rho = _mm256_set1_ps(rho);
  const __m256 v_complement = _mm256_set1_ps(complement);
  const __m256 v_eps = _mm256_set1_ps(eps);
  const __m256 v_sameRate = _mm256_set1_ps(sameChannelRate);
  const __m256 v_otherRate = _mm256_set1_ps(otherChannelRate);

  // Update same-channel weights (indices 0 to s-1)
  for (size_t i = 0; i < s; i += 8) {
    __m256 b = _mm256_load_ps(&buffer[i]);
    __m256 w = _mm256_load_ps(&weights[i]);
    __m256 e = _mm256_load_ps(&eg[i]);

    __m256 gradient = _mm256_mul_ps(v_error, b);
    __m256 grad_sq = _mm256_mul_ps(gradient, gradient);
    e = _mm256_add_ps(_mm256_mul_ps(v_rho, e), _mm256_mul_ps(v_complement, grad_sq));
    _mm256_store_ps(&eg[i], e);

    __m256 denom = _mm256_sqrt_ps(_mm256_add_ps(e, v_eps));
    w = _mm256_add_ps(w, _mm256_mul_ps(v_sameRate, _mm256_div_ps(gradient, denom)));
    _mm256_store_ps(&weights[i], w);
  }

  // Update other-channel weights (indices s to s+d-1)
  for (size_t i = s; i < s + d; i += 8) {
    __m256 b = _mm256_load_ps(&buffer[i]);
    __m256 w = _mm256_load_ps(&weights[i]);
    __m256 e = _mm256_load_ps(&eg[i]);

    __m256 gradient = _mm256_mul_ps(v_error, b);
    __m256 grad_sq = _mm256_mul_ps(gradient, gradient);
    e = _mm256_add_ps(_mm256_mul_ps(v_rho, e), _mm256_mul_ps(v_complement, grad_sq));
    _mm256_store_ps(&eg[i], e);

    __m256 denom = _mm256_sqrt_ps(_mm256_add_ps(e, v_eps));
    w = _mm256_add_ps(w, _mm256_mul_ps(v_otherRate, _mm256_div_ps(gradient, denom)));
    _mm256_store_ps(&weights[i], w);
  }

  // Shift same-channel history
  memmove(&buffer[1], &buffer[0], (s - 1) * sizeof(float));
  buffer[0] = static_cast<float>(sample);
}

#endif
