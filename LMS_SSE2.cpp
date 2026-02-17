#include "LMS_SSE2.hpp"

#ifdef X64_SIMD_AVAILABLE

#if (defined(__GNUC__) || defined(__clang__))
#pragma GCC target("sse2")
#endif

// Static helper functions

static float horizontal_sum(__m128 sum_low, __m128 sum_high) {
  __m128 sum128 = _mm_add_ps(sum_low, sum_high);
  sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
  sum128 = _mm_add_ss(sum128, _mm_shuffle_ps(sum128, sum128, 0x55));
  float sum = _mm_cvtss_f32(sum128);
  return sum;
}

// Member implementations

float LMS_SSE2::predict(const int sample) {
  // Shift other-channel history (the 'd' component)
  memmove(&buffer[s + 1], &buffer[s], (d - 1) * sizeof(float));
  buffer[s] = static_cast<float>(sample);

  // Compute weighted prediction
  __m128 sum_low = _mm_setzero_ps();
  __m128 sum_high = _mm_setzero_ps();
  const size_t len = s + d;

  for (size_t i = 0; i < len; i += 8) {
    __m128 w0 = _mm_load_ps(&weights[i]);
    __m128 b0 = _mm_load_ps(&buffer[i]);
    __m128 prod0 = _mm_mul_ps(w0, b0);
    sum_low = _mm_add_ps(sum_low, prod0);

    __m128 w1 = _mm_load_ps(&weights[i + 4]);
    __m128 b1 = _mm_load_ps(&buffer[i + 4]);
    __m128 prod1 = _mm_mul_ps(w1, b1);
    sum_high = _mm_add_ps(sum_high, prod1);
  }

  float total = horizontal_sum(sum_low, sum_high);
  prediction = total;

  return total;
}

void LMS_SSE2::update(const int sample) {
  const float error = static_cast<float>(sample) - prediction;
  const float complement = 1.0f - rho;

  const __m128 v_error = _mm_set1_ps(error);
  const __m128 v_rho = _mm_set1_ps(rho);
  const __m128 v_complement = _mm_set1_ps(complement);
  const __m128 v_eps = _mm_set1_ps(eps);
  const __m128 v_sameRate = _mm_set1_ps(sameChannelRate);
  const __m128 v_otherRate = _mm_set1_ps(otherChannelRate);

  // Update same-channel weights (indices 0 to s-1)
  for (size_t i = 0; i < s; i += 8) {
    // First 4 floats
    __m128 b0 = _mm_load_ps(&buffer[i]);
    __m128 w0 = _mm_load_ps(&weights[i]);
    __m128 e0 = _mm_load_ps(&eg[i]);

    __m128 gradient0 = _mm_mul_ps(v_error, b0);
    __m128 grad_sq0 = _mm_mul_ps(gradient0, gradient0);
    e0 = _mm_add_ps(_mm_mul_ps(v_rho, e0), _mm_mul_ps(v_complement, grad_sq0));
    _mm_store_ps(&eg[i], e0);

    __m128 denom0 = _mm_sqrt_ps(_mm_add_ps(e0, v_eps));
    w0 = _mm_add_ps(w0, _mm_mul_ps(v_sameRate, _mm_div_ps(gradient0, denom0)));
    _mm_store_ps(&weights[i], w0);

    // Second 4 floats
    __m128 b1 = _mm_load_ps(&buffer[i + 4]);
    __m128 w1 = _mm_load_ps(&weights[i + 4]);
    __m128 e1 = _mm_load_ps(&eg[i + 4]);

    __m128 gradient1 = _mm_mul_ps(v_error, b1);
    __m128 grad_sq1 = _mm_mul_ps(gradient1, gradient1);
    e1 = _mm_add_ps(_mm_mul_ps(v_rho, e1), _mm_mul_ps(v_complement, grad_sq1));
    _mm_store_ps(&eg[i + 4], e1);

    __m128 denom1 = _mm_sqrt_ps(_mm_add_ps(e1, v_eps));
    w1 = _mm_add_ps(w1, _mm_mul_ps(v_sameRate, _mm_div_ps(gradient1, denom1)));
    _mm_store_ps(&weights[i + 4], w1);
  }

  // Update other-channel weights (indices s to s+d-1)
  for (size_t i = s; i < s + d; i += 8) {
    // First 4 floats
    __m128 b0 = _mm_load_ps(&buffer[i]);
    __m128 w0 = _mm_load_ps(&weights[i]);
    __m128 e0 = _mm_load_ps(&eg[i]);

    __m128 gradient0 = _mm_mul_ps(v_error, b0);
    __m128 grad_sq0 = _mm_mul_ps(gradient0, gradient0);
    e0 = _mm_add_ps(_mm_mul_ps(v_rho, e0), _mm_mul_ps(v_complement, grad_sq0));
    _mm_store_ps(&eg[i], e0);

    __m128 denom0 = _mm_sqrt_ps(_mm_add_ps(e0, v_eps));
    w0 = _mm_add_ps(w0, _mm_mul_ps(v_otherRate, _mm_div_ps(gradient0, denom0)));
    _mm_store_ps(&weights[i], w0);

    // Second 4 floats
    __m128 b1 = _mm_load_ps(&buffer[i + 4]);
    __m128 w1 = _mm_load_ps(&weights[i + 4]);
    __m128 e1 = _mm_load_ps(&eg[i + 4]);

    __m128 gradient1 = _mm_mul_ps(v_error, b1);
    __m128 grad_sq1 = _mm_mul_ps(gradient1, gradient1);
    e1 = _mm_add_ps(_mm_mul_ps(v_rho, e1), _mm_mul_ps(v_complement, grad_sq1));
    _mm_store_ps(&eg[i + 4], e1);

    __m128 denom1 = _mm_sqrt_ps(_mm_add_ps(e1, v_eps));
    w1 = _mm_add_ps(w1, _mm_mul_ps(v_otherRate, _mm_div_ps(gradient1, denom1)));
    _mm_store_ps(&weights[i + 4], w1);
  }

  // Shift same-channel history
  memmove(&buffer[1], &buffer[0], (s - 1) * sizeof(float));
  buffer[0] = static_cast<float>(sample);
}

#endif
