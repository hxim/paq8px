#include <cmath>
#include "Adam_SSE2.hpp"

#ifdef X64_SIMD_AVAILABLE

#pragma GCC target("sse2")

void Adam_SSE2::Optimize(float lr_scale, uint64_t training_iterations)
{
  __m128 const zero_vec = _mm_setzero_ps();
  __m128 const vec_beta2 = _mm_set1_ps(beta2);
  __m128 const vec_eps = _mm_set1_ps(eps);
  __m128 const vec_beta2_complement = _mm_set1_ps(1.f - beta2);

  double const t = static_cast<double>(training_iterations);
  float const bias_v = 1.f - static_cast<float>(std::pow(beta2, t));
  __m128 const vec_bias_v = _mm_set1_ps(bias_v);
  __m128 const vec_lr = _mm_set1_ps(base_lr * lr_scale);

  for (size_t i = 0; i < length; i += 4) {
    __m128 vec_gi = _mm_load_ps(&g[i]);
    __m128 vec_vi = _mm_load_ps(&v[i]);

    // v = beta2 * v + (1 - beta2) * g^2
    vec_vi = _mm_mul_ps(vec_vi, vec_beta2);
    __m128 vec_gi_sq = _mm_mul_ps(vec_gi, vec_gi);
    __m128 vec_term = _mm_mul_ps(vec_gi_sq, vec_beta2_complement);
    vec_vi = _mm_add_ps(vec_vi, vec_term);
    _mm_store_ps(&v[i], vec_vi);

    // scaled_gradient = g / (sqrt(v / bias_v) + eps)
    __m128 vec_v_corrected = _mm_div_ps(vec_vi, vec_bias_v);
    __m128 vec_sqrt = _mm_sqrt_ps(vec_v_corrected);
    __m128 vec_denom = _mm_add_ps(vec_sqrt, vec_eps);
    __m128 vec_scaled_grad = _mm_div_ps(vec_gi, vec_denom);
    _mm_store_ps(&g[i], zero_vec);

    // w = w - lr * scaled_gradient
    __m128 vec_wi = _mm_load_ps(&w[i]);
    __m128 vec_update = _mm_mul_ps(vec_lr, vec_scaled_grad);
    vec_wi = _mm_sub_ps(vec_wi, vec_update);
    _mm_store_ps(&w[i], vec_wi);
  }
}

#endif
