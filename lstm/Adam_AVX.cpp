#include <cmath>
#include "Adam_AVX.hpp"

#ifdef X64_SIMD_AVAILABLE

#pragma GCC target("avx")

void Adam_AVX::Optimize(float learning_rate, uint64_t time_step)
{
  __m256 const zero_vec = _mm256_setzero_ps();
  __m256 const vec_beta2 = _mm256_set1_ps(beta2);
  __m256 const vec_eps = _mm256_set1_ps(eps);
  __m256 const vec_beta2_complement = _mm256_set1_ps(1.f - beta2);

  double const t = static_cast<double>(time_step);
  float const bias_v = 1.f - static_cast<float>(std::pow(beta2, t));
  __m256 const vec_bias_v = _mm256_set1_ps(bias_v);
  __m256 const vec_lr = _mm256_set1_ps(learning_rate);

  for (size_t i = 0; i < length; i += 8) {
    __m256 vec_gi = _mm256_load_ps(&g[i]);
    __m256 vec_vi = _mm256_load_ps(&v[i]);
    
    // v = beta2 * v + (1 - beta2) * g^2
    vec_vi = _mm256_mul_ps(vec_vi, vec_beta2);
    __m256 vec_gi_sq = _mm256_mul_ps(vec_gi, vec_gi);
    __m256 vec_term = _mm256_mul_ps(vec_gi_sq, vec_beta2_complement);
    vec_vi = _mm256_add_ps(vec_vi, vec_term);
    _mm256_store_ps(&v[i], vec_vi);

    // scaled_gradient = g / (sqrt(v / bias_v) + eps)
    __m256 vec_v_corrected = _mm256_div_ps(vec_vi, vec_bias_v);
    __m256 vec_sqrt = _mm256_sqrt_ps(vec_v_corrected);
    __m256 vec_denom = _mm256_add_ps(vec_sqrt, vec_eps);
    __m256 vec_scaled_grad = _mm256_div_ps(vec_gi, vec_denom);
    _mm256_store_ps(&g[i], zero_vec);

    // w = w - lr * scaled_gradient
    __m256 vec_wi = _mm256_load_ps(&w[i]);
    __m256 vec_update = _mm256_mul_ps(vec_lr, vec_scaled_grad);
    vec_wi = _mm256_sub_ps(vec_wi, vec_update);
    _mm256_store_ps(&w[i], vec_wi);
  }
}

#endif
