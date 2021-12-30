#pragma once

#include "IOptimizer.hpp"
#include "../Utils.hpp"
#include "../simd.hpp"
#include <cmath>
//#define USE_RSQRT

template <SIMDType simd, std::uint16_t B1, std::uint8_t E1, std::uint16_t B2, std::uint8_t E2, std::uint16_t C, std::uint8_t E3>
class Adam :
  public IOptimizer {
private:
  static constexpr float beta1 = static_cast<float>(static_cast<double>(B1) * neg_pow10<E1>::value);
  static constexpr float beta2 = static_cast<float>(static_cast<double>(B2) * neg_pow10<E2>::value);
  static constexpr float eps = static_cast<float>(static_cast<double>(C) * neg_pow10<E3>::value);
#if (defined(__GNUC__) || defined(__clang__)) && (!defined(__ARM_FEATURE_SIMD32) && !defined(__ARM_NEON))
  __attribute__((target("avx2,fma")))
#endif
  void RunSimdAVX2(
    std::valarray<float>* g,
    std::valarray<float>* m,
    std::valarray<float>* v,
    std::valarray<float>* w,
    float const learning_rate,
    std::uint64_t const time_step) const 
  {
#if !defined(__i386__) && !defined(__x86_64__) && !defined(_M_X64)
    return;
#else
    static constexpr std::size_t SIMDW = 8;
    static __m256 const vec_beta1 = _mm256_set1_ps(beta1);
    static __m256 const vec_beta2 = _mm256_set1_ps(beta2);
    static __m256 const vec_eps = _mm256_set1_ps(eps);
    static __m256 const vec_beta1_complement = _mm256_set1_ps(1.f - beta1);
    static __m256 const vec_beta2_complement = _mm256_set1_ps(1.f - beta2);
# ifdef USE_RSQRT
    static __m256 const vec_three = _mm256_set1_ps(3.f);
    static __m256 const vec_half = _mm256_set1_ps(0.5f);
# endif

    double const t = static_cast<double>(time_step);
    float const bias_m = 1.f - static_cast<float>(std::pow(beta1, t));
    float const bias_v = 1.f - static_cast<float>(std::pow(beta2, t));
    __m256 const vec_bias_m = _mm256_set1_ps(bias_m);
    __m256 const vec_bias_v = _mm256_set1_ps(bias_v);
    __m256 const vec_lr = _mm256_set1_ps(learning_rate);
    std::size_t const len = g->size();
    std::size_t const limit = len & static_cast<std::size_t>(-static_cast<std::ptrdiff_t>(SIMDW));
    std::size_t remainder = len & (SIMDW - 1);

    for (std::size_t i = 0; i < limit; i += SIMDW) {
      __m256 vec_gi = _mm256_loadu_ps(&(*g)[i]);
      __m256 vec_mi = _mm256_fmadd_ps(
        vec_gi,
        vec_beta1_complement,
        _mm256_mul_ps(
          _mm256_loadu_ps(&(*m)[i]),
          vec_beta1
        )
      );
      vec_gi = _mm256_mul_ps(vec_gi, vec_gi);
      _mm256_storeu_ps(&(*m)[i], vec_mi);
      __m256 vec_vi = _mm256_mul_ps(
        _mm256_loadu_ps(&(*v)[i]),
        vec_beta2
      );
      vec_mi = _mm256_div_ps(vec_mi, vec_bias_m);
      vec_vi = _mm256_fmadd_ps(vec_gi, vec_beta2_complement, vec_vi);
      __m256 vec_wi = _mm256_loadu_ps(&(*w)[i]);
      __m256 vec_n = _mm256_add_ps(_mm256_div_ps(vec_vi, vec_bias_v), vec_eps);
# ifdef USE_RSQRT
      __m256 vec_rsqrt = _mm256_rsqrt_ps(vec_n);
      _mm256_storeu_ps(&(*v)[i], vec_vi);
      // one Newton-Raphson iteration to improve precision
      vec_rsqrt = _mm256_mul_ps(
        _mm256_mul_ps(
          vec_half,
          vec_rsqrt
        ),
        _mm256_fnmadd_ps(
          _mm256_mul_ps(
            vec_n,
            vec_rsqrt
          ),
          vec_rsqrt,
          vec_three
        )
      );
      _mm256_storeu_ps(
        &(*w)[i],
        _mm256_fnmadd_ps(
          vec_lr,
          _mm256_mul_ps(
            vec_mi,
            vec_rsqrt
          ),
          vec_wi
        )
      );
# else
      _mm256_storeu_ps(&(*v)[i], vec_vi);
      _mm256_storeu_ps(
        &(*w)[i],
        _mm256_fnmadd_ps(
          vec_lr,
          _mm256_div_ps(
            vec_mi,
            _mm256_sqrt_ps(vec_n)
          ),
          vec_wi
        )
      );
# endif
    }
    for (; remainder > 0; remainder--) {
      const std::size_t i = len - remainder;
      (*m)[i] = (*m)[i] * beta1 + (1.f - beta1) * (*g)[i];
      (*v)[i] = (*v)[i] * beta2 + (1.f - beta2) * (*g)[i] * (*g)[i];
      (*w)[i] -= learning_rate * (((*m)[i] / bias_m) / (std::sqrt((*v)[i] / bias_v) + eps));
    }
#endif
  };
  void RunSimdNone(
    std::valarray<float>* g,
    std::valarray<float>* m,
    std::valarray<float>* v,
    std::valarray<float>* w,
    float const learning_rate,
    std::uint64_t const time_step) const
  {
    float const t = static_cast<float>(time_step);
    float const bias_m = 1.f - std::pow(beta1, t);
    float const bias_v = 1.f - std::pow(beta2, t);
    for (int i = 0; i < g->size(); i++) {
      (*m)[i] = (*m)[i] * beta1 + (1.0f - beta1) * (*g)[i];
      (*v)[i] = (*v)[i] * beta2 + (1.0f - beta2) * (*g)[i] * (*g)[i];
      (*w)[i] -= learning_rate * (
        ((*m)[i] / bias_m) /
        (std::sqrt((*v)[i] / bias_v) + eps)
      );
    }
  }
public:
  void Run(
    std::valarray<float>* g,
    std::valarray<float>* m,
    std::valarray<float>* v,
    std::valarray<float>* w,
    float const learning_rate,
    std::uint64_t const time_step) const
  {
    if constexpr (simd == SIMDType::SIMD_AVX2)
      RunSimdAVX2(g, m, v, w, learning_rate, time_step);
    else
      RunSimdNone(g, m, v, w, learning_rate, time_step);
  }
};
