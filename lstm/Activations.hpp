#ifndef PAQ8PX_ACTIVATIONS_HPP
#define PAQ8PX_ACTIVATIONS_HPP

#include "IActivation.hpp"
#include "../utils.hpp"
#include "../SIMDType.hpp"
#include "SimdFunctions.hpp"
#include <cmath>

template <SIMDType simd>
class Tanh :
  public IActivation {
private:
#if (defined(__GNUC__) || defined(__clang__)) && (!defined(__ARM_FEATURE_SIMD32) && !defined(__ARM_NEON))
  __attribute__((target("avx2,fma")))
#endif
  void RunSimdAVX2(float* f, std::size_t const len) const {
#if !defined(__i386__) && !defined(__x86_64__) && !defined(_M_X64)
    return;
#else
    static constexpr std::size_t SIMDW = 8u;
    __m256 const c1 = _mm256_set1_ps(0.03138777f);
    __m256 const c2 = _mm256_set1_ps(0.276281267f);
    __m256 const c_log2f = _mm256_set1_ps(1.442695022f);
    std::size_t const limit = len & static_cast<std::size_t>(-static_cast<std::ptrdiff_t>(SIMDW));
    std::size_t remainder = len & (SIMDW - 1u);
    for (std::size_t i = 0u; i < limit; i += SIMDW) {
      _mm_prefetch((char*)(f + i + SIMDW), _MM_HINT_T0);
      __m256 v = _mm256_mul_ps(_mm256_loadu_ps(f + i), c_log2f);      
      __m256 x = _mm256_sub_ps(v, _mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      __m256 xx = _mm256_mul_ps(x, x);
      __m256 v1 = _mm256_fmadd_ps(c2, xx, c_log2f);
      __m256 v2 = _mm256_fmadd_ps(xx, _mm256_mul_ps(c1, x), x);
      __m256 v3 = _mm256_castsi256_ps(
        _mm256_add_epi32(
          _mm256_castps_si256(
            _mm256_add_ps(v2, v1)
          ),
          _mm256_slli_epi32(_mm256_cvtps_epi32(v), 24)
        )
      );
      __m256 v4 = _mm256_sub_ps(v2, v1);
      _mm256_storeu_ps(
        f + i,
        _mm256_div_ps(
          _mm256_add_ps(v3, v4),
          _mm256_sub_ps(v3, v4)
        )
      );
    }
    for (; remainder > 0u; remainder--)
      f[len - remainder] = tanha(f[len - remainder]);
#endif
  }
public:
  void Run(float* f, std::size_t const len) const {
    if (simd == SIMD_AVX2)
      RunSimdAVX2(f, len);
    else {
      for (std::size_t i = 0u; i < len; i++)
        f[i] = tanha(f[i]);
    }
  }
};

template <SIMDType simd>
class Logistic :
  public IActivation {
private:
#if (defined(__GNUC__) || defined(__clang__)) && (!defined(__ARM_FEATURE_SIMD32) && !defined(__ARM_NEON))
  __attribute__((target("avx2,fma")))
#endif
    void RunSimdAVX2(float* f, std::size_t const len) const {
#if !defined(__i386__) && !defined(__x86_64__) && !defined(_M_X64)
    return;
#else
    static constexpr std::size_t SIMDW = 8u;
    static __m256 const c1 = _mm256_set1_ps(0.03138777f);
    static __m256 const c2 = _mm256_set1_ps(0.276281267f);
    static __m256 const c_log2f = _mm256_set1_ps(1.442695022f);
    static __m256 const c_log2f_2 = _mm256_set1_ps(0.721347511f);
    static __m256 const vec_half = _mm256_set1_ps(0.5f);
    std::size_t const limit = len & static_cast<std::size_t>(-static_cast<std::ptrdiff_t>(SIMDW));
    std::size_t remainder = len & (SIMDW - 1u);
    for (std::size_t i = 0u; i < limit; i += SIMDW) {
      _mm_prefetch((char*)(f + i + SIMDW), _MM_HINT_T0);
      __m256 v = _mm256_mul_ps(_mm256_loadu_ps(f + i), c_log2f_2);      
      __m256 x = _mm256_sub_ps(v, _mm256_round_ps(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
      __m256 xx = _mm256_mul_ps(x, x);
      __m256 v1 = _mm256_fmadd_ps(c2, xx, c_log2f);
      __m256 v2 = _mm256_fmadd_ps(xx, _mm256_mul_ps(c1, x), x);
      __m256 v3 = _mm256_castsi256_ps(
        _mm256_add_epi32(
          _mm256_castps_si256(
            _mm256_add_ps(v2, v1)
          ),
          _mm256_slli_epi32(_mm256_cvtps_epi32(v), 24)
        )
      );
      __m256 v4 = _mm256_sub_ps(v2, v1);
      _mm256_storeu_ps(
        f + i,
        _mm256_add_ps(
          _mm256_div_ps(
            _mm256_mul_ps(
              _mm256_add_ps(v3, v4),
              vec_half
            ),
            _mm256_sub_ps(v3, v4)
          ),
          vec_half
        )
      );
    }
    for (; remainder > 0u; remainder--)
      f[len - remainder] = (tanha(f[len - remainder] * 0.5f) + 1.0f) * 0.5f;
#endif
  }
public:
  void Run(float* f, std::size_t const len) const {
    if (simd == SIMD_AVX2)
      RunSimdAVX2(f, len);
    else {
      for (std::size_t i = 0u; i < len; i++)
        f[i] = (tanha(f[i] * 0.5f) + 1.0f) * 0.5f;
    }
  }
};

#endif //PAQ8PX_ACTIVATIONS_HPP
