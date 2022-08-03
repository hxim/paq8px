#pragma once

#include <cassert>
#include <cstdint>
#include <cstdio>

#include "cstdint"
#if defined(__i386__) || defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>
#elif defined(__ARM_FEATURE_SIMD32) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

/**
 * Least Mean Squares predictor
 * @tparam F
 * @tparam T
 */
template<typename F, typename T>
class LMS {
private:
  F *weights;
  F *eg;
  F *buffer;
  F rates[2];
  F rho;
  F complement;
  F eps;
  F prediction;
  int s;
  int d;

public:
  /**
    *
    * @param s
    * @param d
    * @param lRate
    * @param rRate
    * @param rho
    * @param eps
    */
  LMS(const int s, const int d, const F lRate, const F rRate, const F rho = (F) 0.95, const F eps = (F) 1e-3) : rates {lRate, rRate},
          rho(rho), complement(1.0f - rho), eps(eps), prediction(0.0f), s(s), d(d) {
    assert(s > 0 && d > 0);
    weights = new F[s + d], eg = new F[s + d], buffer = new F[s + d];
    reset();
  }

  /**
    *
    */
  ~LMS() {
    delete weights, delete eg, delete buffer;
  }

  float rsqrt(const float x) {
#if defined(__ARM_FEATURE_SIMD32) || defined(__ARM_NEON)
    float r = vgetq_lane_f32(vrsqrteq_f32(vdupq_n_f32(x)), 0); //NEON
#else
    float r = _mm_cvtss_f32(_mm_rsqrt_ss(_mm_set_ss(x))); //SSE
#endif
    return (0.5F * (r + 1.0F / (x * r)));
  }

  /**
    *
    * @param sample
    * @return
    */
  F predict(const T sample) {
    memmove(&buffer[s + 1], &buffer[s], (d - 1) * sizeof(F));
    buffer[s] = sample;
    prediction = 0.;
    for( int i = 0; i < s + d; i++ ) {
      prediction += weights[i] * buffer[i];
    }
    return prediction;
  }

  /**
    *
    * @param sample
    */
  void update(const T sample) {
    const F error = sample - prediction;
    int i = 0;
    for( ; i < s; i++ ) {
      const F gradient = error * buffer[i];
      eg[i] = rho * eg[i] + complement * (gradient * gradient);
      weights[i] += (rates[0] * gradient * rsqrt(eg[i] + eps));
    }
    for( ; i < s + d; i++ ) {
      const F gradient = error * buffer[i];
      eg[i] = rho * eg[i] + complement * (gradient * gradient);
      weights[i] += (rates[1] * gradient * rsqrt(eg[i] + eps));
    }
    memmove(&buffer[1], &buffer[0], (s - 1) * sizeof(F));
    buffer[0] = sample;
  }

  /**
    *
    */
  void reset() {
    for( int i = 0; i < s + d; i++ ) {
      weights[i] = eg[i] = buffer[i] = 0.;
    }
  }
};
