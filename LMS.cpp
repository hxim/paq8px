#include <cassert>
#include <cmath>
#include <cstring>

#include "LMS.hpp"
#include "LMS_Scalar.hpp"
#include "LMS_SSE2.hpp"
#include "LMS_AVX.hpp"

LMS::LMS(const int s, const int d,
  const float sameChannelRate,
  const float otherChannelRate) :
  weights(s + d),
  eg(s + d),
  buffer(s + d),
  sameChannelRate(sameChannelRate),
  otherChannelRate(otherChannelRate),
  rho(1.0f - 1.0f / 20.0f),
  eps(0.001f),
  prediction(0.0f),
  s(s),
  d(d)
{
  assert(s > 0 && d > 0 && (s & 7) == 0 && (d & 7) == 0);
}


std::unique_ptr<LMS> LMS::create(
  SIMDType simd,
  int s,
  int d,
  float sameChannelRate,
  float otherChannelRate
) {
#ifdef X64_SIMD_AVAILABLE
  if (simd == SIMDType::SIMD_AVX2 || simd == SIMDType::SIMD_AVX512)
    return std::make_unique<LMS_AVX>(s, d, sameChannelRate, otherChannelRate);
  else if (simd == SIMDType::SIMD_SSE2)
    return std::make_unique<LMS_SSE2>(s, d, sameChannelRate, otherChannelRate);
  else
#endif
    return std::make_unique<LMS_Scalar>(s, d, sameChannelRate, otherChannelRate);
}

// Reference implementation - not in use
float LMS::predict(const int sample) {
  // Shift other-channel history (the 'd' component)
  memmove(&buffer[s + 1], &buffer[s], (d - 1) * sizeof(float));
  buffer[s] = static_cast<float>(sample);

  // Compute weighted prediction
  prediction = 0.0f;
  for (int i = 0; i < s + d; i++) {
    prediction += weights[i] * buffer[i];
  }

  return prediction;
}

// Reference implementation - not in use
void LMS::update(const int sample) {
  const float error = static_cast<float>(sample) - prediction;
  float complement = 1.0f - rho;
  // Update same-channel weights (indices 0 to s-1)
  int i = 0;
  for (; i < s; i++) {
    const float gradient = error * buffer[i];
    eg[i] = rho * eg[i] + complement * (gradient * gradient);
    weights[i] += sameChannelRate * gradient / std::sqrt(eg[i] + eps);
  }

  // Update other-channel weights (indices s to s+d-1)
  for (; i < s + d; i++) {
    const float gradient = error * buffer[i];
    eg[i] = rho * eg[i] + complement * (gradient * gradient);
    weights[i] += otherChannelRate * gradient / std::sqrt(eg[i] + eps);
  }

  // Shift same-channel history
  memmove(&buffer[1], &buffer[0], (s - 1) * sizeof(float));
  buffer[0] = static_cast<float>(sample);
}

void LMS::reset() {
  for (int i = 0; i < s + d; i++) {
    weights[i] = 0.0f;
    eg[i] = 0.0f;
    buffer[i] = 0.0f;
  }
}
