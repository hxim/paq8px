#include "LMS_Scalar.hpp"

// Static helper functions

static float horizontal_sum(float x0, float x1, float x2, float x3, float x4, float x5, float x6, float x7) {
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

// Member implementations

float LMS_Scalar::predict(const int sample) {
  // Shift other-channel history (the 'd' component)
  memmove(&buffer[s + 1], &buffer[s], (d - 1) * sizeof(float));
  buffer[s] = static_cast<float>(sample);

  // Compute weighted prediction - process 8 floats per iteration
  float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
  float sum4 = 0.0f, sum5 = 0.0f, sum6 = 0.0f, sum7 = 0.0f;
  const size_t len = s + d;

  for (size_t i = 0; i < len; i += 8) {
    sum0 += weights[i + 0] * buffer[i + 0];
    sum1 += weights[i + 1] * buffer[i + 1];
    sum2 += weights[i + 2] * buffer[i + 2];
    sum3 += weights[i + 3] * buffer[i + 3];
    sum4 += weights[i + 4] * buffer[i + 4];
    sum5 += weights[i + 5] * buffer[i + 5];
    sum6 += weights[i + 6] * buffer[i + 6];
    sum7 += weights[i + 7] * buffer[i + 7];
  }

  float total = horizontal_sum(sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7);
  prediction = total;

  return total;
}

void LMS_Scalar::update(const int sample) {
  const float error = static_cast<float>(sample) - prediction;
  const float complement = 1.0f - rho;

  // Update same-channel weights (indices 0 to s-1)
  for (size_t i = 0; i < s; i += 8) {
    const float gradient = error * buffer[i];
    eg[i] = rho * eg[i] + complement * (gradient * gradient);
    weights[i] += sameChannelRate * gradient / std::sqrt(eg[i] + eps);
  }

  // Update other-channel weights (indices s to s+d-1)
  for (size_t i = s; i < s + d; i += 4) {
    const float gradient = error * buffer[i];
    eg[i] = rho * eg[i] + complement * (gradient * gradient);
    weights[i] += otherChannelRate * gradient / std::sqrt(eg[i] + eps);
  }

  // Shift same-channel history
  memmove(&buffer[1], &buffer[0], (s - 1) * sizeof(float));
  buffer[0] = static_cast<float>(sample);
}
