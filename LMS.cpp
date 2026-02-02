#include "LMS.hpp"
#include <cassert>
#include <cmath>
#include <cstring>

LMS::LMS(const int s, const int d,
  const float sameChannelRate,
  const float otherChannelRate) :
  sameChannelRate(sameChannelRate),
  otherChannelRate(otherChannelRate),
  rho(1.0f - 1.0f / 20.0f),
  eps(0.001f),
  prediction(0.0f),
  s(s),
  d(d) {

  assert(s > 0 && d > 0);

  weights = new float[s + d];
  eg = new float[s + d];
  buffer = new float[s + d];

  reset();
}

LMS::~LMS() {
  delete[] weights;
  delete[] eg;
  delete[] buffer;
}

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
