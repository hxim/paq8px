#include "VectorFunctions.hpp"

#include <numeric>
#include <cassert>
#include <cstring>

// ============================================================================
// Padé Approximants for Activation Functions
// ============================================================================

// Tanh Padé approximant: tanh(x) ≈ x(27 + x²) / (27 + 9x²)
static float constexpr tanh_pade(float x) {
  float x2 = x * x;
  return (x * (27.0f + x2)) / (27.0f + 9.0f * x2);
}

// Sigmoid via tanh: σ(x) = 0.5 * (1 + tanh(x/2))
static float constexpr sigmoid_pade(float x) {
  float u = 0.5f * x;
  float u2 = u * u;
  float tanh_val = (u * (27.0f + u2)) / (27.0f + 9.0f * u2);
  return 0.5f * (1.0f + tanh_val);
}

static constexpr float SIGMOID_MIN = sigmoid_pade(SIGMOID_CLIP_MIN);
static constexpr float SIGMOID_MAX = sigmoid_pade(SIGMOID_CLIP_MAX);
static constexpr float TANH_MIN = tanh_pade(TANH_CLIP_MIN);
static constexpr float TANH_MAX = tanh_pade(TANH_CLIP_MAX);

static_assert(SIGMOID_MIN > 0.0f);
static_assert(SIGMOID_MAX < 1.0f);
static_assert(TANH_MIN > -1.0f);
static_assert(TANH_MAX < +1.0f);

float tanh_pade_clipped(float x) {
  if (x > TANH_CLIP_MAX) return TANH_MAX;
  if (x < TANH_CLIP_MIN) return TANH_MIN;
  return tanh_pade(x);
}

// Sigmoid via tanh: σ(x) = 0.5 * (1 + tanh(x/2))
float sigmoid_pade_clipped(float x) {
  if (x > SIGMOID_CLIP_MAX) return SIGMOID_MAX;
  if (x < SIGMOID_CLIP_MIN) return SIGMOID_MIN;
  return sigmoid_pade(x);
}

