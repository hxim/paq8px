#include <cmath>
#include "Adam_Scalar.hpp"

void Adam_Scalar::Optimize(float learning_rate, uint64_t time_step)
{
  float const t = static_cast<float>(time_step);
  float const bias_v = 1.f - std::pow(beta2, t);

  for (size_t i = 0; i < length; i++) {
    float g_val = g[i];
    float v_old = v[i];
    float g_sq = g_val * g_val;
    float v_new = v_old * beta2 + (1.0f - beta2) * g_sq;
    v[i] = v_new;
    float v_corrected = v_new / bias_v;
    float denom = std::sqrt(v_corrected) + eps;
    float scaled_gradient = g_val / denom;
    w[i] = w[i] - learning_rate * scaled_gradient;
  }
}
