#include <cmath>
#include "Adam_Scalar.hpp"

void Adam_Scalar::Optimize(float lr_rate, float beta2)
{
  float const lr = base_lr * lr_rate;
  float eps = 1e-6;

  for (size_t i = 0; i < length; i++) {
    float g_val = g[i];
    float v_old = v[i];
    float g_sq = g_val * g_val;
    float v_new = v_old * beta2 + (1.0f - beta2) * g_sq;
    v[i] = v_new;
    float denom = std::sqrt(v_new) + eps;
    float scaled_gradient = g_val / denom;
    g[i] = 0.0f;
    w[i] = w[i] - lr * scaled_gradient;
  }
}

void Adam_Scalar::Rescale(float scale)
{
  for (size_t i = 0; i < length; i++) {
    v[i] = v[i] * scale;
  }
}
