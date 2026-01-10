#pragma once

#include "Adam.hpp"

#ifdef X64_SIMD_AVAILABLE

class Adam_AVX : public Adam
{
public:
  Adam_AVX(size_t length, float* w, float* g, float base_lr) :
    Adam(length, w, g, base_lr)
  {
  }

  virtual void Optimize(float learning_rate, float beta2) override;

  virtual void Rescale(float scale) override;
};
#endif
