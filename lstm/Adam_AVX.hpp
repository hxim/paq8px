#pragma once

#include "Adam.hpp"

#ifdef X64_SIMD_AVAILABLE

class Adam_AVX : public Adam
{
public:
  Adam_AVX(size_t length, float* w, float* g, float beta2Value, float epsilon) :
    Adam(length, w, g, beta2Value, epsilon)
  {
  }

  virtual void Optimize(float learning_rate, uint64_t training_iterations) override;
};
#endif
