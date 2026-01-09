#pragma once

#include "Adam.hpp"

class Adam_Scalar : public Adam
{
public:
  Adam_Scalar(size_t length, float* w, float* g, float base_lr, float beta2Value, float epsilon) :
    Adam(length, w, g, base_lr, beta2Value, epsilon)
  {
  }

  virtual void Optimize(float learning_rate, uint64_t training_iterations) override;

  virtual void Rescale(float scale) override;
};
