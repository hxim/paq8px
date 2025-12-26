#pragma once

#include "Adam.hpp"

class Adam_Scalar : public Adam
{
public:
  Adam_Scalar(size_t length, float* w, float* g, float beta2Value, float epsilon) :
    Adam(length, w, g, beta2Value, epsilon)
  {
  }

  virtual void Optimize(float learning_rate, uint64_t time_step) override;
};
