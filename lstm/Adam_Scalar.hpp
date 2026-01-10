#pragma once

#include "Adam.hpp"

class Adam_Scalar : public Adam
{
public:
  Adam_Scalar(size_t length, float* w, float* g, float base_lr) :
    Adam(length, w, g, base_lr)
  {
  }

  virtual void Optimize(float learning_rate, float beta2) override;

  virtual void Rescale(float scale) override;
};
