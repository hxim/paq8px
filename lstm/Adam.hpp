#pragma once

#include "../Array.hpp"
#include "../Utils.hpp"
#include <cstdint>

class Adam {
protected:
  size_t length;
  float* w;
  float* g;
  Array<float, 32> v;
  float base_lr;
  float beta2;
  float eps;

public:
  Adam(size_t length, float* w, float* g, float base_lr, float beta2, float epsilon);
  ~Adam() = default;

  virtual void Optimize(float learning_rate, uint64_t training_iterations) = 0;

  virtual void Rescale(float scale) = 0;
};
