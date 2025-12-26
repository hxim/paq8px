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
  float beta2;
  float eps;

public:
  Adam(size_t length, float* w, float* g, float beta2, float epsilon);
  ~Adam() = default;

  virtual void Optimize(float learning_rate, uint64_t time_step) = 0;
};
