#pragma once

#include <cstdint>
#include <valarray>

class IOptimizer {
public:
  virtual ~IOptimizer() = default;
  virtual void Run(
    std::valarray<float>* g,
    std::valarray<float>* m,
    std::valarray<float>* v,
    std::valarray<float>* w,
    float const learning_rate,
    std::uint64_t const time_step) const = 0;
};
