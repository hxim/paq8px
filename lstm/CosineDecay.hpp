#pragma once

#include "IDecay.hpp"
#include "../Utils.hpp"

#define _USE_MATH_DEFINES
#include <cmath>

template <std::uint16_t LR, std::uint8_t E1, std::uint8_t Alpha, std::uint64_t Steps>
class CosineDecay :
  public IDecay {
  static_assert((Alpha > 0) && (Alpha < 100), "Cosine decay alpha must be in [1..99]");
  static_assert(Steps > 0, "Cosine decay steps must be > 0");
private:
  static constexpr float alpha = static_cast<float>(Alpha) / 100.f;
  static constexpr double mul = 1. / static_cast<double>(Steps);
  static constexpr float learning_rate = static_cast<float>(static_cast<double>(LR) * neg_pow10<E1>::value);
public:
  ALWAYS_INLINE void Apply(float& rate, std::uint64_t const time_step) const {
    rate = learning_rate * ((1.f - alpha) * (0.5f * (1.f + static_cast<float>(std::cos(std::min<std::uint64_t>(time_step, Steps) * mul * M_PI)))) + alpha);
  }
};

#undef _USE_MATH_DEFINES
