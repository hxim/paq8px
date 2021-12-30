#pragma once

#include "IDecay.hpp"
#include "../Utils.hpp"
#include <cmath>

template <std::uint16_t LR, std::uint8_t E1, std::uint16_t ELR, std::uint8_t E2, std::uint16_t DECAY, std::uint8_t DECAY_E, std::uint8_t N, std::uint8_t D, std::uint64_t Steps = 0>
class PolynomialDecay :
  public IDecay {
  static_assert(D > 0, "Polynomial decay denominator must be > 0");
private:
  static constexpr float power = static_cast<float>(N) / static_cast<float>(D);
  static constexpr float mul = (Steps > 0) ? 1.0f / Steps : 0.0f;
  static constexpr float learning_rate = LR * neg_pow10<E1>::value;
  static constexpr float end_learning_rate = ELR * neg_pow10<E2>::value;
  static constexpr float decay = DECAY * neg_pow10<DECAY_E>::value;
public:
  ALWAYS_INLINE void Apply(float& rate, std::uint64_t const time_step) const {
    if (Steps > 0) {
      if (time_step < Steps)
        rate = (learning_rate - end_learning_rate) * (std::pow((1.0f - time_step * mul), power)) + end_learning_rate;
      else
        rate = end_learning_rate / std::pow(decay * (time_step - Steps) + 1.0f, power);
    }
    else
      rate = std::max<float>(learning_rate / std::pow(decay * time_step + 1.0f, power), end_learning_rate);
  }
};
