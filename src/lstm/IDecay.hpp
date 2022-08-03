#pragma once

#include <cstdint>

class IDecay {
public:
  virtual ~IDecay() = default;
  virtual void Apply(float& rate, uint64_t const time_step) const = 0;
};
