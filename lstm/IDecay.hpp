#pragma once

#include <cstdint>

class IDecay {
public:
  virtual ~IDecay() = default;
  virtual void Apply(float& rate, std::uint64_t const time_step) const = 0;
};
