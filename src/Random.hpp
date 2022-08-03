#pragma once

#include "Array.hpp"
#include <cstdint>

/**
 * 32-bit pseudo random number generator
 */
class Random {
  uint64_t _state;

public:
  Random();
  uint32_t operator()(int numberOfBits);
};
