#include "Random.hpp"
#include "Hash.hpp"

Random::Random() {
  _state = 0;
}

// This pseudo random number generator is a 
// Mixed Congruential Generator with a period of 2^64
// https://en.wikipedia.org/wiki/Linear_congruential_generator

auto Random::operator()(int numberOfBits) -> uint32_t {
  _state = (_state + 1) * PHI64;
  return _state >> (64 - numberOfBits);
}
