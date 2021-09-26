#include <cstdint>
#include "BitCount.hpp"

uint32_t bitCount(uint32_t v) {
  v -= ((v >> 1U) & 0x55555555U);
  v = ((v >> 2U) & 0x33333333U) + (v & 0x33333333U);
  v = ((v >> 4U) + v) & 0x0f0f0f0fU;
  v = ((v >> 8U) + v) & 0x00ff00ffU;
  v = ((v >> 16U) + v) & 0x0000ffffU;
  return v;
}

