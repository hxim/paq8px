#pragma once

#include <cassert>
#include <cstdint>

#ifdef _MSC_VER
#include <intrin.h>
static inline uint32_t clz(uint32_t x) {
  assert(x != 0);
  unsigned long leading_zero;
  _BitScanReverse(&leading_zero, x);
  return 31u - leading_zero;
}
#elif __GNUC__
static inline uint32_t clz(uint32_t x) {
  assert(x != 0);
  return __builtin_clz(x);
}
#else
static inline uint32_t popcnt(uint32_t x) {
  x -= ((x >> 1) & 0x55555555);
  x = (((x >> 2) & 0x33333333) + (x & 0x33333333));
  x = (((x >> 4) + x) & 0x0f0f0f0f);
  x += (x >> 8);
  x += (x >> 16);
  return x & 0x0000003f;
}
static inline uint32_t clz(uint32_t x) {
  assert(x != 0);
  x |= (x >> 1);
  x |= (x >> 2);
  x |= (x >> 4);
  x |= (x >> 8);
  x |= (x >> 16);
  return 32u - popcnt(x);
}
#endif
