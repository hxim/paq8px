#pragma once

#include <cstdint>
#include <cassert>
#include "SystemDefines.hpp"



ALWAYS_INLINE
int max(int a, int b) { return std::max<int>(a, b); }

ALWAYS_INLINE
int min(int a, int b) { return std::min<int>(a, b); }

ALWAYS_INLINE
uint64_t min(uint64_t a, uint64_t b) { return std::min<uint64_t>(a, b); }

ALWAYS_INLINE
uint32_t square(uint32_t x) {
  return x * x;
}
/**
 * Returns floor(log2(x)).
 * 0/1->0, 2->1, 3->1, 4->2 ..., 30->4,  31->4, 32->5,  33->5
 * @param x
 * @return floor(log2(x))
 */
ALWAYS_INLINE
uint32_t ilog2(uint32_t x) {
#ifdef _MSC_VER
  DWORD tmp = 0;
  if (x != 0) {
    _BitScanReverse(&tmp, x);
  }
  return tmp;
#elif (defined(__GNUC__) || defined(__clang__))
  if (x != 0) {
    x = 31 - __builtin_clz(x);
  }
  return x;
#else
  //copy the leading "1" bit to its left (0x03000000 -> 0x03ffffff)
  x |= (x >> 1);
  x |= (x >> 2);
  x |= (x >> 4);
  x |= (x >> 8);
  x |= (x >> 16);
  //how many trailing bits do we have (except the first)?
  return BitCount(x >> 1);
#endif
}


template<typename T>
ALWAYS_INLINE
constexpr bool isPowerOf2(T x) {
  return ((x & (x - 1)) == 0);
}

ALWAYS_INLINE
uint32_t nextPowerOf2(uint32_t x)
{
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x++;
  return x;
}

template <uint8_t e>
struct neg_pow10 {
  static constexpr float value = neg_pow10<e - 1>::value / 10.0f;
};
template <>
struct neg_pow10<0> {
  static constexpr float value = 1.0f;
};

#ifndef NDEBUG
#if defined(UNIX)
#include <execinfo.h>
#define BACKTRACE() \
  { \
    void *callstack[128]; \
    int frames  = backtrace(callstack, 128); \
    char **strs = backtrace_symbols(callstack, frames); \
    for (int i = 0; i < frames; ++i) { \
      printf("%s\n", strs[i]); \
    } \
    free(strs); \
  }
#else
// TODO: How to implement this on Windows?
#define BACKTRACE() \
  { }
#endif
#else
#define BACKTRACE()
#endif

// A basic exception class to let catch() in main() know
// that the exception was thrown intentionally.
class IntentionalException : public std::exception {};

// Error handler: print message if any, and exit
[[noreturn]] static void quit(const char *const message = nullptr) {
  if( message != nullptr ) {
    printf("\n%s", message);
  }
  printf("\n");
  throw IntentionalException();
}

inline uint8_t clip(int const px) {
  if( px > 255 ) {
    return 255;
  }
  if( px < 0 ) {
    return 0;
  }
  return px;
}

inline uint8_t clamp4(const int px, const uint8_t n1, const uint8_t n2, const uint8_t n3, const uint8_t n4) {
  int maximum = n1;
  if( maximum < n2 ) {
    maximum = n2;
  }
  if( maximum < n3 ) {
    maximum = n3;
  }
  if( maximum < n4 ) {
    maximum = n4;
  }
  int minimum = n1;
  if( minimum > n2 ) {
    minimum = n2;
  }
  if( minimum > n3 ) {
    minimum = n3;
  }
  if( minimum > n4 ) {
    minimum = n4;
  }
  if( px < minimum ) {
    return minimum;
  }
  if( px > maximum ) {
    return maximum;
  }
  return px;
}

// Circular/wraparound distance on a 256-value ring
// that is: what is the shortest distance between two byte values in any direction (up or down) when we allow a wrap-around at 255 → 0?
//
// Examples:
//  - Distance between 3 and 10 (or 10 and 3) is 7
//  - Distance between 2 and 200 (or 200 and 2) is 58
// For the latter the true linear distance is 198, but the wraparound distance is only 58
// why: starting from 2 down to 0 then to 255 then further down to 200 takes 58 steps
//
// Where such wrap-around distances are useful:
// For image compression (pixel prediction):
//   images with color-channel transform (b, g, r) -> (g, g-r, g-b): the 2nd and 3rd transformed components experience a wrap-around.
// For multi-byte numeric data prediction:
//   less significant bytes wrap around when the whole multi-byte value is increased/decreased sequentially.
ALWAYS_INLINE static int rabs(int x1, int x2) {
  int d = int8_t(x1 - x2); // -128..127
  return d >= 0 ? d : -d; // abs(d) → 0..128
}
