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
#elif __GNUC__
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

template <std::uint8_t e>
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

inline auto clip(int const px) -> uint8_t {
  if( px > 255 ) {
    return 255;
  }
  if( px < 0 ) {
    return 0;
  }
  return px;
}

inline auto clamp4(const int px, const uint8_t n1, const uint8_t n2, const uint8_t n3, const uint8_t n4) -> uint8_t {
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




//distance (absolute difference) between two 8-bit pixel values, quantized
//used by 8-bit and 24-bit image models
inline auto DiffQt(const uint8_t a, const uint8_t b, const uint8_t limit = 7) -> uint8_t {
  //assert(limit <= 7);
  uint32_t d = abs(a - b);
  if (d <= 2)d = d;
  else if (d <= 5)d = 3; //3..5
  else if (d <= 9)d = 4; //6..9
  else if (d <= 14)d = 5; //10..14
  else if (d <= 23)d = 6; //15..23
  else d = 7; //24..255
  const uint8_t sign = a > b ? 8 : 0;
  return sign | min(d, limit);
}

inline auto logQt(const uint8_t px, const uint8_t bits) -> uint32_t {
  return (uint32_t(0x100 | px)) >> max(0, static_cast<int>(ilog2(px) - bits));
}
