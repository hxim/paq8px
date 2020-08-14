#ifndef PAQ8PX_UTILS_HPP
#define PAQ8PX_UTILS_HPP

#include <cstdint>
#include "BitCount.hpp"

static_assert(sizeof(short) == 2, "sizeof(short)");
static_assert(sizeof(int) == 4, "sizeof(int)");

//////////////////////// Target OS/Compiler ////////////////////////////////

#if defined(_WIN32) || defined(_MSC_VER)
#ifndef WINDOWS
#define WINDOWS  //to compile for Windows
#endif
#endif

#if defined(unix) || defined(__unix__) || defined(__unix) || defined(__APPLE__)
#ifndef UNIX
#define UNIX //to compile for Unix, Linux, Solaris, MacOS / Darwin, etc)
#endif
#endif

#if !defined(WINDOWS) && !defined(UNIX)
#error Unknown target system
#endif

// Floating point operations need IEEE compliance
// Do not use compiler optimization options such as the following:
// gcc : -ffast-math (and -Ofast, -funsafe-math-optimizations, -fno-rounding-math)
// vc++: /fp:fast
#if defined(__FAST_MATH__) || defined(_M_FP_FAST) // gcc vc++
#error Avoid using aggressive floating-point compiler optimization flags
#endif

#if defined(_MSC_VER)
#define ALWAYS_INLINE  __forceinline
#elif defined(__GNUC__)
#define ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define ALWAYS_INLINE inline
#endif

#include <algorithm>
#include <cstdio>
// Determining the proper printf() format specifier for 64 bit unsigned integers:
// - on Windows MSVC and MinGW-w64 use the MSVCRT runtime where it is "%I64u"
// - on Linux it is "%llu"
// The correct value is provided by the PRIu64 macro which is defined here:
#include <cinttypes> //PRIu64

// Platform-specific includes
#ifdef UNIX

#include <cerrno>  //errno
#include <climits> //PATH_MAX (for OSX)
#include <cstring> //strlen(), strcpy(), strcat(), strerror(), memset(), memcpy(), memmove()
#include <unistd.h> //isatty()

#else
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>   //CreateDirectoryW, CommandLineToArgvW, GetConsoleOutputCP, SetConsoleOutputCP
//GetCommandLineW, GetModuleFileNameW, GetStdHandle, GetTempFileName
//MultiByteToWideChar, WideCharToMultiByte,
//FileType, FILE_TYPE_PIPE, FILE_TYPE_DISK,
//uRetVal, DWORD, UINT, TRUE, MAX_PATH, CP_UTF8, etc.
#endif

#define DEFAULT_LEARNING_RATE 7

typedef enum {
  SIMD_NONE, SIMD_SSE2, SIMD_SSSE3, SIMD_AVX2, SIMD_NEON
} SIMD;

struct ErrorInfo {
  uint32_t data[2], sum, mask, collected;

  void reset() {
    memset(this, 0, sizeof(*this));
  }
};

static inline auto square(uint32_t x) -> uint32_t {
  return x * x;
}

static inline auto min(int a, int b) -> int { return std::min<int>(a, b); }

static inline auto min(uint64_t a, uint64_t b) -> uint64_t { return std::min<uint64_t>(a, b); }

static inline auto max(int a, int b) -> int { return std::max<int>(a, b); }

template<typename T>
constexpr auto isPowerOf2(T x) -> bool {
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

typedef enum {
  DEFAULT = 0,
  FILECONTAINER,
  JPEG,
  HDR,
  IMAGE1,
  IMAGE4,
  IMAGE8,
  IMAGE8GRAY,
  IMAGE24,
  IMAGE32,
  AUDIO,
  AUDIO_LE,
  EXE,
  CD,
  ZLIB,
  BASE64,
  GIF,
  PNG8,
  PNG8GRAY,
  PNG24,
  PNG32,
  TEXT,
  TEXT_EOL,
  RLE,
  LZW,
  DEC_ALPHA
} BlockType;

static inline auto hasRecursion(BlockType ft) -> bool {
  return ft == CD || ft == ZLIB || ft == BASE64 || ft == GIF || ft == RLE || ft == LZW || ft == FILECONTAINER;
}

static inline auto hasInfo(BlockType ft) -> bool {
  return ft == IMAGE1 || ft == IMAGE4 || ft == IMAGE8 || ft == IMAGE8GRAY || ft == IMAGE24 || ft == IMAGE32 || ft == AUDIO ||
         ft == AUDIO_LE || ft == PNG8 || ft == PNG8GRAY || ft == PNG24 || ft == PNG32;
}

static inline auto hasTransform(BlockType ft) -> bool {
  return ft == IMAGE24 || ft == IMAGE32 || ft == AUDIO_LE || ft == EXE || ft == CD || ft == ZLIB || ft == BASE64 || ft == GIF ||
         ft == TEXT_EOL || ft == RLE || ft == LZW || ft == DEC_ALPHA;
}

static inline auto isPNG(BlockType ft) -> bool { return ft == PNG8 || ft == PNG8GRAY || ft == PNG24 || ft == PNG32; }

#define OPTION_MULTIPLE_FILE_MODE 1U
#define OPTION_BRUTE 2U
#define OPTION_TRAINEXE 4U
#define OPTION_TRAINTXT 8U
#define OPTION_ADAPTIVE 16U
#define OPTION_SKIPRGB 32U
#define OPTION_LSTM 64U
#define OPTION_LSTM_TRAINING 128U

//////////////////// Cross-platform definitions /////////////////////////////////////

#ifdef _MSC_VER
#define fseeko(a, b, c) _fseeki64(a, b, c)
#define ftello(a) _ftelli64(a)
#else
#ifndef UNIX
#ifndef fseeko
#define fseeko(a, b, c) fseeko64(a, b, c)
#endif
#ifndef ftello
#define ftello(a) ftello64(a)
#endif
#endif
#endif

#ifdef WINDOWS
#define strcasecmp _stricmp
#endif

#if defined(__GNUC__) || defined(__clang__)
#define bswap(x) __builtin_bswap32(x)
#define bswap64(x) __builtin_bswap64(x)
#elif defined(_MSC_VER)
#define bswap(x) _byteswap_ulong(x)
#define bswap64(x) _byteswap_uint64(x)
#else
#define bswap(x) \
  +(((( x ) &0xff000000) >> 24) | +((( x ) &0x00ff0000) >> 8) | +((( x ) &0x0000ff00) << 8) | +((( x ) &0x000000ff) << 24))
#define bswap64(x) \
  +((x) >> 56) |
+   (((x)<<40) & 0x00FF000000000000) | \
+   (((x)<<24) & 0x0000FF0000000000) | \
+   (((x)<<8 ) & 0x000000FF00000000) | \
+   (((x)>>8 ) & 0x00000000FF000000) | \
+   (((x)>>24) & 0x0000000000FF0000) | \
+   (((x)>>40) & 0x000000000000FF00) | \
+   ((x) << 56))
#endif


#define TAB 0x09
#define NEW_LINE 0x0A
#define CARRIAGE_RETURN 0x0D
#define SPACE 0x20
#define QUOTE 0x22
#define APOSTROPHE 0x27


static inline auto clip(int const px) -> uint8_t {
  if( px > 255 ) {
    return 255;
  }
  if( px < 0 ) {
    return 0;
  }
  return px;
}

static inline auto clamp4(const int px, const uint8_t n1, const uint8_t n2, const uint8_t n3, const uint8_t n4) -> uint8_t {
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


/**
 * Returns floor(log2(x)).
 * 0/1->0, 2->1, 3->1, 4->2 ..., 30->4,  31->4, 32->5,  33->5
 * @param x
 * @return floor(log2(x))
 */
static auto ilog2(uint32_t x) -> uint32_t {
#ifdef _MSC_VER
  DWORD tmp = 0;
  if (x != 0) {
    _BitScanReverse(&tmp, x);
  }
  return tmp;
#elif __GNUC__
  if( x != 0 ) {
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

static inline auto logMeanDiffQt(const uint8_t a, const uint8_t b, const uint8_t limit = 7) -> uint8_t {
  if( a == b ) {
    return 0;
  }
  uint8_t sign = a > b ? 8 : 0;
  return sign | min(limit, ilog2((a + b) / max(2, abs(a - b) * 2) + 1));
}

static inline auto logQt(const uint8_t px, const uint8_t bits) -> uint32_t {
  return (uint32_t(0x100 | px)) >> max(0, static_cast<int>(ilog2(px) - bits));
}

#ifdef _MSC_VER
#include <intrin.h>
inline uint32_t clz(uint32_t x) {
  DWORD leading_zero;
  if (x != 0u) {
    _BitScanReverse(&leading_zero, x);
    return 31u - leading_zero;
  }
  return 32u;
}
#elif __GNUC__
inline uint32_t clz(uint32_t x) {
  if (x != 0u)
    return __builtin_clz(x);
  return 32u;
}
#else
inline uint32_t popcnt(uint32_t x) {
  x -= ((x >> 1) & 0x55555555);
  x = (((x >> 2) & 0x33333333) + (x & 0x33333333));
  x = (((x >> 4) + x) & 0x0f0f0f0f);
  x += (x >> 8);
  x += (x >> 16);
  return x & 0x0000003f;
}
inline uint32_t clz(uint32_t x) {
  x |= (x >> 1);
  x |= (x >> 2);
  x |= (x >> 4);
  x |= (x >> 8);
  x |= (x >> 16);
  return 32u - popcnt(x);
}
#endif

#endif //PAQ8PX_UTILS_HPP
