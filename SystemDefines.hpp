#pragma once

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

#if defined(__ARM_NEON)
#define ARM_NEON_AVAILABLE
#include <arm_neon.h>
constexpr bool IS_ARM_NEON_AVAILABLE = true;
#else
constexpr bool IS_ARM_NEON_AVAILABLE = false;
#endif

#if defined(__x86_64__) || defined(_M_X64)
#define X64_SIMD_AVAILABLE
#include <immintrin.h>
constexpr bool IS_X64_SIMD_AVAILABLE = true;
#else
constexpr bool IS_X64_SIMD_AVAILABLE = false;
#endif


// Floating point operations need IEEE compliance
// Do not use unsafe compiler optimization options
// gcc : -ffast-math (and -Ofast, -funsafe-math-optimizations, -fno-rounding-math) => __FAST_MATH__
// MSVC: /fp:fast => _M_FP_FAST
#if defined(__FAST_MATH__) || defined(_M_FP_FAST)
#error Avoid using aggressive floating-point compiler optimization flags
#endif

// MSVC: Verify we're in a safe floating-point mode
#if defined(_MSC_VER)
#if !defined(_M_FP_PRECISE) && !defined(_M_FP_STRICT)
#warning MSVC floating-point mode unclear. Ensure /fp:precise or /fp:strict is set
#endif
#endif


#if defined(_MSC_VER)
#define ALWAYS_INLINE  __forceinline
#elif (defined(__GNUC__) || defined(__clang__))
#define ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define ALWAYS_INLINE inline
#endif

#include <cassert>

#if defined(__clang__)
#define ASSUME(cond) do { assert(cond); __builtin_assume(cond); } while (0)
#elif defined(_MSC_VER)
#define ASSUME(cond) do { assert(cond); __assume(cond); } while (0)
#elif defined(__GNUC__)
#define ASSUME(cond) do { assert(cond); if (!(cond)) __builtin_unreachable(); } while (0)
#else
#define ASSUME(cond) assert(cond)
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
+(((x) << 40) & 0x00FF000000000000) | \
+ (((x) << 24) & 0x0000FF0000000000) | \
+ (((x) << 8) & 0x000000FF00000000) | \
+ (((x) >> 8) & 0x00000000FF000000) | \
+ (((x) >> 24) & 0x0000000000FF0000) | \
+ (((x) >> 40) & 0x000000000000FF00) | \
+ ((x) << 56))
#endif
