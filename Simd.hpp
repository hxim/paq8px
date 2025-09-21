#pragma once

///////////////////////// SIMD Vectorization detection //////////////////////////////////

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define HAS_CPUID 1
#endif

#ifdef HAS_CPUID
#if defined(_MSC_VER)
#include <intrin.h>
#define cpuid(info, x) __cpuidex(info, x, 0)
#elif defined(__GNUC__) || defined(__clang__)
#include <x86intrin.h>
#include <cpuid.h>
#define cpuid(info, x) __cpuid_count(x, 0, (info)[0], (info)[1], (info)[2], (info)[3])
#endif
#endif

// Define interface to xgetbv instruction
#ifdef HAS_CPUID
static inline unsigned long long xgetbv(unsigned long ctr) {
#if (defined(_MSC_FULL_VER) && _MSC_FULL_VER >= 160040000) || (defined(__INTEL_COMPILER) && __INTEL_COMPILER >= 1200)
  return _xgetbv(ctr);
#elif (defined(__GNUC__) || defined(__clang__))
  uint32_t a = 0;
  uint32_t d;
  __asm("xgetbv"
  : "=a"(a), "=d"(d)
  : "c"(ctr)
  :);
  return a | ((static_cast<uint64_t>(d)) << 32);
#else
#error Unknown compiler
#endif
}
#endif

/* Returns system's highest supported SIMD instruction set as
0: none (or unknown)
1: MMX
2: SSE
3: SSE2
4: SSE3
5: SSSE3
6: SSE4.1
7: SSE4.2
 : SSE4A //SSE4A is not supported on Intel, so we will exclude it
8: AVX
9: AVX2
10: AVX512 //AVX512F + AVX512BW
11: NEON
*/
static int simdDetect() {
#ifdef HAS_CPUID
  int cpuidResult[4] = {0, 0, 0, 0};
  cpuid(cpuidResult, 0); // call cpuid function 0 ("Get vendor ID and highest basic calling parameter")
  if( cpuidResult[0] == 0 ) {
    return 0; //cpuid is not supported
  }
  cpuid(cpuidResult, 1); // call cpuid function 1 ("Processor Info and Feature Bits")
  if((cpuidResult[3] & (1 << 23)) == 0 ) {
    return 0; //no MMX
  }
  if((cpuidResult[3] & (1 << 25)) == 0 ) {
    return 1; //no SSE
  }
  //SSE: OK
  if((cpuidResult[3] & (1 << 26)) == 0 ) {
    return 2; //no SSE2
  }
  //SSE2: OK
  if((cpuidResult[2] & (1 << 0)) == 0 ) {
    return 3; //no SSE3
  }
  //SSE3: OK
  if((cpuidResult[2] & (1 << 9)) == 0 ) {
    return 4; //no SSSE3
  }
  //SSSE3: OK
  if((cpuidResult[2] & (1 << 19)) == 0 ) {
    return 5; //no SSE4.1
  }
  //SSE4.1: OK
  if((cpuidResult[2] & (1 << 20)) == 0 ) {
    return 6; //no SSE4.2
  }
  //SSE4.2: OK
  if((cpuidResult[2] & (1 << 27)) == 0 ) {
    return 7; //no OSXSAVE (no XGETBV)
  }
  if((xgetbv(0) & 6) != 6 ) {
    return 7; //AVX is not enabled in OS
  }
  if((cpuidResult[2] & (1 << 28)) == 0 ) {
    return 7; //no AVX
  }
  //AVX: OK
  cpuid(cpuidResult, 7); // call cpuid function 7 ("Extended Feature Bits")
  if((cpuidResult[1] & (1 << 5)) == 0 ) {
    return 8; //no AVX2
  }
  //AVX2: OK
  if((cpuidResult[1] & (1 << 16)) == 0 || (cpuidResult[1] & (1 << 30)) == 0 ) {
    return 9; //no AVX512F + AVX512BW
  }
  //AVX512: OK
  return 10;
#elif defined(ARM_NEON_AVAILABLE)
  return 11;
#else
  return 0; //unknown system - use non-accelerated codepaths
#endif
}
