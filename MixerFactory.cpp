#include "MixerFactory.hpp"
#include "SIMDType.hpp"

auto MixerFactory::createMixer(const Shared* const sh, const int n, const int m, const int s) -> Mixer * {
  if( sh->chosenSimd == SIMDType::SIMD_NONE ) {
    return new SIMDMixer<SIMDType::SIMD_NONE>(sh, n, m, s);
  }
  else if( sh->chosenSimd == SIMDType::SIMD_SSE2 ) {
    return new SIMDMixer<SIMDType::SIMD_SSE2>(sh, n, m, s);
  }
  else if (sh->chosenSimd == SIMDType::SIMD_SSSE3) {
    return new SIMDMixer<SIMDType::SIMD_SSSE3>(sh, n, m, s);
  }
  else if( sh->chosenSimd == SIMDType::SIMD_AVX2 ) {
    return new SIMDMixer<SIMDType::SIMD_AVX2>(sh, n, m, s);
  }
  else if (sh->chosenSimd == SIMDType::SIMD_NEON) {
    return new SIMDMixer<SIMDType::SIMD_NEON>(sh, n, m, s);
  }
  assert(false);
  return nullptr;
}
