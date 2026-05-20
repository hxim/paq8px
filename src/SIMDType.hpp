#pragma once

enum class SIMDType {
  SIMD_NONE, 
  SIMD_SSE2, 
  SIMD_SSE3,
  SIMD_SSE41,
  SIMD_AVX2,
  SIMD_AVX512, 
  SIMD_NEON
};
