#include "SimilarityEmaFunctionsFactory.hpp"

SimilarityEmaUpdateFunction SimilarityEmaFunctionsFactory::getEmaUpdateFunction(const Shared* const shared) {
#ifdef X64_SIMD_AVAILABLE
  const SIMDType chosenSimd = shared->chosenSimd;
  if (chosenSimd >= SIMDType::SIMD_AVX2) {
    return &SimilarityEmaFunctions_AVX2::update_and_find;
  }
  else if (chosenSimd >= SIMDType::SIMD_SSE41) {
    return &SimilarityEmaFunctions_SSE41::update_and_find;
  }
#endif
  return &SimilarityEmaFunctions_Scalar::update_and_find;
}
