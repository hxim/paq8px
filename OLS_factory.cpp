#include "OLS_factory.hpp"
#include "OLS_float_Scalar.hpp"
#include "OLS_double_Scalar.hpp"

#ifdef X64_SIMD_AVAILABLE
#include "OLS_float_SSE3.hpp"
#include "OLS_double_SSE3.hpp"
#endif

std::unique_ptr<OLS_float> create_OLS_float(SIMDType simd, size_t n, size_t solveInterval, float lambda, float nu) {
#ifdef X64_SIMD_AVAILABLE
  if (simd == SIMDType::SIMD_AVX2 || simd == SIMDType::SIMD_AVX512 || simd == SIMDType::SIMD_SSE2)
    return std::make_unique<OLS_float_SSE3>(n, solveInterval, lambda, nu);
  else
#endif
    return std::make_unique<OLS_float_Scalar>(n, solveInterval, lambda, nu);
}

std::unique_ptr<OLS_double> create_OLS_double(SIMDType simd, size_t n, size_t solveInterval, double lambda, double nu) {
#ifdef X64_SIMD_AVAILABLE
  if (simd == SIMDType::SIMD_AVX2 || simd == SIMDType::SIMD_AVX512 || simd == SIMDType::SIMD_SSE2)
    return std::make_unique<OLS_double_SSE3>(n, solveInterval, lambda, nu);
  else
#endif
    return std::make_unique<OLS_double_Scalar>(n, solveInterval, lambda, nu);
}
