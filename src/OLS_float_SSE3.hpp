#pragma once

#include "OLS_float_Scalar.hpp"

#ifdef X64_SIMD_AVAILABLE

/**
 * SSE2 implementation of OLS for float precision
 * Inherits from OLS_float_Scalar and overrides some methods for optimization
 */
class OLS_float_SSE3 : public OLS_float_Scalar
{
public:
  OLS_float_SSE3(size_t n, size_t solveInterval, float lambda, float nu);

  float predict() override;

protected:
  bool factor() override;
};

#endif
