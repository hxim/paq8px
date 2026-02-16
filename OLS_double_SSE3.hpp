#pragma once

#include "OLS_double_Scalar.hpp"

#ifdef X64_SIMD_AVAILABLE

/**
 * SSE2 implementation of OLS for double precision
 * Inherits from OLS_double_Scalar and overrides some methods for optimization
 */
class OLS_double_SSE3 : public OLS_double_Scalar
{
public:
  OLS_double_SSE3(size_t n, size_t solveInterval, double lambda, double nu);

  double predict() override;

protected:
  bool factor() override;
};

#endif
