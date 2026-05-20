#pragma once

#include "OLS.hpp"

/**
 * Scalar implementation of OLS for float precision
 * Inherits from OLS<float> and overrides some methods for optimization
 * Primary goal: order of operations need to match with that of the SSE2 code path for binary compatibility
 * Secondary goal: it's still faster than the naive reference implementation due to better cache locality.
 */
class OLS_float_Scalar : public OLS<float>
{
public:
  OLS_float_Scalar(size_t n, size_t solveInterval, float lambda, float nu);

  float predict() override;
  void update(float y) override;

protected:
  bool factor() override;
  void solve() override;
};
