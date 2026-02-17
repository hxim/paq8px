#pragma once

#include "OLS.hpp"

/**
 * Scalar implementation of OLS for double precision
 * Inherits from OLS<double> and overrides some methods for optimization
 */
class OLS_double_Scalar : public OLS<double>
{
public:
  OLS_double_Scalar(size_t n, size_t solveInterval, double lambda, double nu);

  double predict() override;
  void update(double y) override;

protected:
  bool factor() override;
  void solve() override;
};
