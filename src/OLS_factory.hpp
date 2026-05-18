#pragma once

#include <memory>
#include "Shared.hpp"
#include "OLS.hpp"

std::unique_ptr<OLS_float> create_OLS_float(SIMDType simd, size_t n, size_t solveInterval, float lambda, float nu);
std::unique_ptr<OLS_double> create_OLS_double(SIMDType simd, size_t n, size_t solveInterval, double lambda, double nu);

