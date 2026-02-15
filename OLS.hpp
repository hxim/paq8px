#pragma once

#include <cstdint>

#include "Array.hpp"

/**
 * Exponentially-forgetting OLS (Ordinary Least Squares) predictor
 * Base class with reference implementations
 *
 * @tparam T floating-point type (float or double)
 */

template <typename T>
class OLS
{
protected:
  static constexpr T min_diagonal = static_cast<T>(0.001); // Minimum acceptable pivot magnitude

  const size_t n;               // State dimension
  const size_t nPadded;         // Padded row size in matrices and vectors for SIMD alignment
  const size_t solveInterval;   // Recompute weights after every how many updates
  const T lambda;               // Retention factor (0 < lambda < 1, such as 0.99)
  const T nu;                   // Regularization parameter, such as 0.001
                                
  Array<T, 32> x;               // Current feature vector (padded)
  Array<T, 32> w;               // State estimate (coefficient vector, padded)
  Array<T, 32> b;               // Information vector (padded)
  Array<T, 32> mCovariance;     // Covariance / information matrix (padded rows)
  Array<T, 32> mCholesky;       // Temporary matrix to perform Cholesky decomposition (padded rows)

  size_t samplesSinceLastSolve; // 0 <= samplesSinceLastSolve < solveInterval
  size_t featureIndex;          // Current index for adding features, 0 <= featureIndex < n

  // Helper to compute padded dimension
  static constexpr size_t computePadding(size_t n);

public:
  OLS(size_t n, size_t solveInterval, T lambda, T nu);

  void add(T val);  // Builds up the feature vector for the next prediction, should be called n times
  T predict();      // Makes prediction using the added features
  void update(T y); // Incorporates the previous prediction's true value (y)

protected:
  bool factor();
  void solve();
};

// Explicit instantiation declarations
extern template class OLS<float>;
extern template class OLS<double>;

// Convenience type aliases
using OLS_float = OLS<float>;
using OLS_double = OLS<double>;

