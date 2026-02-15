#include "OLS.hpp"

#include <cmath>

#include "SystemDefines.hpp" // ASSUME, <cassert>

template <typename T>
OLS<T>::OLS(size_t n, size_t solveInterval, T lambda, T nu) :
  n(n),
  nPadded(computePadding(n)),
  solveInterval(solveInterval),
  lambda(lambda),
  nu(nu),
  x(nPadded),
  w(nPadded),
  b(nPadded),
  mCovariance(n * nPadded),
  mCholesky(n * nPadded),
  samplesSinceLastSolve(0),
  featureIndex(0) {
}

// Helper to compute padded dimension for SIMD alignment
template <typename T>
constexpr size_t OLS<T>::computePadding(size_t n) {
  if constexpr (std::is_same_v<T, float>) {
    // Pad to multiple of 4 for SSE (128-bit = 4 floats)
    return (n + 3) & ~3ULL;
  }
  else if constexpr (std::is_same_v<T, double>) {
    // Pad to multiple of 2 for SSE (128-bit = 2 doubles)
    return (n + 1) & ~1ULL;
  }
  else {
    return n; // No padding for other types (not in use)
  }
}

template <typename T>
void OLS<T>::add(T val) {
  assert(featureIndex < n);
  x[featureIndex++] = val;
}

template <typename T>
T OLS<T>::predict() {

  assert(featureIndex == n);
  featureIndex = 0;

  // Prediction: y = xᵀ·w
  T sum = static_cast<T>(0.0);
  for (size_t i = 0; i < n; i++) {
    sum += w[i] * x[i];
  }
  return sum;
}

template <typename T>
void OLS<T>::update(T y) {

  const T alpha = lambda;                       // Retention factor
  const T beta = static_cast<T>(1.0) - lambda;  // Forgetting factor

  // C = λC + (1-λ)x·xᵀ
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      mCovariance[i * nPadded + j] = alpha * mCovariance[i * nPadded + j] + beta * x[i] * x[j];
    }
  }

  // b = λb + (1-λ)y·x
  for (size_t i = 0; i < n; i++) {
    b[i] = alpha * b[i] + beta * y * x[i];
  }

  // Periodic re-solve
  // C·w = b
  samplesSinceLastSolve++;
  if (samplesSinceLastSolve >= solveInterval) {
    bool success = factor();
    if (success) {
      solve();
    }
    samplesSinceLastSolve = 0;
  }
}

template <typename T>
bool OLS<T>::factor() {

  // Copy the matrix
  memcpy(&mCholesky[0], &mCovariance[0], n * nPadded * sizeof(T));

  // Apply regularization on main diagonal
  for (size_t i = 0; i < n; i++) {
    mCholesky[i * nPadded + i] += nu;
  }

  // Cholesky factorization: C = L·Lᵀ
  // Decomposes regularized covariance matrix into lower triangular L
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < i; j++) {
      T sum = mCholesky[i * nPadded + j];
      for (size_t k = 0; k < j; k++) {
        sum -= (mCholesky[i * nPadded + k] * mCholesky[j * nPadded + k]);
      }
      mCholesky[i * nPadded + j] = sum / mCholesky[j * nPadded + j];
    }
    T sum = mCholesky[i * nPadded + i];
    for (size_t k = 0; k < i; k++) {
      sum -= (mCholesky[i * nPadded + k] * mCholesky[i * nPadded + k]);
    }
    if (sum > min_diagonal) {
      mCholesky[i * nPadded + i] = sqrt(sum); // Main diagonal
    }
    else {
      return false; // Factorization failed: matrix not positive definite
    }
  }

  return true; // Success
}

template <typename T>
void OLS<T>::solve() {

  // Forward solve: L·w = b
  for (size_t i = 0; i < n; i++) {
    T sum = b[i];
    for (size_t j = 0; j < i; j++) {
      sum -= (mCholesky[i * nPadded + j] * w[j]);
    }
    w[i] = sum / mCholesky[i * nPadded + i];
  }

  // Backward solve: Lᵀ·w = w
  for (size_t i = n - 1; (int)i >= 0; i--) {
    T sum = w[i];
    for (size_t j = i + 1; j < n; j++) {
      sum -= (mCholesky[j * nPadded + i] * w[j]);
    }
    w[i] = sum / mCholesky[i * nPadded + i];
  }
}


template class OLS<float>;
template class OLS<double>;
