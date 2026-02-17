#include <cmath>

#include "OLS_double_Scalar.hpp"

OLS_double_Scalar::OLS_double_Scalar(size_t n, size_t solveInterval, double lambda, double nu)
  : OLS<double>(n, solveInterval, lambda, nu) {
}

double OLS_double_Scalar::predict() {
  assert(featureIndex == n);
  featureIndex = 0;

  // Prediction: y = xᵀ·w
  // Process in pairs to match SSE2's 2-double processing
  double sum0 = 0.0;
  double sum1 = 0.0;

  for (size_t i = 0; i < nPadded; i += 2) {
    sum0 += x[i] * w[i];
    sum1 += x[i + 1] * w[i + 1];
  }

  // Horizontal sum
  double sum = sum0 + sum1;

  return sum;
}

void OLS_double_Scalar::update(double y) {
  const double alpha = lambda;      // Retention factor
  const double beta = 1.0 - lambda; // Forgetting factor

  // The following code is auto-vectorized to SSE2 on x64 by GCC (optimal)

#if defined(__GNUC__)
  // Tell compiler these pointers are 32-byte aligned
  double* __restrict__ C_aligned = static_cast<double*>(__builtin_assume_aligned(&mCovariance[0], 32));
  double* __restrict__ b_aligned = static_cast<double*>(__builtin_assume_aligned(&b[0], 32));
  const double* __restrict__ x_aligned = static_cast<double*>(__builtin_assume_aligned(&x[0], 32));
#else
  double* C_aligned = &mCovariance[0];
  double* b_aligned = &b[0];
  const double* x_aligned = &x[0];
#endif

  ASSUME((nPadded & 1) == 0); // hint: help the compiler to vectorize better

  // C = λC + (1-λ)x·xᵀ (lower triangle only)
  size_t i_n = 0; // i * nPadded
  for (size_t i = 0; i < n; i++) {
    double xi_beta = x_aligned[i] * beta;
    for (size_t j = 0; j <= i; j++) {  // Only compute lower triangle
      C_aligned[i_n + j] = alpha * C_aligned[i_n + j] + (x_aligned[j] * xi_beta);
    }
    i_n += nPadded;
  }

  // b = λb + (1-λ)y·x
  for (size_t i = 0; i < nPadded; i++) {
    b_aligned[i] = alpha * b_aligned[i] + y * x_aligned[i] * beta;
  }

  // Periodic re-solve
  // C·w = b
  samplesSinceLastSolve++;
  if (samplesSinceLastSolve >= solveInterval) {
    bool isSuccess = factor();
    if (isSuccess) {
      solve();
    }
    samplesSinceLastSolve = 0;
  }
}

bool OLS_double_Scalar::factor() {

  // Copy lower triangle only + apply regularization
  size_t i_n = 0;
  for (size_t i = 0; i < n; i++) {
    // Copy elements below diagonal
    for (size_t j = 0; j < i; j++) {
      mCholesky[i_n + j] = mCovariance[i_n + j];
    }

    // Diagonal with regularization
    mCholesky[i_n + i] = mCovariance[i_n + i] + nu;
    i_n += nPadded;
  }

  // Cholesky factorization: C = L·Lᵀ
  // Decomposes regularized covariance matrix into lower triangular L
  const size_t nb = n >> 1;
  const bool has_boundary = (n & 1);

  double* krow0 = &mCholesky[0];           // row 2k
  double* krow1 = krow0 + nPadded;         // row 2k+1

  for (size_t k = 0; k < nb; k++) {
    size_t kc = 2 * k;

    // --------------------------------------------------
    // 1) Full diagonal update
    // --------------------------------------------------

    // Load values to be updated
    double krow0_kc0 = krow0[kc];
    double krow1_kc0 = krow1[kc];
    double krow1_kc1 = krow1[kc + 1];

    for (size_t p = 0; p < k; p++) {
      size_t pc = 2 * p;

      double a00 = krow0[pc];
      double a01 = krow0[pc + 1];
      double a10 = krow1[pc];
      double a11 = krow1[pc + 1];

      // Update krow0[kc]
      krow0_kc0 -= a00 * a00 + a01 * a01;

      // Update krow1[kc] and krow1[kc+1]:
      krow1_kc0 -= a10 * a00 + a11 * a01;
      krow1_kc1 -= a10 * a10 + a11 * a11;
    }

    // Store results back
    krow0[kc] = krow0_kc0;
    krow1[kc] = krow1_kc0;
    krow1[kc + 1] = krow1_kc1;

    // --------------------------------------------------
    // 2) Factor diagonal block
    // --------------------------------------------------
    double d00 = krow0[kc];
    if (d00 <= min_diagonal)
      goto fail;
    d00 = sqrt(d00);
    krow0[kc] = d00;

    double d10 = krow1[kc] / d00;
    krow1[kc] = d10;

    double d11 = krow1[kc + 1] - d10 * d10;
    if (d11 <= min_diagonal)
      goto fail;
    d11 = sqrt(d11);
    krow1[kc + 1] = d11;

    // --------------------------------------------------
    // 3) Panel update + TRSM
    // --------------------------------------------------
    double* irow0 = krow0 + 2 * nPadded;
    double* irow1 = irow0 + nPadded;

    for (size_t i = k + 1; i < nb; i++) {

      // Load the 2x2 block we're updating
      double irow0_kc0 = irow0[kc];
      double irow0_kc1 = irow0[kc + 1];
      double irow1_kc0 = irow1[kc];
      double irow1_kc1 = irow1[kc + 1];

      for (size_t p = 0; p < k; p++) {
        size_t pc = 2 * p;

        double aik0 = irow0[pc];
        double aik1 = irow0[pc + 1];
        double aik2 = irow1[pc];
        double aik3 = irow1[pc + 1];

        double b00 = krow0[pc];
        double b01 = krow0[pc + 1];
        double b10 = krow1[pc];
        double b11 = krow1[pc + 1];

        // Compute updates for irow0_kc
        irow0_kc0 -= aik0 * b00 + aik1 * b01;
        irow0_kc1 -= aik0 * b10 + aik1 * b11;

        // Compute updates for irow1_kc
        irow1_kc0 -= aik2 * b00 + aik3 * b01;
        irow1_kc1 -= aik2 * b10 + aik3 * b11;
      }

      // TRSM (triangular solve)
      // Step 1: Divide first column by d00
      double col0_0 = irow0_kc0 / d00;
      double col0_1 = irow1_kc0 / d00;

      // Step 2: Update and divide second column
      double col1_0 = (irow0_kc1 - d10 * col0_0) / d11;
      double col1_1 = (irow1_kc1 - d10 * col0_1) / d11;

      // Store results
      irow0[kc] = col0_0;
      irow0[kc + 1] = col1_0;
      irow1[kc] = col0_1;
      irow1[kc + 1] = col1_1;

      irow0 += 2 * nPadded;
      irow1 += 2 * nPadded;
    }

    krow0 += 2 * nPadded;
    krow1 += 2 * nPadded;
  }

  // --------------------------------------------------
  // Boundary scalar
  // --------------------------------------------------
  if (has_boundary) {
    const size_t b = nb << 1;
    double* brow = &mCholesky[0] + b * nPadded;

    double brow_b = brow[b];
    for (size_t p = 0; p < nb; p++) {
      size_t pc = 2 * p;
      double a0 = brow[pc];
      double a1 = brow[pc + 1];
      brow_b -= a0 * a0 + a1 * a1;
    }

    if (brow_b <= min_diagonal)
      goto fail;
    brow[b] = sqrt(brow_b);
  }

  return true; // Success

fail:
  return false; // Factorization failed: matrix not positive definite
}

void OLS_double_Scalar::solve() {

  // The following code is auto-vectorized to SSE2 on x64 by GCC (optimal)

#if defined(__GNUC__)
  // Tell compiler these pointers are 32-byte aligned
  double* __restrict__ C_aligned = static_cast<double*>(__builtin_assume_aligned(&mCholesky[0], 32));
  double* __restrict__ w_aligned = static_cast<double*>(__builtin_assume_aligned(&w[0], 32));
  double* __restrict__ b_aligned = static_cast<double*>(__builtin_assume_aligned(&b[0], 32));
  const double* __restrict__ x_aligned = static_cast<double*>(__builtin_assume_aligned(&x[0], 32));
#else
  double* C_aligned = &mCholesky[0];
  double* w_aligned = &w[0];
  double* b_aligned = &b[0];
  const double* x_aligned = &x[0];
#endif

  ASSUME((nPadded & 1) == 0); // hint: help the compiler to vectorize better

  // Forward solve: L·w = b
  size_t i_n = 0; // i * nPadded
  for (size_t i = 0; i < n; i++) {
    double sum = 0;
    for (size_t j = 0; j < i; j++) {
      sum -= (C_aligned[i_n + j] * w_aligned[j]);
    }
    sum += b_aligned[i];
    w_aligned[i] = sum / C_aligned[i_n + i];
    i_n += nPadded;
  }

  // Backward solve: Lᵀ·w = w
  i_n -= nPadded;
  for (size_t i = n - 1; ((int)i) >= 0; i--) {
    double sum = 0;
    size_t j_n = i_n + nPadded;
    for (size_t j = i + 1; j < n; j++) {
      sum -= (C_aligned[j_n + i] * w_aligned[j]);
      j_n += nPadded;
    }
    sum += w_aligned[i];
    w_aligned[i] = sum / C_aligned[i_n + i];

    i_n -= nPadded;
  }
}
