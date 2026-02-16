#include <cmath>

#include "OLS_float_Scalar.hpp"

OLS_float_Scalar::OLS_float_Scalar(size_t n, size_t solveInterval, float lambda, float nu)
  : OLS<float>(n, solveInterval, lambda, nu) {
}

float OLS_float_Scalar::predict() {
  assert(featureIndex == n);
  featureIndex = 0;

  // Prediction: y = xᵀ·w
  // Process 4 floats at a time to match alignment
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  float sum3 = 0.0f;

  for (size_t i = 0; i < nPadded; i += 4) {
    sum0 += x[i] * w[i];
    sum1 += x[i + 1] * w[i + 1];
    sum2 += x[i + 2] * w[i + 2];
    sum3 += x[i + 3] * w[i + 3];
  }

  // Horizontal sum
  float sum02 = sum0 + sum2;
  float sum13 = sum1 + sum3;
  float sum = sum02 + sum13;
  return sum;
}

void OLS_float_Scalar::update(float y) {
  const float alpha = lambda;      // Retention factor
  const float beta = 1.0f - lambda; // Forgetting factor

  // The following code is auto-vectorized to SSE2 on x64 by GCC (optimal)

#if defined(__GNUC__)
  // Tell compiler these pointers are 32-byte aligned
  float* __restrict__ C_aligned = static_cast<float*>(__builtin_assume_aligned(&mCovariance[0], 32));
  float* __restrict__ b_aligned = static_cast<float*>(__builtin_assume_aligned(&b[0], 32));
  const float* __restrict__ x_aligned = static_cast<float*>(__builtin_assume_aligned(&x[0], 32));
#else
  float* C_aligned = &mCovariance[0];
  float* b_aligned = &b[0];
  const float* x_aligned = &x[0];
#endif

  ASSUME((nPadded & 3) == 0); // hint: help the compiler to vectorize better

  // C = λC + (1-λ)x·xᵀ (lower triangle only)
  size_t i_n = 0; // i * nPadded
  for (size_t i = 0; i < n; i++) {
    float xi_beta = x_aligned[i] * beta;
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

bool OLS_float_Scalar::factor() {

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
  const size_t nb = n >> 2;  // Number of 4x4 blocks
  const size_t b = nb << 2;  // First boundary row

  float* krow0 = &mCholesky[0];           // row 4k
  float* krow1 = krow0 + nPadded;         // row 4k+1
  float* krow2 = krow1 + nPadded;         // row 4k+2
  float* krow3 = krow2 + nPadded;         // row 4k+3

  for (size_t k = 0; k < nb; k++) {
    size_t kc = 4 * k;

    // --------------------------------------------------
    // 1) Full diagonal update (4x4 block)
    // --------------------------------------------------

    // Load diagonal values to be updated
    float krow0_kc0 = krow0[kc];

    float krow1_kc0 = krow1[kc];
    float krow1_kc1 = krow1[kc + 1];

    float krow2_kc0 = krow2[kc];
    float krow2_kc1 = krow2[kc + 1];
    float krow2_kc2 = krow2[kc + 2];

    float krow3_kc0 = krow3[kc];
    float krow3_kc1 = krow3[kc + 1];
    float krow3_kc2 = krow3[kc + 2];
    float krow3_kc3 = krow3[kc + 3];

    for (size_t p = 0; p < k; p++) {
      size_t pc = 4 * p;

      float a00 = krow0[pc];
      float a01 = krow0[pc + 1];
      float a02 = krow0[pc + 2];
      float a03 = krow0[pc + 3];

      float a10 = krow1[pc];
      float a11 = krow1[pc + 1];
      float a12 = krow1[pc + 2];
      float a13 = krow1[pc + 3];

      float a20 = krow2[pc];
      float a21 = krow2[pc + 1];
      float a22 = krow2[pc + 2];
      float a23 = krow2[pc + 3];

      float a30 = krow3[pc];
      float a31 = krow3[pc + 1];
      float a32 = krow3[pc + 2];
      float a33 = krow3[pc + 3];

      // Update krow0[kc]
      krow0_kc0 -= ((a00 * a00 + a01 * a01) + (a02 * a02 + a03 * a03));

      // Update krow1[kc:kc+1]
      krow1_kc0 -= ((a10 * a00 + a11 * a01) + (a12 * a02 + a13 * a03));
      krow1_kc1 -= ((a10 * a10 + a11 * a11) + (a12 * a12 + a13 * a13));

      // Update krow2[kc:kc+2]
      krow2_kc0 -= ((a20 * a00 + a21 * a01) + (a22 * a02 + a23 * a03));
      krow2_kc1 -= ((a20 * a10 + a21 * a11) + (a22 * a12 + a23 * a13));
      krow2_kc2 -= ((a20 * a20 + a21 * a21) + (a22 * a22 + a23 * a23));

      // Update krow3[kc:kc+3]
      krow3_kc0 -= ((a30 * a00 + a31 * a01) + (a32 * a02 + a33 * a03));
      krow3_kc1 -= ((a30 * a10 + a31 * a11) + (a32 * a12 + a33 * a13));
      krow3_kc2 -= ((a30 * a20 + a31 * a21) + (a32 * a22 + a33 * a23));
      krow3_kc3 -= ((a30 * a30 + a31 * a31) + (a32 * a32 + a33 * a33));
    }

    // Store results back
    krow0[kc] = krow0_kc0;

    krow1[kc] = krow1_kc0;
    krow1[kc + 1] = krow1_kc1;

    krow2[kc] = krow2_kc0;
    krow2[kc + 1] = krow2_kc1;
    krow2[kc + 2] = krow2_kc2;

    krow3[kc] = krow3_kc0;
    krow3[kc + 1] = krow3_kc1;
    krow3[kc + 2] = krow3_kc2;
    krow3[kc + 3] = krow3_kc3;

    // --------------------------------------------------
    // 2) Factor diagonal 4x4 block
    // --------------------------------------------------

    // Column 0
    float d00 = krow0[kc];
    if (d00 <= min_diagonal)
      goto fail;
    d00 = sqrt(d00);
    krow0[kc] = d00;

    float d10 = krow1[kc] / d00;
    krow1[kc] = d10;

    float d20 = krow2[kc] / d00;
    krow2[kc] = d20;

    float d30 = krow3[kc] / d00;
    krow3[kc] = d30;

    // Column 1
    float d11 = krow1[kc + 1] - d10 * d10;
    if (d11 <= min_diagonal)
      goto fail;
    d11 = sqrt(d11);
    krow1[kc + 1] = d11;

    float d21 = (krow2[kc + 1] - d20 * d10) / d11;
    krow2[kc + 1] = d21;

    float d31 = (krow3[kc + 1] - d30 * d10) / d11;
    krow3[kc + 1] = d31;

    // Column 2
    float d22 = krow2[kc + 2] - d20 * d20 - d21 * d21;
    if (d22 <= min_diagonal)
      goto fail;
    d22 = sqrt(d22);
    krow2[kc + 2] = d22;

    float d32 = (krow3[kc + 2] - d30 * d20 - d31 * d21) / d22;
    krow3[kc + 2] = d32;

    // Column 3
    float d33 = krow3[kc + 3] - d30 * d30 - d31 * d31 - d32 * d32;
    if (d33 <= min_diagonal)
      goto fail;
    d33 = sqrt(d33);
    krow3[kc + 3] = d33;

    // --------------------------------------------------
    // 3) Panel update + TRSM (4x4 blocks)
    // --------------------------------------------------
    float* irow0 = krow0 + 4 * nPadded;
    float* irow1 = irow0 + nPadded;
    float* irow2 = irow1 + nPadded;
    float* irow3 = irow2 + nPadded;

    for (size_t i = k + 1; i < nb; i++) {

      // Load the 4x4 block we're updating
      float irow0_kc0 = irow0[kc];
      float irow0_kc1 = irow0[kc + 1];
      float irow0_kc2 = irow0[kc + 2];
      float irow0_kc3 = irow0[kc + 3];

      float irow1_kc0 = irow1[kc];
      float irow1_kc1 = irow1[kc + 1];
      float irow1_kc2 = irow1[kc + 2];
      float irow1_kc3 = irow1[kc + 3];

      float irow2_kc0 = irow2[kc];
      float irow2_kc1 = irow2[kc + 1];
      float irow2_kc2 = irow2[kc + 2];
      float irow2_kc3 = irow2[kc + 3];

      float irow3_kc0 = irow3[kc];
      float irow3_kc1 = irow3[kc + 1];
      float irow3_kc2 = irow3[kc + 2];
      float irow3_kc3 = irow3[kc + 3];

      for (size_t p = 0; p < k; p++) {
        size_t pc = 4 * p;

        float aik0 = irow0[pc];
        float aik1 = irow0[pc + 1];
        float aik2 = irow0[pc + 2];
        float aik3 = irow0[pc + 3];

        float aik4 = irow1[pc];
        float aik5 = irow1[pc + 1];
        float aik6 = irow1[pc + 2];
        float aik7 = irow1[pc + 3];

        float aik8 = irow2[pc];
        float aik9 = irow2[pc + 1];
        float aik10 = irow2[pc + 2];
        float aik11 = irow2[pc + 3];

        float aik12 = irow3[pc];
        float aik13 = irow3[pc + 1];
        float aik14 = irow3[pc + 2];
        float aik15 = irow3[pc + 3];

        float b00 = krow0[pc];
        float b01 = krow0[pc + 1];
        float b02 = krow0[pc + 2];
        float b03 = krow0[pc + 3];

        float b10 = krow1[pc];
        float b11 = krow1[pc + 1];
        float b12 = krow1[pc + 2];
        float b13 = krow1[pc + 3];

        float b20 = krow2[pc];
        float b21 = krow2[pc + 1];
        float b22 = krow2[pc + 2];
        float b23 = krow2[pc + 3];

        float b30 = krow3[pc];
        float b31 = krow3[pc + 1];
        float b32 = krow3[pc + 2];
        float b33 = krow3[pc + 3];

        // Update irow0
        irow0_kc0 -= ((aik0 * b00 + aik1 * b01) + (aik2 * b02 + aik3 * b03));
        irow0_kc1 -= ((aik0 * b10 + aik1 * b11) + (aik2 * b12 + aik3 * b13));
        irow0_kc2 -= ((aik0 * b20 + aik1 * b21) + (aik2 * b22 + aik3 * b23));
        irow0_kc3 -= ((aik0 * b30 + aik1 * b31) + (aik2 * b32 + aik3 * b33));

        // Update irow1
        irow1_kc0 -= ((aik4 * b00 + aik5 * b01) + (aik6 * b02 + aik7 * b03));
        irow1_kc1 -= ((aik4 * b10 + aik5 * b11) + (aik6 * b12 + aik7 * b13));
        irow1_kc2 -= ((aik4 * b20 + aik5 * b21) + (aik6 * b22 + aik7 * b23));
        irow1_kc3 -= ((aik4 * b30 + aik5 * b31) + (aik6 * b32 + aik7 * b33));

        // Update irow2
        irow2_kc0 -= ((aik8 * b00 + aik9 * b01) + (aik10 * b02 + aik11 * b03));
        irow2_kc1 -= ((aik8 * b10 + aik9 * b11) + (aik10 * b12 + aik11 * b13));
        irow2_kc2 -= ((aik8 * b20 + aik9 * b21) + (aik10 * b22 + aik11 * b23));
        irow2_kc3 -= ((aik8 * b30 + aik9 * b31) + (aik10 * b32 + aik11 * b33));

        // Update irow3
        irow3_kc0 -= ((aik12 * b00 + aik13 * b01) + (aik14 * b02 + aik15 * b03));
        irow3_kc1 -= ((aik12 * b10 + aik13 * b11) + (aik14 * b12 + aik15 * b13));
        irow3_kc2 -= ((aik12 * b20 + aik13 * b21) + (aik14 * b22 + aik15 * b23));
        irow3_kc3 -= ((aik12 * b30 + aik13 * b31) + (aik14 * b32 + aik15 * b33));
      }

      // TRSM (triangular solve for 4x4 block)
      // Column 0
      float col0_0 = irow0_kc0 / d00;
      float col0_1 = irow1_kc0 / d00;
      float col0_2 = irow2_kc0 / d00;
      float col0_3 = irow3_kc0 / d00;

      // Column 1
      float col1_0 = (irow0_kc1 - d10 * col0_0) / d11;
      float col1_1 = (irow1_kc1 - d10 * col0_1) / d11;
      float col1_2 = (irow2_kc1 - d10 * col0_2) / d11;
      float col1_3 = (irow3_kc1 - d10 * col0_3) / d11;

      // Column 2
      float col2_0 = (irow0_kc2 - d20 * col0_0 - d21 * col1_0) / d22;
      float col2_1 = (irow1_kc2 - d20 * col0_1 - d21 * col1_1) / d22;
      float col2_2 = (irow2_kc2 - d20 * col0_2 - d21 * col1_2) / d22;
      float col2_3 = (irow3_kc2 - d20 * col0_3 - d21 * col1_3) / d22;

      // Column 3
      float col3_0 = (irow0_kc3 - d30 * col0_0 - d31 * col1_0 - d32 * col2_0) / d33;
      float col3_1 = (irow1_kc3 - d30 * col0_1 - d31 * col1_1 - d32 * col2_1) / d33;
      float col3_2 = (irow2_kc3 - d30 * col0_2 - d31 * col1_2 - d32 * col2_2) / d33;
      float col3_3 = (irow3_kc3 - d30 * col0_3 - d31 * col1_3 - d32 * col2_3) / d33;

      // Store results
      irow0[kc] = col0_0;
      irow0[kc + 1] = col1_0;
      irow0[kc + 2] = col2_0;
      irow0[kc + 3] = col3_0;

      irow1[kc] = col0_1;
      irow1[kc + 1] = col1_1;
      irow1[kc + 2] = col2_1;
      irow1[kc + 3] = col3_1;

      irow2[kc] = col0_2;
      irow2[kc + 1] = col1_2;
      irow2[kc + 2] = col2_2;
      irow2[kc + 3] = col3_2;

      irow3[kc] = col0_3;
      irow3[kc + 1] = col1_3;
      irow3[kc + 2] = col2_3;
      irow3[kc + 3] = col3_3;

      irow0 += 4 * nPadded;
      irow1 += 4 * nPadded;
      irow2 += 4 * nPadded;
      irow3 += 4 * nPadded;
    }

    krow0 += 4 * nPadded;
    krow1 += 4 * nPadded;
    krow2 += 4 * nPadded;
    krow3 += 4 * nPadded;
  }

  // --------------------------------------------------
  // Boundary handling for remaining 1-3 rows
  // --------------------------------------------------

  // Process each boundary row
  for (size_t i = b; i < n; i++) {
    float* irow = &mCholesky[i * nPadded];

    // Panel update + TRSM for each 4x4 block
    float* krow0 = &mCholesky[0];
    float* krow1 = krow0 + nPadded;
    float* krow2 = krow1 + nPadded;
    float* krow3 = krow2 + nPadded;

    for (size_t k = 0; k < nb; k++) {
      size_t kc = k * 4;

      // Panel update: subtract contributions from previous blocks
      float sum0 = irow[kc];
      float sum1 = irow[kc + 1];
      float sum2 = irow[kc + 2];
      float sum3 = irow[kc + 3];

      for (size_t p = 0; p < k; p++) {
        size_t pc = p * 4;

        float a0 = irow[pc];
        float a1 = irow[pc + 1];
        float a2 = irow[pc + 2];
        float a3 = irow[pc + 3];

        float b00 = krow0[pc];
        float b01 = krow0[pc + 1];
        float b02 = krow0[pc + 2];
        float b03 = krow0[pc + 3];

        float b10 = krow1[pc];
        float b11 = krow1[pc + 1];
        float b12 = krow1[pc + 2];
        float b13 = krow1[pc + 3];

        float b20 = krow2[pc];
        float b21 = krow2[pc + 1];
        float b22 = krow2[pc + 2];
        float b23 = krow2[pc + 3];

        float b30 = krow3[pc];
        float b31 = krow3[pc + 1];
        float b32 = krow3[pc + 2];
        float b33 = krow3[pc + 3];

        sum0 -= ((a0 * b00 + a1 * b01) + (a2 * b02 + a3 * b03));
        sum1 -= ((a0 * b10 + a1 * b11) + (a2 * b12 + a3 * b13));
        sum2 -= ((a0 * b20 + a1 * b21) + (a2 * b22 + a3 * b23));
        sum3 -= ((a0 * b30 + a1 * b31) + (a2 * b32 + a3 * b33));
      }

      // TRSM: solve against block k's factored diagonal
      float d00 = krow0[kc];
      float d10 = krow1[kc];
      float d11 = krow1[kc + 1];
      float d20 = krow2[kc];
      float d21 = krow2[kc + 1];
      float d22 = krow2[kc + 2];
      float d30 = krow3[kc];
      float d31 = krow3[kc + 1];
      float d32 = krow3[kc + 2];
      float d33 = krow3[kc + 3];

      float col0 = sum0 / d00;
      float col1 = (sum1 - d10 * col0) / d11;
      float col2 = (sum2 - d20 * col0 - d21 * col1) / d22;
      float col3 = (sum3 - d30 * col0 - d31 * col1 - d32 * col2) / d33;

      irow[kc] = col0;
      irow[kc + 1] = col1;
      irow[kc + 2] = col2;
      irow[kc + 3] = col3;

      krow0 += 4 * nPadded;
      krow1 += 4 * nPadded;
      krow2 += 4 * nPadded;
      krow3 += 4 * nPadded;
    }
  }

  // Now scalar Cholesky for boundary rows among themselves
  i_n = b * nPadded;
  for (size_t i = b; i < n; i++) {

    // Off-diagonal: L[i,j] for j in [b, i)
    for (size_t j = b; j < i; j++) {
      size_t j_n = j * nPadded;
      float sum0 = 0.0f;
      float sum1 = 0.0f;
      float sum2 = 0.0f;
      float sum3 = 0.0f;

      size_t k_end = j & ~3ULL;
      size_t k = 0;
      for (; k < k_end; k += 4) {
        float a0 = mCholesky[i_n + k];
        float a1 = mCholesky[i_n + k + 1];
        float a2 = mCholesky[i_n + k + 2];
        float a3 = mCholesky[i_n + k + 3];

        float b0 = mCholesky[j_n + k];
        float b1 = mCholesky[j_n + k + 1];
        float b2 = mCholesky[j_n + k + 2];
        float b3 = mCholesky[j_n + k + 3];

        sum0 += a0 * b0;
        sum1 += a1 * b1;
        sum2 += a2 * b2;
        sum3 += a3 * b3;
      }

      float sum = mCholesky[i_n + j] - ((sum0 + sum2) + (sum1 + sum3));

      for (; k < j; k++) {
        sum -= (mCholesky[i_n + k] * mCholesky[j_n + k]);
      }

      mCholesky[i_n + j] = sum / mCholesky[j_n + j];
    }

    // Diagonal: L[i,i]
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    size_t k_end = i & ~3ULL;
    size_t k = 0;
    for (; k < k_end; k += 4) {
      float a0 = mCholesky[i_n + k];
      float a1 = mCholesky[i_n + k + 1];
      float a2 = mCholesky[i_n + k + 2];
      float a3 = mCholesky[i_n + k + 3];

      sum0 += a0 * a0;
      sum1 += a1 * a1;
      sum2 += a2 * a2;
      sum3 += a3 * a3;
    }

    float sum = mCholesky[i_n + i] - ((sum0 + sum2) + (sum1 + sum3));

    for (; k < i; k++) {
      sum -= (mCholesky[i_n + k] * mCholesky[i_n + k]);
    }

    if (sum > min_diagonal) {
      mCholesky[i_n + i] = sqrt(sum);
    }
    else {
      goto fail;
    }
    i_n += nPadded;
  }

  return true; // Success

fail:
  return false; // Factorization failed: matrix not positive definite
}

void OLS_float_Scalar::solve() {

  // The following code is auto-vectorized to SSE2 on x64 by GCC (optimal)

#if defined(__GNUC__)
  // Tell compiler these pointers are 32-byte aligned
  float* __restrict__ C_aligned = static_cast<float*>(__builtin_assume_aligned(&mCholesky[0], 32));
  float* __restrict__ w_aligned = static_cast<float*>(__builtin_assume_aligned(&w[0], 32));
  float* __restrict__ b_aligned = static_cast<float*>(__builtin_assume_aligned(&b[0], 32));
  const float* __restrict__ x_aligned = static_cast<float*>(__builtin_assume_aligned(&x[0], 32));
#else
  float* C_aligned = &mCholesky[0];
  float* w_aligned = &w[0];
  float* b_aligned = &b[0];
  const float* x_aligned = &x[0];
#endif

  ASSUME((nPadded & 3) == 0); // hint: help the compiler to vectorize better

  // Forward solve: L·w = b
  size_t i_n = 0; // i * nPadded
  for (size_t i = 0; i < n; i++) {
    float sum = 0;
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
    float sum = 0;
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
