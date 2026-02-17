#include "OLS_double_SSE3.hpp"

#ifdef X64_SIMD_AVAILABLE

#include <cmath>

#if (defined(__GNUC__) || defined(__clang__))
#define SSE3_TARGET __attribute__((target("sse3")))
#else
#define SSE3_TARGET
#endif

OLS_double_SSE3::OLS_double_SSE3(size_t n, size_t solveInterval, double lambda, double nu)
  : OLS_double_Scalar(n, solveInterval, lambda, nu) {
}

SSE3_TARGET
double OLS_double_SSE3::predict() {
  assert(featureIndex == n);
  featureIndex = 0;

  // Prediction: y = xᵀ·w
  __m128d v_sum = _mm_setzero_pd();
  for (size_t i = 0; i < nPadded; i += 2) {
    __m128d v_x = _mm_load_pd(&x[i]);
    __m128d v_w = _mm_load_pd(&w[i]);
    v_sum = _mm_add_pd(v_sum, _mm_mul_pd(v_x, v_w));
  }

  // Horizontal sum for doubles
  __m128d v_high = _mm_unpackhi_pd(v_sum, v_sum);
  v_sum = _mm_add_sd(v_sum, v_high);
  double sum = _mm_cvtsd_f64(v_sum);

  return sum;
}

SSE3_TARGET
bool OLS_double_SSE3::factor() {

  // Copy lower triangle only + apply regularization
  size_t i_n = 0;
  for (size_t i = 0; i < n; i++) {

    // Process 2 elements at a time with SSE2
    size_t j = 0;
    for (; j + 1 < i; j += 2) {
      __m128d src = _mm_load_pd(&mCovariance[i_n + j]);
      _mm_store_pd(&mCholesky[i_n + j], src);
    }

    // Handle last element before diagonal
    if (j < i) {
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
    __m128d krow0_kc_vec = _mm_load_sd(&krow0[kc]);        // [krow0_kc0, 0]
    __m128d krow1_kc_vec = _mm_load_pd(&krow1[kc]);        // [krow1_kc0, krow1_kc1]

    for (size_t p = 0; p < k; p++) {
      size_t pc = 2 * p;

      // Load 2x2 blocks from krow0 and krow1
      __m128d krow0_vals = _mm_load_pd(&krow0[pc]);        // [a00, a01]
      __m128d krow1_vals = _mm_load_pd(&krow1[pc]);        // [a10, a11]

      // Update krow0[kc]: krow0_kc0 -= a00 * a00 + a01 * a01
      __m128d prod0 = _mm_mul_pd(krow0_vals, krow0_vals);  // [a00*a00, a01*a01]
      __m128d sum0 = _mm_hadd_pd(prod0, prod0);            // [a00*a00+a01*a01, ...]
      krow0_kc_vec = _mm_sub_sd(krow0_kc_vec, sum0);

      // Update krow1[kc] and krow1[kc+1]:
      // krow1_kc0 -= a10 * a00 + a11 * a01
      // krow1_kc1 -= a10 * a10 + a11 * a11
      __m128d prod1 = _mm_mul_pd(krow1_vals, krow0_vals);  // [a10*a00, a11*a01]
      __m128d prod2 = _mm_mul_pd(krow1_vals, krow1_vals);  // [a10*a10, a11*a11]
      __m128d sum1 = _mm_hadd_pd(prod1, prod2);            // [a10*a00+a11*a01, a10*a10+a11*a11]
      krow1_kc_vec = _mm_sub_pd(krow1_kc_vec, sum1);
    }

    // Store results back
    _mm_store_sd(&krow0[kc], krow0_kc_vec);
    _mm_store_pd(&krow1[kc], krow1_kc_vec);

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

    __m128d d00_vec = _mm_set1_pd(d00);
    __m128d d10_vec = _mm_set1_pd(d10);
    __m128d d11_vec = _mm_set1_pd(d11);

    // --------------------------------------------------
    // 3) Panel update + TRSM
    // --------------------------------------------------
    double* irow0 = krow0 + 2 * nPadded;
    double* irow1 = irow0 + nPadded;

    for (size_t i = k + 1; i < nb; i++) {

      // Load the 2x2 block we're updating
      __m128d irow0_kc = _mm_load_pd(&irow0[kc]);  // [irow0_kc0, irow0_kc1]
      __m128d irow1_kc = _mm_load_pd(&irow1[kc]);  // [irow1_kc0, irow1_kc1]

      for (size_t p = 0; p < k; p++) {
        size_t pc = 2 * p;

        // Load 2x2 blocks from irow0 and irow1
        __m128d aik_row0 = _mm_load_pd(&irow0[pc]);  // [aik0, aik1]
        __m128d aik_row1 = _mm_load_pd(&irow1[pc]);  // [aik2, aik3]

        // Load 2x2 block from krow0 and krow1
        __m128d b_row0 = _mm_load_pd(&krow0[pc]);    // [b00, b01]
        __m128d b_row1 = _mm_load_pd(&krow1[pc]);    // [b10, b11]

        // Compute updates for irow0_kc
        // irow0_kc0 -= aik0 * b00 + aik1 * b01
        // irow0_kc1 -= aik0 * b10 + aik1 * b11
        __m128d prod0 = _mm_mul_pd(aik_row0, b_row0);           // [aik0*b00, aik1*b01]
        __m128d prod1 = _mm_mul_pd(aik_row0, b_row1);           // [aik0*b10, aik1*b11]
        __m128d sum0 = _mm_hadd_pd(prod0, prod1);               // [aik0*b00+aik1*b01, aik0*b10+aik1*b11]
        irow0_kc = _mm_sub_pd(irow0_kc, sum0);

        // Compute updates for irow1_kc
        // irow1_kc0 -= aik2 * b00 + aik3 * b01
        // irow1_kc1 -= aik2 * b10 + aik3 * b11
        __m128d prod2 = _mm_mul_pd(aik_row1, b_row0);           // [aik2*b00, aik3*b01]
        __m128d prod3 = _mm_mul_pd(aik_row1, b_row1);           // [aik2*b10, aik3*b11]
        __m128d sum1 = _mm_hadd_pd(prod2, prod3);               // [aik2*b00+aik3*b01, aik2*b10+aik3*b11]
        irow1_kc = _mm_sub_pd(irow1_kc, sum1);
      }

      // TRSM (triangular solve)
      // Step 1: Divide first column by d00
      __m128d col0_div = _mm_div_pd(_mm_unpacklo_pd(irow0_kc, irow1_kc), d00_vec);
      // col0_div = [irow0_kc0/d00, irow1_kc0/d00]

      // Step 2: Update and divide second column
      // col1 = (col1 - d10 * col0_div) / d11
      __m128d col1 = _mm_unpackhi_pd(irow0_kc, irow1_kc);  // [irow0_kc1, irow1_kc1]
      col1 = _mm_sub_pd(col1, _mm_mul_pd(d10_vec, col0_div));
      col1 = _mm_div_pd(col1, d11_vec);

      // Reconstruct and store
      irow0_kc = _mm_shuffle_pd(col0_div, col1, _MM_SHUFFLE2(0, 0));  // [col0_div[0], col1[0]]
      irow1_kc = _mm_shuffle_pd(col0_div, col1, _MM_SHUFFLE2(1, 1));  // [col0_div[1], col1[1]]

      _mm_store_pd(&irow0[kc], irow0_kc);
      _mm_store_pd(&irow1[kc], irow1_kc);

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
    for (size_t p = 0;  p < nb; p++) {
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

#endif
