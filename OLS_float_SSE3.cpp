#include "OLS_float_SSE3.hpp"

#ifdef X64_SIMD_AVAILABLE

#if (defined(__GNUC__) || defined(__clang__))
#define SSE3_TARGET __attribute__((target("sse3")))
#else
#define SSE3_TARGET
#endif

#include <cmath>

OLS_float_SSE3::OLS_float_SSE3(size_t n, size_t solveInterval, float lambda, float nu)
  : OLS_float_Scalar(n, solveInterval, lambda, nu) {
}

SSE3_TARGET
float OLS_float_SSE3::predict() {
  assert(featureIndex == n);
  featureIndex = 0;

  // Prediction: y = xᵀ·w
  __m128 v_sum = _mm_setzero_ps();
  for (size_t i = 0; i < nPadded; i += 4) {
    __m128 v_x = _mm_load_ps(&x[i]);
    __m128 v_w = _mm_load_ps(&w[i]);
    v_sum = _mm_add_ps(v_sum, _mm_mul_ps(v_x, v_w));
  }

  // Horizontal sum for floats
  v_sum = _mm_add_ps(v_sum, _mm_movehl_ps(v_sum, v_sum));
  v_sum = _mm_add_ss(v_sum, _mm_shuffle_ps(v_sum, v_sum, 0x55));
  float sum = _mm_cvtss_f32(v_sum);

  return sum;
}

SSE3_TARGET
bool OLS_float_SSE3::factor() {

  // Copy lower triangle only + apply regularization
  size_t i_n = 0;
  for (size_t i = 0; i < n; i++) {

    // Copy elements below diagonal
    size_t j_end = i & ~3ULL;
    size_t j = 0;
    for (; j < j_end; j += 4) {
      __m128 src = _mm_load_ps(&mCovariance[i_n + j]);
      _mm_store_ps(&mCholesky[i_n + j], src);
    }

    // Handle remaining elements before diagonal
    for (; j < i; j++) {
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
    // Initialize accumulators as vectors
    __m128 krow0_acc = _mm_load_ss(&krow0[kc]);  // [krow0_kc0, 0, 0, 0]
    __m128 krow1_acc = _mm_setr_ps(krow1[kc], krow1[kc + 1], 0.0f, 0.0f);  // [krow1_kc0, krow1_kc1, 0, 0]
    __m128 krow2_acc = _mm_loadu_ps(&krow2[kc]);  // Load first 3 + junk
    __m128 krow3_acc = _mm_load_ps(&krow3[kc]);

    for (size_t p = 0; p < k; p++) {
      size_t pc = 4 * p;

      __m128 krow0_vals = _mm_load_ps(&krow0[pc]);
      __m128 krow1_vals = _mm_load_ps(&krow1[pc]);
      __m128 krow2_vals = _mm_load_ps(&krow2[pc]);
      __m128 krow3_vals = _mm_load_ps(&krow3[pc]);

      // Compute all updates
      __m128 prod0 = _mm_mul_ps(krow0_vals, krow0_vals);
      __m128 prod1a = _mm_mul_ps(krow1_vals, krow0_vals);
      __m128 prod1b = _mm_mul_ps(krow1_vals, krow1_vals);
      __m128 prod2a = _mm_mul_ps(krow2_vals, krow0_vals);
      __m128 prod2b = _mm_mul_ps(krow2_vals, krow1_vals);
      __m128 prod2c = _mm_mul_ps(krow2_vals, krow2_vals);
      __m128 prod3a = _mm_mul_ps(krow3_vals, krow0_vals);
      __m128 prod3b = _mm_mul_ps(krow3_vals, krow1_vals);
      __m128 prod3c = _mm_mul_ps(krow3_vals, krow2_vals);
      __m128 prod3d = _mm_mul_ps(krow3_vals, krow3_vals);

      // Horizontal adds
      __m128 sum0 = _mm_hadd_ps(prod0, prod0);
      sum0 = _mm_hadd_ps(sum0, sum0);

      __m128 sum1 = _mm_hadd_ps(prod1a, prod1b);
      sum1 = _mm_hadd_ps(sum1, sum1);

      __m128 sum2a = _mm_hadd_ps(prod2a, prod2b);
      __m128 sum2b = _mm_hadd_ps(prod2c, prod2c);
      __m128 sum2 = _mm_hadd_ps(sum2a, sum2b);

      __m128 sum3a = _mm_hadd_ps(prod3a, prod3b);
      __m128 sum3b = _mm_hadd_ps(prod3c, prod3d);
      __m128 sum3 = _mm_hadd_ps(sum3a, sum3b);

      // Accumulate
      krow0_acc = _mm_sub_ss(krow0_acc, sum0);
      krow1_acc = _mm_sub_ps(krow1_acc, sum1);
      krow2_acc = _mm_sub_ps(krow2_acc, sum2);
      krow3_acc = _mm_sub_ps(krow3_acc, sum3);
    }

    // Store results back
    _mm_store_ss(&krow0[kc], krow0_acc);
    _mm_storel_pi((__m64*) & krow1[kc], krow1_acc);

    float temp2[4];
    _mm_store_ps(temp2, krow2_acc);
    krow2[kc] = temp2[0];
    krow2[kc + 1] = temp2[1];
    krow2[kc + 2] = temp2[2];

    _mm_store_ps(&krow3[kc], krow3_acc);

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

    __m128 d00_vec = _mm_set1_ps(d00);
    __m128 d10_vec = _mm_set1_ps(d10);
    __m128 d11_vec = _mm_set1_ps(d11);
    __m128 d20_vec = _mm_set1_ps(d20);
    __m128 d21_vec = _mm_set1_ps(d21);
    __m128 d22_vec = _mm_set1_ps(d22);
    __m128 d30_vec = _mm_set1_ps(d30);
    __m128 d31_vec = _mm_set1_ps(d31);
    __m128 d32_vec = _mm_set1_ps(d32);
    __m128 d33_vec = _mm_set1_ps(d33);

    // --------------------------------------------------
    // 3) Panel update + TRSM (4x4 blocks)
    // --------------------------------------------------
    float* irow0 = krow0 + 4 * nPadded;
    float* irow1 = irow0 + nPadded;
    float* irow2 = irow1 + nPadded;
    float* irow3 = irow2 + nPadded;

    for (size_t i = k + 1; i < nb; i++) {

      // Load the 4x4 block we're updating
      __m128 irow0_kc = _mm_load_ps(&irow0[kc]);
      __m128 irow1_kc = _mm_load_ps(&irow1[kc]);
      __m128 irow2_kc = _mm_load_ps(&irow2[kc]);
      __m128 irow3_kc = _mm_load_ps(&irow3[kc]);

      for (size_t p = 0; p < k; p++) {
        size_t pc = 4 * p;

        // Load 4x4 blocks from each irow
        __m128 aik_row0 = _mm_load_ps(&irow0[pc]);
        __m128 aik_row1 = _mm_load_ps(&irow1[pc]);
        __m128 aik_row2 = _mm_load_ps(&irow2[pc]);
        __m128 aik_row3 = _mm_load_ps(&irow3[pc]);

        // Load 4x4 block from krows
        __m128 b_row0 = _mm_load_ps(&krow0[pc]);
        __m128 b_row1 = _mm_load_ps(&krow1[pc]);
        __m128 b_row2 = _mm_load_ps(&krow2[pc]);
        __m128 b_row3 = _mm_load_ps(&krow3[pc]);

        // Update irow0_kc: 4 dot products
        __m128 prod0a = _mm_mul_ps(aik_row0, b_row0);
        __m128 prod0b = _mm_mul_ps(aik_row0, b_row1);
        __m128 prod0c = _mm_mul_ps(aik_row0, b_row2);
        __m128 prod0d = _mm_mul_ps(aik_row0, b_row3);
        __m128 sum0a = _mm_hadd_ps(prod0a, prod0b);
        __m128 sum0b = _mm_hadd_ps(prod0c, prod0d);
        __m128 sum0 = _mm_hadd_ps(sum0a, sum0b);
        irow0_kc = _mm_sub_ps(irow0_kc, sum0);

        // Update irow1_kc
        __m128 prod1a = _mm_mul_ps(aik_row1, b_row0);
        __m128 prod1b = _mm_mul_ps(aik_row1, b_row1);
        __m128 prod1c = _mm_mul_ps(aik_row1, b_row2);
        __m128 prod1d = _mm_mul_ps(aik_row1, b_row3);
        __m128 sum1a = _mm_hadd_ps(prod1a, prod1b);
        __m128 sum1b = _mm_hadd_ps(prod1c, prod1d);
        __m128 sum1 = _mm_hadd_ps(sum1a, sum1b);
        irow1_kc = _mm_sub_ps(irow1_kc, sum1);

        // Update irow2_kc
        __m128 prod2a = _mm_mul_ps(aik_row2, b_row0);
        __m128 prod2b = _mm_mul_ps(aik_row2, b_row1);
        __m128 prod2c = _mm_mul_ps(aik_row2, b_row2);
        __m128 prod2d = _mm_mul_ps(aik_row2, b_row3);
        __m128 sum2a = _mm_hadd_ps(prod2a, prod2b);
        __m128 sum2b = _mm_hadd_ps(prod2c, prod2d);
        __m128 sum2 = _mm_hadd_ps(sum2a, sum2b);
        irow2_kc = _mm_sub_ps(irow2_kc, sum2);

        // Update irow3_kc
        __m128 prod3a = _mm_mul_ps(aik_row3, b_row0);
        __m128 prod3b = _mm_mul_ps(aik_row3, b_row1);
        __m128 prod3c = _mm_mul_ps(aik_row3, b_row2);
        __m128 prod3d = _mm_mul_ps(aik_row3, b_row3);
        __m128 sum3a = _mm_hadd_ps(prod3a, prod3b);
        __m128 sum3b = _mm_hadd_ps(prod3c, prod3d);
        __m128 sum3 = _mm_hadd_ps(sum3a, sum3b);
        irow3_kc = _mm_sub_ps(irow3_kc, sum3);
      }

      // TRSM (triangular solve for 4x4 block)
      // Extract columns
      __m128 col0 = _mm_shuffle_ps(irow0_kc, irow1_kc, _MM_SHUFFLE(0, 0, 0, 0));
      col0 = _mm_shuffle_ps(col0, _mm_shuffle_ps(irow2_kc, irow3_kc, _MM_SHUFFLE(0, 0, 0, 0)), _MM_SHUFFLE(2, 0, 2, 0));

      __m128 col1 = _mm_shuffle_ps(irow0_kc, irow1_kc, _MM_SHUFFLE(1, 1, 1, 1));
      col1 = _mm_shuffle_ps(col1, _mm_shuffle_ps(irow2_kc, irow3_kc, _MM_SHUFFLE(1, 1, 1, 1)), _MM_SHUFFLE(2, 0, 2, 0));

      __m128 col2 = _mm_shuffle_ps(irow0_kc, irow1_kc, _MM_SHUFFLE(2, 2, 2, 2));
      col2 = _mm_shuffle_ps(col2, _mm_shuffle_ps(irow2_kc, irow3_kc, _MM_SHUFFLE(2, 2, 2, 2)), _MM_SHUFFLE(2, 0, 2, 0));

      __m128 col3 = _mm_shuffle_ps(irow0_kc, irow1_kc, _MM_SHUFFLE(3, 3, 3, 3));
      col3 = _mm_shuffle_ps(col3, _mm_shuffle_ps(irow2_kc, irow3_kc, _MM_SHUFFLE(3, 3, 3, 3)), _MM_SHUFFLE(2, 0, 2, 0));

      // Solve column 0
      col0 = _mm_div_ps(col0, d00_vec);

      // Solve column 1
      col1 = _mm_sub_ps(col1, _mm_mul_ps(d10_vec, col0));
      col1 = _mm_div_ps(col1, d11_vec);

      // Solve column 2
      col2 = _mm_sub_ps(col2, _mm_mul_ps(d20_vec, col0));
      col2 = _mm_sub_ps(col2, _mm_mul_ps(d21_vec, col1));
      col2 = _mm_div_ps(col2, d22_vec);

      // Solve column 3
      col3 = _mm_sub_ps(col3, _mm_mul_ps(d30_vec, col0));
      col3 = _mm_sub_ps(col3, _mm_mul_ps(d31_vec, col1));
      col3 = _mm_sub_ps(col3, _mm_mul_ps(d32_vec, col2));
      col3 = _mm_div_ps(col3, d33_vec);

      // Reconstruct rows from columns
      __m128 tmp0 = _mm_unpacklo_ps(col0, col1);
      __m128 tmp1 = _mm_unpacklo_ps(col2, col3);
      __m128 tmp2 = _mm_unpackhi_ps(col0, col1);
      __m128 tmp3 = _mm_unpackhi_ps(col2, col3);

      irow0_kc = _mm_movelh_ps(tmp0, tmp1);
      irow1_kc = _mm_movehl_ps(tmp1, tmp0);
      irow2_kc = _mm_movelh_ps(tmp2, tmp3);
      irow3_kc = _mm_movehl_ps(tmp3, tmp2);

      // Store results
      _mm_store_ps(&irow0[kc], irow0_kc);
      _mm_store_ps(&irow1[kc], irow1_kc);
      _mm_store_ps(&irow2[kc], irow2_kc);
      _mm_store_ps(&irow3[kc], irow3_kc);

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
      __m128 sum_vec = _mm_load_ps(&irow[kc]);

      for (size_t p = 0; p < k; p++) {
        size_t pc = p * 4;

        __m128 a = _mm_load_ps(&irow[pc]);
        __m128 b_row0 = _mm_load_ps(&krow0[pc]);
        __m128 b_row1 = _mm_load_ps(&krow1[pc]);
        __m128 b_row2 = _mm_load_ps(&krow2[pc]);
        __m128 b_row3 = _mm_load_ps(&krow3[pc]);

        __m128 prod0 = _mm_mul_ps(a, b_row0);
        __m128 prod1 = _mm_mul_ps(a, b_row1);
        __m128 prod2 = _mm_mul_ps(a, b_row2);
        __m128 prod3 = _mm_mul_ps(a, b_row3);

        __m128 sum_a = _mm_hadd_ps(prod0, prod1);
        __m128 sum_b = _mm_hadd_ps(prod2, prod3);
        __m128 update = _mm_hadd_ps(sum_a, sum_b);

        sum_vec = _mm_sub_ps(sum_vec, update);
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

      float temp[4];
      _mm_store_ps(temp, sum_vec);

      float col0 = temp[0] / d00;
      float col1 = (temp[1] - d10 * col0) / d11;
      float col2 = (temp[2] - d20 * col0 - d21 * col1) / d22;
      float col3 = (temp[3] - d30 * col0 - d31 * col1 - d32 * col2) / d33;

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

      __m128 sum_acc = _mm_setzero_ps();

      size_t k_end = j & ~3ULL;
      size_t k = 0;
      for (; k < k_end; k += 4) {
        __m128 a = _mm_load_ps(&mCholesky[i_n + k]);
        __m128 b = _mm_load_ps(&mCholesky[j_n + k]);
        sum_acc = _mm_add_ps(sum_acc, _mm_mul_ps(a, b));
      }

      // Horizontal sum
      sum_acc = _mm_add_ps(sum_acc, _mm_movehl_ps(sum_acc, sum_acc));
      sum_acc = _mm_add_ss(sum_acc, _mm_shuffle_ps(sum_acc, sum_acc, 0x55));
      float sum = mCholesky[i_n + j] - _mm_cvtss_f32(sum_acc);

      for (; k < j; k++) {
        sum -= (mCholesky[i_n + k] * mCholesky[j_n + k]);
      }

      mCholesky[i_n + j] = sum / mCholesky[j_n + j];
    }

    // Diagonal: L[i,i]
    __m128 sum_acc = _mm_setzero_ps();

    size_t k_end = i & ~3ULL;
    size_t k = 0;
    for (; k < k_end; k += 4) {
      __m128 a = _mm_load_ps(&mCholesky[i_n + k]);
      sum_acc = _mm_add_ps(sum_acc, _mm_mul_ps(a, a));
    }

    // Horizontal sum
    sum_acc = _mm_add_ps(sum_acc, _mm_movehl_ps(sum_acc, sum_acc));
    sum_acc = _mm_add_ss(sum_acc, _mm_shuffle_ps(sum_acc, sum_acc, 0x55));
    float sum = mCholesky[i_n + i] - _mm_cvtss_f32(sum_acc);

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

#endif
