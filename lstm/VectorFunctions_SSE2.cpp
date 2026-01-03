#pragma GCC target("sse2")

#include "VectorFunctions_SSE2.hpp"

#ifdef X64_SIMD_AVAILABLE

// Static helper functions

static float horizontal_sum(__m128 sum_low, __m128 sum_high)
{
  __m128 sum128 = _mm_add_ps(sum_low, sum_high);
  sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
  sum128 = _mm_add_ss(sum128, _mm_shuffle_ps(sum128, sum128, 0x55));
  float sum = _mm_cvtss_f32(sum128);
  return sum;
}

float VectorFunctions_SSE2::DotProduct(
  float const* x1,
  float const* x2,
  size_t const len)
{
  __m128 sum_low = _mm_setzero_ps();
  __m128 sum_high = _mm_setzero_ps();

  for (size_t i = 0; i < len; i += 8) {
    __m128 a0 = _mm_load_ps(x1 + i);
    __m128 b0 = _mm_load_ps(x2 + i);
    __m128 prod0 = _mm_mul_ps(a0, b0);
    sum_low = _mm_add_ps(sum_low, prod0);

    __m128 a1 = _mm_load_ps(x1 + i + 4);
    __m128 b1 = _mm_load_ps(x2 + i + 4);
    __m128 prod1 = _mm_mul_ps(a1, b1);
    sum_high = _mm_add_ps(sum_high, prod1);
  }

  return horizontal_sum(sum_low, sum_high);
}

float VectorFunctions_SSE2::SumOfSquares(float* array, size_t array_length)
{
  __m128 sum_vec0 = _mm_setzero_ps();
  __m128 sum_vec1 = _mm_setzero_ps();

  for (size_t i = 0; i < array_length; i += 8)
  {
    __m128 x0 = _mm_load_ps(&array[i]);
    sum_vec0 = _mm_add_ps(sum_vec0, _mm_mul_ps(x0, x0));
    __m128 x1 = _mm_load_ps(&array[i + 4]);
    sum_vec1 = _mm_add_ps(sum_vec1, _mm_mul_ps(x1, x1));
  }

  return horizontal_sum(sum_vec0, sum_vec1);
}

void VectorFunctions_SSE2::NormalizeThenActivate_Sigmoid(
  size_t array_length,
  float* norm,
  float* state,
  float* gamma,
  float* beta,
  float inverse_variance)
{
  __m128 const c_half = _mm_set1_ps(0.5f);
  __m128 const c_27 = _mm_set1_ps(27.0f);
  __m128 const c_9 = _mm_set1_ps(9.0f);
  __m128 const c_clip_lower = _mm_set1_ps(SIGMOID_CLIP_MIN);
  __m128 const c_clip_upper = _mm_set1_ps(SIGMOID_CLIP_MAX);
  __m128 inv_var_vec = _mm_set1_ps(inverse_variance);

  for (size_t i = 0; i < array_length; i += 4) {

    __m128 norm_vec = _mm_load_ps(norm + i);
    norm_vec = _mm_mul_ps(norm_vec, inv_var_vec);
    _mm_store_ps(norm + i, norm_vec);

    __m128 gamma_vec = _mm_load_ps(gamma + i);
    __m128 beta_vec = _mm_load_ps(beta + i);
    __m128 x = _mm_mul_ps(norm_vec, gamma_vec);
    x = _mm_add_ps(x, beta_vec);

    // sigmoid

    x = _mm_max_ps(x, c_clip_lower);
    x = _mm_min_ps(x, c_clip_upper);

    __m128 u = _mm_mul_ps(x, c_half);
    __m128 u2 = _mm_mul_ps(u, u);

    __m128 numer = _mm_mul_ps(u, _mm_add_ps(c_27, u2));
    __m128 denom = _mm_add_ps(c_27, _mm_mul_ps(c_9, u2));
    __m128 tanh_val = _mm_div_ps(numer, denom);

    // sigmoid = 0.5 * (1 + tanh) = 0.5 + 0.5*tanh
    __m128 result = _mm_add_ps(c_half, _mm_mul_ps(c_half, tanh_val));

    _mm_store_ps(state + i, result);
  }
}

void VectorFunctions_SSE2::NormalizeThenActivate_Tanh(
  size_t array_length,
  float* norm,
  float* state,
  float* gamma,
  float* beta,
  float inverse_variance)
{
  __m128 const c_27 = _mm_set1_ps(27.0f);
  __m128 const c_9 = _mm_set1_ps(9.0f);
  __m128 const c_clip_lower = _mm_set1_ps(TANH_CLIP_MIN);
  __m128 const c_clip_upper = _mm_set1_ps(TANH_CLIP_MAX);
  __m128 inv_var_vec = _mm_set1_ps(inverse_variance);

  for (size_t i = 0; i < array_length; i += 4) {

    __m128 norm_vec = _mm_load_ps(norm + i);
    norm_vec = _mm_mul_ps(norm_vec, inv_var_vec);
    _mm_store_ps(norm + i, norm_vec);

    __m128 gamma_vec = _mm_load_ps(gamma + i);
    __m128 beta_vec = _mm_load_ps(beta + i);
    __m128 x = _mm_mul_ps(norm_vec, gamma_vec);
    x = _mm_add_ps(x, beta_vec);

    // tanh

    x = _mm_max_ps(x, c_clip_lower);
    x = _mm_min_ps(x, c_clip_upper);

    __m128 x2 = _mm_mul_ps(x, x);
    __m128 numer = _mm_mul_ps(x, _mm_add_ps(c_27, x2));
    __m128 denom = _mm_add_ps(c_27, _mm_mul_ps(c_9, x2));
    __m128 result = _mm_div_ps(numer, denom);

    _mm_store_ps(state + i, result);
  }
}

void VectorFunctions_SSE2::BackpropagateErrors(
  size_t len,
  size_t base_offset,
  size_t hidden_size,
  float* weights,
  float* error,
  float* grad_store)
{
  for (size_t i = 0; i < len; i += 8) {
    __m128 grad0_vec = _mm_load_ps(&grad_store[i]);
    __m128 grad1_vec = _mm_load_ps(&grad_store[i + 4]);

    // for better precision calculate the sum then add to the existing grads
    __m128 sum0 = _mm_setzero_ps(); 
    __m128 sum1 = _mm_setzero_ps();

    size_t weight_idx = base_offset + i;
    for (size_t i = 0; i < len; i++) {
      __m128 ei = _mm_set1_ps(error[i]);

      __m128 w0 = _mm_load_ps(&weights[weight_idx]);
      __m128 prod0 = _mm_mul_ps(ei, w0);
      sum0 = _mm_add_ps(sum0, prod0);

      __m128 w1 = _mm_load_ps(&weights[weight_idx + 4]);
      __m128 prod1 = _mm_mul_ps(ei, w1);
      sum1 = _mm_add_ps(sum1, prod1);

      weight_idx += hidden_size;
    }

    grad0_vec = _mm_add_ps(grad0_vec, sum0);
    grad1_vec = _mm_add_ps(grad1_vec, sum1);

    _mm_store_ps(&grad_store[i], grad0_vec);
    _mm_store_ps(&grad_store[i + 4], grad1_vec);
  }
}

float VectorFunctions_SSE2::ComputeMaxLogit(
  float* result,
  size_t result_length)
{
  __m128 max_logit_vec = _mm_set1_ps(negative_infinity);

  for (size_t i = 0; i < result_length; i += 16)
  {
    __m128 v0 = _mm_load_ps(result + i + 0);
    __m128 v1 = _mm_load_ps(result + i + 4);
    __m128 v2 = _mm_load_ps(result + i + 8);
    __m128 v3 = _mm_load_ps(result + i + 12);

    max_logit_vec = _mm_max_ps(max_logit_vec, v0);
    max_logit_vec = _mm_max_ps(max_logit_vec, v1);
    max_logit_vec = _mm_max_ps(max_logit_vec, v2);
    max_logit_vec = _mm_max_ps(max_logit_vec, v3);
  }

  // Perform horizontal reduction to find the max value
  __m128 shuf = _mm_shuffle_ps(max_logit_vec, max_logit_vec, _MM_SHUFFLE(2, 3, 0, 1));
  max_logit_vec = _mm_max_ps(max_logit_vec, shuf);
  shuf = _mm_shuffle_ps(max_logit_vec, max_logit_vec, _MM_SHUFFLE(1, 0, 3, 2));
  max_logit_vec = _mm_max_ps(max_logit_vec, shuf);
  shuf = _mm_shuffle_ps(max_logit_vec, max_logit_vec, _MM_SHUFFLE(0, 1, 2, 3));
  max_logit_vec = _mm_max_ps(max_logit_vec, shuf);

  float maxlogit = _mm_cvtss_f32(max_logit_vec);
  return maxlogit;
}

void VectorFunctions_SSE2::MatvecThenSoftmax(
  float* hidden,
  float* logits,
  float* output_layer,
  float* output,
  float* output_bias,
  size_t const hidden_size,
  size_t const output_size,
  size_t const output_offset)
{
  // Compute logits via dot products
  for (size_t i = 0; i < output_size; i++) {
    logits[output_offset + i] = DotProduct(
      &hidden[0],
      &output_layer[(output_offset + i) * hidden_size],
      hidden_size
    ) + output_bias[i];
  }

  // Find max logit for numerical stability
  float max_logit = ComputeMaxLogit(&logits[output_offset], output_size);

  // Compute softmax
  Softmax(
    &logits[output_offset],
    &output[output_offset],
    output_size,
    max_logit);
}


void VectorFunctions_SSE2::Softmax(
  float* logits,
  float* probs,
  size_t len,
  float max_logit)
{
  // Constants for exponential
  const __m128 INV_LN2 = _mm_set1_ps(0x1.715476p+0f);
  const __m128 LN2_HI = _mm_set1_ps(0x1.62e400p-1f);
  const __m128 LN2_LO = _mm_set1_ps(0x1.7f7d1cp-20f);
  const __m128 c2 = _mm_set1_ps(0x1.fffe24p-2f);
  const __m128 c3 = _mm_set1_ps(0x1.5554acp-3f);
  const __m128 c4 = _mm_set1_ps(0x1.5713a4p-5f);
  const __m128 c5 = _mm_set1_ps(0x1.12266ap-7f);
  const __m128 one_vec = _mm_set1_ps(1.0f);
  const __m128i MAGIC_BIAS = _mm_set1_epi32(12582912);
  const __m128i c127 = _mm_set1_epi32(127);
  const __m128 cplus87 = _mm_set1_ps(87.0f);
  const __m128 cminus87 = _mm_set1_ps(-87.0f);
  const __m128 maxlogit_vec = _mm_set1_ps(max_logit);

  __m128 expsum_vec1 = _mm_setzero_ps();
  __m128 expsum_vec2 = _mm_setzero_ps();

  for (size_t i = 0; i < len;) {
    // First block of 4
    {
      //prepare
      __m128 logits_vec = _mm_load_ps(&logits[i]);
      logits_vec = _mm_sub_ps(logits_vec, maxlogit_vec);

      //exp
      logits_vec = _mm_min_ps(logits_vec, cplus87);
      logits_vec = _mm_max_ps(logits_vec, cminus87);

      __m128 z = _mm_mul_ps(logits_vec, INV_LN2);

      const __m128 m = _mm_cvtepi32_ps(MAGIC_BIAS); //due to register pressure
      __m128 t = _mm_add_ps(z, m);
      __m128i n = _mm_sub_epi32(_mm_cvttps_epi32(t), MAGIC_BIAS);
      t = _mm_sub_ps(t, m);

      __m128 r = _mm_sub_ps(logits_vec, _mm_mul_ps(t, LN2_HI));
      r = _mm_sub_ps(r, _mm_mul_ps(t, LN2_LO));

      __m128 r2 = _mm_mul_ps(r, r);
      __m128 q2 = _mm_add_ps(c2, _mm_mul_ps(c3, r));
      __m128 q4 = _mm_add_ps(c4, _mm_mul_ps(c5, r));
      __m128 p = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(q4, r2), q2), r2), _mm_add_ps(r, one_vec));

      __m128i expbits = _mm_add_epi32(n, c127);
      __m128i bits = _mm_slli_epi32(expbits, 23);
      __m128 two_n = _mm_castsi128_ps(bits);

      __m128 result_vec = _mm_mul_ps(p, two_n);

      //finalize, store
      _mm_store_ps(&probs[i], result_vec);
      expsum_vec1 = _mm_add_ps(expsum_vec1, result_vec);
    }
    i += 4;

    // Second block of 4

    {
      //prepare
      __m128 logits_vec = _mm_load_ps(&logits[i]);
      logits_vec = _mm_sub_ps(logits_vec, maxlogit_vec);

      //exp
      logits_vec = _mm_min_ps(logits_vec, cplus87);
      logits_vec = _mm_max_ps(logits_vec, cminus87);

      __m128 z = _mm_mul_ps(logits_vec, INV_LN2);

      const __m128 m = _mm_cvtepi32_ps(MAGIC_BIAS); //due to register pressure
      __m128 t = _mm_add_ps(z, m);
      __m128i n = _mm_sub_epi32(_mm_cvttps_epi32(t), MAGIC_BIAS);
      t = _mm_sub_ps(t, m);

      __m128 r = _mm_sub_ps(logits_vec, _mm_mul_ps(t, LN2_HI));
      r = _mm_sub_ps(r, _mm_mul_ps(t, LN2_LO));

      __m128 r2 = _mm_mul_ps(r, r);
      __m128 q2 = _mm_add_ps(c2, _mm_mul_ps(c3, r));
      __m128 q4 = _mm_add_ps(c4, _mm_mul_ps(c5, r));
      __m128 p = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(q4, r2), q2), r2), _mm_add_ps(r, one_vec));

      __m128i expbits = _mm_add_epi32(n, c127);
      __m128i bits = _mm_slli_epi32(expbits, 23);
      __m128  two_n = _mm_castsi128_ps(bits);

      __m128 result_vec = _mm_mul_ps(p, two_n);

      //finalize, store
      _mm_store_ps(&probs[i], result_vec);
      expsum_vec2 = _mm_add_ps(expsum_vec2, result_vec);
    }

    i += 4;
  }

  float expsum = horizontal_sum(expsum_vec1, expsum_vec2);
  float expsum_reciprocal = 1.0f / expsum;
  __m128 expsum_reciprocal_vec = _mm_set1_ps(expsum_reciprocal);

  for (size_t i = 0; i < len; i += 4)
  {
    __m128 softmax_probs_vec = _mm_load_ps(&probs[i]);
    __m128 result_vec = _mm_mul_ps(softmax_probs_vec, expsum_reciprocal_vec);
    _mm_store_ps(&probs[i], result_vec);
  }
}

#endif
