#pragma GCC target("avx2")

#include "VectorFunctions_AVX2.hpp"

#ifdef X64_SIMD_AVAILABLE

// Static helper functions

static float horizontal_sum(__m256 sum_vec)
{
  __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
  __m128 sum_low = _mm256_castps256_ps128(sum_vec);
  __m128 sum128 = _mm_add_ps(sum_low, sum_high);
  sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
  sum128 = _mm_add_ss(sum128, _mm_shuffle_ps(sum128, sum128, 0x55));
  float sum = _mm_cvtss_f32(sum128);
  return sum;
}

float VectorFunctions_AVX2::DotProduct(
  float const* x1,
  float const* x2,
  size_t const len)
{
  __m256 sum = _mm256_setzero_ps();

  for (size_t i = 0; i < len; i += 8) {
    __m256 a = _mm256_load_ps(x1 + i);
    __m256 b = _mm256_load_ps(x2 + i);
    __m256 prod = _mm256_mul_ps(a, b);
    sum = _mm256_add_ps(sum, prod);
  }

  return horizontal_sum(sum);
}

float VectorFunctions_AVX2::SumOfSquares(float* array, size_t array_length) {
  __m256 sum_vec = _mm256_setzero_ps();
  
  for (size_t i = 0; i < array_length; i += 8) {
    __m256 x = _mm256_load_ps(&array[i]);
    sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(x, x));
  }
  return horizontal_sum(sum_vec);
}

void VectorFunctions_AVX2::NormalizeThenActivate_Sigmoid(
  size_t array_length,
  float* norm,
  float* state,
  float* gamma,
  float* beta,
  float inverse_variance)
{
  __m256 const c_half = _mm256_set1_ps(0.5f);
  __m256 const c_27 = _mm256_set1_ps(27.0f);
  __m256 const c_9 = _mm256_set1_ps(9.0f);
  __m256 const c_clip_lower = _mm256_set1_ps(SIGMOID_CLIP_MIN);
  __m256 const c_clip_upper = _mm256_set1_ps(SIGMOID_CLIP_MAX);
  __m256 inv_var_vec = _mm256_set1_ps(inverse_variance);

  for (size_t i = 0; i < array_length; i += 8) {

    __m256 norm_vec = _mm256_load_ps(norm + i);
    norm_vec = _mm256_mul_ps(norm_vec, inv_var_vec);
    _mm256_store_ps(norm + i, norm_vec);

    __m256 gamma_vec = _mm256_load_ps(gamma + i);
    __m256 beta_vec = _mm256_load_ps(beta + i);
    __m256 x = _mm256_mul_ps(norm_vec, gamma_vec);
    x = _mm256_add_ps(x, beta_vec);

    // sigmoid

    x = _mm256_max_ps(x, c_clip_lower);
    x = _mm256_min_ps(x, c_clip_upper);

    __m256 u = _mm256_mul_ps(x, c_half);
    __m256 u2 = _mm256_mul_ps(u, u);

    __m256 numer = _mm256_mul_ps(u, _mm256_add_ps(c_27, u2));
    __m256 denom = _mm256_add_ps(c_27, _mm256_mul_ps(c_9, u2));
    __m256 tanh_val = _mm256_div_ps(numer, denom);

    // sigmoid = 0.5 * (1 + tanh) = 0.5 + 0.5*tanh
    __m256 result = _mm256_add_ps(c_half, _mm256_mul_ps(c_half, tanh_val));

    _mm256_store_ps(state + i, result);
  }
}

void VectorFunctions_AVX2::NormalizeThenActivate_Tanh(
  size_t array_length,
  float* norm,
  float* state,
  float* gamma,
  float* beta,
  float inverse_variance)
{
  __m256 const c_27 = _mm256_set1_ps(27.0f);
  __m256 const c_9 = _mm256_set1_ps(9.0f);
  __m256 const c_clip_lower = _mm256_set1_ps(TANH_CLIP_MIN);
  __m256 const c_clip_upper = _mm256_set1_ps(TANH_CLIP_MAX);
  __m256 inv_var_vec = _mm256_set1_ps(inverse_variance);

  for (size_t i = 0; i < array_length; i += 8) {

    __m256 norm_vec = _mm256_load_ps(norm + i);
    norm_vec = _mm256_mul_ps(norm_vec, inv_var_vec);
    _mm256_store_ps(norm + i, norm_vec);

    __m256 gamma_vec = _mm256_load_ps(gamma + i);
    __m256 beta_vec = _mm256_load_ps(beta + i);
    __m256 x = _mm256_mul_ps(norm_vec, gamma_vec);
    x = _mm256_add_ps(x, beta_vec);

    // tanh

    x = _mm256_max_ps(x, c_clip_lower);
    x = _mm256_min_ps(x, c_clip_upper);

    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 numer = _mm256_mul_ps(x, _mm256_add_ps(c_27, x2));
    __m256 denom = _mm256_add_ps(c_27, _mm256_mul_ps(c_9, x2));
    __m256 result = _mm256_div_ps(numer, denom);

    _mm256_store_ps(state + i, result);
  }
}

void VectorFunctions_AVX2::BackpropagateErrors(
  size_t len,
  size_t base_offset,
  size_t hidden_size,
  float* weights,
  float* error,
  float* grad_store)
{
  for (size_t i = 0; i < len; i += 8) {
    __m256 grad_vec = _mm256_load_ps(&grad_store[i]);

    // for better precision calculate the sum then add to the existing grads
    __m256 sum = _mm256_setzero_ps();

    size_t weight_idx = base_offset + i;
    for (size_t i = 0; i < len; i++) {
      __m256 ei = _mm256_set1_ps(error[i]);
      __m256 w = _mm256_load_ps(&weights[weight_idx]);
      __m256 prod = _mm256_mul_ps(ei, w);
      sum = _mm256_add_ps(sum, prod);
      weight_idx += hidden_size;
    }

    grad_vec = _mm256_add_ps(grad_vec, sum);
    _mm256_store_ps(&grad_store[i], grad_vec);
  }
}

float VectorFunctions_AVX2::ComputeMaxLogit(
  float* result,
  size_t result_length
)
{
  __m256 max_logit_vec = _mm256_set1_ps(negative_infinity);

  for (size_t i = 0; i < result_length; i += 16)
  {
    __m256 v0 = _mm256_load_ps(result + i);
    __m256 v1 = _mm256_load_ps(result + i + 8);

    max_logit_vec = _mm256_max_ps(max_logit_vec, v0);
    max_logit_vec = _mm256_max_ps(max_logit_vec, v1);
  }

  // Perform horizontal reduction to find the max value
  __m256 shuf = _mm256_permute2f128_ps(max_logit_vec, max_logit_vec, 1);
  max_logit_vec = _mm256_max_ps(max_logit_vec, shuf);
  max_logit_vec = _mm256_max_ps(max_logit_vec, _mm256_permute_ps(max_logit_vec, _MM_SHUFFLE(2, 3, 0, 1)));
  max_logit_vec = _mm256_max_ps(max_logit_vec, _mm256_permute_ps(max_logit_vec, _MM_SHUFFLE(1, 0, 3, 2)));

  float maxlogit = _mm_cvtss_f32(_mm256_castps256_ps128(max_logit_vec));
  return maxlogit;
}

void VectorFunctions_AVX2::MatvecThenSoftmax(
  float* hidden,
  float* logits,
  float* output_layer,
  float* output,
  size_t const hidden_size, // 200*2 = 400
  size_t const output_size, // 256
  size_t const output_offset)
{
  // Compute logits via dot products
  for (size_t i = 0; i < output_size; i++) {  // 256 iterations
    logits[output_offset + i] = DotProduct(
      &hidden[0],
      &output_layer[(output_offset + i) * hidden_size],
      hidden_size                                   // 400
    );
  }

  // Find max logit for numerical stability
  float max_logit = ComputeMaxLogit(&logits[output_offset], output_size);

  // Compute softmax
  Softmax(
    &logits[output_offset],
    &output[output_offset],
    output_size,  // 256
    max_logit);
}

#endif

