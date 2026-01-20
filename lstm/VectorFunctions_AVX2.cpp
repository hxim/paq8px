#include "VectorFunctions_AVX2.hpp"

#ifdef X64_SIMD_AVAILABLE

#if (defined(__GNUC__) || defined(__clang__))
#pragma GCC target("avx2")
#endif

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

// Member implementations

void VectorFunctions_AVX2::Copy(float* dst, const float* src, size_t num_floats) {
  for (size_t i = 0; i < num_floats; i += 8) {
    __m256 vec = _mm256_load_ps(src + i);
    _mm256_store_ps(dst + i, vec);
  }
}

void VectorFunctions_AVX2::Zero(float* dst, size_t num_floats) {
  __m256 zeroes = _mm256_setzero_ps();
  for (size_t i = 0; i < num_floats; i += 8) {
    _mm256_store_ps(dst + i, zeroes);
  }
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
  float* to_be_normalized_values,
  float* activations_out,
  float* gamma,
  float* beta,
  float rms_scale)
{
  __m256 const c_half = _mm256_set1_ps(0.5f);
  __m256 const c_27 = _mm256_set1_ps(27.0f);
  __m256 const c_9 = _mm256_set1_ps(9.0f);
  __m256 const c_clip_lower = _mm256_set1_ps(SIGMOID_CLIP_MIN);
  __m256 const c_clip_upper = _mm256_set1_ps(SIGMOID_CLIP_MAX);
  __m256 inv_var_vec = _mm256_set1_ps(rms_scale);

  for (size_t i = 0; i < array_length; i += 8) {

    __m256 pre_activation_vec = _mm256_load_ps(to_be_normalized_values + i);
    pre_activation_vec = _mm256_mul_ps(pre_activation_vec, inv_var_vec);
    _mm256_store_ps(to_be_normalized_values + i, pre_activation_vec);

    __m256 gamma_vec = _mm256_load_ps(gamma + i);
    __m256 beta_vec = _mm256_load_ps(beta + i);
    __m256 x = _mm256_mul_ps(pre_activation_vec, gamma_vec);
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

    _mm256_store_ps(activations_out + i, result);
  }
}

void VectorFunctions_AVX2::NormalizeThenActivate_Tanh(
  size_t array_length,
  float* to_be_normalized_values,
  float* activations_out,
  float* gamma,
  float* beta,
  float rms_scale)
{
  __m256 const c_27 = _mm256_set1_ps(27.0f);
  __m256 const c_9 = _mm256_set1_ps(9.0f);
  __m256 const c_clip_lower = _mm256_set1_ps(TANH_CLIP_MIN);
  __m256 const c_clip_upper = _mm256_set1_ps(TANH_CLIP_MAX);
  __m256 inv_var_vec = _mm256_set1_ps(rms_scale);

  for (size_t i = 0; i < array_length; i += 8) {

    __m256 pre_activation_vec = _mm256_load_ps(to_be_normalized_values + i);
    pre_activation_vec = _mm256_mul_ps(pre_activation_vec, inv_var_vec);
    _mm256_store_ps(to_be_normalized_values + i, pre_activation_vec);

    __m256 gamma_vec = _mm256_load_ps(gamma + i);
    __m256 beta_vec = _mm256_load_ps(beta + i);
    __m256 x = _mm256_mul_ps(pre_activation_vec, gamma_vec);
    x = _mm256_add_ps(x, beta_vec);

    // tanh

    x = _mm256_max_ps(x, c_clip_lower);
    x = _mm256_min_ps(x, c_clip_upper);

    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 numer = _mm256_mul_ps(x, _mm256_add_ps(c_27, x2));
    __m256 denom = _mm256_add_ps(c_27, _mm256_mul_ps(c_9, x2));
    __m256 result = _mm256_div_ps(numer, denom);

    _mm256_store_ps(activations_out + i, result);
  }
}

void VectorFunctions_AVX2::AccumulateLstmGradients(
  size_t hidden_size,
  size_t concatenated_hidden_size,
  size_t vocabulary_size,
  size_t layer_id,
  float* error_on_output,
  float* hidden_gradient_accumulator,
  float* output_weights)
{
  size_t output_layer_offset = layer_id * hidden_size; // layer_id * 200
  
  for (size_t i = 0; i < vocabulary_size; i += 8) {   // 256 iterations, 8 at a time
    // Load 8 errors as a vector
    __m256 errors = _mm256_load_ps(&error_on_output[i]);
    
    // Broadcast each error to its own vector using AVX2 permutevar8x32
    __m256 error_vec0 = _mm256_permutevar8x32_ps(errors, _mm256_set1_epi32(0));
    __m256 error_vec1 = _mm256_permutevar8x32_ps(errors, _mm256_set1_epi32(1));
    __m256 error_vec2 = _mm256_permutevar8x32_ps(errors, _mm256_set1_epi32(2));
    __m256 error_vec3 = _mm256_permutevar8x32_ps(errors, _mm256_set1_epi32(3));
    __m256 error_vec4 = _mm256_permutevar8x32_ps(errors, _mm256_set1_epi32(4));
    __m256 error_vec5 = _mm256_permutevar8x32_ps(errors, _mm256_set1_epi32(5));
    __m256 error_vec6 = _mm256_permutevar8x32_ps(errors, _mm256_set1_epi32(6));
    __m256 error_vec7 = _mm256_permutevar8x32_ps(errors, _mm256_set1_epi32(7));
    
    for (size_t j = 0; j < hidden_size; j += 8) { // 200 iterations, 8 at a time
      size_t base_offset = output_layer_offset + j;
      
      // Load hidden_gradient_accumulatoronce
      __m256 hidden = _mm256_load_ps(&hidden_gradient_accumulator[j]);
      
      // Load from 8 different output_weights rows and accumulate
      hidden = _mm256_add_ps(hidden, _mm256_mul_ps(_mm256_load_ps(&output_weights[base_offset]), error_vec0)); base_offset += concatenated_hidden_size;
      hidden = _mm256_add_ps(hidden, _mm256_mul_ps(_mm256_load_ps(&output_weights[base_offset]), error_vec1)); base_offset += concatenated_hidden_size;
      hidden = _mm256_add_ps(hidden, _mm256_mul_ps(_mm256_load_ps(&output_weights[base_offset]), error_vec2)); base_offset += concatenated_hidden_size;
      hidden = _mm256_add_ps(hidden, _mm256_mul_ps(_mm256_load_ps(&output_weights[base_offset]), error_vec3)); base_offset += concatenated_hidden_size;
      hidden = _mm256_add_ps(hidden, _mm256_mul_ps(_mm256_load_ps(&output_weights[base_offset]), error_vec4)); base_offset += concatenated_hidden_size;
      hidden = _mm256_add_ps(hidden, _mm256_mul_ps(_mm256_load_ps(&output_weights[base_offset]), error_vec5)); base_offset += concatenated_hidden_size;
      hidden = _mm256_add_ps(hidden, _mm256_mul_ps(_mm256_load_ps(&output_weights[base_offset]), error_vec6)); base_offset += concatenated_hidden_size;
      hidden = _mm256_add_ps(hidden, _mm256_mul_ps(_mm256_load_ps(&output_weights[base_offset]), error_vec7));
      
      // Store back to hidden_gradient_accumulator
      _mm256_store_ps(&hidden_gradient_accumulator[j], hidden);
    }
    
    output_layer_offset += concatenated_hidden_size * 8;
  }
}

void VectorFunctions_AVX2::AccumulateLstmLayerGradients(
  size_t hidden_size,
  size_t timestep_offset,
  float* gradient_from_next_timestep,
  float* hidden_gradient_accumulator,
  float* tanh_state,
  float* forget_gate_activations,
  float* cell_candidate_activations,
  float* output_gate_activations,
  float* output_gate_gradients,
  float* cell_state_gradient,
  float* cell_candidate_gradients,
  float* forget_gate_gradients,
  float* last_cell_state)
{
  const __m256 ones = _mm256_set1_ps(1.0f);
  const __m256 zeros = _mm256_setzero_ps();

  for (size_t i = 0; i < hidden_size; i += 8) {
    __m256 stored_err = _mm256_load_ps(&gradient_from_next_timestep[i]);
    __m256 hidden_err = _mm256_load_ps(&hidden_gradient_accumulator[i]);

    // gradient_from_next_timestep[i] += hidden_gradient_accumulator[i]
    stored_err = _mm256_add_ps(stored_err, hidden_err);
    _mm256_store_ps(&gradient_from_next_timestep[i], stored_err);

    // hidden_gradient_accumulator[i] = 0.0f
    _mm256_store_ps(&hidden_gradient_accumulator[i], zeros);

    // Load states from sequence_position offset
    const size_t idx = timestep_offset + i;
    __m256 tanh_v = _mm256_load_ps(&tanh_state[idx]);
    __m256 forget_gate = _mm256_load_ps(&forget_gate_activations[idx]);
    __m256 cell_candidate = _mm256_load_ps(&cell_candidate_activations[idx]);
    __m256 output_gate = _mm256_load_ps(&output_gate_activations[idx]);
    __m256 input_gate = _mm256_sub_ps(ones, forget_gate);

    // output_gate_gradients[i] = tanh_v * gradient_from_next_timestep[i] * output * (1.0f - output)
    __m256 one_minus_output = _mm256_sub_ps(ones, output_gate);
    __m256 og_err = _mm256_mul_ps(tanh_v, stored_err);
    og_err = _mm256_mul_ps(og_err, output_gate);
    og_err = _mm256_mul_ps(og_err, one_minus_output);
    _mm256_store_ps(&output_gate_gradients[i], og_err);

    // cell_state_gradient[i] += gradient_from_next_timestep[i] * output * (1.0f - tanh_v * tanh_v)
    __m256 state_err = _mm256_load_ps(&cell_state_gradient[i]);
    __m256 tanh_sq = _mm256_mul_ps(tanh_v, tanh_v);
    __m256 one_minus_tanh_sq = _mm256_sub_ps(ones, tanh_sq);
    __m256 temp = _mm256_mul_ps(stored_err, output_gate);
    temp = _mm256_mul_ps(temp, one_minus_tanh_sq);
    state_err = _mm256_add_ps(state_err, temp);

    // cell_candidate_gradients[i] = cell_state_gradient[i] * input_gate * (1.0f - inputv * inputv)
    __m256 inputv_sq = _mm256_mul_ps(cell_candidate, cell_candidate);
    __m256 one_minus_inputv_sq = _mm256_sub_ps(ones, inputv_sq);
    __m256 ig_err = _mm256_mul_ps(state_err, input_gate);
    ig_err = _mm256_mul_ps(ig_err, one_minus_inputv_sq);
    _mm256_store_ps(&cell_candidate_gradients[i], ig_err);

    // forget_gate_gradients[i] = (last_cell_state[idx] - inputv) * cell_state_gradient[i] * forget * input_gate
    __m256 last_st = _mm256_load_ps(&last_cell_state[idx]);
    __m256 fg_err = _mm256_sub_ps(last_st, cell_candidate);
    fg_err = _mm256_mul_ps(fg_err, state_err);
    fg_err = _mm256_mul_ps(fg_err, forget_gate);
    fg_err = _mm256_mul_ps(fg_err, input_gate);
    _mm256_store_ps(&forget_gate_gradients[i], fg_err);

    if (timestep_offset > 0) { // sequence_position > 0
      state_err = _mm256_mul_ps(state_err, forget_gate);
      _mm256_store_ps(&gradient_from_next_timestep[i], zeros);
    }

    _mm256_store_ps(&cell_state_gradient[i], state_err);
  }
}

void VectorFunctions_AVX2::BackpropagateErrors(
  size_t len,                       // hidden_size (200)
  size_t base_offset,               // 0 for temporal, hidden_size for spatial
  size_t component_input_dim,    // Layer 0: 200, Layer 1: 400
  float* weights,                   // Weight matrix
  float* pre_activation_gradients,  // Current layer errors
  float* grad_store)                // Where to accumulate gradients
{
  for (size_t i = 0; i < len; i += 8) {
    __m256 grad_vec = _mm256_load_ps(&grad_store[i]);

    // for better precision calculate the sum then add to the existing grads
    __m256 sum = _mm256_setzero_ps();

    size_t weight_idx = base_offset + i;
    for (size_t i = 0; i < len; i++) {
      __m256 g = _mm256_set1_ps(pre_activation_gradients[i]);
      __m256 w = _mm256_load_ps(&weights[weight_idx]);
      __m256 prod = _mm256_mul_ps(g, w);
      sum = _mm256_add_ps(sum, prod);
      weight_idx += component_input_dim;
    }

    grad_vec = _mm256_add_ps(grad_vec, sum);
    _mm256_store_ps(&grad_store[i], grad_vec);
  }
}

void VectorFunctions_AVX2::AccumulateLayerGradients(
  const size_t hidden_size,
  const size_t vocabulary_size,
  const size_t component_input_dim,
  const float* input,
  const float* pre_activation_gradients,
  float* embedding_ptr,
  float* weight_gradients)
{
  for (size_t i = 0; i < hidden_size; i += 4) {
    // Load 4 errors (using only lower half of AVX register)
    __m128 errors_128 = _mm_load_ps(&pre_activation_gradients[i]);
    __m256 errors = _mm256_castps128_ps256(errors_128);

    // Broadcast each error
    __m256 error_vec0 = _mm256_permutevar8x32_ps(errors, _mm256_set1_epi32(0));
    __m256 error_vec1 = _mm256_permutevar8x32_ps(errors, _mm256_set1_epi32(1));
    __m256 error_vec2 = _mm256_permutevar8x32_ps(errors, _mm256_set1_epi32(2));
    __m256 error_vec3 = _mm256_permutevar8x32_ps(errors, _mm256_set1_epi32(3));

    // Extract scalar values from the broadcast vectors (just get first element)
    float e0 = _mm256_cvtss_f32(error_vec0);
    float e1 = _mm256_cvtss_f32(error_vec1);
    float e2 = _mm256_cvtss_f32(error_vec2);
    float e3 = _mm256_cvtss_f32(error_vec3);

    // Update symbol_embeddings gradient
    size_t emb_offset = i * vocabulary_size;
    embedding_ptr[emb_offset] += e0; emb_offset += vocabulary_size;
    embedding_ptr[emb_offset] += e1; emb_offset += vocabulary_size;
    embedding_ptr[emb_offset] += e2; emb_offset += vocabulary_size;
    embedding_ptr[emb_offset] += e3;

    // Update hidden state weight gradients
    size_t update_offset = i * component_input_dim;
    for (size_t j = 0; j < component_input_dim; j += 8) {
      size_t base_offset = update_offset + j;

      __m256 inp = _mm256_load_ps(&input[j]);

      __m256 upd0 = _mm256_load_ps(&weight_gradients[base_offset]);
      upd0 = _mm256_add_ps(upd0, _mm256_mul_ps(inp, error_vec0)); base_offset += component_input_dim;

      __m256 upd1 = _mm256_load_ps(&weight_gradients[base_offset]);
      upd1 = _mm256_add_ps(upd1, _mm256_mul_ps(inp, error_vec1)); base_offset += component_input_dim;

      __m256 upd2 = _mm256_load_ps(&weight_gradients[base_offset]);
      upd2 = _mm256_add_ps(upd2, _mm256_mul_ps(inp, error_vec2)); base_offset += component_input_dim;

      __m256 upd3 = _mm256_load_ps(&weight_gradients[base_offset]);
      upd3 = _mm256_add_ps(upd3, _mm256_mul_ps(inp, error_vec3));

      base_offset = update_offset + j;
      _mm256_store_ps(&weight_gradients[base_offset], upd0); base_offset += component_input_dim;
      _mm256_store_ps(&weight_gradients[base_offset], upd1); base_offset += component_input_dim;
      _mm256_store_ps(&weight_gradients[base_offset], upd2); base_offset += component_input_dim;
      _mm256_store_ps(&weight_gradients[base_offset], upd3);
    }
  }
}

void VectorFunctions_AVX2::AccumulateOutputLayerGradients(
  size_t previous_output_offset,
  float* error_on_output,
  float* output_weight_gradients,
  float* output_bias_gradients,
  const float* hidden_ptr,
  const size_t vocabulary_size,
  const size_t concatenated_hidden_size,
  const size_t input_symbol)
{
  for (size_t i = 0; i < vocabulary_size; i += 4) {
    // Load 4 errors
    __m128 errors_128 = _mm_load_ps(&error_on_output[i]);
    __m256 errors = _mm256_castps128_ps256(errors_128);

    // Broadcast each error
    __m256 error_vec0 = _mm256_permutevar8x32_ps(errors, _mm256_set1_epi32(0));
    __m256 error_vec1 = _mm256_permutevar8x32_ps(errors, _mm256_set1_epi32(1));
    __m256 error_vec2 = _mm256_permutevar8x32_ps(errors, _mm256_set1_epi32(2));
    __m256 error_vec3 = _mm256_permutevar8x32_ps(errors, _mm256_set1_epi32(3));

    // Update bias (vectorized)
    __m128 bias = _mm_load_ps(&output_bias_gradients[i]);
    bias = _mm_add_ps(bias, errors_128);
    _mm_store_ps(&output_bias_gradients[i], bias);

    // Update output layer weights
    size_t output_offset = i * concatenated_hidden_size;
    for (size_t j = 0; j < concatenated_hidden_size; j += 8) {
      size_t base_offset = output_offset + j;

      __m256 hidden = _mm256_load_ps(&hidden_ptr[j]);

      __m256 out = _mm256_load_ps(&output_weight_gradients[base_offset]);
      out = _mm256_add_ps(out, _mm256_mul_ps(hidden, error_vec0)); base_offset += concatenated_hidden_size;

      __m256 out1 = _mm256_load_ps(&output_weight_gradients[base_offset]);
      out1 = _mm256_add_ps(out1, _mm256_mul_ps(hidden, error_vec1)); base_offset += concatenated_hidden_size;

      __m256 out2 = _mm256_load_ps(&output_weight_gradients[base_offset]);
      out2 = _mm256_add_ps(out2, _mm256_mul_ps(hidden, error_vec2)); base_offset += concatenated_hidden_size;

      __m256 out3 = _mm256_load_ps(&output_weight_gradients[base_offset]);
      out3 = _mm256_add_ps(out3, _mm256_mul_ps(hidden, error_vec3));

      base_offset = output_offset + j;
      _mm256_store_ps(&output_weight_gradients[base_offset], out); base_offset += concatenated_hidden_size;
      _mm256_store_ps(&output_weight_gradients[base_offset], out1); base_offset += concatenated_hidden_size;
      _mm256_store_ps(&output_weight_gradients[base_offset], out2); base_offset += concatenated_hidden_size;
      _mm256_store_ps(&output_weight_gradients[base_offset], out3);
    }
  }
}

float VectorFunctions_AVX2::ComputeMaxLogit(
  float* result,
  size_t result_length
)
{
  __m256 max_logit_vec = _mm256_set1_ps(negative_infinity);

  for (size_t i = 0; i < result_length; i += 16) {
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
  float* output_weights,
  float* output,
  float* output_bias,
  size_t const all_layer_inputs, // 200*2 = 400
  size_t const vocabulary_size, // 256
  size_t const output_offset)
{
  // Compute logits via dot products
  for (size_t i = 0; i < vocabulary_size; i++) {  // 256 iterations
    logits[output_offset + i] = DotProduct(
      &hidden[0],
      &output_weights[i * all_layer_inputs],
      all_layer_inputs // 400
    ) + output_bias[i];
  }

  // Find max logit for numerical stability
  float max_logit = ComputeMaxLogit(&logits[output_offset], vocabulary_size);

  // Compute softmax
  Softmax(
    &logits[output_offset],
    &output[output_offset],
    vocabulary_size,  // 256
    max_logit);
}

void VectorFunctions_AVX2::Softmax(
  float* logits,
  float* probs,
  size_t len,
  float max_logit)
{
  // Constants for exponential
  const __m256 INV_LN2 = _mm256_set1_ps(0x1.715476p+0f);
  const __m256 LN2_HI = _mm256_set1_ps(0x1.62e400p-1f);
  const __m256 LN2_LO = _mm256_set1_ps(0x1.7f7d1cp-20f);
  const __m256 c2 = _mm256_set1_ps(0x1.fffe24p-2f);
  const __m256 c3 = _mm256_set1_ps(0x1.5554acp-3f);
  const __m256 c4 = _mm256_set1_ps(0x1.5713a4p-5f);
  const __m256 c5 = _mm256_set1_ps(0x1.12266ap-7f);
  const __m256 one_vec = _mm256_set1_ps(1.0f);
  const __m256i MAGIC_BIAS = _mm256_set1_epi32(12582912);
  const __m256 MAGIC_BIAS_FLOAT = _mm256_set1_ps(12582912.0f);
  const __m256i c127 = _mm256_set1_epi32(127);
  const __m256 cplus87 = _mm256_set1_ps(87.0f);
  const __m256 cminus87 = _mm256_set1_ps(-87.0f);
  const __m256 maxlogit_vec = _mm256_set1_ps(max_logit);

  __m256 expsum_vec = _mm256_setzero_ps();

  for (size_t i = 0; i < len; i += 8) {
    //prepare
    __m256 logits_vec = _mm256_load_ps(&logits[i]);
    logits_vec = _mm256_sub_ps(logits_vec, maxlogit_vec);

    //exp
    logits_vec = _mm256_min_ps(logits_vec, cplus87);
    logits_vec = _mm256_max_ps(logits_vec, cminus87);

    __m256 z = _mm256_mul_ps(logits_vec, INV_LN2);

    __m256 t = _mm256_add_ps(z, MAGIC_BIAS_FLOAT);
    __m256i n = _mm256_sub_epi32(_mm256_cvttps_epi32(t), MAGIC_BIAS);
    t = _mm256_sub_ps(t, MAGIC_BIAS_FLOAT);

    __m256 r = _mm256_sub_ps(logits_vec, _mm256_mul_ps(t, LN2_HI));
    r = _mm256_sub_ps(r, _mm256_mul_ps(t, LN2_LO));

    __m256 r2 = _mm256_mul_ps(r, r);
    __m256 q2 = _mm256_add_ps(c2, _mm256_mul_ps(c3, r));
    __m256 q4 = _mm256_add_ps(c4, _mm256_mul_ps(c5, r));
    __m256 p = _mm256_add_ps(_mm256_mul_ps(_mm256_add_ps(_mm256_mul_ps(q4, r2), q2), r2), _mm256_add_ps(r, one_vec));

    __m256i expbits = _mm256_add_epi32(n, c127);
    __m256i bits = _mm256_slli_epi32(expbits, 23);
    __m256 two_n = _mm256_castsi256_ps(bits);

    __m256 result_vec = _mm256_mul_ps(p, two_n);

    //finalize, store
    _mm256_store_ps(&probs[i], result_vec);
    expsum_vec = _mm256_add_ps(expsum_vec, result_vec);
  }

  float expsum = horizontal_sum(expsum_vec);
  float expsum_reciprocal = 1.0f / expsum;
  __m256 expsum_reciprocal_vec = _mm256_set1_ps(expsum_reciprocal);

  for (size_t i = 0; i < len; i += 8) {
    __m256 softmax_probs_vec = _mm256_load_ps(&probs[i]);
    __m256 result_vec = _mm256_mul_ps(softmax_probs_vec, expsum_reciprocal_vec);
    _mm256_store_ps(&probs[i], result_vec);
  }
}

#endif

