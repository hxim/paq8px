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
  float* pre_norm_values,
  float* activations_out,
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

    __m256 pre_activation_vec = _mm256_load_ps(pre_norm_values + i);
    pre_activation_vec = _mm256_mul_ps(pre_activation_vec, inv_var_vec);
    _mm256_store_ps(pre_norm_values + i, pre_activation_vec);

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
  float* pre_norm_values,
  float* activations_out,
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

    __m256 pre_activation_vec = _mm256_load_ps(pre_norm_values + i);
    pre_activation_vec = _mm256_mul_ps(pre_activation_vec, inv_var_vec);
    _mm256_store_ps(pre_norm_values + i, pre_activation_vec);

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
  size_t num_cells,
  size_t hidden_size,
  size_t vocabulary_size,
  size_t layer_id,
  float* error_on_output,
  float* hidden_gradient,
  float* output_weights)
{
  size_t output_layer_offset = layer_id * num_cells; // layer_id * 200
  
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
    
    for (size_t j = 0; j < num_cells; j += 8) { // 200 iterations, 8 at a time
      size_t base_offset = output_layer_offset + j;
      
      // Load hidden_gradient once
      __m256 hidden = _mm256_load_ps(&hidden_gradient[j]);
      
      // Load from 8 different output_weights rows and accumulate
      hidden = _mm256_add_ps(hidden, _mm256_mul_ps(_mm256_load_ps(&output_weights[base_offset]), error_vec0)); base_offset += hidden_size;
      hidden = _mm256_add_ps(hidden, _mm256_mul_ps(_mm256_load_ps(&output_weights[base_offset]), error_vec1)); base_offset += hidden_size;
      hidden = _mm256_add_ps(hidden, _mm256_mul_ps(_mm256_load_ps(&output_weights[base_offset]), error_vec2)); base_offset += hidden_size;
      hidden = _mm256_add_ps(hidden, _mm256_mul_ps(_mm256_load_ps(&output_weights[base_offset]), error_vec3)); base_offset += hidden_size;
      hidden = _mm256_add_ps(hidden, _mm256_mul_ps(_mm256_load_ps(&output_weights[base_offset]), error_vec4)); base_offset += hidden_size;
      hidden = _mm256_add_ps(hidden, _mm256_mul_ps(_mm256_load_ps(&output_weights[base_offset]), error_vec5)); base_offset += hidden_size;
      hidden = _mm256_add_ps(hidden, _mm256_mul_ps(_mm256_load_ps(&output_weights[base_offset]), error_vec6)); base_offset += hidden_size;
      hidden = _mm256_add_ps(hidden, _mm256_mul_ps(_mm256_load_ps(&output_weights[base_offset]), error_vec7));
      
      // Store back to hidden_gradient
      _mm256_store_ps(&hidden_gradient[j], hidden);
    }
    
    output_layer_offset += hidden_size * 8;
  }
}

void VectorFunctions_AVX2::AccumulateLstmLayerGradients(
  size_t num_cells,
  size_t sequence_position_offset,
  float* temporal_hidden_gradient,
  float* hidden_gradient,
  float* tanh_state,
  float* forget_gate_activations,
  float* cell_candidate_activations,
  float* output_gate_actications,
  float* input_gate_complement,
  float* output_gate_gradients,
  float* cell_state_gradient,
  float* input_gate_gradients,
  float* forget_gate_gradients,
  float* last_cell_state)
{
  const __m256 ones = _mm256_set1_ps(1.0f);
  const __m256 zeros = _mm256_setzero_ps();

  for (size_t i = 0; i < num_cells; i += 8) {
    __m256 stored_err = _mm256_load_ps(&temporal_hidden_gradient[i]);
    __m256 hidden_err = _mm256_load_ps(&hidden_gradient[i]);

    // temporal_hidden_gradient[i] += hidden_gradient[i]
    stored_err = _mm256_add_ps(stored_err, hidden_err);
    _mm256_store_ps(&temporal_hidden_gradient[i], stored_err);

    // hidden_gradient[i] = 0.0f
    _mm256_store_ps(&hidden_gradient[i], zeros);

    // Load states from sequence_position offset
    const size_t idx = sequence_position_offset + i;
    __m256 tanh_v = _mm256_load_ps(&tanh_state[idx]);
    __m256 forget = _mm256_load_ps(&forget_gate_activations[idx]);
    __m256 inputv = _mm256_load_ps(&cell_candidate_activations[idx]);
    __m256 output = _mm256_load_ps(&output_gate_actications[idx]);
    __m256 input_gate = _mm256_load_ps(&input_gate_complement[idx]);

    // output_gate_gradients[i] = tanh_v * temporal_hidden_gradient[i] * output * (1.0f - output)
    __m256 one_minus_output = _mm256_sub_ps(ones, output);
    __m256 og_err = _mm256_mul_ps(tanh_v, stored_err);
    og_err = _mm256_mul_ps(og_err, output);
    og_err = _mm256_mul_ps(og_err, one_minus_output);
    _mm256_store_ps(&output_gate_gradients[i], og_err);

    // cell_state_gradient[i] += temporal_hidden_gradient[i] * output * (1.0f - tanh_v * tanh_v)
    __m256 state_err = _mm256_load_ps(&cell_state_gradient[i]);
    __m256 tanh_sq = _mm256_mul_ps(tanh_v, tanh_v);
    __m256 one_minus_tanh_sq = _mm256_sub_ps(ones, tanh_sq);
    __m256 temp = _mm256_mul_ps(stored_err, output);
    temp = _mm256_mul_ps(temp, one_minus_tanh_sq);
    state_err = _mm256_add_ps(state_err, temp);

    // input_gate_gradients[i] = cell_state_gradient[i] * input_gate * (1.0f - inputv * inputv)
    __m256 inputv_sq = _mm256_mul_ps(inputv, inputv);
    __m256 one_minus_inputv_sq = _mm256_sub_ps(ones, inputv_sq);
    __m256 ig_err = _mm256_mul_ps(state_err, input_gate);
    ig_err = _mm256_mul_ps(ig_err, one_minus_inputv_sq);
    _mm256_store_ps(&input_gate_gradients[i], ig_err);

    // forget_gate_gradients[i] = (last_cell_state[idx] - inputv) * cell_state_gradient[i] * forget * input_gate
    __m256 last_st = _mm256_load_ps(&last_cell_state[idx]);
    __m256 fg_err = _mm256_sub_ps(last_st, inputv);
    fg_err = _mm256_mul_ps(fg_err, state_err);
    fg_err = _mm256_mul_ps(fg_err, forget);
    fg_err = _mm256_mul_ps(fg_err, input_gate);
    _mm256_store_ps(&forget_gate_gradients[i], fg_err);

    if (sequence_position_offset > 0) { // sequence_position > 0
      state_err = _mm256_mul_ps(state_err, forget);
      _mm256_store_ps(&temporal_hidden_gradient[i], zeros);
    }

    _mm256_store_ps(&cell_state_gradient[i], state_err);
  }
}

void VectorFunctions_AVX2::BackpropagateErrors(
  size_t len,         // num_cells (200)
  size_t base_offset, // 0 for temporal, num_cells for spatial
  size_t hidden_size, // Layer 0: 200, Layer 1: 400
  float* recurrent_weights,    // Weight matrix
  float* gate_gradient_buffer, // Current layer errors
  float* grad_store)    // Where to accumulate gradients
{
  for (size_t i = 0; i < len; i += 8) {
    __m256 grad_vec = _mm256_load_ps(&grad_store[i]);

    // for better precision calculate the sum then add to the existing grads
    __m256 sum = _mm256_setzero_ps();

    size_t weight_idx = base_offset + i;
    for (size_t i = 0; i < len; i++) {
      __m256 g = _mm256_set1_ps(gate_gradient_buffer[i]);
      __m256 w = _mm256_load_ps(&recurrent_weights[weight_idx]);
      __m256 prod = _mm256_mul_ps(g, w);
      sum = _mm256_add_ps(sum, prod);
      weight_idx += hidden_size;
    }

    grad_vec = _mm256_add_ps(grad_vec, sum);
    _mm256_store_ps(&grad_store[i], grad_vec);
  }
}

void VectorFunctions_AVX2::AccumulateLayerGradients(
  const size_t num_cells,
  const size_t vocabulary_size,
  const size_t hidden_size,
  const float* input,
  const float* gate_gradient_buffer,
  float* embedding_ptr,
  float* recurrent_weight_gradients)
{
  for (size_t i = 0; i < num_cells; i += 4) {
    // Load 4 errors (using only lower half of AVX register)
    __m128 errors_128 = _mm_load_ps(&gate_gradient_buffer[i]);
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
    size_t update_offset = i * hidden_size;
    for (size_t j = 0; j < hidden_size; j += 8) {
      size_t base_offset = update_offset + j;

      __m256 inp = _mm256_load_ps(&input[j]);

      __m256 upd0 = _mm256_load_ps(&recurrent_weight_gradients[base_offset]);
      upd0 = _mm256_add_ps(upd0, _mm256_mul_ps(inp, error_vec0)); base_offset += hidden_size;

      __m256 upd1 = _mm256_load_ps(&recurrent_weight_gradients[base_offset]);
      upd1 = _mm256_add_ps(upd1, _mm256_mul_ps(inp, error_vec1)); base_offset += hidden_size;

      __m256 upd2 = _mm256_load_ps(&recurrent_weight_gradients[base_offset]);
      upd2 = _mm256_add_ps(upd2, _mm256_mul_ps(inp, error_vec2)); base_offset += hidden_size;

      __m256 upd3 = _mm256_load_ps(&recurrent_weight_gradients[base_offset]);
      upd3 = _mm256_add_ps(upd3, _mm256_mul_ps(inp, error_vec3));

      base_offset = update_offset + j;
      _mm256_store_ps(&recurrent_weight_gradients[base_offset], upd0); base_offset += hidden_size;
      _mm256_store_ps(&recurrent_weight_gradients[base_offset], upd1); base_offset += hidden_size;
      _mm256_store_ps(&recurrent_weight_gradients[base_offset], upd2); base_offset += hidden_size;
      _mm256_store_ps(&recurrent_weight_gradients[base_offset], upd3);
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
  const size_t hidden_size,
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
    size_t output_offset = i * hidden_size;
    for (size_t j = 0; j < hidden_size; j += 8) {
      size_t base_offset = output_offset + j;

      __m256 hidden = _mm256_load_ps(&hidden_ptr[j]);

      __m256 out = _mm256_load_ps(&output_weight_gradients[base_offset]);
      out = _mm256_add_ps(out, _mm256_mul_ps(hidden, error_vec0)); base_offset += hidden_size;

      __m256 out1 = _mm256_load_ps(&output_weight_gradients[base_offset]);
      out1 = _mm256_add_ps(out1, _mm256_mul_ps(hidden, error_vec1)); base_offset += hidden_size;

      __m256 out2 = _mm256_load_ps(&output_weight_gradients[base_offset]);
      out2 = _mm256_add_ps(out2, _mm256_mul_ps(hidden, error_vec2)); base_offset += hidden_size;

      __m256 out3 = _mm256_load_ps(&output_weight_gradients[base_offset]);
      out3 = _mm256_add_ps(out3, _mm256_mul_ps(hidden, error_vec3));

      base_offset = output_offset + j;
      _mm256_store_ps(&output_weight_gradients[base_offset], out); base_offset += hidden_size;
      _mm256_store_ps(&output_weight_gradients[base_offset], out1); base_offset += hidden_size;
      _mm256_store_ps(&output_weight_gradients[base_offset], out2); base_offset += hidden_size;
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
  float* output_weights,
  float* output,
  float* output_bias,
  size_t const hidden_size_from_all_layers, // 200*2 = 400
  size_t const vocabulary_size, // 256
  size_t const output_offset)
{
  // Compute logits via dot products
  for (size_t i = 0; i < vocabulary_size; i++) {  // 256 iterations
    logits[output_offset + i] = DotProduct(
      &hidden[0],
      &output_weights[i * hidden_size_from_all_layers],
      hidden_size_from_all_layers // 400
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

#endif

