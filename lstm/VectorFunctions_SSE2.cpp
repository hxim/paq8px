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
  float* pre_norm_values,
  float* activations_out,
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

    __m128 pre_activation_vec = _mm_load_ps(pre_norm_values + i);
    pre_activation_vec = _mm_mul_ps(pre_activation_vec, inv_var_vec);
    _mm_store_ps(pre_norm_values + i, pre_activation_vec);

    __m128 gamma_vec = _mm_load_ps(gamma + i);
    __m128 beta_vec = _mm_load_ps(beta + i);
    __m128 x = _mm_mul_ps(pre_activation_vec, gamma_vec);
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

    _mm_store_ps(activations_out + i, result);
  }
}

void VectorFunctions_SSE2::NormalizeThenActivate_Tanh(
  size_t array_length,
  float* pre_norm_values,
  float* activations_out,
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

    __m128 pre_activation_vec = _mm_load_ps(pre_norm_values + i);
    pre_activation_vec = _mm_mul_ps(pre_activation_vec, inv_var_vec);
    _mm_store_ps(pre_norm_values + i, pre_activation_vec);

    __m128 gamma_vec = _mm_load_ps(gamma + i);
    __m128 beta_vec = _mm_load_ps(beta + i);
    __m128 x = _mm_mul_ps(pre_activation_vec, gamma_vec);
    x = _mm_add_ps(x, beta_vec);

    // tanh

    x = _mm_max_ps(x, c_clip_lower);
    x = _mm_min_ps(x, c_clip_upper);

    __m128 x2 = _mm_mul_ps(x, x);
    __m128 numer = _mm_mul_ps(x, _mm_add_ps(c_27, x2));
    __m128 denom = _mm_add_ps(c_27, _mm_mul_ps(c_9, x2));
    __m128 result = _mm_div_ps(numer, denom);

    _mm_store_ps(activations_out + i, result);
  }
}

void VectorFunctions_SSE2::AccumulateLstmGradients(
  size_t num_cells,
  size_t hidden_size,
  size_t vocabulary_size,
  size_t layer_id,
  float* error_on_output,
  float* hidden_gradient,
  float* output_weights)
{
  size_t output_layer_offset = layer_id * num_cells; // layer_id * 200

  for (size_t i = 0; i < vocabulary_size; i += 4) {   // 256 iterations, 4 at a time
    // Load 4 errors as a vector
    __m128 errors = _mm_load_ps(&error_on_output[i]);

    // Broadcast each error to its own vector
    __m128 error_vec0 = _mm_shuffle_ps(errors, errors, _MM_SHUFFLE(0, 0, 0, 0));
    __m128 error_vec1 = _mm_shuffle_ps(errors, errors, _MM_SHUFFLE(1, 1, 1, 1));
    __m128 error_vec2 = _mm_shuffle_ps(errors, errors, _MM_SHUFFLE(2, 2, 2, 2));
    __m128 error_vec3 = _mm_shuffle_ps(errors, errors, _MM_SHUFFLE(3, 3, 3, 3));

    for (size_t j = 0; j < num_cells; j += 4) { // 200 iterations, 4 at a time
      size_t base_offset = output_layer_offset + j;

      // Load hidden_gradient once
      __m128 hidden = _mm_load_ps(&hidden_gradient[j]);

      // Load from 4 different output_weights rows and accumulate
      hidden = _mm_add_ps(hidden, _mm_mul_ps(_mm_load_ps(&output_weights[base_offset]), error_vec0)); base_offset += hidden_size;
      hidden = _mm_add_ps(hidden, _mm_mul_ps(_mm_load_ps(&output_weights[base_offset]), error_vec1)); base_offset += hidden_size;
      hidden = _mm_add_ps(hidden, _mm_mul_ps(_mm_load_ps(&output_weights[base_offset]), error_vec2)); base_offset += hidden_size;
      hidden = _mm_add_ps(hidden, _mm_mul_ps(_mm_load_ps(&output_weights[base_offset]), error_vec3));

      // Store back to hidden_gradient
      _mm_store_ps(&hidden_gradient[j], hidden);
    }

    output_layer_offset += hidden_size * 4;
  }
}

void VectorFunctions_SSE2::AccumulateLstmLayerGradients(
  size_t num_cells,
  size_t sequence_position_offset,
  float* temporal_hidden_gradient,
  float* hidden_gradient,
  float* tanh_state,
  float* forget_gate_activations,
  float* cell_candidate_activations,
  float* output_gate_actications,
  float* output_gate_gradients,
  float* cell_state_gradient,
  float* input_gate_gradients,
  float* forget_gate_gradients,
  float* last_cell_state)
{
  const __m128 ones = _mm_set1_ps(1.0f);
  const __m128 zeros = _mm_setzero_ps();

  for (size_t i = 0; i < num_cells; i += 4) {
    __m128 stored_err = _mm_load_ps(&temporal_hidden_gradient[i]);
    __m128 hidden_err = _mm_load_ps(&hidden_gradient[i]);

    // temporal_hidden_gradient[i] += hidden_gradient[i]
    stored_err = _mm_add_ps(stored_err, hidden_err);
    _mm_store_ps(&temporal_hidden_gradient[i], stored_err);

    // hidden_gradient[i] = 0.0f
    _mm_store_ps(&hidden_gradient[i], zeros);

    // Load states from sequence_position offset
    const size_t idx = sequence_position_offset + i;
    __m128 tanh_v = _mm_load_ps(&tanh_state[idx]);
    __m128 forget_gate = _mm_load_ps(&forget_gate_activations[idx]);
    __m128 cell_candidtae = _mm_load_ps(&cell_candidate_activations[idx]);
    __m128 output_gate = _mm_load_ps(&output_gate_actications[idx]);
    __m128 input_gate = _mm_sub_ps(ones, forget_gate);

    // output_gate_gradients[i] = tanh_v * temporal_hidden_gradient[i] * output * (1.0f - output)
    __m128 one_minus_output = _mm_sub_ps(ones, output_gate);
    __m128 og_err = _mm_mul_ps(tanh_v, stored_err);
    og_err = _mm_mul_ps(og_err, output_gate);
    og_err = _mm_mul_ps(og_err, one_minus_output);
    _mm_store_ps(&output_gate_gradients[i], og_err);

    // cell_state_gradient[i] += temporal_hidden_gradient[i] * output * (1.0f - tanh_v * tanh_v)
    __m128 state_err = _mm_load_ps(&cell_state_gradient[i]);
    __m128 tanh_sq = _mm_mul_ps(tanh_v, tanh_v);
    __m128 one_minus_tanh_sq = _mm_sub_ps(ones, tanh_sq);
    __m128 temp = _mm_mul_ps(stored_err, output_gate);
    temp = _mm_mul_ps(temp, one_minus_tanh_sq);
    state_err = _mm_add_ps(state_err, temp);

    // input_gate_gradients[i] = cell_state_gradient[i] * input_gate * (1.0f - inputv * inputv)
    __m128 inputv_sq = _mm_mul_ps(cell_candidtae, cell_candidtae);
    __m128 one_minus_inputv_sq = _mm_sub_ps(ones, inputv_sq);
    __m128 ig_err = _mm_mul_ps(state_err, input_gate);
    ig_err = _mm_mul_ps(ig_err, one_minus_inputv_sq);
    _mm_store_ps(&input_gate_gradients[i], ig_err);

    // forget_gate_gradients[i] = (last_cell_state[idx] - inputv) * cell_state_gradient[i] * forget * input_gate
    __m128 last_st = _mm_load_ps(&last_cell_state[idx]);
    __m128 fg_err = _mm_sub_ps(last_st, cell_candidtae);
    fg_err = _mm_mul_ps(fg_err, state_err);
    fg_err = _mm_mul_ps(fg_err, forget_gate);
    fg_err = _mm_mul_ps(fg_err, input_gate);
    _mm_store_ps(&forget_gate_gradients[i], fg_err);

    if (sequence_position_offset > 0) { // sequence_position > 0
      state_err = _mm_mul_ps(state_err, forget_gate);
      _mm_store_ps(&temporal_hidden_gradient[i], zeros);
    }

    _mm_store_ps(&cell_state_gradient[i], state_err);
  }
}

void VectorFunctions_SSE2::BackpropagateErrors(
  size_t len,         // num_cells (200)
  size_t base_offset, // 0 for temporal, num_cells for spatial
  size_t total_component_inputs, // Layer 0: 200, Layer 1: 400
  float* weights,     // Weight matrix
  float* gate_gradient_buffer,  // Current layer errors
  float* grad_store)   // Where to accumulate gradients
{
  for (size_t i = 0; i < len; i += 8) {
    __m128 grad0_vec = _mm_load_ps(&grad_store[i]);
    __m128 grad1_vec = _mm_load_ps(&grad_store[i + 4]);

    // for better precision calculate the sum then add to the existing grads
    __m128 sum0 = _mm_setzero_ps(); 
    __m128 sum1 = _mm_setzero_ps();

    size_t weight_idx = base_offset + i;
    for (size_t i = 0; i < len; i++) {
      __m128 g = _mm_set1_ps(gate_gradient_buffer[i]);

      __m128 w0 = _mm_load_ps(&weights[weight_idx]);
      __m128 prod0 = _mm_mul_ps(g, w0);
      sum0 = _mm_add_ps(sum0, prod0);

      __m128 w1 = _mm_load_ps(&weights[weight_idx + 4]);
      __m128 prod1 = _mm_mul_ps(g, w1);
      sum1 = _mm_add_ps(sum1, prod1);

      weight_idx += total_component_inputs;
    }

    grad0_vec = _mm_add_ps(grad0_vec, sum0);
    grad1_vec = _mm_add_ps(grad1_vec, sum1);

    _mm_store_ps(&grad_store[i], grad0_vec);
    _mm_store_ps(&grad_store[i + 4], grad1_vec);
  }
}

void VectorFunctions_SSE2::AccumulateLayerGradients(
  const size_t num_cells,
  const size_t vocabulary_size,
  const size_t total_component_inputs,
  const float* input,
  const float* gate_gradient_buffer,
  float* embedding_ptr,
  float* weight_gradients)
{
  for (size_t i = 0; i < num_cells; i += 4) {
    // Load 4 errors as a vector
    __m128 errors = _mm_load_ps(&gate_gradient_buffer[i]);
    
    // Broadcast each error
    __m128 error_vec0 = _mm_shuffle_ps(errors, errors, _MM_SHUFFLE(0, 0, 0, 0));
    __m128 error_vec1 = _mm_shuffle_ps(errors, errors, _MM_SHUFFLE(1, 1, 1, 1));
    __m128 error_vec2 = _mm_shuffle_ps(errors, errors, _MM_SHUFFLE(2, 2, 2, 2));
    __m128 error_vec3 = _mm_shuffle_ps(errors, errors, _MM_SHUFFLE(3, 3, 3, 3));
    
    // Extract scalar values from the broadcast vectors (just get first element)
    float e0 = _mm_cvtss_f32(error_vec0);
    float e1 = _mm_cvtss_f32(error_vec1);
    float e2 = _mm_cvtss_f32(error_vec2);
    float e3 = _mm_cvtss_f32(error_vec3);
    
    // Update symbol_embeddings gradient
    size_t emb_offset = i * vocabulary_size;
    embedding_ptr[emb_offset] += e0; emb_offset += vocabulary_size;
    embedding_ptr[emb_offset] += e1; emb_offset += vocabulary_size;
    embedding_ptr[emb_offset] += e2; emb_offset += vocabulary_size;
    embedding_ptr[emb_offset] += e3;
    
    // Update hidden state weight gradients
    size_t update_offset = i * total_component_inputs;
    for (size_t j = 0; j < total_component_inputs; j += 4) {
      size_t base_offset = update_offset + j;
      
      __m128 inp = _mm_load_ps(&input[j]);
      
      __m128 upd0 = _mm_load_ps(&weight_gradients[base_offset]);
      upd0 = _mm_add_ps(upd0, _mm_mul_ps(inp, error_vec0)); base_offset += total_component_inputs;
      
      __m128 upd1 = _mm_load_ps(&weight_gradients[base_offset]);
      upd1 = _mm_add_ps(upd1, _mm_mul_ps(inp, error_vec1)); base_offset += total_component_inputs;
      
      __m128 upd2 = _mm_load_ps(&weight_gradients[base_offset]);
      upd2 = _mm_add_ps(upd2, _mm_mul_ps(inp, error_vec2)); base_offset += total_component_inputs;
      
      __m128 upd3 = _mm_load_ps(&weight_gradients[base_offset]);
      upd3 = _mm_add_ps(upd3, _mm_mul_ps(inp, error_vec3));
      
      base_offset = update_offset + j;
      _mm_store_ps(&weight_gradients[base_offset], upd0); base_offset += total_component_inputs;
      _mm_store_ps(&weight_gradients[base_offset], upd1); base_offset += total_component_inputs;
      _mm_store_ps(&weight_gradients[base_offset], upd2); base_offset += total_component_inputs;
      _mm_store_ps(&weight_gradients[base_offset], upd3);
    }
  }
}

void VectorFunctions_SSE2::AccumulateOutputLayerGradients(
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
    __m128 errors = _mm_load_ps(&error_on_output[i]);

    // Broadcast each error
    __m128 error_vec0 = _mm_shuffle_ps(errors, errors, _MM_SHUFFLE(0, 0, 0, 0));
    __m128 error_vec1 = _mm_shuffle_ps(errors, errors, _MM_SHUFFLE(1, 1, 1, 1));
    __m128 error_vec2 = _mm_shuffle_ps(errors, errors, _MM_SHUFFLE(2, 2, 2, 2));
    __m128 error_vec3 = _mm_shuffle_ps(errors, errors, _MM_SHUFFLE(3, 3, 3, 3));

    // Update bias (vectorized)
    __m128 bias = _mm_load_ps(&output_bias_gradients[i]);
    bias = _mm_add_ps(bias, errors);
    _mm_store_ps(&output_bias_gradients[i], bias);

    // Update output layer weights
    size_t output_offset = i * hidden_size;
    for (size_t j = 0; j < hidden_size; j += 4) {
      size_t base_offset = output_offset + j;

      __m128 hidden = _mm_load_ps(&hidden_ptr[j]);

      __m128 out = _mm_load_ps(&output_weight_gradients[base_offset]);
      out = _mm_add_ps(out, _mm_mul_ps(hidden, error_vec0)); base_offset += hidden_size;

      __m128 out1 = _mm_load_ps(&output_weight_gradients[base_offset]);
      out1 = _mm_add_ps(out1, _mm_mul_ps(hidden, error_vec1)); base_offset += hidden_size;

      __m128 out2 = _mm_load_ps(&output_weight_gradients[base_offset]);
      out2 = _mm_add_ps(out2, _mm_mul_ps(hidden, error_vec2)); base_offset += hidden_size;

      __m128 out3 = _mm_load_ps(&output_weight_gradients[base_offset]);
      out3 = _mm_add_ps(out3, _mm_mul_ps(hidden, error_vec3));

      base_offset = output_offset + j;
      _mm_store_ps(&output_weight_gradients[base_offset], out); base_offset += hidden_size;
      _mm_store_ps(&output_weight_gradients[base_offset], out1); base_offset += hidden_size;
      _mm_store_ps(&output_weight_gradients[base_offset], out2); base_offset += hidden_size;
      _mm_store_ps(&output_weight_gradients[base_offset], out3);
    }
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
  float* output_weights,
  float* output,
  float* output_bias,
  size_t const concatenated_layer_outputs_size,
  size_t const vocabulary_size,
  size_t const output_offset)
{
  // Compute logits via dot products
  for (size_t i = 0; i < vocabulary_size; i++) {
    logits[output_offset + i] = DotProduct(
      &hidden[0],
      &output_weights[i * concatenated_layer_outputs_size],
      concatenated_layer_outputs_size
    ) + output_bias[i];
  }

  // Find max logit for numerical stability
  float max_logit = ComputeMaxLogit(&logits[output_offset], vocabulary_size);

  // Compute softmax
  Softmax(
    &logits[output_offset],
    &output[output_offset],
    vocabulary_size,
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
