#include "VectorFunctions_Scalar.hpp"

// Static helper functions

static float horizontal_sum(float x0, float x1, float x2, float x3, float x4, float x5, float x6, float x7)
{
  // Simulate loading of __m256 as an array of 8 floats
  float sum0 = x0 + x4; // Pair 0
  float sum1 = x1 + x5; // Pair 1
  float sum2 = x2 + x6; // Pair 2
  float sum3 = x3 + x7; // Pair 3

  // Combine pairs
  sum0 = sum0 + sum2;
  sum1 = sum1 + sum3;

  // Final horizontal sum
  sum0 = sum0 + sum1;

  return sum0;
}

// Helper for bitcasting
static inline float bitcast_u32_to_f32(uint32_t x) {
  float result;
  memcpy(&result, &x, sizeof(float));
  return result;
}

// expf
// Cody–Waite style reduction with split ln2 (hi+lo) and degree-5 polynomial on the reduced range.
// Preconditions: x is finite; FP rounding mode is round-to-nearest-even. (softmax pipeline guarantees these)
// => no need to check isfinite - for any case we'll check outputs on the result later
// Notes: 
//   Max relative error ≈ 3.37e-7
//   Max ULP error ≈ 4 ULP
//   Median ULP ≈ 1, mean ≈ 0.61
//   Monotonic, strictly positive over the whole domain.
// to get hex literals use: printf("%a\n", x);  // %a for hex float
// see also https://chromium.googlesource.com/external/github.com/google/XNNPACK/+/refs/heads/upstream/test_638074745/src/math/f32-sigmoid-sse2-rr2-p5-div.c

static inline float expf_compat(float x) {
  //keep (n + 127) in [1,254] for a valid exponent field
  if (x < -87)
    x = -87; //final result will be 1.64581131e-38

  // Cody–Waite split of ln2
  const float INV_LN2 = 0x1.715476p+0f;   // 1.4426950216f  // properly rounded 1/ln2
  const float LN2_HI = 0x1.62e400p-1f;    // 0.693145752f   // coarsened value of ln2 i.e. with zeroed 7 last fraction bits
  const float LN2_LO = 0x1.7f7d1cp-20f;   // 1.42860677e-6f // ln2-LN2_HI
  //This way LN2_HI + LN2_LO (in exact real arithmetic) is extremely close to ln2, and when computed in float it reproduces the correctly rounded float value of ln⁡2.
  //There are many valid Cody–Waite splits; this pair (hi = 0x1.62e400p-1f, lo = 0x1.7f7d1cp-20f) is a well-tested single-precision choice that balances reduction error for ∣n∣≲126.

  // n = round(x / ln2)
  // t = (float)n
  float z = x * INV_LN2;
  const float MAGIC_BIAS = 12582912;   // 1.5 x 2^23 : trick for rounding
  float t = z + MAGIC_BIAS;
  int   n = (int)t - 12582912;         // subtract the bias in integer domain;
  t -= MAGIC_BIAS;

  // r = x - n*ln2 using split constants (Cody–Waite)
  float r = x - t * LN2_HI;
  r = r - t * LN2_LO;

  // Taylor coefficients
  // exp(r) ≈ 1 + r + c2 r^2 + c3 r^3 + c4 r^4 + c5 r^5
  const float c2 = 0x1.fffe24p-2f;  // ≈ 0.499992907 (≈ 1/2)
  const float c3 = 0x1.5554acp-3f;  // ≈ 0.166665405 (≈ 1/6)
  const float c4 = 0x1.5713a4p-5f;  // ≈ 0.041879482 (≈ 1/24)
  const float c5 = 0x1.12266ap-7f;  // ≈ 0.008366401 (≈ 1/120)

  // Estrin's Scheme
  // It gives shorter dependency chains than Horner, which usually wins on SIMD and GPUs.
  float r2 = r * r;
  float q2 = c2 + c3 * r;
  float q4 = c4 + c5 * r;
  float p = (q4 * r2 + q2) * r2 + (r + 1.0f);

  // construct 2^n as float via exponent bits: (n + 127) << 23
  uint32_t expbits = (uint32_t)(n + 127);
  // note: expbits must be in [1,254] for normalized; we've clamped x earlier.
  uint32_t bits = expbits << 23;
  float two_n = bitcast_u32_to_f32(bits);

  return p * two_n;
}

// Member implementations

float VectorFunctions_Scalar::DotProduct(
  float const* x1,
  float const* x2,
  size_t const len)
{
  float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
  float sum4 = 0.0f, sum5 = 0.0f, sum6 = 0.0f, sum7 = 0.0f;

  for (size_t i = 0; i < len; i += 8)
  {
    sum0 += x1[i + 0] * x2[i + 0];
    sum1 += x1[i + 1] * x2[i + 1];
    sum2 += x1[i + 2] * x2[i + 2];
    sum3 += x1[i + 3] * x2[i + 3];
    sum4 += x1[i + 4] * x2[i + 4];
    sum5 += x1[i + 5] * x2[i + 5];
    sum6 += x1[i + 6] * x2[i + 6];
    sum7 += x1[i + 7] * x2[i + 7];
  }

  return horizontal_sum(sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7);
}


float VectorFunctions_Scalar::SumOfSquares(float* array, size_t array_length)
{
  float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
  float sum4 = 0.0f, sum5 = 0.0f, sum6 = 0.0f, sum7 = 0.0f;

  for (size_t i = 0; i < array_length; i += 8)
  {
    float x0 = array[i];
    float x1 = array[i + 1];
    float x2 = array[i + 2];
    float x3 = array[i + 3];
    float x4 = array[i + 4];
    float x5 = array[i + 5];
    float x6 = array[i + 6];
    float x7 = array[i + 7];

    sum0 += x0 * x0;
    sum1 += x1 * x1;
    sum2 += x2 * x2;
    sum3 += x3 * x3;
    sum4 += x4 * x4;
    sum5 += x5 * x5;
    sum6 += x6 * x6;
    sum7 += x7 * x7;
  }
  return horizontal_sum(sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7);
}

void VectorFunctions_Scalar::NormalizeThenActivate_Sigmoid(
  size_t array_length,
  float* norm,
  float* state,
  float* gamma,
  float* beta,
  float inverse_variance)
{
  for (size_t i = 0; i < array_length; i++) {
    float n = norm[i] * inverse_variance;
    norm[i] = n;
    state[i] = sigmoid_pade_clipped(n * gamma[i] + beta[i]);
  }
}

void VectorFunctions_Scalar::NormalizeThenActivate_Tanh(
  size_t array_length,
  float* norm,
  float* state,
  float* gamma,
  float* beta,
  float inverse_variance)
{
  for (size_t i = 0; i < array_length; i++) {
    float n = norm[i] * inverse_variance;
    norm[i] = n;
    state[i] = tanh_pade_clipped(n * gamma[i] + beta[i]);
  }
}

void VectorFunctions_Scalar::AccumulateLstmGradients(
  size_t num_cells,
  size_t hidden_size,
  size_t output_size,
  size_t layer,
  float* error_on_output,
  float* hidden_error,
  float* output_weights)
{
  size_t output_layer_offset = layer * num_cells; // layer * 200
  for (size_t i = 0; i < output_size; i++) {   // 256 iterations
    float const error = error_on_output[i];
    for (size_t j = 0; j < num_cells; j++) { // 200 iterations
      hidden_error[j] += output_weights[output_layer_offset + j] * error;
    }
    output_layer_offset += hidden_size;
  }
}

void VectorFunctions_Scalar::AccumulateLstmLayerGradients(
  size_t num_cells,
  size_t ebase,
  float* stored_error,
  float* hidden_error,
  float* tanh_state,
  float* fg_state,
  float* ig_state,
  float* og_state,
  float* input_gate_state,
  float* og_error,
  float* state_error,
  float* ig_error,
  float* fg_error,
  float* last_state)
{
  for (size_t i = 0; i < num_cells; i++) {          // 200 iterations
    stored_error[i] += hidden_error[i];
    hidden_error[i] = 0.0f;

    const size_t idx = ebase + i;                   // epoch*200 + i
    const float tanh_v = tanh_state[idx];
    const float forget = fg_state[idx];
    const float inputv = ig_state[idx];
    const float output = og_state[idx];
    const float input_gate = input_gate_state[idx];

    og_error[i] =
      tanh_v * stored_error[i] *
      output * (1.0f - output); // sigmoid derivative: σ'(x) = σ(x) × (1 - σ(x))

    state_error[i] +=
      stored_error[i] * output *
      (1.0f - tanh_v * tanh_v); // tanh derivative: tanh'(x) = 1 - tanh²(x)

    ig_error[i] =
      state_error[i] * input_gate *
      (1.0f - inputv * inputv); // tanh derivative: tanh'(x) = 1 - tanh²(x)

    fg_error[i] =
      (last_state[idx] - inputv) *
      state_error[i] *
      forget * input_gate; // implicit sigmoid derivative: forget * input_gate where input_gate = 1.0f - forget

    if (ebase > 0) { // epoch > 0
      state_error[i] *= forget;
      stored_error[i] = 0.0f;
    }
  }
}

void VectorFunctions_Scalar::BackpropagateErrors(
  size_t len,         // num_cells (200)
  size_t base_offset, // 0 for temporal, num_cells for spatial
  size_t hidden_size, // Layer 0: 200, Layer 1: 400
  float* weights,     // Weight matrix
  float* error,       // Current layer errors
  float* grad_store)   // Where to accumulate gradients
{
  for (size_t i = 0; i < len; i += 8) { // For each cell in previous layer's hidden state
    // for better precision calculate the sum then add to the existing grads
    float sum0 = 0;
    float sum1 = 0;
    float sum2 = 0;
    float sum3 = 0;
    float sum4 = 0;
    float sum5 = 0;
    float sum6 = 0;
    float sum7 = 0;

    size_t weight_idx = base_offset + i;  // Start at offset for previous layer connections
    for (size_t i = 0; i < len; i++) { // For each current cell
      float ei = error[i];
      sum0 += ei * weights[weight_idx + 0];
      sum1 += ei * weights[weight_idx + 1];
      sum2 += ei * weights[weight_idx + 2];
      sum3 += ei * weights[weight_idx + 3];
      sum4 += ei * weights[weight_idx + 4];
      sum5 += ei * weights[weight_idx + 5];
      sum6 += ei * weights[weight_idx + 6];
      sum7 += ei * weights[weight_idx + 7];
      weight_idx += hidden_size;  // Move to next cell's weights
    }

    grad_store[i + 0] += sum0;
    grad_store[i + 1] += sum1;
    grad_store[i + 2] += sum2;
    grad_store[i + 3] += sum3;
    grad_store[i + 4] += sum4;
    grad_store[i + 5] += sum5;
    grad_store[i + 6] += sum6;
    grad_store[i + 7] += sum7;
  }
}

void VectorFunctions_Scalar::AccumulateLayerGradients(
  const size_t num_cells,
  const size_t embedding_size,
  const size_t hidden_size,
  const float* input,
  const float* error,
  float* embedding_ptr,
  float* weight_gradients)
{
  for (size_t i = 0; i < num_cells; i++) {
    const float ei = error[i];

    // Update embedding gradient
    *embedding_ptr += ei;
    embedding_ptr += embedding_size;

    // Update hidden state weight gradients
    for (size_t j = 0; j < hidden_size; j++)
      weight_gradients[j] += ei * input[j];

    weight_gradients += hidden_size;
  }
}

void VectorFunctions_Scalar::AccumulateOutputLayerGradients(
  size_t previous_output_offset,
  float* error_on_output,
  float* output_weight_gradients,
  float* output_bias_gradients,
  const float* hidden_ptr,
  const size_t output_size,
  const size_t hidden_size,
  const size_t input_symbol)
{

  for (size_t i = 0; i < output_size; i++) {
    float error = error_on_output[i];
    output_bias_gradients[i] += error;

    for (size_t j = 0; j < hidden_size; j++) {
      output_weight_gradients[j] += error * hidden_ptr[j];
    }

    output_weight_gradients += hidden_size;
  }
}

float VectorFunctions_Scalar::ComputeMaxLogit(
  float* result,
  size_t result_length)
{
  float maxlogit = negative_infinity;
  for (size_t i = 0; i < result_length; i++)
  {
    if (result[i] > maxlogit)
      maxlogit = result[i];
  }
  return maxlogit;
}

void VectorFunctions_Scalar::MatvecThenSoftmax(
  float* hidden,
  float* logits,
  float* output_weights,
  float* output,
  float* output_bias,
  size_t const hidden_size_from_all_layers, // 200*2 = 400
  size_t const output_size, // 256
  size_t const output_offset
)
{
  // Compute logits via dot products
  for (size_t i = 0; i < output_size; i++) {   // 256 iterations
    logits[output_offset + i] = DotProduct( // logits[epoch * 256 + i]
      &hidden[0],
      &output_weights[i * hidden_size_from_all_layers],
      hidden_size_from_all_layers // 400
    ) + output_bias[i];
  }

  // Find max logit for numerical stability
  float max_logit = ComputeMaxLogit(&logits[output_offset], output_size);

  // Compute softmax
  Softmax(
    &logits[output_offset],                  // &logits[epoch * 256]
    &output[output_offset],                  // &output[epoch * 256]
    output_size,                             // 256
    max_logit);
}

void VectorFunctions_Scalar::Softmax(
  float* logits,
  float* probs,
  size_t len,
  float max_logit)
{
  float expsum[8]{ 0.0f };
  for (size_t i = 0; i < len; i += 8)
  {
    for (size_t j = 0; j < 8; j++) {
      float x = expf_compat(logits[i + j] - max_logit);
      probs[i + j] = x;
      expsum[j] += x;
    }
  }
  float expsum_reciprocal = 1.0f / horizontal_sum(expsum[0], expsum[1], expsum[2], expsum[3], expsum[4], expsum[5], expsum[6], expsum[7]);
  for (size_t i = 0; i < len; i++)
  {
    probs[i] *= expsum_reciprocal;
  }
}
