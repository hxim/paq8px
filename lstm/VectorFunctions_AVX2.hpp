#pragma once

#include "VectorFunctions_SSE2.hpp"

#ifdef X64_SIMD_AVAILABLE

class VectorFunctions_AVX2: public VectorFunctions_SSE2
{
  virtual float DotProduct(
    float const* x1,
    float const* x2,
    size_t const len
  ) override;

  virtual float SumOfSquares(
    float* array,
    size_t array_length
  ) override;

  virtual void NormalizeThenActivate_Sigmoid(
    size_t array_length,
    float* norm,
    float* state,
    float* gamma,
    float* beta,
    float inverse_variance
  ) override;

  virtual void NormalizeThenActivate_Tanh(
    size_t array_length,
    float* norm,
    float* state,
    float* gamma,
    float* beta,
    float inverse_variance
  ) override;

  virtual void AccumulateLstmGradients(
    size_t num_cells,
    size_t hidden_size,
    size_t output_size,
    size_t layer,
    float* error_on_output,
    float* hidden_error,
    float* output_weights
  ) override;

  virtual void AccumulateLstmLayerGradients(
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
    float* last_state
  ) override;

  virtual void BackpropagateErrors(
    size_t len,         // num_cells (200)
    size_t base_offset, // 0 for temporal, num_cells for spatial
    size_t hidden_size, // Layer 0: 200, Layer 1: 400
    float* weights,     // Weight matrix
    float* error,       // Current layer errors
    float* grad_store   // Where to accumulate gradients
  ) override;

  virtual void AccumulateLayerGradients(
    const size_t num_cells,
    const size_t embedding_size,
    const size_t hidden_size,
    const float* input,
    const float* error,
    float* embedding_ptr,
    float* weight_gradients
  ) override;

  virtual void AccumulateOutputLayerGradients(
    size_t previous_output_offset,
    float* error_on_output,
    float* output_weight_gradients,
    float* output_bias_gradients,
    const float* hidden_ptr,
    const size_t output_size,
    const size_t hidden_size,
    const size_t input_symbol
  ) override;

  virtual float ComputeMaxLogit(
    float* result,
    size_t result_length
  ) override;

  virtual void MatvecThenSoftmax(
    float* hidden,
    float* logits,
    float* output_weights,
    float* output,
    float* output_bias,
    size_t const hidden_size_from_all_layers,
    size_t const output_size,
    size_t const output_offset
  ) override;

  // Softmax: using the SSE2 version
};

#endif
