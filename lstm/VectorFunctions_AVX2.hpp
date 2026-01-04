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

  virtual void BackpropagateErrors(
    size_t len,
    size_t base_offset,
    size_t hidden_size,
    float* weights,
    float* error,
    float* grad_store
  ) override;

  virtual void AccumulateLayerGradients(
    const size_t num_cells,
    const size_t embedding_size,
    const size_t hidden_size,
    const float* input,
    const float* error,
    float* embedding_ptr,
    float* update
  ) override;

  virtual void AccumulateOutputLayerGradients(
    size_t previous_output_offset,
    float* output_ptr,
    float* output_layer_ptr,
    float* output_bias_u,
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
    float* output_layer,
    float* output,
    float* output_bias,
    size_t const hidden_size,
    size_t const output_size,
    size_t const output_offset
  ) override;

  // Softmax: using the SSE2 version
};

#endif
