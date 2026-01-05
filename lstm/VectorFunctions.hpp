#pragma once

#include "../Utils.hpp"
#include "../Simd.hpp"
#include "../SimdType.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>

class VectorFunctions
{
public:
  virtual float DotProduct(
    float const* x1,
    float const* x2,
    size_t const len
  ) = 0;

  virtual float SumOfSquares(
    float* array,
    size_t array_length
  ) = 0;

  virtual void NormalizeThenActivate_Sigmoid(
    size_t array_length,
    float* norm,
    float* state,
    float* gamma,
    float* beta,
    float inverse_variance
  ) = 0;

  virtual void NormalizeThenActivate_Tanh(
    size_t array_length,
    float* norm,
    float* state,
    float* gamma,
    float* beta,
    float inverse_variance
  ) = 0;

  virtual void AccumulateLstmGradients(
    size_t num_cells,
    size_t hidden_size,
    size_t output_size,
    size_t layer,
    float* error_on_output,
    float* hidden_error,
    float* output_layer
  ) = 0;

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
  ) = 0;

  virtual void BackpropagateErrors(
    size_t len,         // num_cells (200)
    size_t base_offset, // 0 for temporal, num_cells for spatial
    size_t hidden_size, // Layer 0: 200, Layer 1: 400
    float* weights,     // Weight matrix
    float* error,       // Current layer errors
    float* grad_store   // Where to accumulate gradients
  ) = 0;

  virtual void AccumulateLayerGradients(
    const size_t num_cells,
    const size_t embedding_size,
    const size_t hidden_size,
    const float* input,
    const float* error,
    float* embedding_ptr,
    float* update
  ) = 0;

  virtual void AccumulateOutputLayerGradients(
    size_t previous_output_offset,
    float* error_on_output,
    float* output_layer_ptr,
    float* output_bias_u,
    const float* hidden_ptr,
    const size_t output_size,
    const size_t hidden_size,
    const size_t input_symbol
  ) = 0;

  virtual float ComputeMaxLogit(
    float* result,
    size_t result_length
  ) = 0;

  virtual void MatvecThenSoftmax(
    float* hidden,
    float* logits,
    float* output_layer,
    float* output,
    float* bias,
    size_t const hidden_size,
    size_t const output_size,
    size_t const output_offset
  ) = 0;

  virtual void Softmax(
    float* logits,
    float* probs,
    size_t len,
    float max_logit
  ) = 0;
};

constexpr float negative_infinity = -std::numeric_limits<float>::max();

// Scalar sigmoid and tanh
float sigmoid_pade_clipped(float x);
float tanh_pade_clipped(float x);

// Clipping constants
static constexpr float SIGMOID_CLIP_MIN = -5.95f;
static constexpr float SIGMOID_CLIP_MAX = +5.95f;
static constexpr float TANH_CLIP_MIN = SIGMOID_CLIP_MIN / 2.0f;
static constexpr float TANH_CLIP_MAX = SIGMOID_CLIP_MAX / 2.0f;


