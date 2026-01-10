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
    float* pre_norm_values,
    float* activations_out,
    float* gamma,
    float* beta,
    float inverse_variance
  ) = 0;

  virtual void NormalizeThenActivate_Tanh(
    size_t array_length,
    float* pre_norm_values,
    float* activations_out,
    float* gamma,
    float* beta,
    float inverse_variance
  ) = 0;

  virtual void AccumulateLstmGradients(
    size_t num_cells,
    size_t hidden_size,
    size_t vocabulary_size,
    size_t layer_id,
    float* error_on_output,
    float* hidden_gradient,
    float* output_weights
  ) = 0;

  virtual void AccumulateLstmLayerGradients(
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
    float* last_cell_state
  ) = 0;

  virtual void BackpropagateErrors(
    size_t len,         // num_cells (200)
    size_t base_offset, // 0 for temporal, num_cells for spatial
    size_t hidden_size, // Layer 0: 200, Layer 1: 400
    float* recurrent_weights,    // Weight matrix
    float* gate_gradient_buffer, // Current layer errors
    float* grad_store     // Where to accumulate gradients
  ) = 0;

  virtual void AccumulateLayerGradients(
    const size_t num_cells,
    const size_t vocabulary_size,
    const size_t hidden_size,
    const float* input,
    const float* gate_gradient_buffer,
    float* embedding_ptr,
    float* recurrent_weight_gradients
  ) = 0;

  virtual void AccumulateOutputLayerGradients(
    size_t previous_output_offset,
    float* error_on_output,
    float* output_weight_gradients,
    float* output_bias_gradients,
    const float* hidden_ptr,
    const size_t vocabulary_size,
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
    float* output_weights,
    float* output,
    float* output_bias,
    size_t const hidden_size_from_all_layers,
    size_t const vocabulary_size,
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


