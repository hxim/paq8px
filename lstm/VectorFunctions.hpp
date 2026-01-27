#pragma once

#include "../Utils.hpp"
#include "../Simd.hpp"
#include "../SIMDType.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>

class VectorFunctions
{
public:

  virtual ~VectorFunctions() = default;

  virtual void Copy(
    float* dst,
    const float* src,
    size_t num_floats
  ) = 0;

  virtual void Zero(
    float* dst,
    size_t num_floats
  ) = 0;

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
    float* to_be_normalized_values,
    float* activations_out,
    float* gamma,
    float* beta,
    float rms_scale
  ) = 0;

  virtual void NormalizeThenActivate_Tanh(
    size_t array_length,
    float* to_be_normalized_values,
    float* activations_out,
    float* gamma,
    float* beta,
    float rms_scale
  ) = 0;

  virtual void AccumulateLstmGradients(
    size_t hidden_size,
    size_t concatenated_hidden_size,
    size_t vocabulary_size,
    size_t layer_id,
    float* error_on_output,
    float* hidden_gradient_accumulator,
    float* output_weights
  ) = 0;

  virtual void AccumulateLstmLayerGradients(
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
    float* last_cell_state
  ) = 0;

  virtual void BackpropagateErrors(
    size_t len,                       // hidden_size (200)
    size_t base_offset,               // 0 for temporal, hidden_size for spatial
    size_t component_input_dim,       // Layer 0: 200, Layer 1: 400
    float* weights,                   // Weight matrix
    float* pre_activation_gradients,  // Current layer errors
    float* grad_store                 // Where to accumulate gradients
  ) = 0;

  virtual void AccumulateLayerGradients(
    const size_t hidden_size,
    const size_t vocabulary_size,
    const size_t component_input_dim,
    const float* input,
    const float* pre_activation_gradients,
    float* embedding_ptr,
    float* weight_gradients
  ) = 0;

  virtual void AccumulateOutputLayerGradients(
    size_t previous_output_offset,
    float* error_on_output,
    float* output_weight_gradients,
    float* output_bias_gradients,
    const float* hidden_ptr,
    const size_t vocabulary_size,
    const size_t concatenated_hidden_size,
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
    size_t const all_layer_inputs,
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


