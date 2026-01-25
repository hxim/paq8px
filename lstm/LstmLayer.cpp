#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <cassert>
#include <cstring>

#include "LstmLayer.hpp"

LstmLayer::LstmLayer(
  SIMDType simdType,
  float tuning_param,
  size_t const layer_id,         // 0, 1
  size_t const vocabulary_size,  // 256
  size_t const hidden_size,      // 200
  size_t const horizon)          // 100
  : cell_state(hidden_size)                   // 200
  , cell_state_gradient(hidden_size)          // 200
  , gradient_from_next_timestep(hidden_size)  // 200
  , tanh_state(horizon * hidden_size)         // 100 * 200 = 20,000
  , last_cell_state(horizon * hidden_size)    // 100 * 200 = 20,000
  , hidden_size(hidden_size)                  // 200
  , component_input_dim(layer_id > 0 ? 2 * hidden_size : hidden_size) // recurrent inputs only or recurrent inputs + hidden state from the previous layer
  , forget_gate(
    simdType,
    vocabulary_size,                      // 256
    component_input_dim,                  // Layer 0: 200, Layer 1: 400
    hidden_size,                          // 200
    horizon,                              // 100
    false,                                // use_tanh
    1.0f,                                 // bias_init
    layer_id == 0 ? 0.0175f : 0.0175f,    // learning_rate_symbol_embeddings
    layer_id == 0 ? 0.005f : 0.007f,      // learning_rate_bias
    layer_id == 0 ? 0.050f : 0.014f,      // learning_rate_recurrent_weights
    layer_id == 0 ? 0.001f : 0.012f,      // learning_rate_rms_gamma
    layer_id == 0 ? 0.011f : 0.011f       // learning_rate_rms_beta
  )
  , cell_candidate(
    simdType,
    vocabulary_size,                      // 256
    component_input_dim,                  // Layer 0: 200, Layer 1: 400
    hidden_size,                          // 200
    horizon,                              // 100
    true,                                 // use_tanh
    0.0f,                                 // bias_init
    layer_id == 0 ? 0.003f : 0.007f,      // learning_rate_symbol_embeddings
    layer_id == 0 ? 0.002f : 0.002f,      // learning_rate_bias
    layer_id == 0 ? 0.00175f : 0.00175f,  // learning_rate_recurrent_weights
    layer_id == 0 ? 0.002f : 0.002f,      // learning_rate_rms_gamma
    layer_id == 0 ? 0.002f : 0.002f       // learning_rate_rms_beta
  )
  , output_gate(
    simdType,
    vocabulary_size,                      // 256
    component_input_dim,                  // Layer 0: 200, Layer 1: 400
    hidden_size,                          // 200
    horizon,                              // 100
    false,                                // use_tanh
    0.0f,                                 // bias_init
    layer_id == 0 ? 0.0225f : 0.0125f,    // learning_rate_symbol_embeddings
    layer_id == 0 ? 0.0130f : 0.0130f,    // learning_rate_bias
    layer_id == 0 ? 0.0125f : 0.0070f,    // learning_rate_recurrent_weights
    layer_id == 0 ? 0.0120f : 0.0130f,    // learning_rate_rms_gamma
    layer_id == 0 ? 0.0120f : 0.0130f     // learning_rate_rms_beta
    )
{

  vectorFunctions = CreateVectorFunctions(simdType);

  // Initialize embedding matrices with random weights in each component
  // All other weights (recurrent, from previous layer, biases) are left as zeroes

  float* forget_emb = &forget_gate.symbol_embeddings[0];
  float* input_emb = &cell_candidate.symbol_embeddings[0];
  float* output_emb = &output_gate.symbol_embeddings[0];

  float fan_in = 1.0f;
  float fan_out = (float)hidden_size;  // 200
  float range = 2.0f * std::sqrt(6.0f / (fan_in + fan_out)); // ~ 0.345; for uniform [-0.5, 0.5]

  // Initialize component embeddings
  for (size_t i = 0; i < hidden_size * vocabulary_size; i++) {
    forget_emb[i] = LstmLayer_Rand(range);
    input_emb[i] = LstmLayer_Rand(range);
    output_emb[i] = LstmLayer_Rand(range);
  }

}

void LstmLayer::ForwardPass(
  float* input,
  uint8_t const input_symbol,
  float* hidden_state_out,
  size_t const sequence_position,
  size_t current_sequence_length)
{
  const size_t seq_pos_offset = sequence_position * hidden_size;            // sequence_position * 200

  float* forget_gate_activations = &forget_gate.activations[0];
  float* cell_candidate_activations = &cell_candidate.activations[0];
  float* output_activations = &output_gate.activations[0];

  // Copy current cell_state to last_cell_state for this sequence_position
  float* src = &cell_state[0];
  float* dst = &last_cell_state[seq_pos_offset];
  vectorFunctions->Copy(dst, src, hidden_size);

  forget_gate.ForwardPass(input, input_symbol, sequence_position);
  cell_candidate.ForwardPass(input, input_symbol, sequence_position);
  output_gate.ForwardPass(input, input_symbol, sequence_position);

  for (size_t i = 0; i < hidden_size; i++) {          // 200 iterations
    const size_t idx = seq_pos_offset + i;         // sequence_position*200 + i
    const float forget_gate_i = forget_gate_activations[idx];
    const float cell_candidate_i = cell_candidate_activations[idx];
    const float output_gate_i = output_activations[idx];
    const float input_gate_i = 1.0f - forget_gate_i;

    cell_state[i] = cell_state[i] * forget_gate_i + cell_candidate_i * input_gate_i;

    const float t = tanh_pade_clipped(cell_state[i]);
    tanh_state[idx] = t;

    hidden_state_out[i] = output_gate_i * t;
  }
}

void LstmLayer::InitializeBackwardPass() {
  vectorFunctions->Zero(&gradient_from_next_timestep[0], hidden_size);
  vectorFunctions->Zero(&cell_state_gradient[0], hidden_size);
}

void LstmLayer::BackwardPass(
  float* layer_input_ptr,
  size_t const sequence_position,
  size_t const layer_id,
  uint8_t const input_symbol,
  float* hidden_gradient_accumulator)
{
  vectorFunctions->AccumulateLstmLayerGradients(
    hidden_size,
    sequence_position * hidden_size, //timestep_offset
    &gradient_from_next_timestep[0],
    &hidden_gradient_accumulator[0],
    &tanh_state[0],
    &forget_gate.activations[0],
    &cell_candidate.activations[0],
    &output_gate.activations[0],
    &output_gate.pre_activation_gradients[0],
    &cell_state_gradient[0], 
    &cell_candidate.pre_activation_gradients[0],
    &forget_gate.pre_activation_gradients[0],
    &last_cell_state[0]
  );

  forget_gate.BackwardPass(
    layer_input_ptr,
    hidden_gradient_accumulator,
    &gradient_from_next_timestep[0],
    sequence_position,
    layer_id,
    input_symbol);
  cell_candidate.BackwardPass(
    layer_input_ptr,
    hidden_gradient_accumulator,
    &gradient_from_next_timestep[0],
    sequence_position,
    layer_id,
    input_symbol);
  output_gate.BackwardPass(
    layer_input_ptr,
    hidden_gradient_accumulator,
    &gradient_from_next_timestep[0],
    sequence_position,
    layer_id,
    input_symbol);
}

void LstmLayer::Optimize(const float lr_scale, const float beta2) {
  forget_gate.Optimize(lr_scale, beta2);
  cell_candidate.Optimize(lr_scale, beta2);
  output_gate.Optimize(lr_scale, beta2);
}

void LstmLayer::Rescale(float scale) {
  forget_gate.Rescale(scale);
  cell_candidate.Rescale(scale);
  output_gate.Rescale(scale);
}

void LstmLayer::SaveWeights(LoadSave& stream) {
  // Save weights for all three gates
  forget_gate.SaveWeights(stream);
  cell_candidate.SaveWeights(stream);
  output_gate.SaveWeights(stream);
}

void LstmLayer::LoadWeights(LoadSave& stream) {
  // Load weights for all three gates
  forget_gate.LoadWeights(stream);
  cell_candidate.LoadWeights(stream);
  output_gate.LoadWeights(stream);
}
