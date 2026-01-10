#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <cassert>
#include <cstring>

#include "LstmLayer.hpp"

float LstmLayer::Rand(float const range) {
  return ((static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX)) - 0.5f) * range;
}

LstmLayer::LstmLayer(
  SIMDType simdType,
  float tuning_param,
  size_t const vocabulary_size,  // 256
  size_t const hidden_size,      // Layer 0: 200 (200*1), Layer 1: 400 (200*2)
  size_t const num_cells,        // 200
  size_t const horizon,          // 100
  float const range)
  : simd(simdType)
  , cell_state(num_cells)                 // 200
  , cell_state_gradient(num_cells)        // 200
  , temporal_hidden_gradient(num_cells)     // 200
  , tanh_state(horizon * num_cells)       // 100 * 200 = 20,000
  , input_gate_complement(horizon * num_cells) // 100 * 200 = 20,000
  , last_cell_state(horizon * num_cells)  // 100 * 200 = 20,000
  , num_cells(num_cells)         // 200
  , hidden_size(hidden_size)     // Layer 0: 200 (200*1), Layer 1: 400 (200*2) - hidden_size = num_cells * (layer_id > 0 ? 2 : 1)
  , forget_gate(
    simdType,
    vocabulary_size,           // 256
    hidden_size,               // Layer 0: 200, Layer 1: 400
    num_cells,                 // 200
    horizon,                   // 100
    false,                     // useTanh
    1.0f,                      // bias_init
    0.01f,                     // learningRate_symbol_embeddings
    0.01f,                     // learningRate_resurrent_weights
    0.01f                      // learningRate_rms
  )
  , input_gate(
    simdType,
    vocabulary_size,           // 256
    hidden_size,               // Layer 0: 200, Layer 1: 400
    num_cells,                 // 200
    horizon,                   // 100
    true,                      // useTanh
    0.0f,                      // bias_init
    0.002f,                    // learningRate_symbol_embeddings
    0.002f,                    // learningRate_resurrent_weights
    0.002f                     // learningRate_rms
  )
  , output_gate(
    simdType,
    vocabulary_size,           // 256
    hidden_size,               // Layer 0: 200, Layer 1: 400
    num_cells,                 // 200
    horizon,                   // 100
    false,                     // useTanh
    0.0f,                      // bias_init
    0.013f,                    // learningRate_symbol_embeddings
    0.013f,                    // learningRate_resurrent_weights
    0.013f                     // learningRate_rms
  )
{

  VectorFunctions = CreateVectorFunctions(simd);

  // Set random weights for each gate
  float* forget_emb = &forget_gate.symbol_embeddings[0];
  float* input_emb = &input_gate.symbol_embeddings[0];
  float* output_emb = &output_gate.symbol_embeddings[0];

  float* forget_w = &forget_gate.recurrent_weights[0];
  float* input_w = &input_gate.recurrent_weights[0];
  float* output_w = &output_gate.recurrent_weights[0];

  // Initialize embeddings
  for (size_t i = 0; i < num_cells; i++) {          // 200 iterations
    for (size_t j = 0; j < vocabulary_size; j++) {   // 256 iterations
      forget_emb[i * vocabulary_size + j] = Rand(range);
      input_emb[i * vocabulary_size + j] = Rand(range);
      output_emb[i * vocabulary_size + j] = Rand(range);
    }
  }

  // Initialize hidden state recurrent_weights
  for (size_t i = 0; i < num_cells; i++) {          // 200 iterations
    for (size_t j = 0; j < hidden_size; j++) {      // Layer 0: 200, Layer 1: 400
      forget_w[i * hidden_size + j] = Rand(range);
      input_w[i * hidden_size + j] = Rand(range);
      output_w[i * hidden_size + j] = Rand(range);
    }
  }
}

void LstmLayer::ForwardPass(
  float* input,
  uint8_t const input_symbol,
  float* hidden,
  size_t const sequence_position,
  size_t sequence_length)
{
  const size_t seq_pos_offset = sequence_position * num_cells;            // sequence_position * 200

  float* forget_gate_outputs = &forget_gate.gate_outputs[0];
  float* input_gate_outputs = &input_gate.gate_outputs[0];
  float* output_gate_outputs = &output_gate.gate_outputs[0];

  // Copy current cell_state to last_cell_state for this sequence_position
  float* src = &cell_state[0];
  float* dst = &last_cell_state[seq_pos_offset];
  memcpy(dst, src, num_cells * sizeof(float));

  forget_gate.ForwardPass(input, input_symbol, sequence_position);
  input_gate.ForwardPass(input, input_symbol, sequence_position);
  output_gate.ForwardPass(input, input_symbol, sequence_position);

  for (size_t i = 0; i < num_cells; i++) {          // 200 iterations
    const size_t idx = seq_pos_offset + i;         // sequence_position*200 + i
    const float forget = forget_gate_outputs[idx];
    const float inputv = input_gate_outputs[idx];
    const float output = output_gate_outputs[idx];

    const float input_gate_value = 1.0f - forget;
    input_gate_complement[idx] = input_gate_value;

    cell_state[i] = cell_state[i] * forget + inputv * input_gate_value;

    const float t = tanh_pade_clipped(cell_state[i]);
    tanh_state[idx] = t;

    hidden[i] = output * t;
  }
}

void LstmLayer::InitializeBackwardPass() {
  memset(&temporal_hidden_gradient[0], 0, num_cells * sizeof(float));
  memset(&cell_state_gradient[0], 0, num_cells * sizeof(float));
}

void LstmLayer::BackwardPass(
  float* layer_input_ptr,
  size_t const sequence_position,
  size_t const layer_id,
  uint8_t const input_symbol,
  float* hidden_gradient)
{
  VectorFunctions->AccumulateLstmLayerGradients(
    num_cells,
    sequence_position * num_cells, //sequence_position_offset
    &temporal_hidden_gradient[0],
    &hidden_gradient[0],
    &tanh_state[0],
    &forget_gate.gate_outputs[0],
    &input_gate.gate_outputs[0],
    &output_gate.gate_outputs[0],
    &input_gate_complement[0],
    &output_gate.gate_gradient_buffer[0],
    &cell_state_gradient[0], 
    &input_gate.gate_gradient_buffer[0],
    &forget_gate.gate_gradient_buffer[0],
    &last_cell_state[0]
  );

  forget_gate.BackwardPass(
    layer_input_ptr,
    hidden_gradient,
    &temporal_hidden_gradient[0],
    sequence_position,
    layer_id,
    input_symbol);
  input_gate.BackwardPass(
    layer_input_ptr,
    hidden_gradient,
    &temporal_hidden_gradient[0],
    sequence_position,
    layer_id,
    input_symbol);
  output_gate.BackwardPass(
    layer_input_ptr,
    hidden_gradient,
    &temporal_hidden_gradient[0],
    sequence_position,
    layer_id,
    input_symbol);
}

void LstmLayer::Optimize(const float lr_scale, const float beta2) {
  forget_gate.Optimize(lr_scale, beta2);
  input_gate.Optimize(lr_scale, beta2);
  output_gate.Optimize(lr_scale, beta2);
}

void LstmLayer::Rescale(float scale) {
  forget_gate.Rescale(scale);
  input_gate.Rescale(scale);
  output_gate.Rescale(scale);
}

void LstmLayer::SaveWeights(LoadSave& stream) {
  // Save weights for all three gates
  forget_gate.SaveWeights(stream);
  input_gate.SaveWeights(stream);
  output_gate.SaveWeights(stream);
}

void LstmLayer::LoadWeights(LoadSave& stream) {
  // Load weights for all three gates
  forget_gate.LoadWeights(stream);
  input_gate.LoadWeights(stream);
  output_gate.LoadWeights(stream);
}
