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
  size_t const embedding_size,   // 256 (vocabulary size)
  size_t const hidden_size,      // Layer 0: 200 (200*1), Layer 1: 400 (200*2)
  size_t const num_cells,        // 200
  size_t const horizon,          // 100
  float const range)
  : simd(simdType)
  , state(num_cells)             // 200
  , state_error(num_cells)       // 200
  , stored_error(num_cells)      // 200
  , tanh_state(horizon * num_cells)       // 100 * 200 = 20,000
  , input_gate_state(horizon * num_cells) // 100 * 200 = 20,000
  , last_state(horizon * num_cells)       // 100 * 200 = 20,000
  , num_cells(num_cells)         // 200
  , epoch(0)
  , horizon(horizon)             // 100
  , forget_gate(
    simdType,
    embedding_size,            // 256
    hidden_size,               // Layer 0: 200, Layer 1: 400
    num_cells,                 // 200
    horizon,                   // 100
    false,                     // useTanh
    1.0f,                      // bias_init
    0.9995f,                   // beta2
    1e-6f,                     // epsilon
    0.007f,                    // learningRate
    0.001f,                    // endLearningRate
    0.0005f,                   // decayMultiplier
    1.0f / 2.0f,               // decayExponent
    0)                         // decaySteps
  , input_node(
    simdType,
    embedding_size,            // 256
    hidden_size,               // Layer 0: 200, Layer 1: 400
    num_cells,                 // 200
    horizon,                   // 100
    true,                      // useTanh
    0.0f,                      // bias_init
    0.9995f,                   // beta2
    1e-6f,                     // epsilon
    0.007f,                    // learningRate
    0.001f,                    // endLearningRate
    0.0005f,                   // decayMultiplier
    1.0f / 2.0f,               // decayExponent
    0)                         // decaySteps
  , output_gate(
    simdType,
    embedding_size,            // 256
    hidden_size,               // Layer 0: 200, Layer 1: 400
    num_cells,                 // 200
    horizon,                   // 100
    false,                     // useTanh
    0.0f,                      // bias_init
    0.9995f,                   // beta2
    1e-6f,                     // epsilon
    0.007f,                    // learningRate
    0.001f,                    // endLearningRate
    0.0005f,                   // decayMultiplier
    1.0f / 2.0f,               // decayExponent
    0)                         // decaySteps
  , update_steps(0)
{
  // Set random weights for each gate
  float* forget_emb = &forget_gate.embedding[0];
  float* input_emb = &input_node.embedding[0];
  float* output_emb = &output_gate.embedding[0];

  float* forget_w = &forget_gate.weights[0];
  float* input_w = &input_node.weights[0];
  float* output_w = &output_gate.weights[0];

  // Initialize embeddings
  for (size_t i = 0; i < num_cells; i++) {          // 200 iterations
    for (size_t j = 0; j < embedding_size; j++) {   // 256 iterations
      forget_emb[i * embedding_size + j] = Rand(range);
      input_emb[i * embedding_size + j] = Rand(range);
      output_emb[i * embedding_size + j] = Rand(range);
    }
  }

  // Initialize hidden state weights
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
  size_t input_size,
  uint8_t const input_symbol,
  float* hidden,
  size_t current_sequence_size_target)
{
  const size_t ebase = epoch * num_cells;            // epoch * 200

  float* fg_state = &forget_gate.state[0];
  float* ig_state = &input_node.state[0];
  float* og_state = &output_gate.state[0];

  // Copy current state to last_state for this epoch
  float* src = &state[0];
  float* dst = &last_state[ebase];
  memcpy(dst, src, num_cells * sizeof(float));

  forget_gate.ForwardPass(input, input_size, input_symbol, epoch);
  input_node.ForwardPass(input, input_size, input_symbol, epoch);
  output_gate.ForwardPass(input, input_size, input_symbol, epoch);

  for (size_t i = 0; i < num_cells; i++) {          // 200 iterations
    const size_t idx = ebase + i;                   // epoch*200 + i
    const float forget = fg_state[idx];
    const float inputv = ig_state[idx];
    const float output = og_state[idx];

    const float input_gate = 1.0f - forget;
    input_gate_state[idx] = input_gate;

    state[i] = state[i] * forget + inputv * input_gate;

    const float t = tanh_pade_clipped(state[i]);
    tanh_state[idx] = t;

    hidden[i] = output * t;
  }

  epoch++;
  if (epoch == current_sequence_size_target)
    epoch = 0;
}

void LstmLayer::BackwardPass(
  float* input,
  size_t input_size,
  size_t const epoch,
  size_t current_sequence_size_target,
  size_t const layer,
  uint8_t const input_symbol,
  float* hidden_error)
{
  const size_t ebase = epoch * num_cells;            // epoch * 200

  float* fg_state = &forget_gate.state[0];
  float* ig_state = &input_node.state[0];
  float* og_state = &output_gate.state[0];

  float* fg_error = &forget_gate.error[0];
  float* ig_error = &input_node.error[0];
  float* og_error = &output_gate.error[0];

  if (epoch == current_sequence_size_target - 1) {
    memcpy(&stored_error[0], &hidden_error[0], num_cells * sizeof(float));
    memset(&state_error[0], 0, num_cells * sizeof(float));
  }

  for (size_t i = 0; i < num_cells; i++) {          // 200 iterations
    const size_t idx = ebase + i;                   // epoch*200 + i

    stored_error[i] += hidden_error[i];

    const float tanh_v = tanh_state[idx];
    const float forget = fg_state[idx];
    const float inputv = ig_state[idx];
    const float output = og_state[idx];
    const float input_gate = input_gate_state[idx];

    og_error[i] =
      tanh_v * stored_error[i] *
      output * (1.0f - output);

    state_error[i] +=
      stored_error[i] * output *
      (1.0f - tanh_v * tanh_v);

    ig_error[i] =
      state_error[i] * input_gate *
      (1.0f - inputv * inputv);

    fg_error[i] =
      (last_state[idx] - inputv) *
      state_error[i] *
      forget * input_gate;

    hidden_error[i] = 0.0f;

    if (epoch > 0) {
      state_error[i] *= forget;
      stored_error[i] = 0.0f;
    }
  }

  if (epoch == 0)
    update_steps++;

  forget_gate.BackwardPass(
    input,
    input_size,
    hidden_error,
    &stored_error[0],
    update_steps,
    epoch,
    layer,
    input_symbol);
  input_node.BackwardPass(
    input,
    input_size,
    hidden_error,
    &stored_error[0],
    update_steps,
    epoch,
    layer,
    input_symbol);
  output_gate.BackwardPass(
    input,
    input_size,
    hidden_error,
    &stored_error[0],
    update_steps,
    epoch,
    layer,
    input_symbol);
}
