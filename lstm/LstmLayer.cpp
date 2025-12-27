#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <cassert>
#include <cstring>

#include "LstmLayer.hpp"
#include "SimdFunctions.hpp"

float LstmLayer::Rand(float const range) {
  return ((static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX)) - 0.5f) * range;
}

LstmLayer::LstmLayer(
  SIMDType simdType,
  size_t const input_size,       // Layer 0: 456, Layer 1: 656
  size_t const output_size,      // 256
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
    input_size,                // Layer 0: 456, Layer 1: 656
    output_size,               // 256
    num_cells,                 // 200
    horizon,                   // 100
    false,                     // useTanh
    0.9999f,                   // beta2
    1e-6f,                     // epsilon
    0.007f,                    // learningRate
    0.001f,                    // endLearningRate
    0.0005f,                   // decayMultiplier
    1.0f,                      // powerNumerator
    2.0f,                      // powerDenominator
    0)                         // decaySteps
  , input_node(
    simdType,
    input_size,                // Layer 0: 456, Layer 1: 656
    output_size,               // 256
    num_cells,                 // 200
    horizon,                   // 100
    true,                      // useTanh
    0.9999f,                   // beta2
    1e-6f,                     // epsilon
    0.007f,                    // learningRate
    0.001f,                    // endLearningRate
    0.0005f,                   // decayMultiplier
    1.0f,                      // powerNumerator
    2.0f,                      // powerDenominator
    0)                         // decaySteps
  , output_gate(
    simdType,
    input_size,                // Layer 0: 456, Layer 1: 656
    output_size,               // 256
    num_cells,                 // 200
    horizon,                   // 100
    false,                     // useTanh
    0.9999f,                   // beta2
    1e-6f,                     // epsilon
    0.007f,                    // learningRate
    0.001f,                    // endLearningRate
    0.0005f,                   // decayMultiplier
    1.0f,                      // powerNumerator
    2.0f,                      // powerDenominator
    0)                         // decaySteps
  , update_steps(0)
{
  // Set random weights for each gate
  float* forget_w = &forget_gate.weights[0];
  float* input_w = &input_node.weights[0];
  float* output_w = &output_gate.weights[0];

  size_t idx = 0;
  for (size_t i = 0; i < num_cells; i++) {          // 200 iterations

    // Set random weights for each gate
    for (size_t j = 0; j < input_size; j++) {       // Layer 0: 456, Layer 1: 656
      forget_w[idx + j] = Rand(range);
      input_w[idx + j] = Rand(range);
      output_w[idx + j] = Rand(range);
    }

    idx += input_size;                              // Layer 0: += 456, Layer 1: += 656
  }

}

void LstmLayer::ForwardPass(
  const Array<float, 32>& input,
  uint8_t const input_symbol,
  Array<float, 32>* hidden,
  size_t const hidden_start)
{
  const size_t ebase = epoch * num_cells;            // epoch * 200

  float* fg_state = &forget_gate.state[0];
  float* ig_state = &input_node.state[0];
  float* og_state = &output_gate.state[0];

  // Copy current state to last_state for this epoch
  for (size_t i = 0; i < num_cells; i++) {          // 200 iterations
    last_state[ebase + i] = state[i];               // last_state[epoch*200 + i]
  }

  forget_gate.ForwardPass(
    input,
    input_symbol,
    epoch);
  input_node.ForwardPass(
    input,
    input_symbol,
    epoch);
  output_gate.ForwardPass(
    input,
    input_symbol,
    epoch);

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

    (*hidden)[hidden_start + i] = output * t;
  }

  epoch++;
  if (epoch == horizon)                              // if epoch == 100
    epoch = 0;
}



void LstmLayer::BackwardPass(
  const Array<float, 32>& input,
  size_t const epoch,
  size_t const layer,
  uint8_t const input_symbol,
  Array<float, 32>* hidden_error)
{
  const size_t ebase = epoch * num_cells;            // epoch * 200

  float* fg_state = &forget_gate.state[0];
  float* ig_state = &input_node.state[0];
  float* og_state = &output_gate.state[0];

  float* fg_error = &forget_gate.error[0];
  float* ig_error = &input_node.error[0];
  float* og_error = &output_gate.error[0];

  for (size_t i = 0; i < num_cells; i++) {          // 200 iterations
    const size_t idx = ebase + i;                   // epoch*200 + i

    if (epoch == horizon - 1) {                     // if epoch == 99
      stored_error[i] = (*hidden_error)[i];
      state_error[i] = 0.0f;
    }
    else {
      stored_error[i] += (*hidden_error)[i];
    }

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

    (*hidden_error)[i] = 0.0f;

    if (epoch > 0) {
      state_error[i] *= forget;
      stored_error[i] = 0.0f;
    }
  }

  if (epoch == 0)
    update_steps++;

  forget_gate.BackwardPass(
    input,
    hidden_error,
    &stored_error,
    update_steps,
    epoch,
    layer,
    input_symbol);
  input_node.BackwardPass(
    input,
    hidden_error,
    &stored_error,
    update_steps,
    epoch,
    layer,
    input_symbol);
  output_gate.BackwardPass(
    input,
    hidden_error,
    &stored_error,
    update_steps,
    epoch,
    layer,
    input_symbol);
}



void LstmLayer::Reset() {
  forget_gate.Reset();
  input_node.Reset();
  output_gate.Reset();

  memset(
    &tanh_state[0],
    0,
    horizon * num_cells * sizeof(float));          // 100 * 200 * 4 = 80,000 bytes
  memset(
    &input_gate_state[0],
    0,
    horizon * num_cells * sizeof(float));          // 100 * 200 * 4 = 80,000 bytes
  memset(
    &last_state[0],
    0,
    horizon * num_cells * sizeof(float));          // 100 * 200 * 4 = 80,000 bytes
  memset(
    &state[0],
    0,
    num_cells * sizeof(float));                    // 200 * 4 = 800 bytes
  memset(
    &state_error[0],
    0,
    num_cells * sizeof(float));                    // 200 * 4 = 800 bytes
  memset(
    &stored_error[0],
    0,
    num_cells * sizeof(float));                    // 200 * 4 = 800 bytes

  epoch = 0;
  update_steps = 0;
}

LstmLayer::WeightArrays LstmLayer::GetWeights() {
  WeightArrays arrays;
  arrays.forget_gate_weights = &forget_gate.weights[0];
  arrays.input_node_weights = &input_node.weights[0];
  arrays.output_gate_weights = &output_gate.weights[0];
  arrays.size_per_gate = forget_gate.weights.size(); // Layer 0: 91,200, Layer 1: 131,200
  return arrays;
}
