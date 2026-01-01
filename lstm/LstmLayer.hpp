#pragma once

#include "Layer.hpp"
#include "../Array.hpp"
#include "../SIMDType.hpp"
#include <vector>
#include <cstdint>

class LstmLayer {
private:
  SIMDType simd;

  Array<float, 32> state;
  Array<float, 32> state_error;
  Array<float, 32> stored_error;

  Array<float, 32> tanh_state;         // Flat: [horizon * num_cells]
  Array<float, 32> input_gate_state;   // Flat: [horizon * num_cells]
  Array<float, 32> last_state;         // Flat: [horizon * num_cells]

  size_t num_cells;
  size_t epoch;
  size_t horizon;

  Layer forget_gate;
  Layer input_node;
  Layer output_gate;

  static float Rand(float range);

public:
  uint64_t update_steps;

  LstmLayer(
    SIMDType simdType,
    size_t embedding_size,
    size_t hidden_size,
    size_t num_cells,
    size_t horizon,
    float range = 0.4f);

  void ForwardPass(
    const Array<float, 32>& input,
    uint8_t input_symbol,
    float* hidden,
    size_t current_sequence_size_target);

  void BackwardPass(
    const Array<float, 32>& input,
    size_t epoch,
    size_t current_sequence_size_target,
    size_t layer,
    uint8_t input_symbol,
    float* hidden_error);
};
