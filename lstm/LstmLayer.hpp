#pragma once

#include "LstmGate.hpp"
#include "../Array.hpp"
#include "../SIMDType.hpp"
#include <vector>
#include <cstdint>

class LstmLayer {
private:
  const SIMDType simd;
  std::unique_ptr<VectorFunctions> VectorFunctions;

  Array<float, 32> cell_state;
  Array<float, 32> cell_state_gradient;
  Array<float, 32> temporal_hidden_gradient;

  Array<float, 32> tanh_state;         // Flat: [horizon * num_cells]
  Array<float, 32> input_gate_complement; // Flat: [horizon * num_cells]
  Array<float, 32> last_cell_state;    // Flat: [horizon * num_cells]

  const size_t hidden_size;
  const size_t num_cells;

  LstmGate forget_gate;
  LstmGate input_gate;
  LstmGate output_gate;

  static float Rand(float range);

public:
  LstmLayer(
    SIMDType simdType,
    float tuning_param,
    size_t vocabulary_size,
    size_t hidden_size,
    size_t num_cells,
    size_t horizon,
    float range = 0.4f);

  void ForwardPass(
    float* input,
    uint8_t const input_symbol,
    float* hidden,
    size_t const sequence_position,
    size_t sequence_length);

  void InitializeBackwardPass();

  void BackwardPass(
    float* input,
    size_t const sequence_position,
    size_t const layer_id,
    uint8_t const input_symbol,
    float* hidden_gradient);

  void Optimize(uint64_t const training_iterations);

  void SaveWeights(LoadSave& stream);
  void LoadWeights(LoadSave& stream);
};
