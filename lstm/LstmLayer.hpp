#pragma once

#include "LstmComponent.hpp"
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
  Array<float, 32> last_cell_state;    // Flat: [horizon * num_cells]

  const size_t total_component_inputs;
  const size_t num_cells;

  LstmComponent forget_gate;
  LstmComponent cell_candidate;
  LstmComponent output_gate;

public:
  LstmLayer(
    SIMDType simdType,
    float tuning_param,
    size_t const layer_id,
    size_t vocabulary_size,
    size_t num_cells,
    size_t horizon);

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

  void Optimize(const float lr_scale, const float beta2);

  void Rescale(float scale);

  void SaveWeights(LoadSave& stream);
  void LoadWeights(LoadSave& stream);
};
