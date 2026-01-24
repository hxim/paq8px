#pragma once

#include "LstmComponent.hpp"
#include "../Array.hpp"
#include "../SIMDType.hpp"
#include <vector>
#include <cstdint>

class LstmLayer {
private:
  std::unique_ptr<VectorFunctions> vectorFunctions;

  Array<float, 32> cell_state;
  Array<float, 32> cell_state_gradient;
  Array<float, 32> gradient_from_next_timestep;

  Array<float, 32> tanh_state;
  Array<float, 32> last_cell_state;

  const size_t component_input_dim;
  const size_t hidden_size;

  LstmComponent forget_gate;
  LstmComponent cell_candidate;
  LstmComponent output_gate;

public:
  LstmLayer(
    SIMDType simdType,
    float tuning_param,
    size_t const layer_id,
    size_t vocabulary_size,
    size_t hidden_size,
    size_t horizon);

  void ForwardPass(
    float* input,
    uint8_t const input_symbol,
    float* hidden_state_out,
    size_t const sequence_position,
    size_t current_sequence_length);

  void InitializeBackwardPass();

  void BackwardPass(
    float* input,
    size_t const sequence_position,
    size_t const layer_id,
    uint8_t const input_symbol,
    float* hidden_gradient_accumulator);

  void Optimize(const float lr_scale, const float beta2);

  void Rescale(float scale);

  void SaveWeights(LoadSave& stream);
  void LoadWeights(LoadSave& stream);
};
