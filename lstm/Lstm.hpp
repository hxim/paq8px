#pragma once

#include "LstmLayer.hpp"

#include "../Utils.hpp"
#include "../Array.hpp"
#include "../SIMDType.hpp"
#include "Adam.hpp"
#ifdef X64_SIMD_AVAILABLE
#include "Adam_AVX.hpp"
#endif
#include "Adam_Scalar.hpp"
#include "PolynomialDecay.hpp"
#include <cstdint>

struct LstmShape {
  size_t vocabulary_size;
  size_t num_cells;
  size_t num_layers;
  size_t horizon;
};

class Lstm {
private:
  SIMDType simd;
  std::unique_ptr<VectorFunctions> VectorFunctions;

  std::vector<std::unique_ptr<LstmLayer>> layers;
  Array<float, 32> all_layer_inputs;             // [horizon * num_layers * max_layer_input_size]

  Array<float, 32> output_weights;          // [vocabulary_size * hidden_size] = 256 * 400
  Array<float, 32> output_weight_gradients; // [vocabulary_size * hidden_size] = 256 * 400 - gradients

  Array<float, 32> output_probabilities;    // [horizon * vocabulary_size] - used both as raw output probs, then when the target_symbol is revealed, it's the error_on_output
  Array<float, 32> logits;                  // [horizon * vocabulary_size]
  Array<float, 32> hidden_states_all_layers;
  Array<float, 32> hidden_gradient;
  Array<float, 32> output_bias;             // [vocabulary_size] = 256
  Array<float, 32> output_bias_gradients;   // [vocabulary_size] = 256

  std::vector<uint8_t> input_symbol_history;

  std::unique_ptr<Adam> output_weights_optimizer;
  std::unique_ptr<Adam> output_bias_optimizer;
  PolynomialDecay learning_rate_scheduler;

  size_t num_cells;
  size_t horizon;
  size_t vocabulary_size;
  size_t num_layers;

  size_t sequence_length;
  size_t sequence_step_target;
  size_t sequence_step_cntr;

  float tuning_param;

public:
  size_t sequence_position; // 0..sequence_length-1
  size_t training_iterations; // 1..n

  Lstm(
    SIMDType simdType,
    LstmShape shape,
    float tuning_param
  );

  float* Predict(uint8_t input_symbol);
  void Perceive(uint8_t target_symbol);

  void SaveModelParameters(LoadSave& stream);
  void LoadModelParameters(LoadSave& stream);
};
