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
#include "SqrtLearningRateDecay.hpp"
#include <cstdint>

struct LstmShape {
  size_t vocabulary_size;
  size_t hidden_size;
  size_t num_layers;
  size_t horizon;
};

class Lstm {
private:
  std::unique_ptr<VectorFunctions> VectorFunctions;

  std::vector<std::unique_ptr<LstmLayer>> layers;
  Array<float, 32> all_layer_inputs;

  Array<float, 32> output_weights;
  Array<float, 32> output_weight_gradients;

  Array<float, 32> output_probabilities;
  Array<float, 32> logits;
  Array<float, 32> concatenated_hidden_states;
  Array<float, 32> hidden_gradient_accumulator;
  Array<float, 32> output_bias;
  Array<float, 32> output_bias_gradients;

  std::vector<uint8_t> input_symbol_history;

  std::unique_ptr<Adam> output_weights_optimizer;
  std::unique_ptr<Adam> output_bias_optimizer;
  SqrtLearningRateDecay learning_rate_scheduler;

  size_t hidden_size;
  size_t horizon;
  size_t vocabulary_size;
  size_t num_layers;

  size_t current_sequence_length;
  size_t sequence_step_target;
  size_t sequence_step_counter;

  float tuning_param;

public:
  size_t sequence_position; // 0..current_sequence_length-1
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
