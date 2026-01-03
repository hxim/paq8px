#pragma once

#include "LstmLayer.hpp"
#include "../file/BitFileDisk.hpp"
#include "../file/OpenFromMyFolder.hpp"
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

namespace LSTM {
  struct Shape {
    size_t output_size;
    size_t num_cells;
    size_t num_layers;
    size_t horizon;
  };
}

class Lstm {
private:
  SIMDType simd;
  std::vector<std::unique_ptr<LstmLayer>> layers;
  Array<float, 32> layer_input;    // [horizon * num_layers * max_layer_input_size]

  Array<float, 32> output_layer;   // [output_size * hidden_size] = 256 * 400
  Array<float, 32> output_layer_u; // [output_size * hidden_size] = 256 * 400 - gradients

  Array<float, 32> output;         // [horizon * output_size]
  Array<float, 32> logits;         // [horizon * output_size]
  Array<float, 32> hidden;
  Array<float, 32> hidden_error;
  Array<float, 32> output_bias;    // [output_size] = 256
  Array<float, 32> output_bias_u;  // [output_size] = 256
  std::vector<uint8_t> input_history;

  std::unique_ptr<Adam> output_weights_optimizer;
  std::unique_ptr<Adam> output_bias_optimizer;
  PolynomialDecay output_decay_func;
  float output_learning_rate;

  uint64_t saved_timestep;
  size_t num_cells;
  size_t horizon;
  size_t current_sequence_size_target = 6; //6..horizon-1
  size_t sequence_step_target = 12;
  size_t sequence_step_cntr = 0;
  size_t output_size;
  size_t num_layers;
  std::unique_ptr<VectorFunctions> VectorFunctions;

public:
  size_t epoch;

  Lstm(
    SIMDType simdType,
    LSTM::Shape shape);

  float* Predict(uint8_t input);
  void Perceive(uint8_t input);
};
