#pragma once

#include "Adam.hpp"
#include "Adam_Scalar.hpp"
#include "VectorFunctions.hpp"
#include "VectorFunctions_Scalar.hpp"
#ifdef X64_SIMD_AVAILABLE
#include "Adam_SSE2.hpp"
#include "Adam_AVX.hpp"
#include "VectorFunctions_SSE2.hpp"
#include "VectorFunctions_AVX2.hpp"
#endif

#include "SqrtLearningRateDecay.hpp"
#include "LoadSave.hpp"
#include "../SIMDType.hpp"
#include "../Array.hpp"
#include <cstdint>
#include <memory>

std::unique_ptr<VectorFunctions> CreateVectorFunctions(SIMDType simd);

std::unique_ptr<Adam> CreateOptimizer(
  SIMDType simd,
  size_t length,
  float* w,
  float* g,
  float base_lr
);

class LstmComponent{
public:
  SIMDType simd;

  Array<float, 32> symbol_embeddings;           // Flat: [num_cells * vocabulary_size] - symbol_embeddings matrix
  Array<float, 32> symbol_embedding_gradients;  // Flat: [num_cells * vocabulary_size] - symbol_embeddings gradients

  Array<float, 32> weights;           // Flat: [num_cells * hidden_size] - hidden state weights only
  Array<float, 32> weight_gradients;  // Flat: [num_cells * hidden_size] - hidden state gradients

  Array<float, 32> pre_norm_values;             // Flat: [horizon * num_cells]
  Array<float, 32> activations;                 // Flat: [horizon * num_cells]

  Array<float, 32> inverse_variance;

  Array<float, 32> gamma;
  Array<float, 32> gamma_gradients;

  Array<float, 32> beta;
  Array<float, 32> beta_gradients;

  Array<float, 32> gate_gradient_buffer;

  // Biases
  Array<float, 32> bias;                  // [num_cells] - gate bias
  Array<float, 32> bias_gradients;        // [num_cells] - gate bias gradients
  std::unique_ptr<Adam> bias_optimizer;

  size_t vocabulary_size;                 // Vocabulary size / symbol_embeddings dimension
  size_t total_component_inputs;          // Layer 0: 200, Layer 1: 400
  size_t num_cells;

  std::unique_ptr<VectorFunctions> VectorFunctions;
  std::unique_ptr<Adam> symbol_embeddings_optimizer;
  std::unique_ptr<Adam> recurrent_weights_optimizer;
  std::unique_ptr<Adam> gamma_optimizer;
  std::unique_ptr<Adam> beta_optimizer;

  bool use_tanh; // true for Tanh, false for Logistic

  LstmComponent(
    SIMDType simdType,
    size_t vocabulary_size,
    size_t total_component_inputs,
    size_t num_cells,
    size_t horizon,
    bool useTanh,
    float bias_init,
    float learningRate_symbol_embeddings,
    float learningRate_recurrent_weights,
    float learningRate_rms
  );

  void ForwardPass(
    float* layer_input_ptr,
    uint8_t const input_symbol,
    size_t const sequence_position
  );

  void BackwardPass(
    float* layer_input_ptr,
    float* hidden_gradient,
    float* temporal_hidden_gradient,
    size_t const sequence_position,
    size_t const layer_id,
    uint8_t const input_symbol
  );

  void Optimize(const float lr_scale, const float beta2);

  void Rescale(float scale);

  void SaveWeights(LoadSave& stream);
  void LoadWeights(LoadSave& stream);
};
