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


float LstmLayer_Rand(float const range);

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
  Array<float, 32> symbol_embeddings;
  Array<float, 32> symbol_embedding_gradients;

  Array<float, 32> weights;
  Array<float, 32> weight_gradients;

  Array<float, 32> normalized_values;
  Array<float, 32> activations;

  Array<float, 32> rms_scale;

  Array<float, 32> gamma;
  Array<float, 32> gamma_gradients;

  Array<float, 32> beta;
  Array<float, 32> beta_gradients;

  Array<float, 32> pre_activation_gradients;

  // Biases
  Array<float, 32> bias;
  Array<float, 32> bias_gradients;
  std::unique_ptr<Adam> bias_optimizer;

  size_t vocabulary_size;
  size_t component_input_dim;
  size_t hidden_size;

  std::unique_ptr<VectorFunctions> VectorFunctions;
  std::unique_ptr<Adam> symbol_embeddings_optimizer;
  std::unique_ptr<Adam> recurrent_weights_optimizer;
  std::unique_ptr<Adam> gamma_optimizer;
  std::unique_ptr<Adam> beta_optimizer;

  bool use_tanh; // true for Tanh, false for Logistic

  LstmComponent(
    SIMDType simdType,
    size_t vocabulary_size,
    size_t component_input_dim,
    size_t hidden_size,
    size_t horizon,
    bool use_tanh,
    float bias_init,
    float learning_rate_symbol_embeddings,
    float learning_rate_bias,
    float learning_rate_recurrent_weights,
    float learning_rate_rms_gamma,
    float learning_rate_rms_beta
  );

  void ForwardPass(
    float* layer_input_ptr,
    uint8_t const input_symbol,
    size_t const sequence_position
  );

  void BackwardPass(
    float* layer_input_ptr,
    float* hidden_gradient_accumulator,
    float* gradient_from_next_timestep,
    size_t const sequence_position,
    size_t const layer_id,
    uint8_t const input_symbol
  );

  void Optimize(const float lr_scale, const float beta2);

  void Rescale(float scale);

  void SaveWeights(LoadSave& stream);
  void LoadWeights(LoadSave& stream);
};
