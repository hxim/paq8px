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

#include "PolynomialDecay.hpp"
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
  float beta2Value,
  float epsilon
);

class Layer {
public:
  SIMDType simd;

  Array<float, 32> embedding;      // Flat: [num_cells * embedding_size] - embedding matrix
  Array<float, 32> embedding_u;    // Flat: [num_cells * embedding_size] - embedding gradients

  Array<float, 32> weights;        // Flat: [num_cells * hidden_size] - hidden state weights only
  Array<float, 32> update;         // Flat: [num_cells * hidden_size] - hidden state gradients

  Array<float, 32> norm;           // Flat: [horizon * num_cells]
  Array<float, 32> state;          // Flat: [horizon * num_cells]

  Array<float, 32> inverse_variance;

  Array<float, 32> gamma;
  Array<float, 32> gamma_u;

  Array<float, 32> beta;
  Array<float, 32> beta_u;

  Array<float, 32> error;

  // Biases
  Array<float, 32> bias;           // [num_cells] - gate bias
  Array<float, 32> bias_u;         // [num_cells] - gate bias gradients
  std::unique_ptr<Adam> bias_optimizer;

  size_t embedding_size;   // Vocabulary size / embedding dimension
  size_t hidden_size;      // Size of hidden state input
  size_t num_cells;

  float learning_rate;

  std::unique_ptr<VectorFunctions> VectorFunctions;
  std::unique_ptr<Adam> embedding_optimizer;
  std::unique_ptr<Adam> weights_optimizer;
  std::unique_ptr<Adam> gamma_optimizer;
  std::unique_ptr<Adam> beta_optimizer;

  PolynomialDecay decayFunc;

  bool use_tanh; // true for Tanh, false for Logistic

  Layer(
    SIMDType simdType,
    size_t embedding_size,
    size_t hidden_size,
    size_t num_cells,
    size_t horizon,
    bool useTanh,
    float bias_init,
    float beta2,
    float epsilon,
    float learningRate,
    float endLearningRate,
    float decayMultiplier,
    float decayExponent,
    uint64_t decaySteps
  );

  void ForwardPass(
    float* input,
    size_t input_size,
    uint8_t const input_symbol,
    size_t const epoch
  );

  void BackwardPass(
    float* input,
    size_t input_size,
    float* hidden_error,
    float* stored_error,
    size_t const epoch,
    size_t const layer,
    uint8_t const input_symbol
  );

  void Optimize(uint64_t const time_step);

};
