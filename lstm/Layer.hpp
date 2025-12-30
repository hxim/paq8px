#pragma once

#include "Adam.hpp"
#include "Adam_Scalar.hpp"
#ifdef X64_SIMD_AVAILABLE
#include "Adam_AVX.hpp"
#endif
#include "SimdFunctions.hpp"
#include "PolynomialDecay.hpp"
#include "../SIMDType.hpp"
#include "../Array.hpp"
#include <cstdint>
#include <memory>

class Layer {
public:
  SIMDType simd;

  Array<float, 32> weights;        // Flat: [num_cells * input_size]
  Array<float, 32> update;         // Flat: [num_cells * input_size]

  Array<float, 32> transpose;      // Flat: [(input_size - output_size) * num_cells]
  Array<float, 32> norm;           // Flat: [horizon * num_cells]
  Array<float, 32> state;          // Flat: [horizon * num_cells]

  Array<float, 32> inverse_variance;

  Array<float, 32> gamma;
  Array<float, 32> gamma_u;

  Array<float, 32> beta;
  Array<float, 32> beta_u;

  Array<float, 32> error;

  size_t input_size;
  size_t output_size;
  size_t num_cells;

  float learning_rate;

  std::unique_ptr<Adam> weights_optimizer;
  std::unique_ptr<Adam> gamma_optimizer;
  std::unique_ptr<Adam> beta_optimizer;

  Tanh activation_tanh;
  Logistic activation_logistic;
  PolynomialDecay decay;

  bool use_tanh; // true for Tanh, false for Logistic

  Layer(
    SIMDType simdType,
    size_t input_size,
    size_t output_size,
    size_t num_cells,
    size_t horizon,
    bool useTanh,
    float beta2,
    float epsilon,
    float learningRate,
    float endLearningRate,
    float decayMultiplier,
    float powerNumerator,
    float powerDenominator,
    uint64_t decaySteps = 0
  );

  void ForwardPass(
    const Array<float, 32>& input,
    uint8_t input_symbol,
    size_t epoch);

  void BeforeBackwardPassAtLastEpoch();

  void BackwardPass(
    const Array<float, 32>& input,
    Array<float, 32>* hidden_error,
    Array<float, 32>* stored_error,
    uint64_t time_step,
    size_t epoch,
    size_t layer,
    uint8_t input_symbol);
};
