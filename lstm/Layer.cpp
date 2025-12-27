#include "Layer.hpp"
#include "SimdFunctions.hpp"
#include <cstring>

Layer::Layer(
  SIMDType simdType,
  size_t input_size,      // Layer 0: 456 (200*1 + 256), Layer 1: 656 (200*2 + 256)
  size_t output_size,     // 256
  size_t num_cells,       // 200
  size_t horizon,         // 100
  bool useTanh,
  float beta2,
  float epsilon,
  float learningRate,
  float endLearningRate,
  float decayMultiplier,
  float powerNumerator,
  float powerDenominator,
  uint64_t decaySteps)
  : simd(simdType)
  , weights(num_cells * input_size) // Layer 0: 200*456=91,200, Layer 1: 200*656=131,200
  , update(num_cells * input_size)  // Layer 0: 200*456=91,200, Layer 1: 200*656=131,200
  , transpose((input_size - output_size) * num_cells) // Layer 0: (256-256)*200=0, Layer 1: (656-256)*200=80,000
  , norm(horizon * num_cells)       // 100*200=20,000
  , state(horizon * num_cells)      // 100*200=20,000
  , inverse_variance(horizon)       // 100
  , gamma(num_cells)                // 200
  , gamma_u(num_cells)              // 200
  , beta(num_cells)                 // 200 (RMSNorm bias)
  , beta_u(num_cells)               // 200 (RMSNorm bias update)
  , error(num_cells)                // 200
  , input_size(input_size)
  , output_size(output_size)
  , num_cells(num_cells)
  , horizon(horizon)
  , learning_rate(0.f)
  , activation_tanh(simdType)
  , activation_logistic(simdType)
  , decay(learningRate, endLearningRate, decayMultiplier, powerNumerator, powerDenominator, decaySteps)
  , use_tanh(useTanh)
{
  // Initialize gamma to 1.0
  for (size_t i = 0; i < num_cells; i++) { // 200 iterations
    gamma[i] = 1.f;
  }

#ifdef X64_SIMD_AVAILABLE
  if (simdType == SIMDType::SIMD_AVX2 || simdType == SIMDType::SIMD_AVX512) {
    weights_optimizer = std::make_unique<Adam_AVX>(
      num_cells * input_size,     // Layer 0: 91,200, Layer 1: 131,200
      &weights[0],
      &update[0],
      beta2,
      epsilon
    );
    gamma_optimizer = std::make_unique<Adam_AVX>(
      num_cells,                  // 200
      &gamma[0],
      &gamma_u[0],
      beta2,
      epsilon
    );
    beta_optimizer = std::make_unique<Adam_AVX>(
      num_cells,                  // 200 (RMSNorm bias)
      &beta[0],
      &beta_u[0],
      beta2,
      epsilon
    );
  }
  else
#endif
  {
    weights_optimizer = std::make_unique<Adam_Scalar>(
      num_cells * input_size,     // Layer 0: 51,200, Layer 1: 131,200
      &weights[0],
      &update[0],
      beta2,
      epsilon
    );
    gamma_optimizer = std::make_unique<Adam_Scalar>(
      num_cells,                  // 200
      &gamma[0],
      &gamma_u[0],
      beta2,
      epsilon
    );
    beta_optimizer = std::make_unique<Adam_Scalar>(
      num_cells,                  // 200 (RMSNorm bias)
      &beta[0],
      &beta_u[0],
      beta2,
      epsilon
    );
  }
}

void Layer::ForwardPass(
  const Array<float, 32>& input,
  uint8_t const input_symbol,
  size_t const epoch)
{
  float* norm_epoch = &norm[epoch * num_cells]; // epoch * 200
  float* state_epoch = &state[epoch * num_cells]; // epoch * 200

  if (simd == SIMDType::SIMD_AVX2 || simd == SIMDType::SIMD_AVX512) {
#ifdef X64_SIMD_AVAILABLE
    for (size_t i = 0; i < num_cells; i++) { // 200 iterations
      float* w = &weights[i * input_size]; // i * (456 or 656)
      norm_epoch[i] = dot256_ps_avx2(
        &input[0],
        w + output_size,          // w + 256
        input.size(),
        w[input_symbol]
      );
    }
  }
#endif
  else {
    for (size_t i = 0; i < num_cells; i++) { // 200 iterations
      float* w = &weights[i * input_size]; // i * (456 or 656)
      float f = w[input_symbol];
      float* wj = w + output_size;        // w + 256
      for (size_t j = 0; j < input.size(); j++)
        f += input[j] * wj[j];
      norm_epoch[i] = f;
    }
  }

  const float ss = SumOfSquares(
    norm_epoch,
    num_cells);                         // 200

  inverse_variance[epoch] = 1.f / std::sqrt(ss / num_cells + 1e-5f); // 1.f / sqrt(ss / 200 + 1e-5f)

  const float inv = inverse_variance[epoch];
  for (size_t i = 0; i < num_cells; i++) { // 200 iterations
    float n = norm_epoch[i] * inv;
    norm_epoch[i] = n;
    state_epoch[i] = n * gamma[i] + beta[i]; // RMSNorm with beta bias
  }

  if (use_tanh) {
    activation_tanh.Run(
      state_epoch,
      num_cells);                       // 200
  }
  else {
    activation_logistic.Run(
      state_epoch,
      num_cells);                       // 200
  }
}

void Layer::BackwardPass(
  const Array<float, 32>& input,
  Array<float, 32>* hidden_error,
  Array<float, 32>* stored_error,
  uint64_t const time_step,
  size_t const epoch,
  size_t const layer,
  uint8_t const input_symbol)
{
  if (epoch == horizon - 1) {             // epoch == 99
    memset(
      &gamma_u[0],
      0,
      num_cells * sizeof(float));       // 200 * 4 = 800 bytes
    memset(
      &beta_u[0],
      0,
      num_cells * sizeof(float));       // 200 * 4 = 800 bytes
    memset(
      &update[0],
      0,
      num_cells * input_size * sizeof(float)); // Layer 0: 91,200*4=364,800 bytes, Layer 1: 131,200*4=524,800 bytes

    // Build transpose
    const size_t rows = transpose.size() / num_cells; // Layer 0: 0/200=0, Layer 1: 80,000/200=400
    for (size_t i = 0; i < num_cells; i++) { // 200 iterations
      float* w = &weights[i * input_size + output_size]; // &weights[i * (456 or 656) + 256]
      float* t = &transpose[i];
      for (size_t j = 0; j < rows; j++)   // Layer 0: 0 iterations, Layer 1: 400 iterations
        t[j * num_cells] = w[j];          // t[j * 200] = w[j]
    }
  }

  float* norm_epoch = &norm[epoch * num_cells]; // epoch * 200

  for (size_t i = 0; i < num_cells; i++) {       // 200 iterations
    beta_u[i] += error[i];                       // RMSNorm bias gradient
    gamma_u[i] += error[i] * norm_epoch[i];
    error[i] *= gamma[i] * inverse_variance[epoch];
  }

  float sop = SumOfProducts(
    &error[0],
    norm_epoch,
    num_cells) / num_cells;             // SumOfProducts(..., 200) / 200

  for (size_t i = 0; i < num_cells; i++)  // 200 iterations
    error[i] -= sop * norm_epoch[i];

  if (layer > 0) {
    for (size_t i = 0; i < num_cells; i++) { // 200 iterations
      float* t = &transpose[(num_cells + i) * num_cells]; // &transpose[(200 + i) * 200]

      if (simd == SIMDType::SIMD_AVX2 || simd == SIMDType::SIMD_AVX512) {
#ifdef X64_SIMD_AVAILABLE
        (*hidden_error)[i] += dot256_ps_avx2(
          &error[0],
          t,
          num_cells,                    // 200
          0.f);
#endif
      }
      else {
        float f = 0.f;
        for (size_t j = 0; j < num_cells; j++) // 200 iterations
          f += error[j] * t[j];
        (*hidden_error)[i] += f;
      }
    }
  }

  if (epoch > 0) {
    for (size_t i = 0; i < num_cells; i++) { // 200 iterations
      float* t = &transpose[i * num_cells]; // &transpose[i * 200]

      if (simd == SIMDType::SIMD_AVX2 || simd == SIMDType::SIMD_AVX512) {
#ifdef X64_SIMD_AVAILABLE
        (*stored_error)[i] += dot256_ps_avx2(
          &error[0],
          t,
          num_cells,                    // 200
          0.f);
#endif
      }
      else {
        float f = 0.f;
        for (size_t j = 0; j < num_cells; j++) // 200 iterations
          f += error[j] * t[j];
        (*stored_error)[i] += f;
      }
    }
  }

  for (size_t i = 0; i < num_cells; i++) { // 200 iterations
    float* u = &update[i * input_size];   // &update[i * (456 or 656)]
    float ei = error[i];

    float* uj = u + output_size;          // u + 256
    for (size_t j = 0; j < input.size(); j++)
      uj[j] += ei * input[j];

    u[input_symbol] += ei;
  }

  if (epoch == 0) {
    decay.Apply(learning_rate, time_step);
    weights_optimizer->Optimize(learning_rate, time_step);
    gamma_optimizer->Optimize(learning_rate, time_step);
    beta_optimizer->Optimize(learning_rate, time_step);
  }
}

void Layer::Reset() {
  for (size_t i = 0; i < horizon; i++) {  // 100 iterations
    inverse_variance[i] = 0.f;
    float* norm_epoch = &norm[i * num_cells];   // &norm[i * 200]
    float* state_epoch = &state[i * num_cells]; // &state[i * 200]
    for (size_t j = 0; j < num_cells; j++) {    // 200 iterations
      state_epoch[j] = 0.f;
      norm_epoch[j] = 0.f;
    }
  }

  for (size_t i = 0; i < num_cells; i++) { // 200 iterations
    gamma[i] = 1.f;
  }

  memset(
    &gamma_u[0],
    0,
    num_cells * sizeof(float));         // 200 * 4 = 800 bytes
  memset(
    &beta[0],
    0,
    num_cells * sizeof(float));         // 200 * 4 = 800 bytes (RMSNorm bias)
  memset(
    &beta_u[0],
    0,
    num_cells * sizeof(float));         // 200 * 4 = 800 bytes (RMSNorm bias)
  memset(
    &update[0],
    0,
    num_cells * input_size * sizeof(float)); // Layer 0: 91,200*4, Layer 1: 131,200*4 bytes
  memset(
    &transpose[0],
    0,
    transpose.size() * sizeof(float));  // Layer 0: 0*4=0 bytes, Layer 1: 80,000*4=320,000 bytes
}
