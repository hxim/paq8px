#include "Layer.hpp"
#include "SimdFunctions.hpp"
#include <cstring>

Layer::Layer(
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
  uint64_t decaySteps)
  : simd(simdType)
  , weights(num_cells * input_size)
  , update(num_cells * input_size)
  , transpose((input_size - output_size) * num_cells)
  , norm(horizon * num_cells)
  , state(horizon * num_cells)
  , inverse_variance(horizon)
  , gamma(num_cells)
  , gamma_u(num_cells)
  , beta(num_cells)
  , beta_u(num_cells)
  , error(num_cells)
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
  for (size_t i = 0; i < num_cells; i++) {
    gamma[i] = 1.f;
  }

#ifdef X64_SIMD_AVAILABLE
  if (simdType == SIMDType::SIMD_AVX2 || simdType == SIMDType::SIMD_AVX512) {
    weights_optimizer = std::make_unique<Adam_AVX>(
      num_cells * input_size,
      &weights[0],
      &update[0],
      beta2,
      epsilon
    );
    gamma_optimizer = std::make_unique<Adam_AVX>(
      num_cells,
      &gamma[0],
      &gamma_u[0],
      beta2,
      epsilon
    );
    beta_optimizer = std::make_unique<Adam_AVX>(
      num_cells,
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
      num_cells * input_size,
      &weights[0],
      &update[0],
      beta2,
      epsilon
    );
    gamma_optimizer = std::make_unique<Adam_Scalar>(
      num_cells,
      &gamma[0],
      &gamma_u[0],
      beta2,
      epsilon
    );
    beta_optimizer = std::make_unique<Adam_Scalar>(
      num_cells,
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
  float* norm_epoch = &norm[epoch * num_cells];
  float* state_epoch = &state[epoch * num_cells];

  if (simd == SIMDType::SIMD_AVX2 || simd == SIMDType::SIMD_AVX512) {
#ifdef X64_SIMD_AVAILABLE
    for (size_t i = 0; i < num_cells; i++) {
      float* w = &weights[i * input_size];
      norm_epoch[i] = dot256_ps_avx2(
        &input[0],
        w + output_size,
        input.size(),
        w[input_symbol]
      );
    }
  }
#endif
  else {
    for (size_t i = 0; i < num_cells; i++) {
      float* w = &weights[i * input_size];
      float f = w[input_symbol];
      float* wj = w + output_size;
      for (size_t j = 0; j < input.size(); j++)
        f += input[j] * wj[j];
      norm_epoch[i] = f;
    }
  }

  const float ss = SumOfSquares(norm_epoch, num_cells);
  inverse_variance[epoch] = 1.f / std::sqrt(ss / num_cells + 1e-5f);

  const float inv = inverse_variance[epoch];
  for (size_t i = 0; i < num_cells; i++) {
    float n = norm_epoch[i] * inv;
    norm_epoch[i] = n;
    state_epoch[i] = n * gamma[i] + beta[i];
  }

  if (use_tanh) {
    activation_tanh.Run(state_epoch, num_cells);
  }
  else {
    activation_logistic.Run(state_epoch, num_cells);
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
  if (epoch == horizon - 1) {
    memset(&gamma_u[0], 0, num_cells * sizeof(float));
    memset(&beta_u[0], 0, num_cells * sizeof(float));
    memset(&update[0], 0, num_cells * input_size * sizeof(float));

    // Build transpose
    const size_t rows = transpose.size() / num_cells;
    for (size_t i = 0; i < num_cells; i++) {
      float* w = &weights[i * input_size + output_size];
      float* t = &transpose[i];
      for (size_t j = 0; j < rows; j++)
        t[j * num_cells] = w[j];
    }
  }

  float* norm_epoch = &norm[epoch * num_cells];

  for (size_t i = 0; i < num_cells; i++) {
    beta_u[i] += error[i];
    gamma_u[i] += error[i] * norm_epoch[i];
    error[i] *= gamma[i] * inverse_variance[epoch];
  }

  float sop = SumOfProducts(&error[0], norm_epoch, num_cells) / num_cells;
  for (size_t i = 0; i < num_cells; i++)
    error[i] -= sop * norm_epoch[i];

  if (layer > 0) {
    for (size_t i = 0; i < num_cells; i++) {
      float* t = &transpose[(num_cells + i) * num_cells];

      if (simd == SIMDType::SIMD_AVX2 || simd == SIMDType::SIMD_AVX512) {
#ifdef X64_SIMD_AVAILABLE
        (*hidden_error)[i] += dot256_ps_avx2(&error[0], t, num_cells, 0.f);
#endif
      }
      else {
        float f = 0.f;
        for (size_t j = 0; j < num_cells; j++)
          f += error[j] * t[j];
        (*hidden_error)[i] += f;
      }
    }
  }

  if (epoch > 0) {
    for (size_t i = 0; i < num_cells; i++) {
      float* t = &transpose[i * num_cells];

      if (simd == SIMDType::SIMD_AVX2 || simd == SIMDType::SIMD_AVX512) {
#ifdef X64_SIMD_AVAILABLE
        (*stored_error)[i] += dot256_ps_avx2(&error[0], t, num_cells, 0.f);
#endif
      }
      else {
        float f = 0.f;
        for (size_t j = 0; j < num_cells; j++)
          f += error[j] * t[j];
        (*stored_error)[i] += f;
      }
    }
  }

  for (size_t i = 0; i < num_cells; i++) {
    float* u = &update[i * input_size];
    float ei = error[i];

    float* uj = u + output_size;
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
  for (size_t i = 0; i < horizon; i++) {
    inverse_variance[i] = 0.f;
    float* norm_epoch = &norm[i * num_cells];
    float* state_epoch = &state[i * num_cells];
    for (size_t j = 0; j < num_cells; j++) {
      state_epoch[j] = 0.f;
      norm_epoch[j] = 0.f;
    }
  }

  for (size_t i = 0; i < num_cells; i++) {
    gamma[i] = 1.f;
  }

  memset(&gamma_u[0], 0, num_cells * sizeof(float));
  memset(&beta[0], 0, num_cells * sizeof(float));
  memset(&beta_u[0], 0, num_cells * sizeof(float));
  memset(&update[0], 0, num_cells * input_size * sizeof(float));
  memset(&transpose[0], 0, transpose.size() * sizeof(float));
}
