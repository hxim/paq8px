#include "Layer.hpp"
#include "SimdFunctions.hpp"
#include <cstring>

Layer::Layer(
  SIMDType simdType,
  size_t embedding_size,  // 256 (vocabulary size)
  size_t hidden_size,     // Layer 0: 200 (200*1), Layer 1: 400 (200*2)
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
  , embedding(num_cells * embedding_size)           // 200*256=51,200 - embedding matrix
  , embedding_u(num_cells * embedding_size)         // 200*256=51,200 - embedding gradients
  , weights(num_cells * hidden_size)                // Layer 0: 200*200=40,000, Layer 1: 200*400=80,000 - hidden weights only
  , update(num_cells * hidden_size)                 // Layer 0: 200*200=40,000, Layer 1: 200*400=80,000 - hidden gradients
  , transpose(hidden_size * num_cells)              // Layer 0: 200*200=40,000, Layer 1: 400*200=80,000
  , norm(horizon * num_cells)                       // 100*200=20,000
  , state(horizon * num_cells)                      // 100*200=20,000
  , inverse_variance(horizon)                       // 100
  , gamma(num_cells)                                // 200
  , gamma_u(num_cells)                              // 200
  , beta(num_cells)                                 // 200 (RMSNorm bias)
  , beta_u(num_cells)                               // 200 (RMSNorm bias update)
  , error(num_cells)                                // 200
  , embedding_size(embedding_size)
  , hidden_size(hidden_size)
  , num_cells(num_cells)
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
    embedding_optimizer = std::make_unique<Adam_AVX>(
      num_cells * embedding_size,       // 200*256=51,200 - embedding parameters
      &embedding[0],
      &embedding_u[0],
      beta2,
      epsilon
    );
    weights_optimizer = std::make_unique<Adam_AVX>(
      num_cells * hidden_size,          // Layer 0: 200*200=40,000, Layer 1: 200*400=80,000
      &weights[0],
      &update[0],
      beta2,
      epsilon
    );
    gamma_optimizer = std::make_unique<Adam_AVX>(
      num_cells,                        // 200
      &gamma[0],
      &gamma_u[0],
      beta2,
      epsilon
    );
    beta_optimizer = std::make_unique<Adam_AVX>(
      num_cells,                        // 200 (RMSNorm bias)
      &beta[0],
      &beta_u[0],
      beta2,
      epsilon
    );
  }
  else
#endif
  {
    embedding_optimizer = std::make_unique<Adam_Scalar>(
      num_cells * embedding_size,       // 200*256=51,200 - embedding parameters
      &embedding[0],
      &embedding_u[0],
      beta2,
      epsilon
    );
    weights_optimizer = std::make_unique<Adam_Scalar>(
      num_cells * hidden_size,          // Layer 0: 200*200=40,000, Layer 1: 200*400=80,000
      &weights[0],
      &update[0],
      beta2,
      epsilon
    );
    gamma_optimizer = std::make_unique<Adam_Scalar>(
      num_cells,                        // 200
      &gamma[0],
      &gamma_u[0],
      beta2,
      epsilon
    );
    beta_optimizer = std::make_unique<Adam_Scalar>(
      num_cells,                        // 200 (RMSNorm bias)
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
      // Embedding lookup for this cell
      float embed_value = embedding[i * embedding_size + input_symbol]; // embedding[i*256 + input_symbol]

      // Hidden state weights for this cell
      float* w = &weights[i * hidden_size]; // i * (200 or 400)

      // Compute: embedding_value + dot(input, hidden_weights)
      norm_epoch[i] = dot256_ps_avx2(
        &input[0],
        w,
        input.size(),   // Size of hidden state input array
        embed_value     // Start with embedding value
      );
    }
#endif
  }
  else {
    for (size_t i = 0; i < num_cells; i++) { // 200 iterations
      // Embedding lookup for this cell
      float f = embedding[i * embedding_size + input_symbol]; // embedding[i*256 + input_symbol]

      // Hidden state weights for this cell
      float* w = &weights[i * hidden_size]; // i * (200 or 400)

      // Accumulate hidden state contributions
      for (size_t j = 0; j < input.size(); j++)  // input.size() = hidden_size
        f += input[j] * w[j];

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

void Layer::BeforeBackwardPassAtLastEpoch() {
  memset(&gamma_u[0], 0, num_cells * sizeof(float));             // 200 * 4 = 800 bytes
  memset(&beta_u[0], 0, num_cells * sizeof(float));              // 200 * 4 = 800 bytes
  memset(&embedding_u[0], 0, num_cells * embedding_size * sizeof(float)); // 200*256*4=204,800 bytes - embedding gradients
  memset(&update[0], 0, num_cells * hidden_size * sizeof(float)); // Layer 0: 200*200*4=160,000 bytes, Layer 1: 200*400*4=320,000 bytes

  // Build transpose for hidden weights only
  for (size_t i = 0; i < num_cells; i++) { // 200 iterations
    float* w = &weights[i * hidden_size]; // &weights[i * (200 or 400)]
    float* t = &transpose[i];
    for (size_t j = 0; j < hidden_size; j++)   // Layer 0: 200 iterations, Layer 1: 400 iterations
      t[j * num_cells] = w[j];          // t[j * 200] = w[j]
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

  // Backpropagate to previous layer's hidden state
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

  // Backpropagate to previous timestep's hidden state
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

  // Update gradients
  for (size_t i = 0; i < num_cells; i++) { // 200 iterations
    float ei = error[i];

    // Update embedding gradient for this input symbol
    embedding_u[i * embedding_size + input_symbol] += ei; // embedding_u[i*256 + input_symbol]

    // Update hidden state weight gradients
    float* u = &update[i * hidden_size];   // &update[i * (200 or 400)]
    for (size_t j = 0; j < input.size(); j++)  // input.size() = hidden_size
      u[j] += ei * input[j];
  }

  // Optimize at the first epoch
  if (epoch == 0) {
    decay.Apply(learning_rate, time_step);
    embedding_optimizer->Optimize(learning_rate, time_step);
    weights_optimizer->Optimize(learning_rate, time_step);
    gamma_optimizer->Optimize(learning_rate, time_step);
    beta_optimizer->Optimize(learning_rate, time_step);
  }
}
