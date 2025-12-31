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
  float decayExponent,
  uint64_t decaySteps)
  : simd(simdType)
  , embedding(num_cells * embedding_size)           // 200*256=51,200 - embedding matrix
  , embedding_u(num_cells * embedding_size)         // 200*256=51,200 - embedding gradients
  , weights(num_cells * hidden_size)                // Layer 0: 200*200=40,000, Layer 1: 200*400=80,000 - hidden weights only
  , update(num_cells * hidden_size)                 // Layer 0: 200*200=40,000, Layer 1: 200*400=80,000 - hidden gradients
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
  , decayMultiplier(learningRate, endLearningRate, decayMultiplier, decayExponent, decaySteps)
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

  inverse_variance[epoch] = std::sqrt(num_cells / ss); // 1.f / sqrt(ss / 200)

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
  float* hidden_error,
  float* stored_error,
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

  // Layer backprop: backpropagate to previous layer's hidden state
  // The first num_cells weights are temporal connections, next num_cells are from previous layer
  // weights[i * hidden_size + j] where j >= num_cells connects to previous layer
  if (layer > 0) {
    if (simd == SIMDType::SIMD_AVX2 || simd == SIMDType::SIMD_AVX512) {
#ifdef X64_SIMD_AVAILABLE
      backpropagate_errors_avx(num_cells, num_cells, hidden_size, &weights[0], &error[0], hidden_error);
#endif
    }
    else {

      for (size_t j = 0; j < num_cells; j += 8) { // For each cell in previous layer's hidden state
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        float sum4 = 0.0f, sum5 = 0.0f, sum6 = 0.0f, sum7 = 0.0f;

        size_t weight_idx = num_cells + j;  // Start at offset for previous layer connections
        for (size_t i = 0; i < num_cells; i++) { // For each current cell
          float ei = error[i];
          sum0 += ei * weights[weight_idx + 0];
          sum1 += ei * weights[weight_idx + 1];
          sum2 += ei * weights[weight_idx + 2];
          sum3 += ei * weights[weight_idx + 3];
          sum4 += ei * weights[weight_idx + 4];
          sum5 += ei * weights[weight_idx + 5];
          sum6 += ei * weights[weight_idx + 6];
          sum7 += ei * weights[weight_idx + 7];
          weight_idx += hidden_size;  // Move to next cell's weights
        }

        hidden_error[j + 0] += sum0;
        hidden_error[j + 1] += sum1;
        hidden_error[j + 2] += sum2;
        hidden_error[j + 3] += sum3;
        hidden_error[j + 4] += sum4;
        hidden_error[j + 5] += sum5;
        hidden_error[j + 6] += sum6;
        hidden_error[j + 7] += sum7;
      }
    }
  }

  // Temporal backprop: backpropagate to previous timestep's hidden state
  // stored_error is for the previous timestep (size: num_cells)
  // The previous timestep's output feeds back as input to current cell
  // weights[i * hidden_size + j] where j < num_cells for temporal connections
  if (epoch > 0) {
    if (simd == SIMDType::SIMD_AVX2 || simd == SIMDType::SIMD_AVX512) {
#ifdef X64_SIMD_AVAILABLE
      backpropagate_errors_avx(num_cells, 0, hidden_size, &weights[0], &error[0], stored_error);
#endif
    }
    else {

      for (size_t j = 0; j < num_cells; j += 8) {     // For each cell in previous timestep
        float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        float sum4 = 0.0f, sum5 = 0.0f, sum6 = 0.0f, sum7 = 0.0f;

        size_t weight_idx = j; // Start at offset for previous timestep's connections
        for (size_t i = 0; i < num_cells; i++) {   // For each current cell
          float ei = error[i];
          sum0 += ei * weights[weight_idx + 0];
          sum1 += ei * weights[weight_idx + 1];
          sum2 += ei * weights[weight_idx + 2];
          sum3 += ei * weights[weight_idx + 3];
          sum4 += ei * weights[weight_idx + 4];
          sum5 += ei * weights[weight_idx + 5];
          sum6 += ei * weights[weight_idx + 6];
          sum7 += ei * weights[weight_idx + 7];
          weight_idx += hidden_size;  // Move to next cell's weights
        }
        stored_error[j + 0] += sum0;
        stored_error[j + 1] += sum1;
        stored_error[j + 2] += sum2;
        stored_error[j + 3] += sum3;
        stored_error[j + 4] += sum4;
        stored_error[j + 5] += sum5;
        stored_error[j + 6] += sum6;
        stored_error[j + 7] += sum7;
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
    decayMultiplier.Apply(learning_rate, time_step);
    embedding_optimizer->Optimize(learning_rate, time_step);
    weights_optimizer->Optimize(learning_rate, time_step);
    gamma_optimizer->Optimize(learning_rate, time_step);
    beta_optimizer->Optimize(learning_rate, time_step);
  }
}
