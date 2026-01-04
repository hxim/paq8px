#include "Layer.hpp"
#include <cstring>

std::unique_ptr<VectorFunctions> CreateVectorFunctions(SIMDType simd) {
  if (simd == SIMDType::SIMD_AVX2 || simd == SIMDType::SIMD_AVX512)
    return std::make_unique<VectorFunctions_AVX2>();
  else if (simd == SIMDType::SIMD_SSE2)
    return std::make_unique<VectorFunctions_SSE2>();
  else
    return std::make_unique<VectorFunctions_Scalar>();
}

std::unique_ptr<Adam> CreateOptimizer(
  SIMDType simd,
  size_t length,
  float* w,
  float* g,
  float beta2Value,
  float epsilon)
{
  if (simd == SIMDType::SIMD_AVX2 || simd == SIMDType::SIMD_AVX512)
    return std::make_unique<Adam_AVX>(length, w, g, beta2Value, epsilon);
  else if (simd == SIMDType::SIMD_SSE2)
    return std::make_unique<Adam_SSE2>(length, w, g, beta2Value, epsilon);
  else
    return std::make_unique<Adam_Scalar>(length, w, g, beta2Value, epsilon);
}

Layer::Layer(
  SIMDType simdType,
  size_t embedding_size,  // 256 (vocabulary size)
  size_t hidden_size,     // Layer 0: 200 (200*1), Layer 1: 400 (200*2)
  size_t num_cells,       // 200
  size_t horizon,         // 100
  bool useTanh,
  float bias_init,
  float beta2,
  float epsilon,
  float learningRate,
  float endLearningRate,
  float decayMultiplier,
  float decayExponent,
  uint64_t decaySteps)
  : simd(simdType)
  , embedding(num_cells * embedding_size)           // 200*256 - embedding matrix
  , embedding_u(num_cells * embedding_size)         // 200*256 - embedding gradients
  , weights(num_cells * hidden_size)                // Layer 0: 200*200, Layer 1: 200*400 - hidden weights
  , update(num_cells * hidden_size)                 // Layer 0: 200*200, Layer 1: 200*400 - hidden gradients
  , norm(horizon * num_cells)                       // 100*200
  , state(horizon * num_cells)                      // 100*200
  , inverse_variance(horizon)                       // 100
  , gamma(num_cells)                                // 200 (RMSNorm scale)
  , gamma_u(num_cells)                              // 200 (RMSNorm scale update)
  , beta(num_cells)                                 // 200 (RMSNorm bias)
  , beta_u(num_cells)                               // 200 (RMSNorm bias update)
  , bias(num_cells)                                 // 200
  , bias_u(num_cells)                               // 200
  , error(num_cells)                                // 200
  , embedding_size(embedding_size)
  , hidden_size(hidden_size)
  , num_cells(num_cells)
  , learning_rate(0.f)
  , decayFunc(learningRate, endLearningRate, decayMultiplier, decayExponent, decaySteps)
  , use_tanh(useTanh)
{

  VectorFunctions = CreateVectorFunctions(simd);

  // Initialize RMS gamma and weigth bias
  for (size_t i = 0; i < num_cells; i++) { // 200 iterations
    gamma[i] = 1.f;
    bias[i] = bias_init;
  }

  embedding_optimizer = CreateOptimizer(
    simdType,
    num_cells * embedding_size,       // 200*256 - embedding parameters
    &embedding[0],
    &embedding_u[0],
    beta2,
    epsilon
  );
  weights_optimizer = CreateOptimizer(
    simdType,
    num_cells * hidden_size,          // Layer 0: 200*200, Layer 1: 200*400
    &weights[0],
    &update[0],
    beta2,
    epsilon
  );
  gamma_optimizer = CreateOptimizer(
    simdType,
    num_cells,                        // 200 (RMS scale)
    &gamma[0],
    &gamma_u[0],
    beta2,
    epsilon
  );
  beta_optimizer = CreateOptimizer(
    simdType,
    num_cells,                        // 200 (RMSNorm bias)
    &beta[0],
    &beta_u[0],
    beta2,
    epsilon
  );
  bias_optimizer = CreateOptimizer(
    simdType,
    num_cells,                        // 200 (bias)
    &bias[0],
    &bias_u[0],
    beta2,
    epsilon
  );
}

void Layer::ForwardPass(
  float* input,
  size_t input_size,
  uint8_t const input_symbol,
  size_t const epoch)
{
  float* norm_epoch = &norm[epoch * num_cells]; // epoch * 200
  float* state_epoch = &state[epoch * num_cells]; // epoch * 200

  for (size_t i = 0; i < num_cells; i++) { // 200 iterations
    // Embedding lookup for this cell
    float embed_value = embedding[i * embedding_size + input_symbol]; // embedding[i*256 + input_symbol]

    // Hidden state weights for this cell
    float* w = &weights[i * hidden_size]; // i * (200 or 400)

    // Compute: embedding_value + dot(input, hidden_weights)
    norm_epoch[i] = VectorFunctions->DotProduct(
      input,
      w,
      input_size     // Size of hidden state input array, = hidden_size
    ) + embed_value + bias[i];
  }

  const float ss = VectorFunctions->SumOfSquares(norm_epoch, num_cells);

  const float inv_var = std::sqrt(num_cells / ss); // 1.f / sqrt(ss / 200)
  inverse_variance[epoch] = inv_var;

  if(use_tanh)
    VectorFunctions->NormalizeThenActivate_Tanh(
      num_cells,
      norm_epoch,
      state_epoch,
      &gamma[0],
      &beta[0],
      inv_var);
  else
    VectorFunctions->NormalizeThenActivate_Sigmoid(
      num_cells,
      norm_epoch,
      state_epoch,
      &gamma[0],
      &beta[0],
      inv_var);
}

void Layer::BackwardPass(
  float* input,
  size_t input_size,
  float* hidden_error,
  float* stored_error,
  uint64_t const time_step,
  size_t const epoch,
  size_t const layer,
  uint8_t const input_symbol)
{
  float* norm_epoch = &norm[epoch * num_cells]; // epoch * 200

  for (size_t i = 0; i < num_cells; i++) {       // 200 iterations
    bias_u[i] += error[i];
    beta_u[i] += error[i];                       // RMSNorm bias gradient
    gamma_u[i] += error[i] * norm_epoch[i];
    error[i] *= gamma[i] * inverse_variance[epoch];
  }

  const float dop = VectorFunctions->DotProduct(
    &error[0],
    norm_epoch,
    num_cells) / num_cells;             // SumOfProducts(..., 200) / 200

  for (size_t i = 0; i < num_cells; i++)  // 200 iterations
    error[i] -= dop * norm_epoch[i];

  // Layer backprop: backpropagate to previous layer's hidden state
  // The first num_cells weights are temporal connections, next num_cells are from previous layer
  // weights[i * hidden_size + j] where j >= num_cells connects to previous layer
  if (layer > 0) {
    VectorFunctions->BackpropagateErrors(
      num_cells,
      num_cells, // base_offset
      hidden_size,
      &weights[0],
      &error[0],
      hidden_error);
  }

  // Temporal backprop: backpropagate to previous timestep's hidden state
  // stored_error is for the previous timestep (size: num_cells)
  // The previous timestep's output feeds back as input to current cell
  // weights[i * hidden_size + j] where j < num_cells for temporal connections
  if (epoch > 0) {
    VectorFunctions->BackpropagateErrors(
      num_cells,
      0, // base_offset
      hidden_size,
      &weights[0],
      &error[0],
      stored_error);
  }

  VectorFunctions->AccumulateLayerGradients(
    num_cells,
    embedding_size,
    hidden_size, // same as input_size
    input,
    &error[0],
    &embedding_u[input_symbol],
    &update[0]
  );
  
  // Optimize at the first epoch
  if (epoch == 0) {
    decayFunc.Apply(learning_rate, time_step);
    embedding_optimizer->Optimize(learning_rate, time_step);
    weights_optimizer->Optimize(learning_rate, time_step);
    gamma_optimizer->Optimize(learning_rate, time_step);
    beta_optimizer->Optimize(learning_rate, time_step);
    bias_optimizer->Optimize(learning_rate, time_step);
  }
}
