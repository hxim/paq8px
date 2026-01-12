#include "LstmComponent.hpp"
#include <cstring>

float LstmLayer_Rand(float const range) {
  return ((static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX)) - 0.5f) * range;
}

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
  float base_lr)
{
  if (simd == SIMDType::SIMD_AVX2 || simd == SIMDType::SIMD_AVX512)
    return std::make_unique<Adam_AVX>(length, w, g, base_lr);
  else if (simd == SIMDType::SIMD_SSE2)
    return std::make_unique<Adam_SSE2>(length, w, g, base_lr);
  else
    return std::make_unique<Adam_Scalar>(length, w, g, base_lr);
}

// LstmComponent: A learnable transformation unit that can function as either
// a gate (sigmoid) or a value computer (tanh), e.g., forget gate or cell candidate.

LstmComponent::LstmComponent(
  SIMDType simdType,
  size_t vocabulary_size,         // 256 (vocabulary size)
  size_t component_input_dim,     // 0, 1
  size_t hidden_size,             // 200
  size_t horizon,                 // 100
  bool use_tanh,
  float bias_init,
  float learning_rate_symbol_embeddings,
  float learning_rate_recurrent_weights,
  float learning_rate_rms)
  : vocabulary_size(vocabulary_size)
  , hidden_size(hidden_size)
  , use_tanh(use_tanh)
  // component_input_dim:
  // The size of the 2 input dimensions for a layer:
  // hidden state from previous timestep + optionally hidden state from previous layer
  , component_input_dim(component_input_dim)
  , symbol_embeddings(hidden_size * vocabulary_size)          // 200*256
  , symbol_embedding_gradients(hidden_size * vocabulary_size) // 200*256
  // weights:
  // Input-to-component weight matrix. Inputs are:
  // For layer 0: only temporal connections (200)
  // For layer 1+: temporal + inter-layer connections (200+200)
  , weights(hidden_size * component_input_dim)                // Layer 0: 200*200, Layer 1: 200*400
  , weight_gradients(hidden_size * component_input_dim)       // Layer 0: 200*200, Layer 1: 200*400
  , normalized_values(horizon * hidden_size)                  // 100*200
  , activations(horizon * hidden_size)                        // 100*200
  , rms_scale(horizon)                                        // 100
  , gamma(hidden_size)                                        // 200 (RMSNorm scale)
  , gamma_gradients(hidden_size)                              // 200 (RMSNorm scale update)
  , beta(hidden_size)                                         // 200 (RMSNorm bias)
  , beta_gradients(hidden_size)                               // 200 (RMSNorm bias update)
  , bias(hidden_size)                                         // 200
  , bias_gradients(hidden_size)                               // 200
  // pre_activation_gradients:
  // Gradients with respect to the normalized values (post-RMSNorm i.e. pre-sigmoid/tanh)
  , pre_activation_gradients(hidden_size)                     // 200
{

  VectorFunctions = CreateVectorFunctions(simdType);

  // Initialize RMS gamma and weight bias
  for (size_t i = 0; i < hidden_size; i++) { // 200 iterations
    gamma[i] = 1.f;
    bias[i] = bias_init;
  }

  symbol_embeddings_optimizer = CreateOptimizer(
    simdType,
    hidden_size * vocabulary_size,        // 200*256 = symbol_embeddings parameters
    &symbol_embeddings[0],
    &symbol_embedding_gradients[0],
    learning_rate_symbol_embeddings
  );
  recurrent_weights_optimizer = CreateOptimizer(
    simdType,
    hidden_size * component_input_dim,    // Layer 0: 200*200, Layer 1: 200*400
    &weights[0],
    &weight_gradients[0],
    learning_rate_recurrent_weights
  );
  gamma_optimizer = CreateOptimizer(
    simdType,
    hidden_size,                          // 200 (RMS scale)
    &gamma[0],
    &gamma_gradients[0],
    learning_rate_rms
  );
  beta_optimizer = CreateOptimizer(
    simdType,
    hidden_size,                          // 200 (RMSNorm bias)
    &beta[0],
    &beta_gradients[0],
    learning_rate_rms
  );
  bias_optimizer = CreateOptimizer(
    simdType,
    hidden_size,                          // 200 (bias)
    &bias[0],
    &bias_gradients[0],
    learning_rate_rms
  );
}

void LstmComponent::ForwardPass(
  float* layer_input_ptr,
  uint8_t const input_symbol,
  size_t const sequence_position)
{
  float* normalized_values_at_seq_pos = &normalized_values[sequence_position * hidden_size];    // sequence_position * 200
  float* activations_at_seq_pos = &activations[sequence_position * hidden_size];                // sequence_position * 200

  const float* embed_ptr = &symbol_embeddings[input_symbol]; // Embedding lookup for this cell
  const float* weight_ptr = &weights[0];
  const float* bias_ptr = &bias[0];

  for (size_t i = 0; i < hidden_size; i++) { // 200 iterations
    // Compute: embedding_value + bias_value + DotProduct(input, hidden_weights)
    normalized_values_at_seq_pos[i] = VectorFunctions->DotProduct(layer_input_ptr, weight_ptr, component_input_dim) + (*embed_ptr) + (*bias_ptr);

    embed_ptr += vocabulary_size; // + 256
    weight_ptr += component_input_dim;   // + (200 or 400)
    bias_ptr++;
  }

  const float sum_of_squares = VectorFunctions->SumOfSquares(normalized_values_at_seq_pos, hidden_size);

  const float inverse_rms = std::sqrt(hidden_size / sum_of_squares); // 1.f / sqrt(sum_of_squares / 200)
  rms_scale[sequence_position] = inverse_rms;

  if (use_tanh)
    VectorFunctions->NormalizeThenActivate_Tanh(
      hidden_size,
      normalized_values_at_seq_pos, // normalized
      activations_at_seq_pos,       // out
      &gamma[0],
      &beta[0],
      inverse_rms);
  else
    VectorFunctions->NormalizeThenActivate_Sigmoid(
      hidden_size,
      normalized_values_at_seq_pos, // normalized
      activations_at_seq_pos,       // out
      &gamma[0],
      &beta[0],
      inverse_rms);
}

void LstmComponent::BackwardPass(
  float* layer_input_ptr,
  float* hidden_gradient_accumulator,
  float* gradient_from_next_timestep,
  size_t const sequence_position,
  size_t const layer_id,
  uint8_t const input_symbol)
{
  float* pre_activation_at_seq_pos = &normalized_values[sequence_position * hidden_size]; // sequence_position * 200

  for (size_t i = 0; i < hidden_size; i++) {          // 200 iterations
    bias_gradients[i] += pre_activation_gradients[i];
    beta_gradients[i] += pre_activation_gradients[i]; // RMSNorm bias gradient
    gamma_gradients[i] += pre_activation_gradients[i] * pre_activation_at_seq_pos[i];
    pre_activation_gradients[i] *= gamma[i] * rms_scale[sequence_position];
  }

  const float dop = VectorFunctions->DotProduct(
    &pre_activation_gradients[0],
    pre_activation_at_seq_pos,
    hidden_size) / hidden_size; // DotProduct(..., 200) / 200

  for (size_t i = 0; i < hidden_size; i++)  // 200 iterations
    pre_activation_gradients[i] -= dop * pre_activation_at_seq_pos[i];

  // Layer backprop: backpropagate to previous layer's hidden state
  // The first hidden_size weights are temporal connections, next hidden_size are from previous layer
  // weights[i * component_input_dim + j] where j >= hidden_size connects to previous layer
  if (layer_id > 0) {
    VectorFunctions->BackpropagateErrors(
      hidden_size,
      hidden_size, // base_offset
      component_input_dim,
      &weights[0],
      &pre_activation_gradients[0],
      hidden_gradient_accumulator);
  }

  // Temporal backprop: backpropagate to previous seq_pos's hidden state
  // gradient_from_next_timestep is for the previous seq_pos (size: hidden_size)
  // Output from the previous seq_pos feeds back as input to current cell
  // weights[i * component_input_dim + j] where j < hidden_size for temporal connections
  if (sequence_position > 0) {
    VectorFunctions->BackpropagateErrors(
      hidden_size,
      0, // base_offset
      component_input_dim,
      &weights[0],
      &pre_activation_gradients[0],
      gradient_from_next_timestep);
  }

  VectorFunctions->AccumulateLayerGradients(
    hidden_size,
    vocabulary_size,
    component_input_dim,
    layer_input_ptr,
    &pre_activation_gradients[0],
    &symbol_embedding_gradients[input_symbol],
    &weight_gradients[0]
  );
}

void LstmComponent::Optimize(const float lr_scale, const float beta2) {
  symbol_embeddings_optimizer->Optimize(lr_scale, beta2);
  recurrent_weights_optimizer->Optimize(lr_scale, beta2);
  gamma_optimizer->Optimize(lr_scale, beta2);
  beta_optimizer->Optimize(lr_scale, beta2);
  bias_optimizer->Optimize(lr_scale, beta2);
}

void LstmComponent::Rescale(float scale) {
  symbol_embeddings_optimizer->Rescale(scale);
  recurrent_weights_optimizer->Rescale(scale);
  gamma_optimizer->Rescale(scale);
  beta_optimizer->Rescale(scale);
  bias_optimizer->Rescale(scale);
}

void LstmComponent::SaveWeights(LoadSave& stream) {
  // Save learned parameters
  stream.WriteFloatArray(&symbol_embeddings[0], symbol_embeddings.size());
  stream.WriteFloatArray(&weights[0], weights.size());
  stream.WriteFloatArray(&bias[0], bias.size());
  stream.WriteFloatArray(&gamma[0], gamma.size());
  stream.WriteFloatArray(&beta[0], beta.size());
}

void LstmComponent::LoadWeights(LoadSave& stream) {
  // Load learned parameters
  stream.ReadFloatArray(&symbol_embeddings[0], symbol_embeddings.size());
  stream.ReadFloatArray(&weights[0], weights.size());
  stream.ReadFloatArray(&bias[0], bias.size());
  stream.ReadFloatArray(&gamma[0], gamma.size());
  stream.ReadFloatArray(&beta[0], beta.size());
}
