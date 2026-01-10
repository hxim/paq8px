#include "LstmComponent.hpp"
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
  size_t vocabulary_size, // 256 (vocabulary size)
  size_t total_component_inputs,        // 0, 1
  size_t num_cells,       // 200
  size_t horizon,         // 100
  bool useTanh,
  float bias_init,
  float learningRate_symbol_embeddings,
  float learningRate_recurrent_weights,
  float learningRate_rms)
  : simd(simdType)
  , vocabulary_size(vocabulary_size)
  , num_cells(num_cells)
  , use_tanh(useTanh)
  , total_component_inputs(total_component_inputs)
  , symbol_embeddings(num_cells * vocabulary_size)   // 200*256
  , symbol_embedding_gradients(num_cells * vocabulary_size) // 200*256
  , weights(num_cells * total_component_inputs)      // Layer 0: 200*200, Layer 1: 200*400
  , weight_gradients(num_cells * total_component_inputs) // Layer 0: 200*200, Layer 1: 200*400
  , pre_norm_values(horizon * num_cells)            // 100*200
  , activations(horizon * num_cells)                // 100*200
  , inverse_variance(horizon)                       // 100
  , gamma(num_cells)                                // 200 (RMSNorm scale)
  , gamma_gradients(num_cells)                      // 200 (RMSNorm scale update)
  , beta(num_cells)                                 // 200 (RMSNorm bias)
  , beta_gradients(num_cells)                       // 200 (RMSNorm bias update)
  , bias(num_cells)                                 // 200
  , bias_gradients(num_cells)                       // 200
  , gate_gradient_buffer(num_cells)                 // 200
{

  VectorFunctions = CreateVectorFunctions(simd);

  // Initialize RMS gamma and weight bias
  for (size_t i = 0; i < num_cells; i++) { // 200 iterations
    gamma[i] = 1.f;
    bias[i] = bias_init;
  }

  symbol_embeddings_optimizer = CreateOptimizer(
    simdType,
    num_cells * vocabulary_size,       // 200*256 = symbol_embeddings parameters
    &symbol_embeddings[0],
    &symbol_embedding_gradients[0],
    learningRate_symbol_embeddings
  );
  recurrent_weights_optimizer = CreateOptimizer(
    simdType,
    num_cells * total_component_inputs,// Layer 0: 200*200, Layer 1: 200*400
    &weights[0],
    &weight_gradients[0],
    learningRate_recurrent_weights
  );
  gamma_optimizer = CreateOptimizer(
    simdType,
    num_cells,                        // 200 (RMS scale)
    &gamma[0],
    &gamma_gradients[0],
    learningRate_rms
  );
  beta_optimizer = CreateOptimizer(
    simdType,
    num_cells,                        // 200 (RMSNorm bias)
    &beta[0],
    &beta_gradients[0],
    learningRate_rms
  );
  bias_optimizer = CreateOptimizer(
    simdType,
    num_cells,                        // 200 (bias)
    &bias[0],
    &bias_gradients[0],
    learningRate_rms
  );
}

void LstmComponent::ForwardPass(
  float* layer_input_ptr,
  uint8_t const input_symbol,
  size_t const sequence_position)
{
  float* pre_norm_values_at_seq_pos = &pre_norm_values[sequence_position * num_cells];   // sequence_position * 200
  float* activations_at_seq_pos = &activations[sequence_position * num_cells];      // sequence_position * 200

  const float* embed_ptr = &symbol_embeddings[input_symbol]; // Embedding lookup for this cell
  const float* weight_ptr = &weights[0];
  const float* bias_ptr = &bias[0];

  for (size_t i = 0; i < num_cells; i++) { // 200 iterations
    // Compute: embedding_value + bias_value + dot(input, hidden_weights)
    pre_norm_values_at_seq_pos[i] = VectorFunctions->DotProduct(layer_input_ptr, weight_ptr, total_component_inputs) + (*embed_ptr) + (*bias_ptr);

    embed_ptr += vocabulary_size; // + 256
    weight_ptr += total_component_inputs;   // + (200 or 400)
    bias_ptr++;
  }

  const float ss = VectorFunctions->SumOfSquares(pre_norm_values_at_seq_pos, num_cells);

  const float inv_var = std::sqrt(num_cells / ss); // 1.f / sqrt(ss / 200)
  inverse_variance[sequence_position] = inv_var;

  if (use_tanh)
    VectorFunctions->NormalizeThenActivate_Tanh(
      num_cells,
      pre_norm_values_at_seq_pos, // updated, normalized
      activations_at_seq_pos,     // out
      &gamma[0],
      &beta[0],
      inv_var);
  else
    VectorFunctions->NormalizeThenActivate_Sigmoid(
      num_cells,
      pre_norm_values_at_seq_pos, // updated, normalized
      activations_at_seq_pos,     // out
      &gamma[0],
      &beta[0],
      inv_var);
}

void LstmComponent::BackwardPass(
  float* layer_input_ptr,
  float* hidden_gradient,
  float* temporal_hidden_gradient,
  size_t const sequence_position,
  size_t const layer_id,
  uint8_t const input_symbol)
{
  float* pre_activation_at_seq_pos = &pre_norm_values[sequence_position * num_cells]; // sequence_position * 200

  for (size_t i = 0; i < num_cells; i++) {       // 200 iterations
    bias_gradients[i] += gate_gradient_buffer[i];
    beta_gradients[i] += gate_gradient_buffer[i];                       // RMSNorm bias gradient
    gamma_gradients[i] += gate_gradient_buffer[i] * pre_activation_at_seq_pos[i];
    gate_gradient_buffer[i] *= gamma[i] * inverse_variance[sequence_position];
  }

  const float dop = VectorFunctions->DotProduct(
    &gate_gradient_buffer[0],
    pre_activation_at_seq_pos,
    num_cells) / num_cells; // DotProduct(..., 200) / 200

  for (size_t i = 0; i < num_cells; i++)  // 200 iterations
    gate_gradient_buffer[i] -= dop * pre_activation_at_seq_pos[i];

  // Layer backprop: backpropagate to previous layer's hidden state
  // The first num_cells weights are temporal connections, next num_cells are from previous layer
  // weights[i * total_component_inputs + j] where j >= num_cells connects to previous layer
  if (layer_id > 0) {
    VectorFunctions->BackpropagateErrors(
      num_cells,
      num_cells, // base_offset
      total_component_inputs,
      &weights[0],
      &gate_gradient_buffer[0],
      hidden_gradient);
  }

  // Temporal backprop: backpropagate to previous seq_pos's hidden state
  // temporal_hidden_gradient is for the previous seq_pos (size: num_cells)
  // Output from the previous seq_pos feeds back as input to current cell
  // weights[i * total_component_inputs + j] where j < num_cells for temporal connections
  if (sequence_position > 0) {
    VectorFunctions->BackpropagateErrors(
      num_cells,
      0, // base_offset
      total_component_inputs,
      &weights[0],
      &gate_gradient_buffer[0],
      temporal_hidden_gradient);
  }

  VectorFunctions->AccumulateLayerGradients(
    num_cells,
    vocabulary_size,
    total_component_inputs,
    layer_input_ptr,
    &gate_gradient_buffer[0],
    &symbol_embedding_gradients[input_symbol],
    &weight_gradients[0]
  );
}

void LstmComponent::Optimize(const float lr_scale, const float beta2) {
  // Optimize all parameters
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
