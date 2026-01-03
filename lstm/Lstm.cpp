#include "Lstm.hpp"
#include <cstring>
#include <cmath>

Lstm::Lstm(
  SIMDType simdType,
  LSTM::Shape shape)
  : simd(simdType)
  , layer_input(shape.horizon * shape.num_layers * (shape.num_cells * 2)) // 100 * 2 * (200*2) = 100 * 2 * 400
  , output_layer(shape.output_size * (shape.num_cells * shape.num_layers)) // 256 * 400
  , output_layer_u(shape.output_size * (shape.num_cells * shape.num_layers)) // 256 * 400
  , output(shape.horizon * shape.output_size)         // 100 * 256 = 25,600
  , logits(shape.horizon * shape.output_size)         // 100 * 256 = 25,600
  , hidden(shape.num_cells * shape.num_layers)        // 200 * 2 = 400
  , hidden_error(shape.num_cells)                     // 200
  , output_bias(shape.output_size)                    // 256
  , output_bias_u(shape.output_size)                  // 256
  , input_history(shape.horizon)                      // 100
  , saved_timestep(0)
  , num_cells(shape.num_cells)
  , horizon(shape.horizon)
  , output_size(shape.output_size)
  , num_layers(shape.num_layers)
  , epoch(0)
  , output_learning_rate(0.f)
  , output_decay_func(
    0.015f,      // learningRate
    0.005f,      // endLearningRate
    0.0005f,     // decayMultiplier
    1.0f / 2.0f, // decayExponent
    0)           // decaySteps
{

  VectorFunctions = CreateVectorFunctions(simd);

#ifdef X64_SIMD_AVAILABLE
  if (simdType == SIMDType::SIMD_AVX2 || simdType == SIMDType::SIMD_AVX512) {
    output_weights_optimizer = std::make_unique<Adam_AVX>(
      shape.output_size * (shape.num_cells * shape.num_layers), // 256 * 400
      &output_layer[0],
      &output_layer_u[0],
      0.9995f,  // beta2
      1e-6f     // epsilon
    );
    output_bias_optimizer = std::make_unique<Adam_AVX>(
      shape.output_size,
      &output_bias[0],
      &output_bias_u[0],
      0.9995f,
      1e-6f
    );
  }
  else
#endif
  {
    output_weights_optimizer = std::make_unique<Adam_Scalar>(
      shape.output_size * (shape.num_cells * shape.num_layers), // 256 * 400
      &output_layer[0],
      &output_layer_u[0],
      0.9995f,
      1e-6f
    );
    output_bias_optimizer = std::make_unique<Adam_Scalar>(
      shape.output_size,
      &output_bias[0],
      &output_bias_u[0],
      0.9995f,
      1e-6f
    );
  }

  // Create LSTM layers
  for (size_t i = 0; i < num_layers; i++) {           // 2 iterations
    size_t hidden_size = num_cells * (i > 0 ? 2 : 1); // Layer 0: 200 (200*1), Layer 1: 400 (200*2)
    layers.push_back(
      std::make_unique<LstmLayer>(
        simdType,
        output_size,              // 256
        hidden_size,              // Layer 0: 200 (200*1), Layer 1: 400 (200*2)
        num_cells,                // 200
        horizon                   // 100
      )
    );
  }
}

float* Lstm::Predict(uint8_t const input) {
  size_t const max_layer_size = num_cells * 2;      // 200*2 = 400

  for (size_t i = 0; i < layers.size(); i++) {      // 2 iterations
    float* hidden_i = &hidden[i * num_cells]; // i * 200

    // Copy from hidden to layer_input
    float* src = hidden_i;
    float* dst = &layer_input[epoch * num_layers * max_layer_size + i * max_layer_size];
    memcpy(dst, src, num_cells * sizeof(float));

    // Get pointer to this layer's input
    size_t layer_input_size = num_cells * (i > 0 ? 2 : 1); // Layer 0: 200, Layer 1: 400
    size_t base_idx = epoch * num_layers * max_layer_size + i * max_layer_size;
    float* layer_input_ptr = &layer_input[base_idx];

    layers[i]->ForwardPass(
      layer_input_ptr,
      layer_input_size,
      input,
      hidden_i,
      current_sequence_size_target);

    // Copy hidden to next layer's input if not last layer
    if (i < layers.size() - 1) {
      float* src = hidden_i;
      float* dst = &layer_input[epoch * num_layers * max_layer_size + (i + 1) * max_layer_size + num_cells];
      memcpy(dst, src, num_cells * sizeof(float));
    }
  }

  size_t const hidden_size = num_cells * num_layers; // 200 * 2 = 400, same as hidden.size()
  size_t const output_offset = epoch * output_size; // epoch * 256

  VectorFunctions->MatvecThenSoftmax(
    &hidden[0],
    &logits[0],
    &output_layer[0],
    &output[0],
    &output_bias[0],
    hidden_size,
    output_size,
    output_offset
  );

  size_t const epoch_ = epoch;
  epoch++;
  if (epoch == current_sequence_size_target) epoch = 0;

  // Return pointer to the output slice for this epoch in the persistent output array
  return &output[epoch_ * output_size];              // &output[epoch_ * 256]
}

void Lstm::Perceive(const uint8_t input) {
  size_t const last_epoch = ((epoch > 0) ? epoch : current_sequence_size_target) - 1; // ((epoch > 0) ? epoch : 100) - 1
  uint8_t const old_input = input_history[last_epoch];
  input_history[last_epoch] = input;

  size_t const max_layer_size = num_cells * 2;      // 200*2 = 400
  size_t const hidden_size = num_cells * num_layers; // 200 * 2 = 400

  if (epoch == 0) {
    for (int epoch_ = static_cast<int>(current_sequence_size_target) - 1; epoch_ >= 0; epoch_--) {
      for (int layer = static_cast<int>(layers.size()) - 1; layer >= 0; layer--) {
        int offset = layer * static_cast<int>(num_cells); // layer * 200
        for (size_t i = 0; i < output_size; i++) {   // 256 iterations
          float const error = (i == input_history[epoch_]) ? output[epoch_ * output_size + i] - 1.f : output[epoch_ * output_size + i];

          for (size_t j = 0; j < hidden_error.size(); j++) { // 200 iterations
            hidden_error[j] += output_layer[i * hidden_size + (j + offset)] * error;
          }
        }

        size_t const prev_epoch = ((epoch_ > 0) ? epoch_ : current_sequence_size_target) - 1;
        uint8_t const input_symbol = (epoch_ > 0) ? input_history[prev_epoch] : old_input;

        // Get pointer to this layer's input
        size_t layer_input_size = num_cells * (layer > 0 ? 2 : 1); // Layer 0: 200, Layer 1: 400
        float* layer_input_ptr = &layer_input[epoch_ * num_layers * max_layer_size + layer * max_layer_size];

        layers[layer]->BackwardPass(
          layer_input_ptr,
          layer_input_size,
          epoch_,
          current_sequence_size_target,
          layer,
          input_symbol,
          &hidden_error[0]);
      }
    }

    sequence_step_cntr++;
    if (sequence_step_cntr >= sequence_step_target) { //target sequence size has been reached
      sequence_step_cntr = 0;
      if (current_sequence_size_target < horizon) {
        current_sequence_size_target++;
        //debug:
        //printf("current_sequence_size_target: %d\n", (int)current_sequence_size_target);
        sequence_step_target = 12 + 1 * (current_sequence_size_target - 1);
      }
    }
  }

  size_t previous_output_offset = last_epoch * output_size;
  float* output_ptr = &output[previous_output_offset];

  float* output_layer_ptr = &output_layer_u[0];
  const float* hidden_ptr = &hidden[0];

  VectorFunctions->AccumulateOutputLayerGradients(
    previous_output_offset,
    output_ptr,
    output_layer_ptr,
    &output_bias_u[0],
    hidden_ptr,
    output_size,
    hidden_size,
    input);

  if (epoch == 0) {
    // Get current timestep from first layer
    uint64_t time_step = layers[0]->update_steps;

    output_decay_func.Apply(output_learning_rate, time_step);
    output_weights_optimizer->Optimize(output_learning_rate, time_step);
    output_bias_optimizer->Optimize(output_learning_rate, time_step);
  }
}
