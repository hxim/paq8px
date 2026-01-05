#include "Lstm.hpp"
#include <cstring>
#include <cmath>

Lstm::Lstm(
  SIMDType simdType,
  LSTM::Shape shape)
  : simd(simdType)
  // For the first layer we need num_cells, for every subsequent layer we need num_cells*2 hidden inputs
  , layer_input(shape.horizon * (shape.num_cells + (shape.num_layers - 1) * shape.num_cells * 2)) // 100 * (200 + 1*400)
  , output_layer(shape.output_size * (shape.num_cells * shape.num_layers)) // 256 * 400
  , output_layer_u(shape.output_size * (shape.num_cells * shape.num_layers)) // 256 * 400
  , output(shape.horizon * shape.output_size)         // 100 * 256
  , logits(shape.horizon * shape.output_size)         // 100 * 256
  , hidden(shape.num_cells * shape.num_layers)        // 200 * 2
  , hidden_error(shape.num_cells)                     // 200
  , output_bias(shape.output_size)                    // 256
  , output_bias_u(shape.output_size)                  // 256
  , input_symbol_history(shape.horizon + 1)           // 101: at position i it's both the input symbol for epoch i and also the target symbol for epoch i-1
  , saved_timestep(0)
  , num_cells(shape.num_cells)
  , horizon(shape.horizon)
  , output_size(shape.output_size)
  , num_layers(shape.num_layers)
  , epoch(0)
  , time_step(1)
  , output_learning_rate(0.f)
  , output_decay_func(
    0.015f,      // learningRate
    0.005f,      // endLearningRate
    0.0005f,     // decayMultiplier
    1.0f / 2.0f, // decayExponent
    0)           // decaySteps
{

  VectorFunctions = CreateVectorFunctions(simd);

  output_weights_optimizer = CreateOptimizer(
    simdType,
    shape.output_size * (shape.num_cells * shape.num_layers), // 256 * 400
    &output_layer[0],
    &output_layer_u[0],
    0.9995f,  // beta2
    1e-6f     // epsilon
  );
  output_bias_optimizer = CreateOptimizer(
    simdType,
    shape.output_size,
    &output_bias[0],
    &output_bias_u[0],
    0.9995f,
    1e-6f
  );


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

static size_t GetLayerInputOffset(size_t num_cells, size_t num_layers, size_t epoch, size_t current_layer_idx) {
  // Calculate offset for this epoch and layer
  size_t size_per_epoch = num_cells + (num_layers - 1) * num_cells * 2;
  size_t offset = epoch * size_per_epoch;   // Start of this epoch

  // Add offset for layers before layer i
  for (size_t j = 0; j < current_layer_idx; j++) {
    offset += num_cells * (j > 0 ? 2 : 1);
  }
  return offset;
}

float* Lstm::Predict(uint8_t const input_symbol) {

  input_symbol_history[epoch] = input_symbol;

  for (size_t i = 0; i < num_layers; i++) {
    float* hidden_i = &hidden[i * num_cells];

    size_t base_idx = GetLayerInputOffset(num_cells, num_layers, epoch, i);
    float* layer_input_ptr = &layer_input[base_idx];

    // First copy own previous hidden to position 0
    // Then copy previous layer's output to position num_cells

    // Copy own previous hidden state (always)
    memcpy(layer_input_ptr, hidden_i, num_cells * sizeof(float));
    // Copy previous layer's output (when there's a previous layer)
    if (i > 0) {
      // Layer i: [own_prev_hidden | layer_(i-1)_current_output]
      memcpy(layer_input_ptr + num_cells, &hidden[(i - 1) * num_cells], num_cells * sizeof(float));
    }

    size_t layer_input_size = num_cells * (i > 0 ? 2 : 1);

    layers[i]->ForwardPass(
      layer_input_ptr,
      layer_input_size,
      input_symbol,
      hidden_i,
      epoch,
      current_sequence_size_target);
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

  // Return pointer to the output slice for this epoch in the persistent output array
  return &output[epoch * output_size];              // &output[epoch * 256]
}

void Lstm::Perceive(const uint8_t target_symbol) {
  input_symbol_history[epoch + 1] = target_symbol;

  bool is_last_epoch = epoch == current_sequence_size_target - 1;

  size_t const hidden_size = num_cells * num_layers; // 200 * 2 = 400

  // Calculate error_on_output
  size_t output_offset = epoch * output_size;
  output[output_offset + target_symbol] -= 1;

  if (is_last_epoch) {

    for (size_t layer = 0; layer < num_layers; layer++) {
      layers[layer]->InitializeBackwardPass();;
    }

    // Backward pass through all timesteps in reverse order
    for (int epoch_ = static_cast<int>(current_sequence_size_target) - 1; epoch_ >= 0; epoch_--) {

      size_t output_offset_this_epoch = epoch_ * output_size;


      // Backward pass through all layers in reverse order
      for (int layer = static_cast<int>(num_layers) - 1; layer >= 0; layer--) {
        // The hidden_error buffer is reused to accumulate gradients from multiple sources,
        // we must not clear them here - it would break the gradient flow between layers.
        //
        //   Output Layer
        //       /    \
        //      /      \
        //     v        v
        //   Layer 1   Direct gradient to Layer 0
        //     |
        //     v
        //   Layer 0

        // Backpropagate from output layer to this LSTM layer
        VectorFunctions->AccumulateLstmGradients(
          num_cells,
          hidden_size,
          output_size,
          layer,
          &output[output_offset_this_epoch],
          &hidden_error[0],
          &output_layer[0]
        );

        // Get the input symbol for this timestep
        uint8_t const input_symbol = input_symbol_history[epoch_];

        // Get pointer to this layer's input
        size_t layer_input_offset = GetLayerInputOffset(num_cells, num_layers, epoch_, layer);
        float* layer_input_ptr = &layer_input[layer_input_offset];
        size_t layer_input_size = num_cells * (layer > 0 ? 2 : 1); // Layer 0: 200, Layer 1: 400

        layers[layer]->BackwardPass(
          layer_input_ptr,
          layer_input_size,
          epoch_,
          layer,
          input_symbol,
          &hidden_error[0]);
      }
    }

    // After full backward pass, optimize all layers
    for (size_t layer = 0; layer < num_layers; layer++) {
      layers[layer]->Optimize(time_step);
    }
  }

  // Accumulate output layer gradients for the current timestep

  float* output_layer_ptr = &output_layer_u[0];
  const float* hidden_ptr = &hidden[0];

  VectorFunctions->AccumulateOutputLayerGradients(
    output_offset,
    &output[output_offset],
    output_layer_ptr,
    &output_bias_u[0],
    hidden_ptr,
    output_size,
    hidden_size,
    target_symbol);

  if (is_last_epoch) {
    // Optimize output layer after full sequence
    output_decay_func.Apply(output_learning_rate, time_step);
    output_weights_optimizer->Optimize(output_learning_rate, time_step);
    output_bias_optimizer->Optimize(output_learning_rate, time_step);

    // Increase sequence size
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

    time_step++;
    epoch = 0;
  }
  else {
    epoch++;
  }
}
