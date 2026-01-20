#include "Lstm.hpp"
#include <cstring>
#include <cmath>
#include <cstdlib>

Lstm::Lstm(
  SIMDType simdType,
  LstmShape shape,
  float tuning_param)
  : tuning_param(tuning_param)
    // all_layer_inputs:
    // This buffer stores the concatenated inputs that will be fed into each layer (own previous hidden + previous layer's output).
    // So, for the first layer we need hidden_size, for every subsequent layer we need hidden_size*2 hidden inputs
  , all_layer_inputs(shape.horizon * (shape.hidden_size + (shape.num_layers - 1) * shape.hidden_size * 2)) // 100 * (200 + 1*400)
  , output_weights(shape.vocabulary_size* (shape.hidden_size * shape.num_layers))          // 256 * 400
  , output_weight_gradients(shape.vocabulary_size* (shape.hidden_size * shape.num_layers)) // 256 * 400
    // output_probabilities:
    // This array serves dual purposes fo efficiency: first, it's the calculated probabilities by SoftMax.
    // Then, we calculate the error signal, which is just a change of a single value.
    // Thus, we keep everything in here, we just repurpose this as 'error_on_output'
  , output_probabilities(shape.horizon * shape.vocabulary_size)       // 100 * 256
  , logits(shape.horizon * shape.vocabulary_size)                     // 100 * 256
  , concatenated_hidden_states(shape.hidden_size * shape.num_layers)  // 200 * 2
  , hidden_gradient_accumulator(shape.hidden_size)                    // 200
  , output_bias(shape.vocabulary_size)                // 256
  , output_bias_gradients(shape.vocabulary_size)      // 256
  // input_symbol_history[i] serves dual purpose:
  // - Input symbol for forward pass at sequence position i
  // - Target symbol for backward pass at sequence position i-1
  // Size is horizon+1 to accommodate the target for the last position
  , input_symbol_history(shape.horizon + 1)           // 101
  , hidden_size(shape.hidden_size)
  , horizon(shape.horizon)
  , vocabulary_size(shape.vocabulary_size)
  , num_layers(shape.num_layers)
  , current_sequence_length(4)        // 4..horizon
  , sequence_step_target(12)
  , sequence_step_counter(0)          // 0..sequence_step_target-1
  , sequence_position(0)
  , training_iterations(1)
  , learning_rate_scheduler(
    1.0f,         // initial_lr
    0.30f,        // final_lr
    0.00040f)     // decay_rate - it reaches the final_lr in ((1.0 / 0.333333)² - 1) / 0.0005 = (9 - 1) / 0.0005 = 16'000 iterations
{
  assert((hidden_size & 7) == 0); // hidden_size must be a power of 8

  VectorFunctions = CreateVectorFunctions(simdType);

  output_weights_optimizer = CreateOptimizer(
    simdType,
    shape.vocabulary_size * (shape.hidden_size * shape.num_layers), // 256 * 400
    &output_weights[0],
    &output_weight_gradients[0],
    0.018f   // base_lr
  );
  output_bias_optimizer = CreateOptimizer(
    simdType,
    shape.vocabulary_size,
    &output_bias[0],
    &output_bias_gradients[0],
    0.018f
  );


  // Create LSTM layers
  for (size_t i = 0; i < num_layers; i++) {           // 2 iterations
    layers.push_back(
      std::make_unique<LstmLayer>(
        simdType,
        tuning_param,
        i,                        // layer_id
        vocabulary_size,          // 256
        hidden_size,              // 200
        horizon                   // 100
      )
    );
  }

  // Set random weights for the output layer
  // Output biases are left as zeroes

  size_t fan_in = shape.hidden_size * shape.num_layers;
  size_t fan_out = vocabulary_size;
  float xavier_range = 2.0f * std::sqrt(6.0f / (fan_in + fan_out)); // ~ 0.191; for uniform [-0.5, 0.5]

  for (size_t i = 0; i < output_weights.size(); i++) {
    output_weights[i] = LstmLayer_Rand(xavier_range);
  }
}

static size_t GetLayerInputOffset(size_t hidden_size, size_t num_layers, size_t sequence_position, size_t current_layer_idx) {
  // Calculate offset for sequence_position and layer
  size_t layer_input_stride = hidden_size + (num_layers - 1) * hidden_size * 2;
  size_t offset = sequence_position * layer_input_stride;   // Start of sequence_position

  // Add offset for layers before layer i
  for (size_t j = 0; j < current_layer_idx; j++) {
    offset += hidden_size * (j > 0 ? 2 : 1);
  }
  return offset;
}

float* Lstm::Predict(uint8_t const input_symbol) {

  input_symbol_history[sequence_position] = input_symbol;

  for (size_t i = 0; i < num_layers; i++) {
    float* hidden_state_ptr = &concatenated_hidden_states[i * hidden_size];

    size_t base_idx = GetLayerInputOffset(hidden_size, num_layers, sequence_position, i);
    float* layer_input_ptr = &all_layer_inputs[base_idx];

    // First copy own previous hidden to position 0
    // Then copy previous layer's output to position hidden_size

    // Copy own previous hidden state (always)
    VectorFunctions->Copy(layer_input_ptr, hidden_state_ptr, hidden_size);
    // Copy previous layer's output (when there's a previous layer)
    if (i > 0) {
      // Layer i: [own_prev_hidden | layer_(i-1)_current_output]
      VectorFunctions->Copy(layer_input_ptr + hidden_size, &concatenated_hidden_states[(i - 1) * hidden_size], hidden_size);
    }

    layers[i]->ForwardPass(
      layer_input_ptr,
      input_symbol,
      hidden_state_ptr, // out
      sequence_position,
      current_sequence_length);
  }

  size_t const concatenated_hidden_size = hidden_size * num_layers; // 200 * 2 = 400, same as hidden.size()
  size_t const output_offset = sequence_position * vocabulary_size; // sequence_position * 256

  VectorFunctions->MatvecThenSoftmax(
    &concatenated_hidden_states[0],
    &logits[0],
    &output_weights[0],
    &output_probabilities[0],
    &output_bias[0],
    concatenated_hidden_size,
    vocabulary_size,
    output_offset
  );

  // Return pointer to the output slice for sequence_position in the persistent output array
  return &output_probabilities[sequence_position * vocabulary_size]; // &output_probabilities[sequence_position * 256]
}

void Lstm::Perceive(const uint8_t target_symbol) {
  input_symbol_history[sequence_position + 1] = target_symbol;

  bool is_last_seq_pos = sequence_position == current_sequence_length - 1;

  size_t const concatenated_hidden_size = hidden_size * num_layers; // 200 * 2 = 400

  // Calculate error_on_output
  size_t output_offset = sequence_position * vocabulary_size;
  output_probabilities[output_offset + target_symbol] -= 1.0f;

  if (is_last_seq_pos) {

    for (size_t layer = 0; layer < num_layers; layer++) {
      layers[layer]->InitializeBackwardPass();;
    }

    // Backward pass through all sequence_positions in reverse order
    for (int seq_pos = static_cast<int>(current_sequence_length) - 1; seq_pos >= 0; seq_pos--) {

      size_t output_offset_at_seq_pos = seq_pos * vocabulary_size;


      // Backward pass through all layers in reverse order
      for (int layer_id = static_cast<int>(num_layers) - 1; layer_id >= 0; layer_id--) {
        // The hidden_gradient_accumulatorbuffer is reused to accumulate gradients from multiple sources,
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

        float* error_on_output = &output_probabilities[output_offset_at_seq_pos];

        VectorFunctions->AccumulateLstmGradients(
          hidden_size,
          concatenated_hidden_size,
          vocabulary_size,
          layer_id,
          error_on_output,
          &hidden_gradient_accumulator[0],
          &output_weights[0]
        );

        // Get the input symbol for sequence_position
        uint8_t const input_symbol = input_symbol_history[seq_pos];

        // Get pointer to this layer's input
        size_t layer_input_offset = GetLayerInputOffset(hidden_size, num_layers, seq_pos, layer_id);
        float* layer_input_ptr = &all_layer_inputs[layer_input_offset];

        layers[layer_id]->BackwardPass(
          layer_input_ptr,
          seq_pos,
          layer_id,
          input_symbol,
          &hidden_gradient_accumulator[0]);
      }
    }
  }

  // Accumulate output layer gradients for the sequence_position

  float* error_on_output = &output_probabilities[output_offset];

  VectorFunctions->AccumulateOutputLayerGradients(
    output_offset,
    error_on_output, 
    &output_weight_gradients[0],
    &output_bias_gradients[0],
    &concatenated_hidden_states[0],
    vocabulary_size,
    concatenated_hidden_size,
    target_symbol);

  // After full backward pass, optimize
  if (is_last_seq_pos) {

    // Increase sequence size
    sequence_step_counter++;
    if (sequence_step_counter >= sequence_step_target) { //target sequence size has been reached
      sequence_step_counter = 0;
      if (current_sequence_length < horizon) {
        float prev_sequence_length = (float)current_sequence_length;
        current_sequence_length++;
        float scale = (float)current_sequence_length / prev_sequence_length;

        //debug:
        //printf("current_sequence_length: %d\n", (int)current_sequence_length);
        sequence_step_target = 12 + 1 * (current_sequence_length - 1);

        for (size_t layer = 0; layer < num_layers; layer++) {
          layers[layer]->Rescale(scale);
        }
        output_weights_optimizer->Rescale(scale);
        output_bias_optimizer->Rescale(scale);
      }
    }

    constexpr float max_beta2 = 1.0f - 1.0f / 2048.0f;
    float beta2;
    if (training_iterations >= 4095)
      beta2 = max_beta2;
    else {
      // Forget the initial high gradients, i.e. adapt to a stable (more typical) second moment
      float n = ((training_iterations - 1) / 2.0f) + 1; // 1 .. 2047.5
      beta2 = 1.0f - 1.0f / n; // 0.0f .. 1.0f - 1.0f / 2047.5f 
    }

    float lr_scale = learning_rate_scheduler.GetLearningRate(training_iterations);

    for (size_t layer = 0; layer < num_layers; layer++) {
      layers[layer]->Optimize(lr_scale, beta2);
    }

    output_weights_optimizer->Optimize(lr_scale, beta2);
    output_bias_optimizer->Optimize(lr_scale, beta2);

    training_iterations++;
    sequence_position = 0;
  }
  else {
    sequence_position++;
  }
}

void Lstm::SaveModelParameters(LoadSave& stream) {

  // Write ASCII header
  stream.WriteTextLine("paq8pxlstmparams");

  char buffer[64];
  snprintf(buffer, sizeof(buffer), "%zu", vocabulary_size);
  stream.WriteTextLine(buffer);

  snprintf(buffer, sizeof(buffer), "%zu", hidden_size);
  stream.WriteTextLine(buffer);

  snprintf(buffer, sizeof(buffer), "%zu", num_layers);
  stream.WriteTextLine(buffer);

  snprintf(buffer, sizeof(buffer), "%zu", horizon);
  stream.WriteTextLine(buffer);

  // Write ESC character to mark end of text header
  fputc(0x1B, stream.file);
  // From here on binary data follows

  // Save all LSTM layers
  for (size_t i = 0; i < num_layers; i++) {
    layers[i]->SaveWeights(stream);
  }

  // Save output layer weights and biases
  stream.WriteFloatArray(&output_weights[0], output_weights.size());
  stream.WriteFloatArray(&output_bias[0], output_bias.size());
}

void Lstm::LoadModelParameters(LoadSave& stream) {
  char buffer[256];

  // Read and verify magic string
  if (!stream.ReadTextLine(buffer, sizeof(buffer)) ||
    strcmp(buffer, "paq8pxlstmparams") != 0) {
    fprintf(stderr, "Error: Invalid model file - wrong magic string\n");
    exit(1);
  }

  // Read shape parameters
  if (!stream.ReadTextLine(buffer, sizeof(buffer))) {
    fprintf(stderr, "Error: Failed to read vocabulary_size\n");
    exit(1);
  }
  size_t saved_vocabulary_size = strtoull(buffer, nullptr, 10);

  if (!stream.ReadTextLine(buffer, sizeof(buffer))) {
    fprintf(stderr, "Error: Failed to read hidden_size\n");
    exit(1);
  }
  size_t saved_num_cells = strtoull(buffer, nullptr, 10);

  if (!stream.ReadTextLine(buffer, sizeof(buffer))) {
    fprintf(stderr, "Error: Failed to read num_layers\n");
    exit(1);
  }
  size_t saved_num_layers = strtoull(buffer, nullptr, 10);

  if (!stream.ReadTextLine(buffer, sizeof(buffer))) {
    fprintf(stderr, "Error: Failed to read horizon\n");
    exit(1);
  }
  size_t saved_horizon = strtoull(buffer, nullptr, 10);

  // Read and verify ESC character
  int esc = fgetc(stream.file);
  if (esc != 0x1B) {
    fprintf(stderr, "Error: Missing ESC marker after header (got 0x%02X)\n", esc);
    exit(1);
  }

  // Verify shape matches
  if (saved_vocabulary_size != vocabulary_size ||
    saved_num_cells != hidden_size ||
    saved_num_layers != num_layers ||
    saved_horizon != horizon) {
    fprintf(stderr, "Error: Model shape mismatch\n");
    fprintf(stderr, "Expected: vocabulary_size=%zu, hidden_size=%zu, num_layers=%zu, horizon=%zu\n",
      vocabulary_size, hidden_size, num_layers, horizon);
    fprintf(stderr, "Got:      vocabulary_size=%zu, hidden_size=%zu, num_layers=%zu, horizon=%zu\n",
      saved_vocabulary_size, saved_num_cells, saved_num_layers, saved_horizon);
    exit(1);
  }

  // Load all LSTM layers
  for (size_t i = 0; i < num_layers; i++) {
    layers[i]->LoadWeights(stream);
  }

  // Load output layer weights and biases
  stream.ReadFloatArray(&output_weights[0], output_weights.size());
  stream.ReadFloatArray(&output_bias[0], output_bias.size());
}
