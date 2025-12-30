#include "Lstm.hpp"
#include "SimdFunctions.hpp"
#include <cstring>
#include <cmath>

Lstm::Lstm(
  SIMDType simdType,
  LSTM::Shape shape,
  float const learning_rate)
  : simd(simdType)
  , layer_input(shape.horizon * shape.num_layers * (shape.num_cells * 2)) // 100 * 2 * (200*2) = 100 * 2 * 400 (no biases)
  , output_layer(shape.horizon * shape.output_size * (shape.num_cells * shape.num_layers)) // 100 * 256 * (200*2) = 100 * 256 * 400 (no biases)
  , output(shape.horizon * shape.output_size)         // 100 * 256 = 25,600
  , logits(shape.horizon * shape.output_size)         // 100 * 256 = 25,600
  , hidden(shape.num_cells * shape.num_layers)        // 200 * 2 = 400
  , hidden_error(shape.num_cells)                     // 200
  , input_history(shape.horizon)                      // 100
  , saved_timestep(0)
  , learning_rate(learning_rate)
  , num_cells(shape.num_cells)                        // 200
  , horizon(shape.horizon)                            // 100
  , output_size(shape.output_size)                    // 256
  , num_layers(shape.num_layers)                      // 2
  , epoch(0)
{

  size_t max_layer_size = num_cells * 2;              // 200*2 = 400

  // Initialize output probabilities
  float init_prob = 1.0f / output_size;               // 1.0f / 256
  for (size_t e = 0; e < horizon; e++) {             // 100 iterations
    for (size_t i = 0; i < output_size; i++) {       // 256 iterations
      output[e * output_size + i] = init_prob;       // output[e * 256 + i]
    }
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

#ifdef X64_SIMD_AVAILABLE

#if (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
#endif
void Lstm::SoftMaxSimdAVX2() {
  size_t const len = hidden.size();                  // 400
  size_t const hidden_size = num_cells * num_layers; // 200 * 2 = 400

  // Compute logits via dot products
  for (size_t i = 0; i < output_size; i++) {         // 256 iterations
    logits[epoch * output_size + i] = dot256_ps_avx2( // logits[epoch * 256 + i]
      &hidden[0],
      &output_layer[epoch * output_size * hidden_size + i * hidden_size], // &output_layer[epoch * 256 * 400 + i * 400]
      len,                                           // 400
      0.f
    );
  }

  // Find max logit for numerical stability
  float max_logit = logits[epoch * output_size];     // logits[epoch * 256]
  for (size_t i = 1; i < output_size; i++) {         // 255 iterations (starting from 1)
    if (logits[epoch * output_size + i] > max_logit) // logits[epoch * 256 + i]
      max_logit = logits[epoch * output_size + i];
  }

  // Compute softmax
  softmax_avx2(
    &logits[epoch * output_size],                  // &logits[epoch * 256]
    &output[epoch * output_size],                  // &output[epoch * 256]
    output_size,                                   // 256
    max_logit);
}

#endif

void Lstm::SoftMaxSimdNone() {
  size_t const len = hidden.size();                  // 400
  size_t const hidden_size = num_cells * num_layers; // 200 * 2 = 400

  // Compute logits via dot products
  for (size_t i = 0; i < output_size; i++) {         // 256 iterations
    logits[epoch * output_size + i] = SumOfProducts( // logits[epoch * 256 + i]
      &hidden[0],
      &output_layer[epoch * output_size * hidden_size + i * hidden_size], // &output_layer[epoch * 256 * 400 + i * 400]
      len                                            // 400
    );
  }

  // Find max logit for numerical stability
  float max_logit = logits[epoch * output_size];     // logits[epoch * 256]
  for (size_t i = 1; i < output_size; i++) {         // 255 iterations
    if (logits[epoch * output_size + i] > max_logit) // logits[epoch * 256 + i]
      max_logit = logits[epoch * output_size + i];
  }

  // Compute softmax
  softmax_scalar(
    &logits[epoch * output_size],                  // &logits[epoch * 256]
    &output[epoch * output_size],                  // &output[epoch * 256]
    output_size,                                   // 256
    max_logit);
}

float* Lstm::Predict(uint8_t const input) {
  size_t const max_layer_size = num_cells * 2;      // 200*2 = 400

  // Create temporary array to pass a slice of layer_input to ForwardPass
  Array<float, 32> temp_input(num_cells * 2 + output_size); // 200*2 + 256 = 656

  for (size_t i = 0; i < layers.size(); i++) {      // 2 iterations
    // Copy from hidden to layer_input
    size_t layer_input_size = num_cells * (i > 0 ? 2 : 1); // Layer 0: 200, Layer 1: 400
    for (size_t j = 0; j < num_cells && j < layer_input_size; j++) { // 200 iterations
      layer_input[epoch * num_layers * max_layer_size + i * max_layer_size + j] = hidden[i * num_cells + j];
    }

    // Prepare temp_input for this layer
    for (size_t j = 0; j < layer_input_size; j++) { // Layer 0: 200, Layer 1: 400
      temp_input[j] = layer_input[epoch * num_layers * max_layer_size + i * max_layer_size + j];
    }
    for (size_t j = 0; j < output_size; j++) {  // 256 iterations
      temp_input[layer_input_size + j] = 0.f;   // Will be filled by ForwardPass
    }

    layers[i]->ForwardPass(
      temp_input,
      input,
      &hidden,
      i * num_cells,                              // i * 200
      current_sequence_size_target);

    // Copy hidden to next layer's input if not last layer
    if (i < layers.size() - 1) {                    // if i < 1
      for (size_t j = 0; j < num_cells; j++) {      // 200 iterations
        layer_input[epoch * num_layers * max_layer_size + (i + 1) * max_layer_size + num_cells + j] = hidden[i * num_cells + j];
      }
    }
  }

  if (simd == SIMDType::SIMD_AVX2 || simd == SIMDType::SIMD_AVX512) {
#ifdef X64_SIMD_AVAILABLE
    SoftMaxSimdAVX2();
#endif
  }
  else {
    SoftMaxSimdNone();
  }

  size_t const epoch_ = epoch;
  epoch++;
  if (epoch == current_sequence_size_target) epoch = 0;                   // if epoch == 100

  // Return pointer to the output slice for this epoch in the persistent output array
  return &output[epoch_ * output_size];              // &output[epoch_ * 256]
}

void Lstm::Perceive(const uint8_t input) {
  size_t const last_epoch = ((epoch > 0) ? epoch : current_sequence_size_target) - 1; // ((epoch > 0) ? epoch : 100) - 1
  uint8_t const old_input = input_history[last_epoch];
  input_history[last_epoch] = input;

  size_t const max_layer_size = num_cells * 2;      // 200*2 = 400
  size_t const hidden_size = num_cells * num_layers; // 200 * 2 = 400

  Array<float, 32> temp_input(num_cells * 2 + output_size); // 200*2 + 256 = 656

  if (epoch == 0) {
    for (int epoch_ = static_cast<int>(current_sequence_size_target) - 1; epoch_ >= 0; epoch_--) { // starts at 99, goes to 0
      for (int layer = static_cast<int>(layers.size()) - 1; layer >= 0; layer--) { // starts at 1, goes to 0
        int offset = layer * static_cast<int>(num_cells); // layer * 200
        for (size_t i = 0; i < output_size; i++) {   // 256 iterations
          float const error = (i == input_history[epoch_]) ? output[epoch_ * output_size + i] - 1.f : output[epoch_ * output_size + i];

          for (size_t j = 0; j < hidden_error.size(); j++) { // 200 iterations
            hidden_error[j] += output_layer[epoch_ * output_size * hidden_size + i * hidden_size + (j + offset)] * error;
          }
        }

        size_t const prev_epoch = ((epoch_ > 0) ? epoch_ : current_sequence_size_target) - 1; // ((epoch_ > 0) ? epoch_ : 100) - 1
        uint8_t const input_symbol = (epoch_ > 0) ? input_history[prev_epoch] : old_input;

        // Prepare temp_input
        size_t layer_input_size = num_cells * (layer > 0 ? 2 : 1); // Layer 0: 200, Layer 1: 400
        for (size_t j = 0; j < layer_input_size; j++) { // Layer 0: 200, Layer 1: 400
          temp_input[j] = layer_input[epoch_ * num_layers * max_layer_size + layer * max_layer_size + j];
        }
        for (size_t j = 0; j < output_size; j++) {   // 256 iterations
          temp_input[layer_input_size + j] = 0.f;
        }

        layers[layer]->BackwardPass(
          temp_input,
          epoch_,
          current_sequence_size_target,
          layer,
          input_symbol,
          &hidden_error);
      }
    }

    sequence_step_cntr++;
    if (sequence_step_cntr >= sequence_step_target) //target sequence size has been reached
    {
      sequence_step_cntr = 0;
      if (current_sequence_size_target < horizon) {
        current_sequence_size_target++;
        //debug:
        //printf("current_sequence_size_target: %d\n", (int)current_sequence_size_target);
        sequence_step_target = 12 + 1 * (current_sequence_size_target - 1);
      }
    }
  }

  for (size_t i = 0; i < output_size; i++) {         // 256 iterations
    float const error = (i == input) ? output[last_epoch * output_size + i] - 1.f : output[last_epoch * output_size + i];

    for (size_t j = 0; j < hidden.size(); j++) {     // 400 iterations
      output_layer[epoch * output_size * hidden_size + i * hidden_size + j] =
        output_layer[last_epoch * output_size * hidden_size + i * hidden_size + j] - learning_rate * error * hidden[j];
    }
  }
}

uint64_t Lstm::GetCurrentTimeStep() const {
  return layers[0]->update_steps;
}

void Lstm::SetTimeStep(uint64_t const t) {
  for (size_t i = 0; i < layers.size(); i++)         // 2 iterations
    layers[i]->update_steps = t;
}
