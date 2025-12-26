#include "Lstm.hpp"
#include "SimdFunctions.hpp"
#include <cstring>
#include <cmath>

// LSTM::Model Implementation

LSTM::Model::Model(LSTM::Shape const shape)
  : timestep(0)
  , shape(shape)
  , weights(0)  // Initialize with size 0, will resize below
  , output(shape.output_size * (shape.num_cells * shape.num_layers + 1))
{
  // Calculate total size for all layers
  size_t total_weight_size = 0;
  for (size_t i = 0; i < shape.num_layers; i++) {
    size_t weight_size = 1 + shape.num_cells * (i > 0 ? 2 : 1) + shape.output_size;
    total_weight_size += 3 * shape.num_cells * weight_size;
  }
  weights.resize(total_weight_size);
}

void LSTM::Model::LoadFromDisk(const char* const dictionary, int32_t bits, int32_t exp) {
  BitFileDisk file(true);
  OpenFromMyFolder::anotherFile(&file, dictionary);

  size_t hidden_size = shape.num_cells * shape.num_layers + 1;

  if ((bits > 0) && (bits <= 16)) {
    float scale = Posit<9, 1>::Decode(file.getBits(8));

    // Load output
    for (size_t i = 0; i < shape.output_size; i++) {
      for (size_t j = 0; j < hidden_size; j++)
        output[i * hidden_size + j] = Posit<16, 1>::Decode(file.getBits(bits)) * scale;
    }

    // Load weights
    size_t weight_offset = 0;
    for (size_t layer = 0; layer < shape.num_layers; layer++) {
      size_t weight_size = 1 + shape.num_cells * (layer > 0 ? 2 : 1) + shape.output_size;
      for (size_t gate = 0; gate < 3; gate++) {
        for (size_t cell = 0; cell < shape.num_cells; cell++) {
          for (size_t idx = 0; idx < weight_size; idx++) {
            size_t index = weight_offset + gate * shape.num_cells * weight_size + cell * weight_size + idx;
            weights[index] = Posit<16, 1>::Decode(file.getBits(bits)) * scale;
          }
        }
      }
      weight_offset += 3 * shape.num_cells * weight_size;
    }
  }
  else {
    float v;
    // Load output
    for (size_t i = 0; i < shape.output_size; i++) {
      for (size_t j = 0; j < hidden_size; j++) {
        if (file.blockRead(reinterpret_cast<uint8_t*>(&v), sizeof(float)) != sizeof(float)) break;
        output[i * hidden_size + j] = v;
      }
    }

    // Load weights
    size_t weight_offset = 0;
    for (size_t layer = 0; layer < shape.num_layers; layer++) {
      size_t weight_size = 1 + shape.num_cells * (layer > 0 ? 2 : 1) + shape.output_size;
      for (size_t gate = 0; gate < 3; gate++) {
        for (size_t cell = 0; cell < shape.num_cells; cell++) {
          for (size_t idx = 0; idx < weight_size; idx++) {
            if (file.blockRead(reinterpret_cast<uint8_t*>(&v), sizeof(float)) != sizeof(float)) break;
            size_t index = weight_offset + gate * shape.num_cells * weight_size + cell * weight_size + idx;
            weights[index] = v;
          }
        }
      }
      weight_offset += 3 * shape.num_cells * weight_size;
    }
  }
  file.close();
}

void LSTM::Model::SaveToDisk(const char* const dictionary, int32_t bits, int32_t exp) {
  BitFileDisk file(false);
  file.create(dictionary);

  size_t hidden_size = shape.num_cells * shape.num_layers + 1;

  if ((bits > 0) && (bits <= 16)) {
    float const s = std::pow(2.f, (1 << exp) * (bits - 2));
    float max_w = 0.f, w, scale;

    // Find max weight in output
    for (size_t i = 0; i < shape.output_size; i++) {
      for (size_t j = 0; j < hidden_size; j++) {
        if ((w = std::fabs(output[i * hidden_size + j])) > max_w)
          max_w = w;
      }
    }

    // Find max weight in all layers
    size_t weight_offset = 0;
    for (size_t layer = 0; layer < shape.num_layers; layer++) {
      size_t weight_size = 1 + shape.num_cells * (layer > 0 ? 2 : 1) + shape.output_size;
      size_t layer_total = 3 * shape.num_cells * weight_size;
      for (size_t i = 0; i < layer_total; i++) {
        if ((w = std::fabs(weights[weight_offset + i])) > max_w)
          max_w = w;
      }
      weight_offset += layer_total;
    }

    scale = Posit<9, 1>::Decode(Posit<9, 1>::Encode(std::max<float>(1.f, max_w / s)));
    file.putBits(Posit<9, 1>::Encode(scale), 8);

    // Save output
    for (size_t i = 0; i < shape.output_size; i++) {
      for (size_t j = 0; j < hidden_size; j++)
        file.putBits(Posit<16, 1>::Encode(output[i * hidden_size + j] / scale), bits);
    }

    // Save weights
    weight_offset = 0;
    for (size_t layer = 0; layer < shape.num_layers; layer++) {
      size_t weight_size = 1 + shape.num_cells * (layer > 0 ? 2 : 1) + shape.output_size;
      for (size_t gate = 0; gate < 3; gate++) {
        for (size_t cell = 0; cell < shape.num_cells; cell++) {
          for (size_t idx = 0; idx < weight_size; idx++) {
            size_t index = weight_offset + gate * shape.num_cells * weight_size + cell * weight_size + idx;
            file.putBits(Posit<16, 1>::Encode(weights[index] / scale), bits);
          }
        }
      }
      weight_offset += 3 * shape.num_cells * weight_size;
    }
    file.flush();
  }
  else {
    float v;
    // Save output
    for (size_t i = 0; i < shape.output_size; i++) {
      for (size_t j = 0; j < hidden_size; j++) {
        v = output[i * hidden_size + j];
        file.blockWrite(reinterpret_cast<uint8_t*>(&v), sizeof(float));
      }
    }

    // Save weights
    size_t weight_offset = 0;
    for (size_t layer = 0; layer < shape.num_layers; layer++) {
      size_t weight_size = 1 + shape.num_cells * (layer > 0 ? 2 : 1) + shape.output_size;
      for (size_t gate = 0; gate < 3; gate++) {
        for (size_t cell = 0; cell < shape.num_cells; cell++) {
          for (size_t idx = 0; idx < weight_size; idx++) {
            size_t index = weight_offset + gate * shape.num_cells * weight_size + cell * weight_size + idx;
            v = weights[index];
            file.blockWrite(reinterpret_cast<uint8_t*>(&v), sizeof(float));
          }
        }
      }
      weight_offset += 3 * shape.num_cells * weight_size;
    }
  }
  file.close();
}

Lstm::Lstm(
  SIMDType simdType,
  LSTM::Shape shape,
  float const learning_rate)
  : simd(simdType)
  , layer_input(shape.horizon* shape.num_layers* (1 + shape.num_cells * 2))
  , output_layer(shape.horizon* shape.output_size* (shape.num_cells* shape.num_layers + 1))
  , output(shape.horizon* shape.output_size)
  , logits(shape.horizon* shape.output_size)
  , hidden(shape.num_cells* shape.num_layers + 1)
  , hidden_error(shape.num_cells)
  , input_history(shape.horizon)
  , saved_timestep(0)
  , learning_rate(learning_rate)
  , num_cells(shape.num_cells)
  , horizon(shape.horizon)
  , output_size(shape.output_size)
  , num_layers(shape.num_layers)
  , epoch(0)
{
  hidden[hidden.size() - 1] = 1.f; // bias

  size_t max_layer_size = 1 + num_cells * 2;

  // Initialize layer inputs and biases
  for (size_t e = 0; e < horizon; e++) {
    for (size_t layer = 0; layer < num_layers; layer++) {
      size_t layer_input_size = 1 + num_cells * (layer > 0 ? 2 : 1);
      // layer_in(e, layer, layer_input_size - 1) = 1.f; // bias
      layer_input[e * num_layers * max_layer_size + layer * max_layer_size + (layer_input_size - 1)] = 1.f;
    }
  }

  // Initialize output probabilities
  float init_prob = 1.0f / output_size;
  for (size_t e = 0; e < horizon; e++) {
    for (size_t i = 0; i < output_size; i++) {
      // out(e, i) = init_prob;
      output[e * output_size + i] = init_prob;
    }
  }

  // Create LSTM layers
  for (size_t i = 0; i < num_layers; i++) {
    size_t layer_input_size = 1 + num_cells * (i > 0 ? 2 : 1);
    layers.push_back(
      std::make_unique<LstmLayer>(
        simdType,
        layer_input_size + output_size,
        output_size,
        num_cells,
        horizon
      )
    );
  }
}

#ifdef X64_SIMD_AVAILABLE

#if (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
#endif
void Lstm::SoftMaxSimdAVX2() {
  size_t const len = hidden.size();
  size_t const hidden_size = num_cells * num_layers + 1;

  // Compute logits via dot products
  for (size_t i = 0; i < output_size; i++) {
    // logit(epoch, i) = dot256_ps_avx2(&hidden[0], &out_layer(epoch, i, 0), len, 0.f);
    logits[epoch * output_size + i] = dot256_ps_avx2(
      &hidden[0],
      &output_layer[epoch * output_size * hidden_size + i * hidden_size],
      len,
      0.f
    );
  }

  // Find max logit for numerical stability
  float max_logit = logits[epoch * output_size];
  for (size_t i = 1; i < output_size; i++) {
    if (logits[epoch * output_size + i] > max_logit)
      max_logit = logits[epoch * output_size + i];
  }

  // Compute softmax
  softmax_avx2(&logits[epoch * output_size], &output[epoch * output_size], output_size, max_logit);
}

#endif

void Lstm::SoftMaxSimdNone() {
  size_t const len = hidden.size();
  size_t const hidden_size = num_cells * num_layers + 1;

  // Compute logits via dot products
  for (size_t i = 0; i < output_size; i++) {
    // logit(epoch, i) = SumOfProducts(&hidden[0], &out_layer(epoch, i, 0), len);
    logits[epoch * output_size + i] = SumOfProducts(
      &hidden[0],
      &output_layer[epoch * output_size * hidden_size + i * hidden_size],
      len
    );
  }

  // Find max logit for numerical stability
  float max_logit = logits[epoch * output_size];
  for (size_t i = 1; i < output_size; i++) {
    if (logits[epoch * output_size + i] > max_logit)
      max_logit = logits[epoch * output_size + i];
  }

  // Compute softmax
  softmax_scalar(&logits[epoch * output_size], &output[epoch * output_size], output_size, max_logit);
}

float* Lstm::Predict(uint8_t const input) {
  size_t const max_layer_size = 1 + num_cells * 2;

  // Create temporary array to pass a slice of layer_input to ForwardPass
  Array<float, 32> temp_input(1 + num_cells * 2 + output_size);

  for (size_t i = 0; i < layers.size(); i++) {
    // Copy from hidden to layer_input
    size_t layer_input_size = 1 + num_cells * (i > 0 ? 2 : 1);
    for (size_t j = 0; j < num_cells && j < layer_input_size - 1; j++) {
      // layer_in(epoch, i, j) = hidden[i * num_cells + j];
      layer_input[epoch * num_layers * max_layer_size + i * max_layer_size + j] = hidden[i * num_cells + j];
    }

    // Prepare temp_input for this layer
    for (size_t j = 0; j < layer_input_size; j++) {
      // temp_input[j] = layer_in(epoch, i, j);
      temp_input[j] = layer_input[epoch * num_layers * max_layer_size + i * max_layer_size + j];
    }
    for (size_t j = 0; j < output_size; j++) {
      temp_input[layer_input_size + j] = 0.f; // Will be filled by ForwardPass
    }

    layers[i]->ForwardPass(temp_input, input, &hidden, i * num_cells);

    // Copy hidden to next layer's input if not last layer
    if (i < layers.size() - 1) {
      for (size_t j = 0; j < num_cells; j++) {
        // layer_in(epoch, i + 1, num_cells + j) = hidden[i * num_cells + j];
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
  if (epoch == horizon) epoch = 0;

  // Return pointer to the output slice for this epoch in the persistent output array
  return &output[epoch_ * output_size];
}

void Lstm::Perceive(const uint8_t input) {
  size_t const last_epoch = ((epoch > 0) ? epoch : horizon) - 1;
  uint8_t const old_input = input_history[last_epoch];
  input_history[last_epoch] = input;

  size_t const max_layer_size = 1 + num_cells * 2;
  size_t const hidden_size = num_cells * num_layers + 1;

  Array<float, 32> temp_input(1 + num_cells * 2 + output_size);

  if (epoch == 0) {
    for (int epoch_ = static_cast<int>(horizon) - 1; epoch_ >= 0; epoch_--) {
      for (int layer = static_cast<int>(layers.size()) - 1; layer >= 0; layer--) {
        int offset = layer * static_cast<int>(num_cells);
        for (size_t i = 0; i < output_size; i++) {
          // float const error = (i == input_history[epoch_]) ? out(epoch_, i) - 1.f : out(epoch_, i);
          float const error = (i == input_history[epoch_]) ? output[epoch_ * output_size + i] - 1.f : output[epoch_ * output_size + i];
          for (size_t j = 0; j < hidden_error.size(); j++) {
            // hidden_error[j] += out_layer(epoch_, i, j + offset) * error;
            hidden_error[j] += output_layer[epoch_ * output_size * hidden_size + i * hidden_size + (j + offset)] * error;
          }
        }

        size_t const prev_epoch = ((epoch_ > 0) ? epoch_ : horizon) - 1;
        uint8_t const input_symbol = (epoch_ > 0) ? input_history[prev_epoch] : old_input;

        // Prepare temp_input
        size_t layer_input_size = 1 + num_cells * (layer > 0 ? 2 : 1);
        for (size_t j = 0; j < layer_input_size; j++) {
          // temp_input[j] = layer_in(epoch_, layer, j);
          temp_input[j] = layer_input[epoch_ * num_layers * max_layer_size + layer * max_layer_size + j];
        }
        for (size_t j = 0; j < output_size; j++) {
          temp_input[layer_input_size + j] = 0.f;
        }

        layers[layer]->BackwardPass(temp_input, epoch_, layer, input_symbol, &hidden_error);
      }
    }
  }

  for (size_t i = 0; i < output_size; i++) {
    // float const error = (i == input) ? out(last_epoch, i) - 1.f : out(last_epoch, i);
    float const error = (i == input) ? output[last_epoch * output_size + i] - 1.f : output[last_epoch * output_size + i];
    for (size_t j = 0; j < hidden.size(); j++) {
      // out_layer(epoch, i, j) = out_layer(last_epoch, i, j) - learning_rate * error * hidden[j];
      output_layer[epoch * output_size * hidden_size + i * hidden_size + j] =
        output_layer[last_epoch * output_size * hidden_size + i * hidden_size + j] - learning_rate * error * hidden[j];
    }
  }
}

uint64_t Lstm::GetCurrentTimeStep() const {
  return layers[0]->update_steps;
}

void Lstm::SetTimeStep(uint64_t const t) {
  for (size_t i = 0; i < layers.size(); i++)
    layers[i]->update_steps = t;
}

void Lstm::Reset() {
  size_t const max_layer_size = 1 + num_cells * 2;
  size_t const hidden_size = num_cells * num_layers + 1;

  memset(&output_layer[0], 0, horizon * output_size * hidden_size * sizeof(float));
  memset(&hidden[0], 0, (hidden.size() - 1) * sizeof(float));
  hidden[hidden.size() - 1] = 1.f;

  float init_prob = 1.0f / output_size;
  for (size_t e = 0; e < horizon; e++) {
    for (size_t i = 0; i < output_size; i++) {
      // out(e, i) = init_prob;
      output[e * output_size + i] = init_prob;
      // logit(e, i) = 0.f;
      logits[e * output_size + i] = 0.f;
    }
    for (size_t layer = 0; layer < num_layers; layer++) {
      size_t layer_input_size = 1 + num_cells * (layer > 0 ? 2 : 1);
      for (size_t k = 0; k < layer_input_size - 1; k++) {
        // layer_in(e, layer, k) = 0.f;
        layer_input[e * num_layers * max_layer_size + layer * max_layer_size + k] = 0.f;
      }
      // layer_in(e, layer, layer_input_size - 1) = 1.f;
      layer_input[e * num_layers * max_layer_size + layer * max_layer_size + (layer_input_size - 1)] = 1.f;
    }
  }

  memset(&hidden_error[0], 0, num_cells * sizeof(float));

  for (size_t i = 0; i < layers.size(); i++)
    layers[i]->Reset();

  epoch = 0;
}

void Lstm::LoadModel(LSTM::Model& model) {
  Reset();
  SetTimeStep(model.timestep);

  size_t const last_epoch = ((epoch > 0) ? epoch : horizon) - 1;
  size_t const hidden_size = num_cells * num_layers + 1;

  for (size_t i = 0; i < output_size; i++) {
    for (size_t j = 0; j < hidden_size; j++) {
      output_layer[last_epoch * output_size * hidden_size + i * hidden_size + j] = model.output[i * hidden_size + j];
    }
  }

  size_t weight_offset = 0;
  for (size_t layer = 0; layer < layers.size(); layer++) {
    auto weight_arrays = layers[layer]->GetWeights();
    size_t weight_size = 1 + num_cells * (layer > 0 ? 2 : 1) + output_size;

    for (size_t gate = 0; gate < 3; gate++) {
      float* gate_weights = nullptr;
      if (gate == 0) gate_weights = weight_arrays.forget_gate_weights;
      else if (gate == 1) gate_weights = weight_arrays.input_node_weights;
      else gate_weights = weight_arrays.output_gate_weights;

      for (size_t cell = 0; cell < num_cells; cell++) {
        for (size_t idx = 0; idx < weight_size; idx++) {
          size_t index = weight_offset + gate * num_cells * weight_size + cell * weight_size + idx;
          gate_weights[cell * weight_size + idx] = model.weights[index];
        }
      }
    }
    weight_offset += 3 * num_cells * weight_size;
  }
}

void Lstm::SaveModel(LSTM::Model& model) {
  model.timestep = GetCurrentTimeStep();
  size_t const last_epoch = ((epoch > 0) ? epoch : horizon) - 1;
  size_t const hidden_size = num_cells * num_layers + 1;

  for (size_t i = 0; i < output_size; i++) {
    for (size_t j = 0; j < hidden_size; j++) {
      model.output[i * hidden_size + j] = output_layer[last_epoch * output_size * hidden_size + i * hidden_size + j];
    }
  }

  size_t weight_offset = 0;
  for (size_t layer = 0; layer < layers.size(); layer++) {
    auto weight_arrays = layers[layer]->GetWeights();
    size_t weight_size = 1 + num_cells * (layer > 0 ? 2 : 1) + output_size;

    for (size_t gate = 0; gate < 3; gate++) {
      float* gate_weights = nullptr;
      if (gate == 0) gate_weights = weight_arrays.forget_gate_weights;
      else if (gate == 1) gate_weights = weight_arrays.input_node_weights;
      else gate_weights = weight_arrays.output_gate_weights;

      for (size_t cell = 0; cell < num_cells; cell++) {
        for (size_t idx = 0; idx < weight_size; idx++) {
          size_t index = weight_offset + gate * num_cells * weight_size + cell * weight_size + idx;
          model.weights[index] = gate_weights[cell * weight_size + idx];
        }
      }
    }
    weight_offset += 3 * num_cells * weight_size;
  }
}
