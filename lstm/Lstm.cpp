#include "Lstm.hpp"
#include "SimdFunctions.hpp"
#include <cstring>
#include <cmath>

// LSTM::Model Implementation

LSTM::Model::Model(LSTM::Shape const shape)
  : timestep(0)
  , shape(shape)
  , output(std::valarray<float>(shape.num_cells* shape.num_layers + 1), shape.output_size)
{
  for (size_t i = 0; i < shape.num_layers; i++) {
    weights.push_back(std::unique_ptr<std::array<std::valarray<std::valarray<float>>, 3>>(
      new std::array<std::valarray<std::valarray<float>>, 3>));
    for (size_t j = 0; j < 3; j++) {
      (*weights[i])[j].resize(shape.num_cells);
      for (size_t k = 0; k < shape.num_cells; k++)
        (*weights[i])[j][k].resize(1 + shape.input_size + shape.num_cells * (i > 0 ? 2 : 1) + shape.output_size);
    }
  }
}

void LSTM::Model::LoadFromDisk(const char* const dictionary, int32_t bits, int32_t exp) {
  BitFileDisk file(true);
  OpenFromMyFolder::anotherFile(&file, dictionary);

  if ((bits > 0) && (bits <= 16)) {
    float scale = Posit<9, 1>::Decode(file.getBits(8));
    for (size_t i = 0; i < shape.output_size; i++) {
      for (size_t j = 0; j < output[i].size(); j++)
        output[i][j] = Posit<16, 1>::Decode(file.getBits(bits)) * scale;
    }
    for (size_t i = 0; i < shape.num_layers; i++) {
      for (size_t j = 0; j < weights[i]->size(); j++) {
        for (size_t k = 0; k < (*weights[i])[j].size(); k++) {
          for (size_t l = 0; l < (*weights[i])[j][k].size(); l++)
            (*weights[i])[j][k][l] = Posit<16, 1>::Decode(file.getBits(bits)) * scale;
        }
      }
    }
  }
  else {
    float v;
    for (size_t i = 0; i < shape.output_size; i++) {
      for (size_t j = 0; j < output[i].size(); j++) {
        if (file.blockRead(reinterpret_cast<uint8_t*>(&v), sizeof(float)) != sizeof(float)) break;
        output[i][j] = v;
      }
    }
    for (size_t i = 0; i < shape.num_layers; i++) {
      for (size_t j = 0; j < weights[i]->size(); j++) {
        for (size_t k = 0; k < (*weights[i])[j].size(); k++) {
          for (size_t l = 0; l < (*weights[i])[j][k].size(); l++) {
            if (file.blockRead(reinterpret_cast<uint8_t*>(&v), sizeof(float)) != sizeof(float)) break;
            (*weights[i])[j][k][l] = v;
          }
        }
      }
    }
  }
  file.close();
}

void LSTM::Model::SaveToDisk(const char* const dictionary, int32_t bits, int32_t exp) {
  BitFileDisk file(false);
  file.create(dictionary);

  if ((bits > 0) && (bits <= 16)) {
    float const s = std::pow(2.f, (1 << exp) * (bits - 2));
    float max_w = 0.f, w, scale;

    for (size_t i = 0; i < shape.output_size; i++) {
      for (size_t j = 0; j < output[i].size(); j++) {
        if ((w = std::fabs(output[i][j])) > max_w)
          max_w = w;
      }
    }
    for (size_t i = 0; i < shape.num_layers; i++) {
      for (size_t j = 0; j < weights[i]->size(); j++) {
        for (size_t k = 0; k < (*weights[i])[j].size(); k++) {
          for (size_t l = 0; l < (*weights[i])[j][k].size(); l++) {
            if ((w = std::fabs((*weights[i])[j][k][l])) > max_w)
              max_w = w;
          }
        }
      }
    }

    scale = Posit<9, 1>::Decode(Posit<9, 1>::Encode(std::max<float>(1.f, max_w / s)));
    file.putBits(Posit<9, 1>::Encode(scale), 8);

    for (size_t i = 0; i < shape.output_size; i++) {
      for (size_t j = 0; j < output[i].size(); j++)
        file.putBits(Posit<16, 1>::Encode(output[i][j] / scale), bits);
    }
    for (size_t i = 0; i < shape.num_layers; i++) {
      for (size_t j = 0; j < weights[i]->size(); j++) {
        for (size_t k = 0; k < (*weights[i])[j].size(); k++) {
          for (size_t l = 0; l < (*weights[i])[j][k].size(); l++)
            file.putBits(Posit<16, 1>::Encode((*weights[i])[j][k][l] / scale), bits);
        }
      }
    }
    file.flush();
  }
  else {
    float v;
    for (size_t i = 0; i < shape.output_size; i++) {
      for (size_t j = 0; j < output[i].size(); j++) {
        v = output[i][j];
        file.blockWrite(reinterpret_cast<uint8_t*>(&v), sizeof(float));
      }
    }
    for (size_t i = 0; i < shape.num_layers; i++) {
      for (size_t j = 0; j < weights[i]->size(); j++) {
        for (size_t k = 0; k < (*weights[i])[j].size(); k++) {
          for (size_t l = 0; l < (*weights[i])[j][k].size(); l++) {
            v = (*weights[i])[j][k][l];
            file.blockWrite(reinterpret_cast<uint8_t*>(&v), sizeof(float));
          }
        }
      }
    }
  }
  file.close();
}

Lstm::Lstm(
  SIMDType simdType,
  LSTM::Shape shape,
  float const learning_rate,
  float const gradient_clip)
  : simd(simdType)
  , layer_input(std::valarray<std::valarray<float>>(std::valarray<float>(shape.input_size + 1 + shape.num_cells * 2), shape.num_layers), shape.horizon)
  , output_layer(std::valarray<std::valarray<float>>(std::valarray<float>(shape.num_cells* shape.num_layers + 1), shape.output_size), shape.horizon)
  , output(std::valarray<float>(1.0f / shape.output_size, shape.output_size), shape.horizon)
  , logits(std::valarray<float>(shape.output_size), shape.horizon)
  , hidden(shape.num_cells* shape.num_layers + 1)
  , hidden_error(shape.num_cells)
  , input_history(shape.horizon)
  , saved_timestep(0)
  , learning_rate(learning_rate)
  , num_cells(shape.num_cells)
  , horizon(shape.horizon)
  , input_size(shape.input_size)
  , output_size(shape.output_size)
  , epoch(0)
{
  hidden[hidden.size() - 1] = 1.f; // bias

  for (size_t epoch = 0; epoch < horizon; epoch++) {
    layer_input[epoch][0].resize(1 + num_cells + input_size);
    for (size_t i = 0; i < shape.num_layers; i++)
      layer_input[epoch][i][layer_input[epoch][i].size() - 1] = 1.f; // bias
  }

  for (size_t i = 0; i < shape.num_layers; i++) {
    layers.push_back(
      std::unique_ptr<LstmLayer>(
        new LstmLayer(
          simdType,
          layer_input[0][i].size() + output_size,
          input_size,
          output_size,
          num_cells,
          horizon,
          gradient_clip
        )
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

  // Compute logits via dot products
  for (size_t i = 0; i < output_size; i++)
    logits[epoch][i] = dot256_ps_avx2(&hidden[0], &output_layer[epoch][i][0], len, 0.f);

  // Find max logit for numerical stability
  float max_logit = logits[epoch][0];
  for (size_t i = 1; i < output_size; i++) {
    if (logits[epoch][i] > max_logit)
      max_logit = logits[epoch][i];
  }

  // Compute softmax
  softmax_avx2(&logits[epoch][0], &output[epoch][0], output_size, max_logit);
}

#endif

void Lstm::SoftMaxSimdNone() {
  size_t const len = hidden.size();

  // Compute logits via dot products
  for (size_t i = 0; i < output_size; i++)
    logits[epoch][i] = SumOfProducts(&hidden[0], &output_layer[epoch][i][0], len);

  // Find max logit for numerical stability
  float max_logit = logits[epoch][0];
  for (size_t i = 1; i < output_size; i++) {
    if (logits[epoch][i] > max_logit)
      max_logit = logits[epoch][i];
  }

  // Compute softmax
  softmax_scalar(&logits[epoch][0], &output[epoch][0], output_size, max_logit);
}

std::valarray<float>& Lstm::Predict(uint8_t const input) {
  for (size_t i = 0; i < layers.size(); i++) {
    memcpy(&layer_input[epoch][i][input_size], &hidden[i * num_cells], num_cells * sizeof(float));
    layers[i]->ForwardPass(layer_input[epoch][i], input, &hidden, i * num_cells);
    if (i < layers.size() - 1) {
      memcpy(&layer_input[epoch][i + 1][num_cells + input_size], &hidden[i * num_cells], num_cells * sizeof(float));
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

  return output[epoch_];
}

void Lstm::Perceive(const uint8_t input) {
  size_t const last_epoch = ((epoch > 0) ? epoch : horizon) - 1;
  uint8_t const old_input = input_history[last_epoch];
  input_history[last_epoch] = input;

  if (epoch == 0) {
    for (int epoch_ = static_cast<int>(horizon) - 1; epoch_ >= 0; epoch_--) {
      for (int layer = static_cast<int>(layers.size()) - 1; layer >= 0; layer--) {
        int offset = layer * static_cast<int>(num_cells);
        for (size_t i = 0; i < output_size; i++) {
          float const error = (i == input_history[epoch_]) ? output[epoch_][i] - 1.f : output[epoch_][i];
          for (size_t j = 0; j < hidden_error.size(); j++)
            hidden_error[j] += output_layer[epoch_][i][j + offset] * error;
        }
        size_t const prev_epoch = ((epoch_ > 0) ? epoch_ : horizon) - 1;
        uint8_t const input_symbol = (epoch_ > 0) ? input_history[prev_epoch] : old_input;
        layers[layer]->BackwardPass(layer_input[epoch_][layer], epoch_, layer, input_symbol, &hidden_error);
      }
    }
  }

  for (size_t i = 0; i < output_size; i++) {
    float const error = (i == input) ? output[last_epoch][i] - 1.f : output[last_epoch][i];
    for (size_t j = 0; j < hidden.size(); j++) {
      output_layer[epoch][i][j] = output_layer[last_epoch][i][j] - learning_rate * error * hidden[j];
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
  for (size_t i = 0; i < output_layer.size(); i++) {
    for (size_t j = 0; j < output_size; j++) {
      for (size_t k = 0; k < output_layer[0][j].size(); k++)
        output_layer[i][j][k] = 0.f;
    }
  }

  for (size_t i = 0; i < hidden.size() - 1; i++)
    hidden[i] = 0.f;

  hidden[hidden.size() - 1] = 1.f;

  for (size_t i = 0; i < horizon; i++) {
    for (size_t j = 0; j < output_size; j++) {
      output[i][j] = 1.0f / output_size;
      logits[i][j] = 0.f;
    }
    for (size_t j = 0; j < layers.size(); j++) {
      for (size_t k = 0; k < layer_input[i][j].size() - 1; k++)
        layer_input[i][j][k] = 0.f;
      layer_input[i][j][layer_input[i][j].size() - 1] = 1.f;
    }
  }

  for (size_t i = 0; i < num_cells; i++)
    hidden_error[i] = 0.f;

  for (size_t i = 0; i < layers.size(); i++)
    layers[i]->Reset();

  epoch = 0;
}

void Lstm::LoadModel(LSTM::Model& model) {
  Reset();
  SetTimeStep(model.timestep);

  size_t const last_epoch = ((epoch > 0) ? epoch : horizon) - 1;

  for (size_t i = 0; i < output_size; i++) {
    for (size_t j = 0; j < output_layer[0][i].size(); j++)
      output_layer[last_epoch][i][j] = model.output[i][j];
  }

  for (size_t i = 0; i < layers.size(); i++) {
    auto weights = layers[i]->Weights();
    for (size_t j = 0; j < weights.size(); j++) {
      for (size_t k = 0; k < weights[j]->size(); k++) {
        for (size_t l = 0; l < (*weights[j])[k].size(); l++)
          (*weights[j])[k][l] = (*model.weights[i])[j][k][l];
      }
    }
  }
}

void Lstm::SaveModel(LSTM::Model& model) {
  model.timestep = GetCurrentTimeStep();
  size_t const last_epoch = ((epoch > 0) ? epoch : horizon) - 1;

  for (size_t i = 0; i < output_size; i++) {
    for (size_t j = 0; j < output_layer[0][i].size(); j++)
      model.output[i][j] = output_layer[last_epoch][i][j];
  }

  for (size_t i = 0; i < layers.size(); i++) {
    auto weights = layers[i]->Weights();
    for (size_t j = 0; j < weights.size(); j++) {
      for (size_t k = 0; k < weights[j]->size(); k++) {
        for (size_t l = 0; l < (*weights[j])[k].size(); l++)
          (*model.weights[i])[j][k][l] = (*weights[j])[k][l];
      }
    }
  }
}
