#pragma once

#include "LstmLayer.hpp"
#include "SimdFunctions.hpp"
#include "Posit.hpp"
#include "../file/BitFileDisk.hpp"
#include "../file/OpenFromMyFolder.hpp"
#include "../Utils.hpp"
#include <vector>
#include <array>
#include <unordered_map>
#include <memory>

namespace LSTM {
  struct Shape {
    std::size_t input_size;
    std::size_t output_size;
    std::size_t num_cells;
    std::size_t num_layers;
    std::size_t horizon;
  };

  class Model {
  public:
    enum class Type {
      Default,
      English,
      x86_64
    };
    std::uint64_t timestep;
    LSTM::Shape shape;
    std::vector<std::unique_ptr<std::array<std::valarray<std::valarray<float>>, 3>>> weights;
    std::valarray<std::valarray<float>> output;
    Model(LSTM::Shape const shape) :
      timestep(0u),
      shape(shape),
      output(std::valarray<float>(shape.num_cells * shape.num_layers + 1), shape.output_size)
    {
      for (std::size_t i = 0u; i < shape.num_layers; i++) {
        weights.push_back(std::unique_ptr<std::array<std::valarray<std::valarray<float>>, 3>>(new std::array<std::valarray<std::valarray<float>>, 3>));
        for (std::size_t j = 0u; j < 3u; j++) {
          (*weights[i])[j].resize(shape.num_cells);
          for (std::size_t k = 0u; k < shape.num_cells; k++)
            (*weights[i])[j][k].resize( 1 + shape.input_size + shape.num_cells*(i > 0u? 2u : 1u) + shape.output_size);
        }
      }
    }
    template<std::int32_t bits = 0, std::int32_t exp = 0>
    void LoadFromDisk(const char* const dictionary) {
      static_assert((bits >= 0) && (bits <= 16), "LSTM::Model::LoadFromDisk template parameter @bits must be in range [0..16]");
      BitFileDisk file(true);
      OpenFromMyFolder::anotherFile(&file, dictionary);
      if ((bits > 0) && (bits <= 16)) {
        float scale = Posit<9, 1>::Decode(file.getBits(8u));
        for (std::size_t i = 0u; i < shape.output_size; i++) {
          for (std::size_t j = 0u; j < output[i].size(); j++)
            output[i][j] = Posit<bits, exp>::Decode(file.getBits(bits)) * scale;
        }
        for (std::size_t i = 0u; i < shape.num_layers; i++) {
          for (std::size_t j = 0u; j < weights[i]->size(); j++) {
            for (std::size_t k = 0u; k < (*weights[i])[j].size(); k++) {
              for (std::size_t l = 0u; l < (*weights[i])[j][k].size(); l++)
                (*weights[i])[j][k][l] = Posit<bits, exp>::Decode(file.getBits(bits)) * scale;
            }
          }
        }
      }
      else {
        float v;
        for (std::size_t i = 0u; i < shape.output_size; i++) {
          for (std::size_t j = 0u; j < output[i].size(); j++) {
            if (file.blockRead(reinterpret_cast<std::uint8_t*>(&v), sizeof(float)) != sizeof(float)) break;
            output[i][j] = v;
          }
        }
        for (std::size_t i = 0u; i < shape.num_layers; i++) {
          for (std::size_t j = 0u; j < weights[i]->size(); j++) {
            for (std::size_t k = 0u; k < (*weights[i])[j].size(); k++) {
              for (std::size_t l = 0u; l < (*weights[i])[j][k].size(); l++) {
                if (file.blockRead(reinterpret_cast<std::uint8_t*>(&v), sizeof(float)) != sizeof(float)) break;
                (*weights[i])[j][k][l] = v;
              }
            }
          }
        }
      }
      file.close();
    }
    template<std::int32_t bits = 0, std::int32_t exp = 0>
    void SaveToDisk(const char* const dictionary) {
      static_assert((bits >= 0) && (bits <= 16), "LSTM::Model::SaveToDisk template parameter @bits must be in range [0..16]");
      BitFileDisk file(false);
      file.create(dictionary);
      if ((bits > 0) && (bits <= 16)) {
        std::uint32_t buf = 0u;
        std::int32_t constexpr buf_width = static_cast<std::int32_t>(sizeof(buf)) * 8;
        std::int32_t bits_left = buf_width;
        float const s = std::pow(2.f, (1 << exp) * (bits - 2));
        float max_w = 0.f, w, scale;
        for (std::size_t i = 0u; i < shape.output_size; i++) {
          for (std::size_t j = 0u; j < output[i].size(); j++) {
            if ((w = std::fabs(output[i][j])) > max_w)
              max_w = w;
          }
        }
        for (std::size_t i = 0u; i < shape.num_layers; i++) {
          for (std::size_t j = 0u; j < weights[i]->size(); j++) {
            for (std::size_t k = 0u; k < (*weights[i])[j].size(); k++) {
              for (std::size_t l = 0u; l < (*weights[i])[j][k].size(); l++) {
                if ((w = std::fabs((*weights[i])[j][k][l])) > max_w)
                  max_w = w;
              }
            }
          }
        }
        scale = Posit<9, 1>::Decode(Posit<9, 1>::Encode(std::max<float>(1.f, max_w / s)));
        file.putBits(Posit<9, 1>::Encode(scale), 8u);
        for (std::size_t i = 0u; i < shape.output_size; i++) {
          for (std::size_t j = 0u; j < output[i].size(); j++)
            file.putBits(Posit<bits, exp>::Encode(output[i][j] / scale), bits);
        }
        for (std::size_t i = 0u; i < shape.num_layers; i++) {
          for (std::size_t j = 0u; j < weights[i]->size(); j++) {
            for (std::size_t k = 0u; k < (*weights[i])[j].size(); k++) {
              for (std::size_t l = 0u; l < (*weights[i])[j][k].size(); l++)
                file.putBits(Posit<bits, exp>::Encode((*weights[i])[j][k][l] / scale), bits);
            }
          }
        }
        file.flush();
      }
      else {
        float v;
        for (std::size_t i = 0u; i < shape.output_size; i++) {
          for (std::size_t j = 0u; j < output[i].size(); j++) {
            v = output[i][j];
            file.blockWrite(reinterpret_cast<std::uint8_t*>(&v), sizeof(float));
          }
        }
        for (std::size_t i = 0u; i < shape.num_layers; i++) {
          for (std::size_t j = 0u; j < weights[i]->size(); j++) {
            for (std::size_t k = 0u; k < (*weights[i])[j].size(); k++) {
              for (std::size_t l = 0u; l < (*weights[i])[j][k].size(); l++) {
                v = (*weights[i])[j][k][l];
                file.blockWrite(reinterpret_cast<std::uint8_t*>(&v), sizeof(float));
              }
            }
          }
        }
      }
      file.close();
    }
  }; // class Model

  using Repository = typename std::unordered_map<LSTM::Model::Type, std::unique_ptr<LSTM::Model>>;
} // namespace LSTM

/**
 * Long Short-Term Memory neural network.
 * Based on the LSTM implementation in cmix by Byron Knoll.
 */
template <SIMDType simd, typename T>
class Lstm {
  static_assert(std::is_integral<T>::value && (!std::is_same<T, bool>::value), "LSTM input type must be integral and non-boolean");
private:
  std::vector<std::unique_ptr<LstmLayer<simd, T>>> layers;
  std::valarray<std::valarray<std::valarray<float>>> layer_input, output_layer;
  std::valarray<std::valarray<float>> output;
  std::valarray<float> hidden, hidden_error;
  std::vector<T> input_history;
  std::uint64_t saved_timestep;
  float learning_rate;
  std::size_t num_cells, horizon, input_size, output_size;

#if (defined(__GNUC__) || defined(__clang__)) && (!defined(__ARM_FEATURE_SIMD32) && !defined(__ARM_NEON))
  __attribute__((target("avx2,fma")))
#endif
  void SoftMaxSimdAVX2() {
#if !defined(__i386__) && !defined(__x86_64__) && !defined(_M_X64)
    return;
#else
    static constexpr std::size_t SIMDW = 8;
    std::size_t const limit = output_size & static_cast<std::size_t>(-static_cast<std::ptrdiff_t>(SIMDW)), len = hidden.size();
    std::size_t remainder = output_size & (SIMDW - 1);
    __m256 v_sum = _mm256_setzero_ps();
    for (std::size_t i = 0; i < limit; i++)
      output[epoch][i] = dot256_ps_fma3(&hidden[0], &output_layer[epoch][i][0], len, 0.f);
    for (std::size_t i = 0; i < limit; i += SIMDW) {
      __m256 v_exp = exp256_ps_fma3(_mm256_loadu_ps(&output[epoch][i]));
      _mm256_storeu_ps(&output[epoch][i], v_exp);
      v_sum = _mm256_add_ps(v_sum, v_exp);
    }
    float sum = hsum256_ps_avx(v_sum);
    for (; remainder > 0; remainder--) {
      const std::size_t i = output_size - remainder;
      output[epoch][i] = expa(dot256_ps_fma3(&hidden[0], &output_layer[epoch][i][0], len, 0.f));
      sum += output[epoch][i];
    }
    output[epoch] /= sum;
#endif
  }
  void SoftMaxSimdNone() {
    for (unsigned int i = 0; i < output_size; ++i)
      output[epoch][i] = expa(SumOfProducts(&hidden[0], &output_layer[epoch][i][0], hidden.size()));
    float s = 0.0f;
    for (int i = 0; i < output[epoch].size(); i++) s += output[epoch][i];
    for (int i = 0; i < output[epoch].size(); i++) output[epoch][i] /= s;
  }
public:
  std::size_t epoch;
  Lstm(
    LSTM::Shape shape,
    float const learning_rate,
    float const gradient_clip) :
    layer_input(std::valarray<std::valarray<float>>(std::valarray<float>(shape.input_size + 1 + shape.num_cells * 2), shape.num_layers), shape.horizon),
    output_layer(std::valarray<std::valarray<float>>(std::valarray<float>(shape.num_cells * shape.num_layers + 1), shape.output_size), shape.horizon),
    output(std::valarray<float>(1.0f / shape.output_size, shape.output_size), shape.horizon),
    hidden(shape.num_cells * shape.num_layers + 1),
    hidden_error(shape.num_cells),
    input_history(shape.horizon),
    saved_timestep(0),
    learning_rate(learning_rate),
    num_cells(shape.num_cells),
    horizon(shape.horizon),
    input_size(shape.input_size),
    output_size(shape.output_size),
    epoch(0)
  {
    hidden[hidden.size() - 1] = 1.f;
    for (std::size_t epoch = 0; epoch < horizon; epoch++) {
      layer_input[epoch][0].resize(1 + num_cells + input_size);
      for (std::size_t i = 0; i < shape.num_layers; i++)
        layer_input[epoch][i][layer_input[epoch][i].size() - 1] = 1.f;
    }
    for (std::size_t i = 0; i < shape.num_layers; i++) {
      layers.push_back(std::unique_ptr<LstmLayer<simd, T>>(new LstmLayer<simd, T>(
        layer_input[0][i].size() + output_size,
        input_size, output_size,
        num_cells, horizon, gradient_clip, learning_rate
        )));
    }
  }

  void SetInput(std::valarray<float> const& input) {
    for (std::size_t i = 0; i < layers.size(); i++)
      memcpy(&layer_input[epoch][i][0], &input[0], input_size * sizeof(float));
  }

  std::valarray<float>& Predict(T const input) {
    for (std::size_t i = 0; i < layers.size(); i++) {
      memcpy(&layer_input[epoch][i][input_size], &hidden[i * num_cells], num_cells * sizeof(float));
      layers[i]->ForwardPass(layer_input[epoch][i], input, &hidden, i * num_cells);
      if (i < layers.size() - 1) {
        memcpy(&layer_input[epoch][i + 1][num_cells + input_size], &hidden[i * num_cells], num_cells * sizeof(float));
      }
    }
    if (simd == SIMDType::SIMD_AVX2)
      SoftMaxSimdAVX2();
    else
      SoftMaxSimdNone();
    std::size_t const epoch_ = epoch;
    epoch++;
    if (epoch == horizon) epoch = 0;
    return output[epoch_];
  }

  std::valarray<float>& Perceive(const T input) {
    std::size_t const last_epoch = ((epoch > 0) ? epoch : horizon) - 1;
    T const old_input = input_history[last_epoch];
    input_history[last_epoch] = input;
    if (epoch == 0) {
      for (int epoch_ = static_cast<int>(horizon) - 1; epoch_ >= 0; epoch_--) {
        for (int layer = static_cast<int>(layers.size()) - 1; layer >= 0; layer--) {
          int offset = layer * static_cast<int>(num_cells);
          for (std::size_t i = 0; i < output_size; i++) {
            float const error = (i == input_history[epoch_]) ? output[epoch_][i] - 1.f : output[epoch_][i];
            for (std::size_t j = 0; j < hidden_error.size(); j++)
              hidden_error[j] += output_layer[epoch_][i][j + offset] * error;
          }
          std::size_t const prev_epoch = ((epoch_ > 0) ? epoch_ : horizon) - 1;
          T const input_symbol = (epoch_ > 0) ? input_history[prev_epoch] : old_input;
          layers[layer]->BackwardPass(layer_input[epoch_][layer], epoch_, layer, input_symbol, &hidden_error);
        }
      }
    }
    for (std::size_t i = 0; i < output_size; i++) {
      float const error = (i == input) ? output[last_epoch][i] - 1.f : output[last_epoch][i];
      for (int j = 0; j < hidden.size(); j++) {
        output_layer[epoch][i][j] = output_layer[last_epoch][i][j]- learning_rate * error * hidden[j];
      }
    }
    return Predict(input);
  }

  void SaveTimeStep() {
    saved_timestep = layers[0]->update_steps;
  }

  void RestoreTimeStep() {
    for (std::size_t i = 0; i < layers.size(); i++)
      layers[i]->update_steps = saved_timestep;
  }

  void SetTimeStep(std::uint64_t const t) {
    for (std::size_t i = 0; i < layers.size(); i++)
      layers[i]->update_steps = t;
  }

  void Reset() {
    for (std::size_t i = 0u; i < output_layer.size(); i++) {
      for (std::size_t j = 0u; j < output_size; j++) {
        for (std::size_t k = 0u; k < output_layer[0][j].size(); k++)
          output_layer[i][j][k] = 0.f;
      }
    }
    for (std::size_t i = 0u; i < hidden.size() - 1; i++)
      hidden[i] = 0.f;
    hidden[hidden.size() - 1u] = 1.f;
    for (std::size_t i = 0u; i < horizon; i++) {
      for (std::size_t j = 0u; j < output_size; j++)
        output[i][j] = 1.0f / output_size;
      for (std::size_t j = 0u; j < layers.size(); j++) {
        for (std::size_t k = 0u; k < layer_input[i][j].size() - 1; k++)
          layer_input[i][j][k] = 0.f;
        layer_input[i][j][layer_input[i][j].size() - 1u] = 1.f;
      }
    }
    for (std::size_t i = 0u; i < num_cells; i++)
      hidden_error[i] = 0.f;
    for (std::size_t i = 0u; i < layers.size(); i++)
      layers[i]->Reset();
    epoch = 0u;
  }

  void LoadModel(LSTM::Model& model) {
    Reset();
    SetTimeStep(model.timestep);
    std::size_t const last_epoch = ((epoch > 0) ? epoch : horizon) - 1;
    for (std::size_t i = 0u; i < output_size; i++) {
      for (std::size_t j = 0u; j < output_layer[0][i].size(); j++)
        output_layer[last_epoch][i][j] = model.output[i][j];
    }
    for (std::size_t i = 0u; i < layers.size(); i++) {
      auto weights = layers[i]->Weights();
      for (std::size_t j = 0u; j < weights.size(); j++) {
        for (std::size_t k = 0u; k < weights[j]->size(); k++) {
          for (std::size_t l = 0u; l < (*weights[j])[k].size(); l++)
            (*weights[j])[k][l] = (*model.weights[i])[j][k][l];
        }
      }
    }
  }

  void SaveModel(LSTM::Model& model) {
    model.timestep = layers[0]->update_steps;
    std::size_t const last_epoch = ((epoch > 0) ? epoch : horizon) - 1;
    for (std::size_t i = 0u; i < output_size; i++) {
      for (std::size_t j = 0u; j < output_layer[0][i].size(); j++)
        model.output[i][j] = output_layer[last_epoch][i][j];
    }
    for (std::size_t i = 0u; i < layers.size(); i++) {
      auto weights = layers[i]->Weights();
      for (std::size_t j = 0u; j < weights.size(); j++) {
        for (std::size_t k = 0u; k < weights[j]->size(); k++) {
          for (std::size_t l = 0u; l < (*weights[j])[k].size(); l++)
            (*model.weights[i])[j][k][l] = (*weights[j])[k][l];
        }
      }
    }
  }
};
