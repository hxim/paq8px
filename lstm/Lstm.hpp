#pragma once

#include "LstmLayer.hpp"
#include "SimdFunctions.hpp"
#include "Posit.hpp"
#include "../file/BitFileDisk.hpp"
#include "../file/OpenFromMyFolder.hpp"
#include "../Utils.hpp"
#include "../Array.hpp"
#include "../SIMDType.hpp"
#include <vector>
#include <array>
#include <unordered_map>
#include <memory>
#include <cstdint>
#include <cstddef>
namespace LSTM {
  struct Shape {
    size_t output_size;
    size_t num_cells;
    size_t num_layers;
    size_t horizon;
  };

  class Model {
  public:
    enum class Type {
      Default,
      English,
      x86_64
    };

    uint64_t timestep;
    LSTM::Shape shape;
    // Flattened weights array
    // weights[layer * 3 * num_cells * weight_size + gate * num_cells * weight_size + cell * weight_size + weight_index]
    Array<float, 32> weights;
    Array<float, 32> output;  // Flat: [output_size * (num_cells * num_layers + 1)]

    Model(LSTM::Shape const shape);

    void LoadFromDisk(const char* const dictionary, int32_t bits = 0, int32_t exp = 0);
    void SaveToDisk(const char* const dictionary, int32_t bits = 0, int32_t exp = 0);
  };

  using Repository = typename std::unordered_map<LSTM::Model::Type, std::unique_ptr<LSTM::Model>>;
}

class Lstm {
private:
  SIMDType simd;
  std::vector<std::unique_ptr<LstmLayer>> layers;
  Array<float, 32> layer_input;    // Flat: [horizon * num_layers * max_layer_input_size]
  Array<float, 32> output_layer;   // Flat: [horizon * output_size * (num_cells * num_layers + 1)]
  Array<float, 32> output;         // Flat: [horizon * output_size]
  Array<float, 32> logits;         // Flat: [horizon * output_size]
  Array<float, 32> hidden;
  Array<float, 32> hidden_error;
  std::vector<uint8_t> input_history;
  uint64_t saved_timestep;
  float learning_rate;
  size_t num_cells;
  size_t horizon;
  size_t output_size;
  size_t num_layers;

#ifdef X64_SIMD_AVAILABLE
  void SoftMaxSimdAVX2();
#endif

  void SoftMaxSimdNone();

public:
  size_t epoch;

  Lstm(
    SIMDType simdType,
    LSTM::Shape shape,
    float learning_rate);

  float* Predict(uint8_t input);
  void Perceive(uint8_t input);
  uint64_t GetCurrentTimeStep() const;
  void SetTimeStep(uint64_t t);
  void Reset();
  void LoadModel(LSTM::Model& model);
  void SaveModel(LSTM::Model& model);
};
