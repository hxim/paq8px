#pragma once

#include "LstmLayer.hpp"
#include "SimdFunctions.hpp"
#include "Posit.hpp"
#include "../file/BitFileDisk.hpp"
#include "../file/OpenFromMyFolder.hpp"
#include "../Utils.hpp"
#include "../SIMDType.hpp"
#include <vector>
#include <array>
#include <unordered_map>
#include <memory>
#include <valarray>
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
        std::vector<std::unique_ptr<std::array<std::valarray<std::valarray<float>>, 3>>> weights;
        std::valarray<std::valarray<float>> output;
        
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
    std::valarray<std::valarray<std::valarray<float>>> layer_input, output_layer;
    std::valarray<std::valarray<float>> output;
    std::valarray<std::valarray<float>> logits;
    std::valarray<float> hidden, hidden_error;
    std::vector<uint8_t> input_history;
    uint64_t saved_timestep;
    float learning_rate;
    size_t num_cells, horizon, input_size, output_size;

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

    std::valarray<float>& Predict(uint8_t input);
    void Perceive(uint8_t input);
    uint64_t GetCurrentTimeStep() const;
    void SetTimeStep(uint64_t t);
    void Reset();
    void LoadModel(LSTM::Model& model);
    void SaveModel(LSTM::Model& model);
};
