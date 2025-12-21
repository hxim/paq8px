#pragma once

#include "Layer.hpp"
#include "../SIMDType.hpp"
#include <valarray>
#include <vector>
#include <cstdint>

class LstmLayer {
private:
    SIMDType simd;
    
    std::valarray<float> state;
    std::valarray<float> state_error;
    std::valarray<float> stored_error;
    
    std::valarray<std::valarray<float>> tanh_state;
    std::valarray<std::valarray<float>> input_gate_state;
    std::valarray<std::valarray<float>> last_state;
    
    size_t num_cells;
    size_t epoch;
    size_t horizon;

    Layer forget_gate;
    Layer input_node;
    Layer output_gate;

    static float Rand(float range);

public:
    uint64_t update_steps;

    LstmLayer(
        SIMDType simdType,
        size_t input_size,
        size_t output_size,
        size_t num_cells,
        size_t horizon,
        float range = 0.4f);

    void ForwardPass(
        std::valarray<float> const& input,
        uint8_t input_symbol,
        std::valarray<float>* hidden,
        size_t hidden_start);

    void BackwardPass(
        std::valarray<float> const& input,
        size_t epoch,
        size_t layer,
        uint8_t input_symbol,
        std::valarray<float>* hidden_error);

    void Reset();
    
    std::vector<std::valarray<std::valarray<float>>*> Weights();
};
