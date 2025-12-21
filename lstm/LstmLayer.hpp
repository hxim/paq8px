#pragma once

#include "Layer.hpp"
#include "../SIMDType.hpp"
#include <valarray>
#include <vector>
#include <cstdint>

template <typename T>
class LstmLayer {
private:
    SIMDType simd;
    
    std::valarray<float> state;
    std::valarray<float> state_error;
    std::valarray<float> stored_error;
    
    std::valarray<std::valarray<float>> tanh_state;
    std::valarray<std::valarray<float>> input_gate_state;
    std::valarray<std::valarray<float>> last_state;
    
    float gradient_clip;
    
    size_t num_cells;
    size_t epoch;
    size_t horizon;
    size_t input_size;
    size_t output_size;

    Layer<T> forget_gate;
    Layer<T> input_node;
    Layer<T> output_gate;

    void Clamp(std::valarray<float>* x);
    static float Rand(float range);

public:
    uint64_t update_steps;

    LstmLayer(
        SIMDType simdType,
        size_t input_size,
        size_t auxiliary_input_size,
        size_t output_size,
        size_t num_cells,
        size_t horizon,
        float gradient_clip,
        float range = 0.4f);

    void ForwardPass(
        std::valarray<float> const& input,
        T input_symbol,
        std::valarray<float>* hidden,
        size_t hidden_start);

    void BackwardPass(
        std::valarray<float> const& input,
        size_t epoch,
        size_t layer,
        T input_symbol,
        std::valarray<float>* hidden_error);

    void Reset();
    
    std::vector<std::valarray<std::valarray<float>>*> Weights();
};