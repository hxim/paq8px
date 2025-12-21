#pragma once

#include "Adam.hpp"
#include "SimdFunctions.hpp"
#include "PolynomialDecay.hpp"
#include "../SIMDType.hpp"
#include <cstdint>
#include <valarray>

class Layer {
private:
    SIMDType simd;
    
    std::valarray<std::valarray<float>> update;
    std::valarray<std::valarray<float>> v;
    std::valarray<std::valarray<float>> transpose;
    std::valarray<std::valarray<float>> norm;
    
    std::valarray<float> inverse_variance;
    
    std::valarray<float> gamma; 
    std::valarray<float> gamma_u; 
    std::valarray<float> gamma_v;

    std::valarray<float> beta;
    std::valarray<float> beta_u;
    std::valarray<float> beta_v;

    size_t input_size;
    size_t output_size;
    size_t num_cells;
    size_t horizon;

    float learning_rate;

    Adam optimizer;
    Tanh activation_tanh;
    Logistic activation_logistic;
    PolynomialDecay decay;
    
    bool use_tanh; // true for Tanh, false for Logistic

public:
    std::valarray<std::valarray<float>> weights;
    std::valarray<std::valarray<float>> state;
    std::valarray<float> error;

    Layer(
        SIMDType simdType,
        size_t input_size,
        size_t output_size,
        size_t num_cells,
        size_t horizon,
        bool useTanh,
        float beta2,
        float epsilon,
        float learningRate,
        float endLearningRate,
        float decayMultiplier,
        float powerNumerator,
        float powerDenominator,
        uint64_t decaySteps = 0
    );

    void ForwardPass(
        std::valarray<float> const& input,
        uint8_t input_symbol,
        size_t epoch);

    void BackwardPass(
        std::valarray<float> const& input,
        std::valarray<float>* hidden_error,
        std::valarray<float>* stored_error,
        uint64_t time_step,
        size_t epoch,
        size_t layer,
        uint8_t input_symbol);

    void Reset();
};
