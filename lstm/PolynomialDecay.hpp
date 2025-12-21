#pragma once

#include "../Utils.hpp"
#include <cstdint>

class PolynomialDecay {
private:
    float power;
    float mul;
    float learning_rate;
    float end_learning_rate;
    float decay;
    uint64_t steps;

public:
    PolynomialDecay(
        float learningRate,
        float endLearningRate,
        float decayMultiplier,
        float powerNumerator,
        float powerDenominator,
        uint64_t decaySteps = 0
    );
    
    ~PolynomialDecay() = default;
    
    void Apply(float& rate, uint64_t time_step) const;
};