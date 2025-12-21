#include "PolynomialDecay.hpp"
#include <cmath>
#include <algorithm>

PolynomialDecay::PolynomialDecay(
    float learningRate,
    float endLearningRate,
    float decayMultiplier,
    float powerNumerator,
    float powerDenominator,
    uint64_t decaySteps)
    : power(powerNumerator / powerDenominator)
    , mul((decaySteps > 0) ? 1.0f / decaySteps : 0.0f)
    , learning_rate(learningRate)
    , end_learning_rate(endLearningRate)
    , decay(decayMultiplier)
    , steps(decaySteps)
{
}

void PolynomialDecay::Apply(float& rate, uint64_t const time_step) const {
    if (steps > 0) {
        if (time_step < steps) {
            rate = (learning_rate - end_learning_rate) * 
                   (std::pow((1.0f - time_step * mul), power)) + end_learning_rate;
        } else {
            rate = end_learning_rate / std::pow(decay * (time_step - steps) + 1.0f, power);
        }
    } else {
        rate = std::max<float>(
            learning_rate / std::pow(decay * time_step + 1.0f, power), 
            end_learning_rate
        );
    }
}