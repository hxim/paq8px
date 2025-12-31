#include "PolynomialDecay.hpp"
#include <cmath>
#include <algorithm>

PolynomialDecay::PolynomialDecay(
    float learningRate,
    float endLearningRate,
    float decayMultiplier,
    float exponent,
    uint64_t decaySteps)
    : exponent(exponent)
    , mul((decaySteps > 0) ? 1.0f / decaySteps : 0.0f)
    , learning_rate(learningRate)
    , end_learning_rate(endLearningRate)
    , decayMultiplier(decayMultiplier)
    , steps(decaySteps)
{
}

void PolynomialDecay::Apply(float& rate, uint64_t const time_step) const {
    if (steps > 0) {
        if (time_step < steps) {
            rate = (learning_rate - end_learning_rate) * 
                   (std::pow((1.0f - time_step * mul), exponent)) + end_learning_rate;
        } else {
            rate = end_learning_rate / std::pow(decayMultiplier * (time_step - steps) + 1.0f, exponent);
        }
    } else {
        rate = std::max<float>(
            learning_rate / std::pow(decayMultiplier * time_step + 1.0f, exponent),
            end_learning_rate
        );
    }
}
