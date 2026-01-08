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

void PolynomialDecay::Apply(float& rate, uint64_t const training_iterations) const {
    if (steps > 0) {
        if (training_iterations < steps) {
            rate = (learning_rate - end_learning_rate) * 
                   (std::pow((1.0f - training_iterations * mul), exponent)) + end_learning_rate;
        } else {
            rate = end_learning_rate / std::pow(decayMultiplier * (training_iterations - steps) + 1.0f, exponent);
        }
    } else {
        rate = std::max<float>(
            learning_rate / std::pow(decayMultiplier * training_iterations + 1.0f, exponent),
            end_learning_rate
        );
    }
}
