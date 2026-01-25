#include "SqrtLearningRateDecay.hpp"
#include <cmath>
#include <algorithm>

SqrtLearningRateDecay::SqrtLearningRateDecay(
  float initial_lr,
  float final_lr,
  float decay_rate)
  : initial_lr(initial_lr)
  , final_lr(final_lr)
  , decay_rate(decay_rate)
{
}

float SqrtLearningRateDecay::GetLearningRate(uint64_t iteration) const {
  float lr = initial_lr / std::sqrt(decay_rate * iteration + 1.0f);
  return std::max(lr, final_lr);
}
