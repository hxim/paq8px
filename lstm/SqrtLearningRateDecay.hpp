#pragma once

#include <cstdint>

class SqrtLearningRateDecay {
private:
  float initial_lr;
  float final_lr;
  float decay_rate;

public:
  SqrtLearningRateDecay(float initial_lr, float final_lr, float decay_rate);
  float GetLearningRate(uint64_t iteration) const;
};
