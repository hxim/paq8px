#pragma once

#include "Shared.hpp"
#include "Predictor.hpp"
#include "model/ContextModelBlock.hpp"

/**
 * A Predictor which estimates the probability that the next bit of metadata is 1 
 * with the help of a model and a mixer.
 * Metadata = (block type, block info and block size) 
 */
class PredictorBlock : public Predictor {
private:
  MixerFactory mixerFactory;
  ContextModelBlock contextModelBlock;
public:
  PredictorBlock(Shared* const sh);
  uint32_t p();
};
