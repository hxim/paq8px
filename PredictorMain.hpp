#pragma once

#include "Models.hpp"
#include "SSE.hpp"
#include "Shared.hpp"
#include "Predictor.hpp"
#include "model/ContextModel.hpp"

/**
 * A Predictor which estimates the probability that the next bit of uncompressed data is 1 
 * in the main data stream with the help of model(s), mixer(s) and an SSE stage. 
 */
class PredictorMain : public Predictor {
private:
  Models *models;
  MixerFactory* mixerFactory;
  ContextModel* contextModel;
  SSE sse;
public:
  PredictorMain(Shared* const sh);
  ~PredictorMain();
  uint32_t p();
};
