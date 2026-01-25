#pragma once

#include "Models.hpp"
#include "Shared.hpp"
#include "Predictor.hpp"

/**
 * A Predictor which estimates the probability that the next bit of uncompressed data is 1
 * in the main data stream with the help of model(s), mixer(s) and an SSE stage.
 */
class PredictorMainLstmOnly : public Predictor {
private:
  Models* models;
public:
  PredictorMainLstmOnly(Shared* const sh);
  ~PredictorMainLstmOnly();
  uint32_t p();
};
