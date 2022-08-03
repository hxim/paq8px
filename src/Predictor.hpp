#pragma once

#include "Shared.hpp"

class Predictor {
public:
  Shared* shared;
  Predictor(Shared* const sh);
  /**
    * Returns the final probability that the next bit is a 1: P(1).
    * This prediction can be entropy encoded.
    * It is a fixed point number between 0.0 and 1.0 scaled by 31 bits (this comes from the PRECISION requirement of the arithmetic encoder)
    * @return the prediction
    */
  virtual uint32_t p() = 0;
};
