#include "PredictorBlock.hpp"
#include "ArithmeticEncoder.hpp"

PredictorBlock::PredictorBlock(Shared* const sh) : Predictor(sh), mixerFactory(sh), contextModelBlock(sh, &mixerFactory) {}

uint32_t PredictorBlock::p() {
  uint32_t pr = contextModelBlock.p(); // pr is 12 bits
  pr <<= (ArithmeticEncoder::PRECISION - 12); // scale up for the arithmetic encoder
  return pr;
}
