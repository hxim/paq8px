#include "PredictorMainLstmOnly.hpp"
#include "ArithmeticEncoder.hpp"

PredictorMainLstmOnly::PredictorMainLstmOnly(Shared* const sh) : Predictor(sh) {
  models = new Models(sh, nullptr);
}

PredictorMainLstmOnly::~PredictorMainLstmOnly() {
  delete models;
}

uint32_t PredictorMainLstmOnly::p() {
  LstmModelContainer& lstm = models->lstmModelText(); // use the instance for Text so that loading/saving parameters would work properly
  lstm.next();
  constexpr uint32_t scale = 1u << ArithmeticEncoder::PRECISION;
  uint32_t p = static_cast<uint32_t>(roundf(lstm.getp() * scale));
  p = std::clamp(p, 1u, scale - 1u);

  return p;
}
