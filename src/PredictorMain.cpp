#include "PredictorMain.hpp"
#include "ArithmeticEncoder.hpp"

PredictorMain::PredictorMain(Shared* const sh) : Predictor(sh), sse(sh) {
  mixerFactory = new MixerFactory(shared);
  models = new Models(sh, mixerFactory);
  models->trainModelsWhenNeeded();
  contextModel = new ContextModel(sh, models, mixerFactory);
}

PredictorMain::~PredictorMain() {
  delete models;
  delete contextModel;
  delete mixerFactory;
}

uint32_t PredictorMain::p() {
  uint32_t pr = contextModel->p();
  pr = sse.p(pr);
  return pr;
}