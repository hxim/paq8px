#include "LstmModelContainer.hpp"
#include "../Stretch.hpp"
#include <cmath>

LstmModelContainer::LstmModelContainer(Shared* const sh)
  : shared(sh)
  , simd(sh->chosenSimd)
  , shape{ alphabetSize, shared->LstmSettings.hidden_size, shared->LstmSettings.num_layers, shared->LstmSettings.horizon }
  , lstm(sh->chosenSimd, shape, sh->tuning_param)
  , probs(nullptr)
  , byteModelToBitModel()
  , apm1{ sh, 1<<16, 24, 255 }
  , apm2{ sh, 1<<12, 24, 1023 }
  , expectedByte(0)
{
}

void LstmModelContainer::next() {
  shared->GetUpdateBroadcaster()->subscribe(this);

  INJECT_SHARED_bpos
  if (bpos == 0) {
    uint8_t const c1 = shared->State.c1;
    probs = const_cast<float*>(lstm.Predict(c1));
    byteModelToBitModel.CalculateByteProbabilities(probs, alphabetSize);
    expectedByte = (uint8_t)byteModelToBitModel.GetExpectedByte(probs, alphabetSize);
  }
}

float LstmModelContainer::getp() {
  float prob = byteModelToBitModel.p();
  return prob;
}

void LstmModelContainer::mix(Mixer& m) {
  next();

  auto y = shared->State.y;
  auto c0 = shared->State.c0;
  INJECT_SHARED_bpos

  const float prob = getp();
  int p = static_cast<int32_t>(roundf(prob * 4096.0f));
  p = std::clamp(p, 1, 4095);

  int st = stretch(p);
  m.promote(st);
  m.add(st);
  m.add((p - 2048) >> 2);

  uint32_t misses = shared->State.misses & 0xFF;
  uint32_t certain = (p == 1) || (p == 4095);

  int const pr1 = apm1.p(p, c0 << 8 | misses); // 16 bits
  int const pr2 = apm2.p(p, expectedByte << 4 | certain << 3 | bpos); // 12 bits

  m.add(stretch(pr1) >> 1);
  m.add(stretch(pr2) >> 1);

  m.set(bpos << 8 | expectedByte, 8 * 256); // 12 bit
}

void LstmModelContainer::update() {
  INJECT_SHARED_bpos
  if (bpos == 0) {
    uint8_t c = shared->State.c1;
    lstm.Perceive(c);
  }
  else {
    byteModelToBitModel.SliceForNextBit(probs, shared->State.y);
  }
}

void LstmModelContainer::LoadModelParameters(FILE* file) {
  LoadSave stream(file);
  lstm.LoadModelParameters(stream);
}

void LstmModelContainer::SaveModelParameters(FILE* file) {
  LoadSave stream(file);
  lstm.SaveModelParameters(stream);
}
