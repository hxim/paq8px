#include "LstmModelContainer.hpp"
#include "../Stretch.hpp"
#include <cmath>

LstmModelContainer::LstmModelContainer(
  Shared* const sh,
  size_t const hidden_size,
  size_t const num_layers,
  size_t const horizon)
  : shared(sh)
  , simd(sh->chosenSimd)
  , shape{ alphabetSize, hidden_size, num_layers, horizon }
  , lstm(sh->chosenSimd, shape, sh->tuning_param)
  , probs(nullptr)
  , byteModelToBitModel()
  , apm1{ sh, 0x10000, 24, 255 }
  , apm2{ sh, 0x800, 24, 255 }
  , apm3{ sh, 1024, 24, 255 }
  , iCtx{ 11, 1, 9 }
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

int LstmModelContainer::getp() {
  int p = static_cast<int32_t>(roundf(byteModelToBitModel.p() * 4096.0f));
  p = std::clamp(p, 1, 4095);
  return p;
}

void LstmModelContainer::mix(Mixer& m) {
  next();

  auto y = shared->State.y;
  auto c0 = shared->State.c0;
  INJECT_SHARED_bpos

  iCtx += y;
  iCtx = (bpos << 8) | expectedByte;
  uint32_t ctx = iCtx();

  int p = getp();
  m.promote(stretch(p) / 2);
  m.add(stretch(p));
  m.add((p - 2048) >> 2);

  int const pr1 = apm1.p(p, (c0 << 8) | (shared->State.misses & 0xFF));
  int const pr2 = apm2.p(p, (bpos << 8) | expectedByte);
  int const pr3 = apm3.p(pr2, ctx);

  m.add(stretch(pr1) >> 1);
  m.add(stretch(pr2) >> 1);
  m.add(stretch(pr3) >> 1);
  m.set((bpos << 8) | expectedByte, 8 * 256);
  m.set(static_cast<uint32_t>(lstm.sequence_position) << 3 | bpos, 100 * 8);
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
