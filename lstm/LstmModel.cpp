#include "LstmModel.hpp"
#include "../Stretch.hpp"
#include <cstring>
#include <algorithm>

LstmModel::LstmModel(
  const Shared* const sh,
  SIMDType simdType,
  size_t const num_cells,
  size_t const num_layers,
  size_t const horizon,
  float const learning_rate)
  : shared(sh)
  , probs(alphabetSize)
  , apm1{ sh, 0x10000, 24, 255 }
  , apm2{ sh, 0x800, 24, 255 }
  , apm3{ sh, 1024, 24, 255 }
  , iCtx{ 11, 1, 9 }
  , top(alphabetSize - 1)
  , mid(0)
  , bot(0)
  , expected(0)
  , simd(simdType)
  , shape{ alphabetSize, num_cells, num_layers, horizon }
  , lstm(simdType, shape, learning_rate)
  , modelType(LSTM::Model::Type::Default)
  , pModelType(LSTM::Model::Type::Default)
  , pBlockType(BlockType::Count)
{
  float const init_value = 1.f / alphabetSize;
  for (size_t i = 0; i < alphabetSize; i++) {
    probs[i] = init_value;
  }

  if (shared->GetOptionTrainLSTM()) {
    repo[LSTM::Model::Type::Default] = std::unique_ptr<LSTM::Model>(new LSTM::Model(shape));
    lstm.SaveModel(*repo[LSTM::Model::Type::Default]);

    repo[LSTM::Model::Type::English] = std::unique_ptr<LSTM::Model>(new LSTM::Model(shape));
    repo[LSTM::Model::Type::English]->LoadFromDisk("english.rnn", 4, 1);

    repo[LSTM::Model::Type::x86_64] = std::unique_ptr<LSTM::Model>(new LSTM::Model(shape));
    repo[LSTM::Model::Type::x86_64]->LoadFromDisk("x86_64.rnn", 4, 1);
  }
}

void LstmModel::mix(Mixer& m) {
  uint8_t const bpos = shared->State.bitPosition;
  uint8_t const y = shared->State.y;
  uint8_t const c0 = shared->State.c0;

  if (y)
    bot = mid + 1;
  else
    top = mid;

  if (bpos == 0) {
    uint8_t const c1 = shared->State.c1;
    lstm.Perceive(c1);
    auto const& output = lstm.Predict(c1);
    memcpy(&probs[0], &output[0], alphabetSize * sizeof(float));
    top = alphabetSize - 1;
    bot = 0;

    if ((shared->GetOptionTrainLSTM()) && (shared->State.blockPos == 0)) {
      BlockType const blockType = static_cast<BlockType>(shared->State.blockType);
      if (blockType != pBlockType) {
        switch (blockType) {
        case BlockType::TEXT:
        case BlockType::TEXT_EOL:
          modelType = LSTM::Model::Type::English;
          break;
        case BlockType::EXE:
          modelType = LSTM::Model::Type::x86_64;
          break;
        default:
          modelType = LSTM::Model::Type::Default;
        }

        if (modelType != pModelType) {
          if ((pModelType == LSTM::Model::Type::x86_64) && (blockType == BlockType::DEFAULT)) {
            // Skip switching
          }
          else {
            lstm.SaveModel(*repo[pModelType]);
            lstm.LoadModel(*repo[modelType]);
            pModelType = modelType;
          }
        }
      }
      pBlockType = blockType;
    }
  }

  mid = (bot + top) >> 1;
  float prediction, num, denom;

  if (simd == SIMDType::SIMD_AVX2 || simd == SIMDType::SIMD_AVX512) {
#ifdef X64_SIMD_AVAILABLE
    num = sum256_ps_avx2(&probs[mid + 1], top - mid, 0.f);
    denom = sum256_ps_avx2(&probs[bot], mid + 1 - bot, num);
#endif
  }
  else {
    num = 0.0f;
    for (size_t i = mid + 1; i <= top; i++)
      num += probs[i];
    denom = num;
    for (size_t i = bot; i <= mid; i++)
      denom += probs[i];
  }

  prediction = (denom == 0.f) ? 0.5f : num / denom;

  expected = static_cast<uint8_t>(bot);
  float max_prob_val = probs[bot];
  for (size_t i = bot + 1; i <= top; i++) {
    if (probs[i] > max_prob_val) {
      max_prob_val = probs[i];
      expected = static_cast<uint8_t>(i);
    }
  }

  iCtx += y;
  iCtx = (bpos << 8) | expected;
  uint32_t ctx = iCtx();

  int const p = std::min(std::max(std::lround(prediction * 4096.0f), 1L), 4095L);
  m.promote(stretch(p) / 2);
  m.add(stretch(p));
  m.add((p - 2048) >> 2);

  int const pr1 = apm1.p(p, (c0 << 8) | (shared->State.misses & 0xFF));
  int const pr2 = apm2.p(p, (bpos << 8) | expected);
  int const pr3 = apm3.p(pr2, ctx);

  m.add(stretch(pr1) >> 1);
  m.add(stretch(pr2) >> 1);
  m.add(stretch(pr3) >> 1);
  m.set((bpos << 8) | expected, 8 * 256);
  m.set(static_cast<uint32_t>(lstm.epoch) << 3 | bpos, 100 * 8);
}
