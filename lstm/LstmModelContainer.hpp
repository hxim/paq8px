#pragma once

#include "../Shared.hpp"
#include "../Mixer.hpp"
#include "../APM.hpp"
#include "../IndirectContext.hpp"
#include "../UpdateBroadcaster.hpp"
#include "Lstm.hpp"
#include "ByteModelToBitModel.hpp"
#include <cstdint>

/**
 * Encapsulating the LSTM model into a paq-compatible container
 */
class LstmModelContainer : IPredictor {
private:
  Shared* const shared;

  SIMDType simd;
  LstmShape shape;
  Lstm lstm;

  float* probs;
  ByteModelToBitModel byteModelToBitModel;
  APM apm1, apm2, apm3;
  IndirectContext<std::uint16_t> iCtx;
  uint8_t expectedByte;

public:
  static constexpr int MIXERINPUTS = 5;
  static constexpr int MIXERCONTEXTS = 8 * 256 + 8 * 100;
  static constexpr int MIXERCONTEXTSETS = 2;
  static constexpr size_t alphabetSize = 1llu << 8;

  explicit LstmModelContainer(
    Shared* const sh,
    size_t hidden_size,
    size_t num_layers,
    size_t horizon);

  void next();
  int getp();
  void mix(Mixer& m);
  void update() override;

  void SaveModelParameters(FILE* file);
  void LoadModelParameters(FILE* file);
};
