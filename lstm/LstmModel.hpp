#pragma once

#include "../Shared.hpp"
#include "../Mixer.hpp"
#include "../APM.hpp"
#include "../IndirectContext.hpp"
#include "../Array.hpp"
#include <cstdint>

class LstmModel {
public:
  static constexpr int MIXERINPUTS = 5;
  static constexpr int MIXERCONTEXTS = 8 * 256 + 8 * 100;
  static constexpr int MIXERCONTEXTSETS = 2;
  static constexpr size_t alphabetSize = 1llu << 8;

protected:
  const Shared* const shared;
  Array<float, 32> probs;
  APM apm1, apm2, apm3;
  IndirectContext<std::uint16_t> iCtx;
  size_t top, mid, bot;
  uint8_t expected;

public:
  LstmModel(const Shared* const sh);
  virtual ~LstmModel() = default;
  virtual void mix(Mixer& m) = 0;
};
