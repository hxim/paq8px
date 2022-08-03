#pragma once

#include "IPredictor.hpp"
#include "Hash.hpp"
#include "Mixer.hpp"
#include "Shared.hpp"
#include "StateMap.hpp"
#include "UpdateBroadcaster.hpp"
#include <cassert>
#include <cstdint>

class IndirectMap : IPredictor {
public:
  static constexpr int MIXERINPUTS = 2;

private:
  const Shared * const shared;
  Random rnd;
  Array<uint8_t> data;
  StateMap sm;
  const uint32_t mask;
  const uint32_t maskBits;
  const uint32_t stride;
  const uint32_t bTotal;
  uint32_t b {};
  uint32_t bCount {};
  uint32_t context {};
  uint8_t *cp;
  int scale;

public:
  IndirectMap(const Shared* const sh, int bitsOfContext, int inputBits, int scale, int limit);
  void setDirect(uint32_t ctx);
  void set(uint64_t ctx);
  void update() override;
  void setScale(int Scale);
  void mix(Mixer &m);
};
