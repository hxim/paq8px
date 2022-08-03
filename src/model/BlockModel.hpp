#pragma once

#include "../ContextMap2.hpp"

class BlockModel {
private:
  static constexpr int nCM = 5;
  Shared* const shared;
  ContextMap2 cm;
public:
  static constexpr int MIXERINPUTS =
    nCM * (ContextMap2::MIXERINPUTS + ContextMap2::MIXERINPUTS_BYTE_HISTORY); //30
  static constexpr int MIXERCONTEXTS = ((int)BlockType::Count + 1) + (9 * 2 * 8) + (256); //428
  static constexpr int MIXERCONTEXTSETS = 3;
  BlockModel(Shared* const sh, const uint64_t cmSize);

  void mix(Mixer& m);

};
