#pragma once

#include "../ContextMap2.hpp"
#include "../Mixer.hpp"
#include "../Shared.hpp"
#include <cassert>
#include <cstdint>

/**
 * Model order 1-2-3 contexts with gaps.
 */
class SparseModel {
private:
  static constexpr int nCM = 28 + 3;
  static constexpr int nCM_TEXT = 3;
  const Shared * const shared;
  ContextMap2 cm;
  uint32_t ctx = 0;
public:
  static constexpr int MIXERINPUTS = nCM * (ContextMap2::MIXERINPUTS + ContextMap2::MIXERINPUTS_RUN_STATS + ContextMap2::MIXERINPUTS_BYTE_HISTORY); // 217
  static constexpr int MIXERINPUTS_TEXT = nCM_TEXT * (ContextMap2::MIXERINPUTS + ContextMap2::MIXERINPUTS_RUN_STATS + ContextMap2::MIXERINPUTS_BYTE_HISTORY); // 21
  static constexpr int MIXERCONTEXTS = 4 * 256;
  static constexpr int MIXERCONTEXTS_TEXT = 0;
  static constexpr int MIXERCONTEXTSETS = 1;
  static constexpr int MIXERCONTEXTSETS_TEXT = 0;
  explicit SparseModel(const Shared* const sh, uint64_t size);
  void mix(Mixer &m);
};
