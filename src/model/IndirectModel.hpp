#pragma once

#include "../ContextMap2.hpp"
#include "../IndirectContext.hpp"
#include "../LargeIndirectContext.hpp"
#include "../Mixer.hpp"
#include "../Shared.hpp"
#include <cctype>

/**
 * The context is a byte string history that occurs within a 1 or 2 byte context.
 */
class IndirectModel {
private:
  static constexpr int nCM = 27;
  const Shared * const shared;
  ContextMap2 cm;
  Array<uint32_t> t1 {256}; // 1K
  Array<uint16_t> t2 {256*256}; // 128K
  Array<uint16_t> t3 {32*32*32}; // 64K
  Array<uint16_t> t4 {16*16*16*16}; // 128K
  Array<uint32_t> t5 {256*256}; // 256K
  LargeIndirectContext<uint32_t> iCtxLarge{ 18,8 }; // 11MB // hashBits, inputBits
  uint32_t chars4 {0};

public:
  static constexpr int MIXERINPUTS = nCM * (ContextMap2::MIXERINPUTS + ContextMap2::MIXERINPUTS_RUN_STATS); // 135
  static constexpr int MIXERCONTEXTS = 0;
  static constexpr int MIXERCONTEXTSETS = 0;
  explicit IndirectModel(const Shared* const sh, uint64_t size);
  void mix(Mixer &m);
};
