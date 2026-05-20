#pragma once

#include "../ContextMap2.hpp"
#include "../Shared.hpp"
#include <cstdint>

/**
 * modeling ascii character sequences
 * Although there have been similar approaches in wordModel (f4, mask), and different "masks" in @ref TextModel,
 * so there is some overlap, but there is novelty in charGroupModel.
 * Words (letter-sequences) and numbers (digit-sequences) are collapsed into one character.
 * So
 *     "A (fairly simple) sentence." becomes "A (a a) a.",
 * and
 *     "A picture is worth a 1000 words!" becomes "A a a a a 1 a!"
 * And the model uses these simplified "strings" as contexts for a @ref ContextMap2.
 * a "mask" is a simplification of the last some bytes of the input sequence emphasizing some feature that looks promising to use it as a context.
 */
class CharGroupModel {
private:
  static constexpr int nCM = 7;
  const Shared * const shared;
  ContextMap2 cm;
  uint32_t g1 = 0; /**< collapsed group identifiers of the last 12 (4+4+4) characters; the most recent is @ref gAscii1 **/
  uint32_t g2 = 0;
  uint32_t g3 = 0;
public:
  static constexpr int MIXERINPUTS =
          nCM * (ContextMap2::MIXERINPUTS + ContextMap2::MIXERINPUTS_RUN_STATS + ContextMap2::MIXERINPUTS_BYTE_HISTORY); // 49
  static constexpr int MIXERCONTEXTS = 0;
  static constexpr int MIXERCONTEXTSETS = 0;
  explicit CharGroupModel(const Shared* const sh, uint64_t size);
  void mix(Mixer &m);
};
