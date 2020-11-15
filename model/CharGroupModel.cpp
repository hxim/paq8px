#include "CharGroupModel.hpp"

CharGroupModel::CharGroupModel(const Shared* const sh, const uint64_t size) : shared(sh), cm(sh, size, nCM, 64, CM_USE_RUN_STATS | CM_USE_BYTE_HISTORY) {}

void CharGroupModel::mix(Mixer &m) {
  INJECT_SHARED_bpos
  if( bpos == 0 ) {
    INJECT_SHARED_c4
    uint8_t g = c4; // group identifier
    if( '0' <= g && g <= '9' ) {
      g = '0'; //all digits are in one group
    } else if( 'A' <= g && g <= 'Z' ) {
      g = 'A'; //all uppercase letters are in one group
    } else if( 'a' <= g && g <= 'z' ) {
      g = 'a'; //all lowercase letters are in one group
    } else if( g >= 128 ) {
      g = 128;
    }

    const bool toBeCollapsed = (g == '0' || g == 'A' || g == 'a') && (g == (g1 & 0xff));
    if( !toBeCollapsed ) {
      g3 <<= 8;
      g3 |= g2 >> (32 - 8);
      g2 <<= 8;
      g2 |= g1 >> (32 - 8);
      g1 <<= 8;
      g1 |= g;
    }

    uint64_t i = static_cast<uint64_t>(toBeCollapsed) * nCM;
    cm.set(hash(++i, g3, g2,            g1)); // last 12 groups
    cm.set(hash(++i,     g2,            g1)); // last 8 groups
    cm.set(hash(++i,     g2 & 0x00ffff, g1)); // last 6 groups
    cm.set(hash(++i,                    g1)); // last 4 groups
    cm.set(hash(++i,                    g1 & 0x0000ffff)); // last 2 groups
    cm.set(hash(++i,     g2 & 0xffffff, g1, c4 & 0x0000ffff)); // last 7 groups + last 2 chars
    cm.set(hash(++i,     g2 & 0x0000ff, g1, c4 & 0x00ffffff)); // last 5 groups + last 3 chars
  }
  cm.mix(m);
}
