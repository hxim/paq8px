#include "NormalModel.hpp"

NormalModel::NormalModel(Shared* const sh, const uint64_t cmSize) : 
  shared(sh), cm(sh, cmSize, nCM, 64),
  smOrder0Slow(sh, 1, 255, 1023, StateMap::Generic), 
  smOrder1Slow(sh, 1, 255 * 256, 1023, StateMap::Generic),
  smOrder1Fast(sh, 1, 255 * 256, 64, StateMap::Generic) // 64->16 is also ok
{
  assert(isPowerOf2(cmSize));
}

void NormalModel::reset() {
  memset(&cxt[0], 0, sizeof(cxt));
}

void NormalModel::updateHashes() {
  INJECT_SHARED_c1
  INJECT_SHARED_blockType
  const uint64_t blocktype_c1 = blockType << 8 | c1;
  /* todo: let blocktype represent simply the blocktype without any transformation used:
      blockType == AUDIO_LE = AUDIO
      blockType == TEXT_EOL = TEXT
  */
  for( uint64_t i = 14; i > 0; --i ) {
    cxt[i] = (cxt[i - 1] + blocktype_c1 + i) * PHI64;
  }
}

void NormalModel::mix(Mixer &m) {
  INJECT_SHARED_bpos
  if( bpos == 0 ) {
    updateHashes();
    const uint8_t RH = CM_USE_RUN_STATS | CM_USE_BYTE_HISTORY;
    for(uint64_t i = 1; i <= 7; ++i ) {
      cm.set(RH, cxt[i]);
    }
    cm.set(RH, cxt[9]);
    cm.set(RH, cxt[14]);
  }
  cm.mix(m);
  INJECT_SHARED_c0
  INJECT_SHARED_c1
  m.add((stretch(smOrder0Slow.p1(c0 - 1))) >> 2U); //order 0
  m.add((stretch(smOrder1Fast.p1((c0 - 1) << 8U | c1))) >> 2U); //order 1
  m.add((stretch(smOrder1Slow.p1((c0 - 1) << 8U | c1))) >> 2U); //order 1

  const int order = max(0, cm.order - (nCM - 7)); //0-7
  assert(0 <= order && order <= 7);
  m.set(order << 3U | bpos, 64);
  shared->State.order = order;
}

void NormalModel::mixPost(Mixer &m) {
  INJECT_SHARED_c4
  uint32_t c2 = (c4 >> 8U) & 0xffU;
  uint32_t c3 = (c4 >> 16U) & 0xffU;
  uint32_t c;

  INJECT_SHARED_c0
  INJECT_SHARED_c1
  INJECT_SHARED_bpos
  INJECT_SHARED_blockType
  m.set((c1 | static_cast<int>(bpos > 5) << 8U | static_cast<int>(((c0 & ((1U << bpos) - 1)) == 0) || (c0 == ((2 << bpos) - 1))) << 9U), 1024);
  m.set(c0, 256);
  uint32_t bt = blockType == DEFAULT ? 0 : blockType == TEXT || blockType == TEXT_EOL ? 1 : blockType == EXE || blockType == DEC_ALPHA ? 2 : 3;
  m.set(shared->State.order | ((c1 >> 6U) & 3U) << 3U | static_cast<int>(bpos == 0) << 5U | static_cast<int>(c1 == c2) << 6U | bt << 7U, 512);
  m.set(c2, 256);
  m.set(c3, 256);
  if( bpos != 0 ) {
    c = c0 << (8 - bpos);
    if( bpos == 1 ) {
      c |= c3 >> 1U;
    }
    c = min(bpos, 5) << 8U | c1 >> 5U | (c2 >> 5U) << 3U | (c & 192U);
  } else {
    c = c3 >> 7U | (c4 >> 31U) << 1U | (c2 >> 6U) << 2U | (c1 & 240U);
  }
  m.set(c, 1536);
}
