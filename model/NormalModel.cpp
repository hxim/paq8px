#include "NormalModel.hpp"

NormalModel::NormalModel(Shared* const sh, const uint64_t cmSize) :
  shared(sh), cm(sh, cmSize, nCM, 64),
  smOrder0(sh, 255, 1023, 16),
  smOrder1(sh, 255 * 256, 1023, 32)
{
  assert(isPowerOf2(cmSize));
}

void NormalModel::reset() {
  memset(&shared->State.NormalModel.cxt[0], 0, sizeof(shared->State.NormalModel.cxt));
}

void NormalModel::updateHashes() {
  INJECT_SHARED_c1
  INJECT_SHARED_blockType
    BlockType normalizedBlockType = blockType;
  /* todo: let blocktype represent simply the blocktype without any transformation used:
      blockType == BlockType::AUDIO_LE = BlockType::AUDIO
      blockType == BlockType::TEXT_EOL = BlockType::TEXT
  */
  if (isTEXT(blockType) || (blockType == BlockType::JPEG && shared->State.JPEG.state == 0))
    normalizedBlockType = BlockType::DEFAULT;
  else if (blockType == BlockType::AUDIO_LE)
    normalizedBlockType = BlockType::AUDIO;
  const uint64_t blocktype_c1 = normalizedBlockType << 8 | c1;
  uint64_t* cxt = shared->State.NormalModel.cxt;
  for( uint64_t i = 14; i > 0; --i ) {
    cxt[i] = (cxt[i - 1] + blocktype_c1 + i) * PHI64;
  }
}

void NormalModel::mix(Mixer &m) {
  INJECT_SHARED_bpos
  if( bpos == 0 ) {
    updateHashes();
    uint64_t* cxtHashes = shared->State.NormalModel.cxt;
    const uint8_t RH = CM_USE_RUN_STATS | CM_USE_BYTE_HISTORY;
    for(uint64_t i = 1; i <= 6; ++i ) {
      cm.set(RH, cxtHashes[i]);
    }
    cm.set(RH, cxtHashes[8]); 
    cm.set(RH, cxtHashes[11]);
    cm.set(RH, cxtHashes[14]);
  }
  cm.mix(m);

  INJECT_SHARED_c0
  INJECT_SHARED_c1

  int pr_slow, pr_fast, st;
  int ct1 = (c0 - 1);
  smOrder0.p(ct1, pr_slow, pr_fast);
  m.add((pr_slow - 2048) >> 3); st = stretch(pr_slow); m.add(st >> 2);
  m.add((pr_fast - 2048) >> 3); st = stretch(pr_fast); m.add(st >> 2);
  int ct2 = ct1 << 8 | c1;
  smOrder1.p(ct2, pr_slow, pr_fast);
  m.add((pr_slow - 2048) >> 3); st = stretch(pr_slow); m.add(st >> 2);
  m.add((pr_fast - 2048) >> 3); st = stretch(pr_fast); m.add(st >> 2);

  const int order = max(0, cm.order - (nCM - 7)); //0-7
  assert(0 <= order && order <= 7);
  m.set(order << 3U | bpos, 64);
  shared->State.NormalModel.order = order;
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
  uint32_t bt = blockType == BlockType::DEFAULT ? 0 : isTEXT(blockType) ? 1 : blockType == BlockType::EXE || blockType == BlockType::DEC_ALPHA ? 2 : 3;
  m.set(shared->State.NormalModel.order | ((c1 >> 6U) & 3U) << 3U | static_cast<int>(bpos == 0) << 5U | static_cast<int>(c1 == c2) << 6U | bt << 7U, 512);
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
