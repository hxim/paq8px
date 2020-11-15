#include "SparseBitModel.hpp"

SparseBitModel::SparseBitModel(const Shared* const sh, const uint64_t size) : shared(sh), cm(sh, size, nCM, 64, CM_USE_NONE) {}

void SparseBitModel::mix(Mixer &m) {

  INJECT_SHARED_blockType
  const bool isText = blockType == TEXT || blockType == TEXT_EOL;

  INJECT_SHARED_bpos
  if (bpos == 0) {
    INJECT_SHARED_c4
    uint64_t i = 0;
    if (isText) {
      cm.skipn(4);
      i += 4;
    }
    else {
      cm.set(hash(++i, c4 & 0x00f8f8f8));
      cm.set(hash(++i, c4 & 0xf8f8f8f8));
      cm.set(hash(++i, c4 & 0xc0ccc0cc));
      cm.set(hash(++i, c4 & 0x00c0ccc0));
    }
    //for (some) text files they are useful (but week) contexts
    cm.set(hash(++i, c4 & 0x00f0f0ff));
    cm.set(hash(++i, c4 & 0x00e0e0e0));
    cm.set(hash(++i, c4 & 0xe0e0e0e0));
    cm.set(hash(++i, c4 & 0xe00000e0));
  }

  cm.mix(m);
}
