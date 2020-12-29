#include "SparseBitModel.hpp"

SparseBitModel::SparseBitModel(const Shared* const sh, const uint64_t size) : shared(sh), cm(sh, size, nCM, 64) {}

void SparseBitModel::mix(Mixer &m) {

  INJECT_SHARED_blockType
  const bool isText = blockType == TEXT || blockType == TEXT_EOL;

  INJECT_SHARED_bpos
  if (bpos == 0) {
    INJECT_SHARED_c4
    const uint8_t __ = CM_USE_NONE;
    uint64_t i = 0;
    if (isText) {
      cm.skipn(__, 4);
      i += 4;
    }
    else {
      cm.set(__, hash(++i, c4 & 0x0000f8f8));
      cm.set(__, hash(++i, c4 & 0x00f8f8f8));
      cm.set(__, hash(++i, c4 & 0xf8f8f8f8));
      cm.set(__, hash(++i, c4 & 0x0f0f0f0f));
    }
    cm.set(__, hash(++i, c4 & 0x00f0f0ff));
    cm.set(__, hash(++i, c4 & 0xdfdfdfe0));
    cm.set(__, hash(++i, c4 & 0xffc0ffc0));
    cm.set(__, hash(++i, c4 & 0xe0ffffe0));
    cm.set(__, hash(++i, c4 & 0x00e0e0e0));
    cm.set(__, hash(++i, c4 & 0xe0e0e0e0));
    assert(i == nCM);
  }

  cm.mix(m);
}
