#include "BlockModel.hpp"

/**
 * Model for metadata: blocktype, blockinfo and blocksize
**/

BlockModel::BlockModel(Shared* const sh, const uint64_t cmSize) :
  shared(sh), cm(sh, cmSize, nCM, 64)
{
  assert(isPowerOf2(cmSize));
}

void BlockModel::mix(Mixer& m) {
  const uint32_t blockTypeHistory = shared->State.blockTypeHistory; // xx xx xx xx
  const uint8_t blockStateID = shared->State.blockStateID; //0x10, 0x20..0x23, 0x30..0x33
  uint8_t bytePos = (blockStateID & 0x0f); //0,1,2,3
  INJECT_SHARED_c4
  uint32_t extrabytes = c4 & (0xffffff >> ((3 - bytePos) * 8)); //last 0-1-2-3 bytes (depending on bytePos) from c4 
  bytePos = (blockStateID == 0x10) ? 0 : (bytePos + 1); //0,1,2,3,4
  const uint8_t pos9 = bytePos + (blockStateID >= 0x30 ? 4 : 0); //0,1,2,3,4,5,6,7,8
  INJECT_SHARED_bpos
  if (bpos == 0) {
    const uint8_t _H = CM_USE_BYTE_HISTORY;
    uint64_t i = 0;
    if (blockStateID == 0x10) { //blocktype
      cm.set(_H, hash(++i));
      cm.set(_H, hash(++i, blockTypeHistory & 0xff)); //1 history
      cm.set(_H, hash(++i, blockTypeHistory & 0xffff)); //2 history
      cm.set(_H, hash(++i, blockTypeHistory & 0xff00)); //1 sparse history
      cm.set(0, hash(++i, blockTypeHistory & 0xffffff)); //3 history
    }
    else { //blockSize and blockInfo
      if (extrabytes == 0) {
        cm.set(_H, hash(++i, blockStateID));
      }
      else {
        cm.skip(_H); ++i;
      }
      cm.set(_H, hash(++i, blockStateID, extrabytes));
      cm.set(_H, hash(++i, blockStateID, extrabytes, blockTypeHistory & 0xff)); //1 history
      cm.set(_H, hash(++i, blockStateID, extrabytes, blockTypeHistory & 0xffff)); //2 history
      cm.set(0, hash(++i, blockStateID, extrabytes, blockTypeHistory & 0xffffff)); //3 history
    }
  }
  cm.mix(m);

  m.set(pos9 == 0 ? 0 : 1 + (blockTypeHistory & 0xff), 1 + (int)BlockType::Count);
  m.set(pos9 << 4 | (extrabytes == 0) << 3 | bpos, 9 * 2 * 8);
  m.set(finalize64(hash(pos9, blockTypeHistory & 0xffff), 8), 256);
}

