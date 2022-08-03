#include "SparseModel.hpp"

SparseModel::SparseModel(const Shared* const sh, const uint64_t size) : shared(sh), cm(sh, size, nCM, 64) {}

void SparseModel::mix(Mixer &m) {
  INJECT_SHARED_blockType
  const bool isText = isTEXT(blockType);

  INJECT_SHARED_bpos
  if( bpos == 0 ) {
    INJECT_SHARED_c4
    uint64_t i = 0;
    const uint8_t RH = CM_USE_RUN_STATS | CM_USE_BYTE_HISTORY;

    //textfiles benefit from these contexts - these strong(er) models for text files
    cm.set(RH, hash(++i, c4 & 0x00ff00ff)); //buf(1) + buf(3)
    cm.set(RH, hash(++i, c4 & 0xff0000ff)); //buf(1) + buf(4)
    cm.set(RH, hash(++i, c4 & 0x0000ff00)); //buf(2)

    //The following sparse contexts are usually just generating noise for text content - so skip if not text
    if (!isText) {
      INJECT_SHARED_buf

      //context for 4-byte structures and 
      //positions of zeroes in the last 16 bytes
      ctx <<= 1;
      ctx |= (c4 & 0xff) == buf(5); //column matches in a 4-byte fixed structure
      ctx <<= 1;
      ctx |= (c4 & 0xff) == 0; //zeroes

      cm.set(RH, hash(++i, ctx)); // calgary/obj2, calgary/pic, cantenbury/kennedy.xls, cantenbury/sum, etc.

      //combination of two bytes
      cm.set(RH, hash(++i, buf(1) | buf(6) << 8));
      cm.set(RH, hash(++i, buf(3) | buf(6) << 8));
      
      //combination of three bytes
      cm.set(RH, hash(++i, buf(1) | buf(3) << 8 | buf(5) << 16));
      cm.set(RH, hash(++i, buf(2) | buf(4) << 8 | buf(6) << 16));

      //special model for some 4-byte fixed length structures
      cm.set(RH, hash(++i, c4 & 0xffe00000 | (ctx&0xff))); // calgary/geo, calgary/obj2, etc.

      //single bytes
      cm.set(RH, hash(++i, (c4 & 0x00ff0000) >> 16)); //buf(3)
      cm.set(RH, hash(++i, c4 >> 24)); //buf(4)
      cm.set(RH, hash(++i, buf(5)));
      cm.set(RH, hash(++i, buf(6)));
      cm.set(RH, hash(++i, buf(7)));
      cm.set(RH, hash(++i, buf(8)));

      //two consecutive bytes
      cm.set(RH, hash(++i, buf(3) << 8 | buf(2)));
      cm.set(RH, hash(++i, buf(4) << 8 | buf(3)));
      cm.set(RH, hash(++i, buf(5) << 8 | buf(4)));
      cm.set(RH, hash(++i, buf(6) << 8 | buf(5)));
      cm.set(RH, hash(++i, buf(7) << 8 | buf(6)));
      
      //two bytes with a gap
      cm.set(RH, hash(++i, buf(4) << 8 | buf(2)));
      cm.set(RH, hash(++i, buf(5) << 8 | buf(3)));
      cm.set(RH, hash(++i, buf(6) << 8 | buf(4)));
      cm.set(RH, hash(++i, buf(7) << 8 | buf(5)));
      cm.set(RH, hash(++i, buf(8) << 8 | buf(6)));

      //combination of two bytes
      
      cm.set(RH, hash(++i, buf(1) | buf(5) << 8));
      cm.set(RH, hash(++i, buf(4) | buf(8) << 8));

      //two consecutive bytes
      cm.set(RH, hash(++i, buf(8) << 8 | buf(7))); // in case of text files only useful for silesia/nci
      cm.set(RH, hash(++i, buf(9) << 8 | buf(8))); // silesia/nci

      //two bytes with a gap
      cm.set(RH, hash(++i, buf(9) << 8 | buf(7)));  // silesia/nci
      cm.set(RH, hash(++i, buf(10) << 8 | buf(8))); // silesia/nci
    }
    assert((int) i == (isText ? nCM_TEXT : nCM));
  }
  cm.mix(m);

  if (!isText) {
    //support fixed structures of 4 bytes
    INJECT_SHARED_blockPos
    m.set((blockPos & 3)<<8 | (ctx&0xff), 4 * 256);
  }


}
