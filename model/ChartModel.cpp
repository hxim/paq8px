#include "ChartModel.hpp"
#include "../BlockType.hpp"

///////////////////////////// chartModel ////////////////////////////

// The second to last byte ((c4>>8)&0xff) decides which slot in the chart to place
// the context which consists of selected bits from the last 3 bytes.
// Note: some of the (indirect) contexts below are independent from the chart model 

ChartModel::ChartModel(const Shared* const sh, const uint64_t size) : shared(sh),
  cm(sh, size, nCM, 64)
{}


void ChartModel::mix(Mixer& m) {
  INJECT_SHARED_bpos
  if (bpos == 0) {
    
    INJECT_SHARED_c4
    uint32_t w, w0, c1, w2, w3, w4;
    c1 = c4 & 0xff;
    w = c4 & 0x0000ffff;
    w0 = c4 & 0x00ffffff;
    w4 = w0 << 8;
    w2 = w4 & 0x0000ff00;

    INJECT_SHARED_blockType
    const bool isText = isTEXT(blockType);
    
    if (!isText) { //for binary blocks we are using sparse contexts, for text blocks continuous contexts
      w3 = w4 & 0x00ff0000;
      w4 = w4 & 0xff000000;
    }

    uint32_t a0;
    if (isText) { // in case of text blocks we use the following 3-bit category, in case of binary files, it will be simply the top 3 bits (see below)
      uint8_t g = c4; // group identifier
      if ('a' <= g && g <= 'z') g = 0;
      else if ('A' <= g && g <= 'Z') g = 1;
      else if ('0' <= g && g <= '9') g = 2;
      else if (0 == g || g == ' ') g = 3;
      else if (g <= 31) g = 4;
      else if (g <= 63) g = 5;
      else if (g <= 127) g = 6;
      else g = 7;
      charGroup = charGroup << 8 | g;
      a0 = charGroup;
    }
    else 
      a0 = (c4 >> 5) & 0x00070707; // bits 0-2 (top 3 bits)

    const uint8_t __ = CM_USE_NONE;

    uint64_t h = 0; // context index for hash

    if (!isText) {
      // sparse indirect bit contexts
      uint32_t 
        b[3] = { // content: 9 selected bits from c4&0xffffff00
            (c4 >> 23) & 0x01c0 |
            (c4 >> 18) & 0x38 |
            (c4 >> 13) & 0x07,
            (c4 >> 20) & 0x1c0 |
            (c4 >> 15) & 0x38 |
            (c4 >> 10) & 0x07,
            (c4 >> 18) & 0x30 |
            (c4 >> 12) & 0x0c |
            (c4 >> 8) & 3 },
        d[3] = {  // content: 9 selected bits from c4&0x00ffffff
            (c4 >> 15) & 0x1c0 |
            (c4 >> 10) & 0x38 |
            (c4 >> 5) & 0x07,
            (c4 >> 12) & 0x1c0 |
            (c4 >> 7) & 0x38 |
            (c4 >> 2) & 0x07,
            (c4 >> 10) & 0x30 |
            (c4 >> 4) & 0x0c |
            c4 & 3 };

      for (int i = 0; i < 3; i++) {
        uint32_t f = i << 9 | b[i];
        const uint8_t g = indirect1[i << 9 | d[i]]; // context from indirect
        indirect1[f] = c1; // update indirect
        cm.set(__, hash(++h, g)); 
        cm.set(__, hash(++h, w2 | g));
        cm.set(__, hash(++h, w3 | g));
        cm.set(__, hash(++h, w4 | g));
      }
    }

    const uint8_t buf2 = (c4 >> 8) & 0xff;
    const uint8_t buf3 = (c4 >> 16) & 0xff;

    //sparse indirect 1-byte contexts
    uint8_t g = indirect2[c1]; // context from indirect
    indirect2[buf2] = c1; // update indirect
    cm.set(__, hash(++h, w4 | g));

    //sparse indirect 2-byte contexts
    g = indirect3[buf2 << 8 | c1]; // context from indirect
    indirect3[buf3 << 8 | buf2] = c1; // update indirect
    cm.set(__, hash(++h, w4 | g));
    if (!isText) {
      cm.set(__, hash(++h, w3 | g));
    }

    uint32_t
      a[3] = { // content: 3 selected bits of 3 consecutive bytes that is
            a0 , //charGroup (text) or top 3 bits (binary)
            (c4 >> 2) & 0x00070707, //bits 3-5 (3 bits)
            c4 & 0x00070707 }; // bits 5-7 (bottom 3 bits)

    uint32_t cnt = isText ? 1 : 3; // for text files we use the charGroup (1 context), for binary files we use the bit contexts (3 contexts)
    
    //update chart
    for (uint32_t i = 0; i < cnt; i++) {
      uint32_t e = a[i];
      chart[i << 3 | ((e >> 8) & 7)] = w0;
    }

    cnt *= 8; // the chart has 8+8+8 slots: for text blocks we use only the first 8, for binary blocks we use all the 24 

    //chart model
    for (uint32_t i = 0; i < cnt; i++) {
      uint32_t s = i >> 3;   // selector: which bits are selected: s: 0 for text; 0..2 for binary
      uint32_t e = a[s];     // content: 3 selected bits from 3 consecutive bytes
      uint32_t k = chart[i]; // i: text: 0..7; binary: 0..23
                                                              //   k   e
      cm.set(__, hash(++h, k));                               //  111 000
      cm.set(__, hash(++h, e, k & 0xffff));                   //  011 111
      cm.set(__, hash(++h, (e & 7) << 8 | (k & 0xff00ff)));   //  101 001
    }

    assert(h == isText ? nCM_TEXT : nCM);
  }
  
  cm.mix(m);
}
