#include "ChartModel.hpp"


///////////////////////////// chartModel ////////////////////////////

// The second to last byte decides which slot in the chart to place
// the last byte, along with itself. Then the contexts are hashed
// with the positions of the slots that have been updated recently or
// will be updated next.

ChartModel::ChartModel(const Shared* const sh, const uint64_t size1, const uint64_t size2) : shared(sh),
  cm(sh, size1, nCM1, 64),
  cn(sh, size2, nCM2, 64)
{}


void ChartModel::mix(Mixer& m) {
  INJECT_SHARED_bpos
  if (bpos == 0) {
    INJECT_SHARED_c4

    const uint32_t
       w = c4 & 0x0000ffff,
      w0 = c4 & 0x00ffffff,
      w1 = c4 & 0x000000ff,
      w2 = (c4 << 8) & 0x0000ff00,
      w3 = (c4 << 8) & 0x00ff0000,
      w4 = (c4 << 8) & 0xff000000,
      a[3] = { // content: 2-3 bits of 3 consecutive bytes that is
            (c4 >> 5) & 0x00070707, //bits 0-2 (top 3 bits)
            (c4 >> 2) & 0x00070707, //bits 3-5
            c4 & 0x00030303 }, // bits 6-7 (bottom 2 bits)
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
      d[3] = {  // // content: 9 selected bits from c4&0x00ffffff
           (c4 >> 15) & 0x1c0 |
            (c4 >> 10) & 0x38 |
            (c4 >> 5) & 0x07,
            (c4 >> 12) & 0x1c0 |
            (c4 >> 7) & 0x38 |
            (c4 >> 2) & 0x07,
            (c4 >> 10) & 0x30 |
            (c4 >> 4) & 0x0c |
            c4 & 3 };


    INJECT_SHARED_blockType
    const bool isText = blockType == TEXT || blockType == TEXT_EOL;

    const uint8_t __ = CM_USE_NONE;
    uint64_t q1 = 0;
    uint64_t q2 = 0;

    uint32_t cnt = isText ? 1 : 3; // for text files only the top bits are good predictors - no need to go for the others

    for (int i = 0; i < cnt; i++) {
      uint32_t j = i << 9;
      uint32_t f = j | b[i];
      uint32_t e = a[i];
      indirect1[f] = w1; // <--Update indirect
      const uint32_t g =indirect1[j|d[i]];
      chart[i << 3 | ((e >> 16) & 7)] = w0; // <--Fix chart
      chart[i << 3 | ((e >> 8) & 7)] = w << 8; // <--Update chart
      cn.set(__, hash(++q2, e)); // <--Model previous/current/next slot
      cn.set(__, hash(++q2, g)); // <--Guesses next "c4&0xFF"
      cn.set(__, hash(++q2, w2|g)); // <--Guesses next "c4&0xFFFF"
      cn.set(__, hash(++q2, w3|g)); // <--Guesses next "c4&0xFF00FF"
      cn.set(__, hash(++q2, w4|g)); // <--Guesses next "c4&0xFF0000FF"
    }
    const uint8_t buf1 = c4 & 0xff;
    const uint8_t buf2 = (c4>>8) & 0xff;
    const uint8_t buf3 = (c4 >> 16) & 0xff;
    
    indirect2[buf2] = buf1;
    uint32_t g = indirect2[buf1];
          cn.set(__, hash(++q2, g)); // <--Guesses next "c4&0xFF"
          cn.set(__, hash(++q2, w2|g)); // <--Guesses next "c4&0xFFFF"
          cn.set(__, hash(++q2, w3|g)); // <--Guesses next "c4&0xFF00FF"
          cn.set(__, hash(++q2, w4|g)); // <--Guesses next "c4&0xFF0000FF"

    indirect3[buf3 << 8 | buf2] = buf1;
    g = indirect3[buf2 << 8 | buf1];
    cn.set(__, hash(++q2, g)); // <--Guesses next "c4&0xFF"
    cn.set(__, hash(++q2, w2|g)); // <--Guesses next "c4&0xFFFF"
    cn.set(__, hash(++q2, w3|g)); // <--Guesses next "c4&0xFF00FF"
    cn.set(__, hash(++q2, w4|g)); // <--Guesses next "c4&0xFF0000FF"

    assert(q2 <= nCM2);

    cnt = isText ? 8 : 2*8+4;

    for (int i = 0; i < cnt; i++) {
      uint32_t s = i >> 3;   // selector: which bits are selected
      uint32_t e = a[s];     // content: 2-3 selected bits from 3 consecutive bytes
      uint32_t k = chart[i]; 
                                                              //   k   e
      cm.set(__, hash(++q1, k, s));                           //  111 000
      cm.set(__, hash(++q1, e, k, s));                        //  111 111
      cm.set(__, hash(++q1, e & 255, k >> 16, k & 255, s));   //  101 001
    }

    assert(q1 <= nCM1);
  }
  cn.mix(m);
  cm.mix(m);
}
