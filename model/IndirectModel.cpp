#include "IndirectModel.hpp"

IndirectModel::IndirectModel(const Shared* const sh, const uint64_t size) : shared(sh), cm(sh, size, nCM, 64) {}

void IndirectModel::mix(Mixer &m) {
  INJECT_SHARED_bpos
  if( bpos == 0 ) {
    INJECT_SHARED_buf
    INJECT_SHARED_c4

    //
    // small contexts (with direct table lookup)
    //

    //current contexts
    uint32_t c1 = c4 & 0xff; //8 bits
    uint32_t d2 = c4 & 0xffff; //16 bits
    uint32_t d3 = (buf(1) >> 3) | (buf(2) >> 3) << 5 | (buf(3) >> 3) << 10; //15 bits
    uint32_t d4 = (buf(1) >> 4) | (buf(2) >> 4) << 4 | (buf(3) >> 4) << 8 | (buf(4) >> 4) << 12; //16 bits

    //... and their byte histories
    const uint32_t h1 = c1 | t1[c1] << 8;
    const uint32_t h2 = d2 | t2[d2] << 16;
    const uint32_t h3 = d3 | t3[d3] << 16;
    const uint32_t h4 = d4 | t4[d4] << 16;

    //update their context history tables
    uint32_t& r1 = t1[d2 >> 8];
    r1 = r1 << 8 | c1;
    uint16_t& r2 = t2[(c4 >> 8) & 0xffff];
    r2 = r2 << 8 | c1;
    uint16_t& r3 = t3[(buf(2) >> 3) | (buf(3) >> 3) << 5 | (buf(4) >> 3) << 10];
    r3 = r3 << 8 | c1;
    uint16_t& r4 = t4[(buf(2) >> 4) | (buf(3) >> 4) << 4 | (buf(4) >> 4) << 8 | (buf(5) >> 4) << 12];
    r4 = r4 << 8 | c1;

    const uint8_t R_ = CM_USE_RUN_STATS; 
    uint64_t i = 0;

    
    cm.set(R_,hash(++i, h1));              // last 1 byte + 3 history bytes {in a 1-byte context}
    cm.set(R_, hash(++i, h1 & 0xffffff00)); // 3 history bytes {in a 1-byte context}
    cm.set(R_, hash(++i, t1[c1]));          // 4 history bytes {in a 1-byte context}
    cm.set(R_, hash(++i, h2));              // last 2 bytes + 2 history bytes {in a 2-byte context}
    cm.set(R_, hash(++i, h3));              // last 3 top 5 bits + 2 history bytes {in a "last 3 top 5 bits" context}
    cm.set(R_, hash(++i, h4));              // last 4 top 4 bits + 2 history bytes {in a "last 4 top 4 bits" context}
    
    cm.set(R_, hash(++i, h1 & 0x0000ff00)); // 1 history byte {in a 1-byte context}
    cm.set(R_, hash(++i, h2 & 0x00ff0000)); // 1 history byte {in a 2-byte context}
    cm.set(R_, hash(++i, h3 & 0x00ff0000)); // last 3 top 5 bits + 2 history bytes {in a "last 3 top 5 bits" context}
    cm.set(R_, hash(++i, h4 & 0x00ff0000)); // last 4 top 4 bits + 2 history bytes {in a "last 4 top 4 bits" context}
   
    cm.set(R_, hash(++i, h1 & 0x0000ffff)); // last byte + 1 history byte {in a 1-byte context}
    cm.set(R_, hash(++i, h2 & 0x00ffffff)); // last 2 bytes + 1 history byte {in a 2-byte context}
    cm.set(R_, hash(++i, h3 & 0x00ffffff)); // last 3 top 5 bits + 1 history byte {in a "last 3 top 5 bits" context}
    cm.set(R_, hash(++i, h4 & 0x00ffffff)); // last 4 top 4 bits + 1 history byte {in a "last 4 top 4 bits" context}

    // context with 2 characters converted to lowercase (context table: t5, byte history: h5)

    const uint8_t c = tolower(c1);
    chars4 = chars4 << 8 | c;

    const uint32_t h5 = t5[chars4 & 0xffff];
    uint32_t& r5 = t5[(chars4 >> 8) & 0xffff];
    r5 = r5 << 8 | c;

    cm.set(R_, hash(++i, (h5 & 0xff) | c << 8)); //last char + 1 history char
    cm.set(R_, hash(++i, h5 & 0xffff)); //2 history chars
    cm.set(R_, hash(++i, h5 & 0xffffff)); //3 history chars
    cm.set(R_, hash(++i, h5)); //4 history chars

    //
    // large contexts (with hashtable)
    //

    uint32_t context, h6;

    iCtxLarge.set(hash(i, c4 >> 8), c1);
    context = c4 & 0xffffff; //24 bits
    h6 = iCtxLarge.get(hash(i, context));
    cm.set(R_, hash(++i, h6 & 0xff, context & 0xff));
    cm.set(R_, hash(++i, h6 & 0xff, context & 0xffff));
    cm.set(R_, hash(++i, h6, context & 0xff)); //without context is also OK (not for news and obj2 however)

    iCtxLarge.set(hash(i, chars4 >> 8), c1);
    context = chars4 & 0xffffff; //24 bits lowercase
    h6 = iCtxLarge.get(hash(i, context));
    cm.set(R_, hash(++i, h6 & 0xff, context));
    cm.set(R_, hash(++i, h6 & 0xffff, context));
    cm.set(R_, hash(++i, h6));

    iCtxLarge.set(hash(i, buf(5) << 24 | c4 >> 8), c1);
    context = c4; //32 bits
    h6 = iCtxLarge.get(hash(i, context));
    cm.set(R_, hash(++i, h6 & 0xff, context &0xffff));
    cm.set(R_, hash(++i, h6 & 0xffff, context & 0xff));
    cm.set(R_, hash(++i, h6));

    assert(i == nCM);
  }
  cm.mix(m);
}
