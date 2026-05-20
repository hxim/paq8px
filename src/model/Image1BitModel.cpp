#include "Image1BitModel.hpp"
#include "../Stretch.hpp"
#include "../BitCount.hpp"
#include "../Hash.hpp"

Image1BitModel::Image1BitModel(const Shared* const sh) : 
  shared(sh), 
  mapL(sh, nLSM, 16, 128), //contexts, hashBits, scale = 64, rate = 16
  stateMap(sh, N, 256, 1023, StateMapType::BitHistory)
  {}
void Image1BitModel::setParam(int widthInBytes) {
  w = widthInBytes;
}

void Image1BitModel::add(Mixer& m, uint32_t n0, uint32_t n1)
{
  int p1 = ((n1 + 1) << 12) / ((n0 + n1) + 2);
  m.add(stretch(p1) >> 1);
  m.add((p1 - 2048) >> 2);
}

void Image1BitModel::update() {
  INJECT_SHARED_y
  for (int i = 0; i < N; ++i) {
    StateTable::update(&t[cxt[i]], y, rnd);

    uint32_t c = counts[cxt[i]];
    int cnt0 = (c >> 16) & 65535;
    int cnt1 = (c & 65535);

    if (y == 0) cnt0++; else cnt1++;
    if (cnt0 + cnt1 >= 65536) { cnt0 >>= 1; cnt1 >>= 1; } //prevent overflow in add()
      
    c = cnt0 << 16 | cnt1;
    counts[cxt[i]] = c;
  }
}

void Image1BitModel::mix(Mixer& m) {
  update();

  INJECT_SHARED_y
  INJECT_SHARED_buf
  INJECT_SHARED_bpos
  
  /* rows, y = last known pixel
  *                                N,NN,NNN
  * r4:   ...                      ↓      ...
  * r3:   xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx
  * r2:   xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx
  * r1:   xxxxxxxx xxxxxxxx xxxxxxxx xxxxxxxx
  * r0: xxxxxxxx xxxxxxxx xxxxxxxy ?
  */

  r0 += r0 + y;
  r1 += r1 + ((buf(1 * w - 1) >> (7 - bpos)) & 1);
  r2 += r2 + ((buf(2 * w - 1) >> (7 - bpos)) & 1);
  r3 += r3 + ((buf(3 * w - 1) >> (7 - bpos)) & 1);
  r4 += r4 + ((buf(4 * w - 1) >> (7 - bpos)) & 1);
  r5 += r5 + ((buf(5 * w - 1) >> (7 - bpos)) & 1);
  r6 += r6 + ((buf(6 * w - 1) >> (7 - bpos)) & 1);
  r7 += r7 + ((buf(7 * w - 1) >> (7 - bpos)) & 1);
  r8 += r8 + ((buf(8 * w - 1) >> (7 - bpos)) & 1);

  int c = 0; //base for each context
  int i = 0; //context index
  
  //   x
  //  x?
  cxt[i++] = c + (y | (r1 >> 8 & 1) << 1);
  c += 1 << 2;

  //  xxx
  //  x?
  uint32_t surrounding4 = y | (r1 >> 7 & 7) << 1;
  cxt[i++] = c + surrounding4;
  c += 1 << 4;

  //    x
  //    x
  //  xx?
  cxt[i++] = c + ((r0 & 3) | (r1 >> 8 & 1) << 2 | ((r2 >> 8) & 1) << 3);
  c += 1 << 4;

  //    x
  //    x
  //    x
  // xxx?
  uint32_t mCtx6 = ((r0 & 7) | (r1 >> 8 & 1) << 3 | (r2 >> 8 & 1) << 4 | (r3 >> 8 & 1) << 5);
  cxt[i++] = c + mCtx6;
  c += 1 << 6;

  //  xx 
  //   xxx
  // xxx?
  cxt[i++] = c + (r0 & 7) | (r1 >> 7 & 7) << 3 | (r2 >> 9 & 3) << 6;
  c += 1 << 8; 
  
  //   x
  //   x
  //  xxxxx
  //  x?
  cxt[i++] = c + (y | (r1 >> 5 & 0x1f) << 1 | (r2 >> 8 & 1) << 6 | (r3 >> 8 & 1) << 7);
  c += 1 << 8;

  //   xx
  //   xx
  //   xxx
  //  x?
  cxt[i++] = c + (y | (r1 >> 6 & 7) << 1 | (r2 >> 7 & 3) << 4 | (r3 >> 7 & 3) << 6);
  c += 1 << 8;

  //  x
  //  x
  //  x
  //  x
  //  x
  //  x
  //  x
  //  x
  //  ?
  cxt[i++] = c + ((r1 >> 8 & 1) << 7 | (r2 >> 8 & 1) << 6 | (r3 >> 8 & 1) << 5 | (r4 >> 8 & 1) << 4 | (r5 >> 8 & 1) << 3 | (r6 >> 8 & 1) << 2 | (r7 >> 8 & 1) << 1 | (r8 >> 8 & 1));
  c += 1 << 8;

  // xxxxxxxx?
  cxt[i++] = c + (r0 & 0xff);
  c += 1 << 8;

  //  xx
  //  xx
  //  xx
  //  xx
  //  x?
  cxt[i++] = c + (y | (r1 >> 8 & 3) << 1 | (r2 >> 8 & 3) << 3 | (r3 >> 8 & 3) << 5 | (r4 >> 8 & 3) << 7);
  c += 1 << 9;

  //  xxxxxx
  //   xxxx?
  cxt[i++] = c + ((r0 & 0x0f) | (r1 >> 8 & 0x3f) << 4);
  c += 1 << 10;
  
  //       xx
  //   xxxxxxx
  //  xxx?
  cxt[i++] = c + ((r0 & 7) | (r1 >> 4 & 0x7f) << 3 | ((r2 >> 5) & 3) << 10);
  c += 1 << 12;
  
  // xxxxx   
  // xxxxx
  // xx?
  uint32_t surrounding12 = (r0 & 3) | (r1 >> 6 & 0x1f) << 2 | (r2 >> 6 & 0x1f) << 7;
  cxt[i++] = c + surrounding12;
  c += 1 << 12;

  // xxxxxxx   
  // xxxxxxx   
  // xxxxxxx
  // xxx?
  uint32_t surrounding24 = (r0 & 7) | (r1 >> 5 & 0x7f) << 3 | (r2 >> 5 & 0x7f) << 10 | (r3 >> 5 & 0x7f) << 17;
  
  // xxxxxxxxx
  // x.......x
  // x.......x
  // x.......x
  // x...?
  uint32_t frame16 = (r0 >> 3 & 1) | (r1 >> 12 & 1) << 1 | (r2 >> 12 & 1) << 2 | (r3 >> 12 & 1) << 3 | (r4 >> 4 & 0x1ff) << 4 | (r3 >> 4 & 1) << 13 | (r2 >> 4 & 1) << 14 | (r1 >> 4 & 1) << 15;

  //contexts: surrounding pixel counts (most useful for dithered images)
  int bitcount04 = bitCount(surrounding4); //bitcount for the surrounding 4 pixels
  int bitcount12 = bitCount(surrounding12); //... 12 pixels
  int bitcount24 = bitCount(surrounding24); //... 24 pixels
  int bitcount40 = bitCount(frame16) + bitcount24; //...40 pixels
  
  cxt[i++] = c + bitcount04; 
  c += 5;  

  cxt[i++] = c + bitcount12; 
  c += 13; 

  cxt[i++] = c + bitcount24; 
  c += 25; 

  cxt[i++] = c + bitcount40; 
  c += 41; 

  assert(i == N);
  assert(c == C);

  //     xxxxx
  //    xxxxxxx
  // xxxxxxxxxxxxxx
  // xxxxxx?
  mapL.set(hash((r0 & 0x3f), (r1 >> 1 & 0x1fff), (r2 >> 5 & 0x7f), (r3 >> 6 & 0x1f))); // 6+13+7+5=31 bits

  //         xx
  //         xx
  //   xxxxxxxx
  // xxxxxxxx?
  mapL.set(hash((r0 & 0xff), (r1 >> 7 & 0xff), (r2 >> 7 & 3)<<2 | (r3 >> 7 & 3))); // 8+8+2+2=20 bits

  //    xx
  //    xx
  //    xx
  //    xx
  //    xx
  //    xx
  //    xx
  //    xx
  //   x?
  mapL.set(hash(y | (r1 >> 7 & 3) << 1, (r2 >> 7 & 3) | (r3 >> 7 & 3) << 2, (r4 >> 7 & 3) | (r5 >> 7 & 3) << 2, (r6 >> 7 & 3) | (r7 >> 7 & 3) << 2, (r8 >> 7 & 3)));  // 8*2+1=17 bits hashed to 16 bits
  
  //  xxxxxxx
  //  xxxxxxx
  //  xxxxxxx
  //  xxx?
  mapL.set(hash(surrounding24)); // 24 bits
  
  // xxxxxxxxx
  // xxxxxxxxx
  // xxxxxxxxx
  // xxxxxxxxx
  // xxxx?
  mapL.set(hash(surrounding24, frame16)); // 40 bits

  mapL.mix(m);

  stateMap.subscribe();
  for (int i = 0; i < N; ++i) {
    const uint8_t state = t[cxt[i]];
    if (state == 0) {
      stateMap.skip(i);
      m.add(0);
      m.add(0);
      m.add(0);
    }
    else {
      m.add(stretch(stateMap.p2(i, state)) >> 1);
      uint32_t n0 = (counts[cxt[i]] >> 16) & 65535;
      uint32_t n1 = (counts[cxt[i]]) & 65535;
      add(m, n0, n1);
    }
  }

  //for dithered images
  add(m, bitcount04, 04 - bitcount04);
  add(m, bitcount12, 12 - bitcount12);
  add(m, bitcount24, 24 - bitcount24);
  add(m, bitcount40, 40 - bitcount40);

  m.set(surrounding4, 16);
  m.set(mCtx6, 64);
  m.set(bitcount40, 41);
}
