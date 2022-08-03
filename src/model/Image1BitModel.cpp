#include "Image1BitModel.hpp"
#include "../Stretch.hpp"

Image1BitModel::Image1BitModel(const Shared* const sh) : shared(sh), sm {sh, s, 256, 1023, StateMapType::BitHistory} {}

void Image1BitModel::setParam(int widthInBytes) {
  w = widthInBytes;
}

void Image1BitModel::mix(Mixer &m) {
  // update the model
  INJECT_SHARED_y
  for( int i = 0; i < s; ++i ) {
    StateTable::update(&t[cxt[i]], y, rnd);
  }

  INJECT_SHARED_buf
  INJECT_SHARED_bpos
  // update the contexts (pixels surrounding the predicted one)
  r0 += r0 + y;
  r1 += r1 + ((buf(w - 1) >> (7 - bpos)) & 1);
  r2 += r2 + ((buf(w + w - 1) >> (7 - bpos)) & 1);
  r3 += r3 + ((buf(w + w + w - 1) >> (7 - bpos)) & 1);
  cxt[0] = (r0 & 0x7) | (r1 >> 4 & 0x38) | (r2 >> 3 & 0xc0);
  cxt[1] = 0x100 + ((r0 & 1) | (r1 >> 4 & 0x3e) | (r2 >> 2 & 0x40) | (r3 >> 1 & 0x80));
  cxt[2] = 0x200 + ((r0 & 1) | (r1 >> 4 & 0x1d) | (r2 >> 1 & 0x60) | (r3 & 0xC0));
  cxt[3] = 0x300 + (y | ((r0 << 1) & 4) | ((r1 >> 1) & 0xF0) | ((r2 >> 3) & 0x0A));
  cxt[4] = 0x400 + ((r0 >> 4 & 0x2AC) | (r1 & 0xA4) | (r2 & 0x349) | static_cast<uint32_t>((r3 & 0x14D) == 0));
  cxt[5] = 0x800 + (y | ((r1 >> 4) & 0x0E) | ((r2 >> 1) & 0x70) | ((r3 << 2) & 0x380));
  cxt[6] = 0xC00 + (((r1 & 0x30) ^ (r3 & 0x0c0c)) | (r0 & 3));
  cxt[7] = 0x1000 + (static_cast<uint32_t>((r0 & 0x444) == 0) | (r1 & 0xC0C) | (r2 & 0xAE3) | (r3 & 0x51C));
  cxt[8] = 0x2000 + ((r0 & 0x07) | ((r1 >> 1) & 0x3F8) | ((r2 << 5) & 0xC00));
  cxt[9] = 0x3000 + ((r0 & 0x3f) ^ (r1 & 0x3ffe) ^ (r2 << 2 & 0x7f00) ^ (r3 << 5 & 0xf800));
  cxt[10] = 0x13000 + ((r0 & 0x3e) ^ (r1 & 0x0c0c) ^ (r2 & 0xc800));

  // predict
  sm.subscribe();
  for( int i = 0; i < s; ++i ) {
    const uint8_t s = t[cxt[i]];
    m.add(stretch(sm.p2(i, s)));
  }

  m.set(y << 1 | ((buf(w - 1) >> (7 - bpos)) & 1), 4);
}
