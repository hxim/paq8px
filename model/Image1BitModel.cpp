#include "Image1BitModel.hpp"
#include "../Stretch.hpp"

Image1BitModel::Image1BitModel(const Shared* const sh) : shared(sh), sm {sh, s, 256, 1023, StateMap::BitHistory} {}

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
  r1 += r1 + ((buf(w - 1) >> (7 - bpos)) & 1U);
  r2 += r2 + ((buf(w + w - 1) >> (7 - bpos)) & 1U);
  r3 += r3 + ((buf(w + w + w - 1) >> (7 - bpos)) & 1U);
  cxt[0] = (r0 & 0x7U) | (r1 >> 4U & 0x38U) | (r2 >> 3U & 0xc0U);
  cxt[1] = 0x100 + ((r0 & 1U) | (r1 >> 4U & 0x3eU) | (r2 >> 2U & 0x40U) | (r3 >> 1U & 0x80U));
  cxt[2] = 0x200 + ((r0 & 1U) | (r1 >> 4U & 0x1dU) | (r2 >> 1U & 0x60U) | (r3 & 0xC0U));
  cxt[3] = 0x300 + (y | ((r0 << 1U) & 4U) | ((r1 >> 1U) & 0xF0U) | ((r2 >> 3U) & 0xAU));
  cxt[4] = 0x400 + ((r0 >> 4U & 0x2ACU) | (r1 & 0xA4U) | (r2 & 0x349U) | static_cast<uint32_t>((r3 & 0x14DU) == 0u));
  cxt[5] = 0x800 + (y | ((r1 >> 4U) & 0xEU) | ((r2 >> 1U) & 0x70U) | ((r3 << 2U) & 0x380U));
  cxt[6] = 0xC00 + (((r1 & 0x30U) ^ (r3 & 0x0c0cU)) | (r0 & 3U));
  cxt[7] = 0x1000 + (static_cast<uint32_t>((r0 & 0x444U) == 0u) | (r1 & 0xC0CU) | (r2 & 0xAE3U) | (r3 & 0x51CU));
  cxt[8] = 0x2000 + ((r0 & 7U) | ((r1 >> 1U) & 0x3F8U) | ((r2 << 5U) & 0xC00U));
  cxt[9] = 0x3000 + ((r0 & 0x3fU) ^ (r1 & 0x3ffeU) ^ (r2 << 2U & 0x7f00U) ^ (r3 << 5U & 0xf800U));
  cxt[10] = 0x13000 + ((r0 & 0x3eU) ^ (r1 & 0x0c0cU) ^ (r2 & 0xc800U));

  // predict
  sm.subscribe();
  for( int i = 0; i < s; ++i ) {
    const uint8_t s = t[cxt[i]];
    m.add(stretch(sm.p2(i, s)));
  }

  m.set(y << 1 | ((buf(w - 1) >> (7 - bpos)) & 1U), 4);
}
