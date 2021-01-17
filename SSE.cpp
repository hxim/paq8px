#include "SSE.hpp"

SSE::SSE(Shared* const sh) : shared(sh),
  Text {
    { /*APM:*/  {sh,0x10000,24}, {sh,0x10000,24}, {sh,0x10000,24}, {sh,0x10000,24}}, /* APM: contexts, steps */
    { /*APM1:*/ {sh,0x10000,7}, {sh,0x10000,6}, {sh,0x10000,6}} /* APM1: contexts, rate */
  },
  Image {
    { /*APM:*/ {{sh,0x1000,24}, {sh,0x10000,24}, {sh,0x10000,24}, {sh,0x10000,24}}, /*APM1:*/ {{sh,0x10000,7}, {sh,0x10000,7}} }, // color
    { /*APM:*/ {{sh,0x1000,24}, {sh,0x10000,24}, {sh,0x10000,24}, {sh,0x10000,24}}, /*APM1:*/ {{sh,0x10000,5}, {sh,0x10000,6}} }, // palette
    { /*APM:*/ {{sh,0x1000,24}, {sh,0x10000,24}, {sh,0x10000,24}} } //gray
  },
  Jpeg{
    { /*APM:*/ {sh,0x2000,24} }
  },
  DEC{
    { /*APM:*/ {sh,25*26,20} }
  },
  x86_64{
    { /*APM:*/ {sh,0x800,20}, {sh,0x10000,16}, {sh,0x10000,16} }
  },
  Generic {
    /*APM1:*/ {{sh,0x2000,7}, {sh,0x10000,7}, {sh,0x10000,7}, {sh,0x10000,7}, {sh,0x10000,7}, {sh,0x10000,7}, {sh,0x10000,7}}
  }
{}
auto SSE::p(int pr0) -> int {
  INJECT_SHARED_c0
  INJECT_SHARED_bpos
  INJECT_SHARED_c4
  INJECT_SHARED_blockPos
  int pr = 0;
  int pr1 = 0;
  int pr2 = 0;
  int pr3 = 0;
  INJECT_SHARED_blockType
    switch( blockType ) {
    case TEXT:
    case TEXT_EOL: {
      int limit = 0x3FFU >> (static_cast<int>(blockPos < 0xFFF) * 2);
      pr = Text.APMs[0].p(pr0, (c0 << 8U) | (shared->State.Text.mask & 0xFU) | ((shared->State.misses & 0xFU) << 4U), limit);
      pr1 = Text.APMs[1].p(pr0, finalize64(hash(bpos, shared->State.misses & 3U, c4 & 0xffffU, shared->State.Text.mask >> 4U), 16), limit);
      pr2 = Text.APMs[2].p(pr0, finalize64(hash(c0, shared->State.Match.expectedByte << 2 | shared->State.Match.length3), 16), limit);
      pr3 = Text.APMs[3].p(pr0, finalize64(hash(c0, c4 & 0xffffU, shared->State.Text.firstLetter), 16), limit);

      pr0 = (pr0 + pr1 + pr2 + pr3 + 2) >> 2U;

      pr1 = Text.APM1s[0].p(pr0, finalize64(hash(shared->State.Match.expectedByte << 2 | shared->State.Match.length3, c4 & 0xffU), 16));
      pr2 = Text.APM1s[1].p(pr, finalize64(hash(c0, c4 & 0x00ffffffU), 16));
      pr3 = Text.APM1s[2].p(pr, finalize64(hash(c0, c4 & 0xffffff00U), 16));

      pr = (pr + pr1 + pr2 + pr3 + 2) >> 2U;
      pr = (pr + pr0 + 1) >> 1U;
      break;
    }
    case IMAGE24:
    case IMAGE32: {
      int limit = 0x3FFU >> (static_cast<int>(blockPos < 0xFFFU) * 4);
      pr = Image.Color.APMs[0].p(pr0, (c0 << 4U) | (shared->State.misses & 0xFU), limit);
      pr1 = Image.Color.APMs[1].p(pr0, finalize64(hash(c0, shared->State.Image.pixels.W, shared->State.Image.pixels.WW), 16), limit);
      pr2 = Image.Color.APMs[2].p(pr0, finalize64(hash(c0, shared->State.Image.pixels.N, shared->State.Image.pixels.NN), 16), limit);
      pr3 = Image.Color.APMs[3].p(pr0, (c0 << 8U) | shared->State.Image.ctx, limit);

      pr0 = (pr0 + pr1 + pr2 + pr3 + 2) >> 2U;
      pr1 = Image.Color.APM1s[0].p(pr, finalize64(hash(c0, shared->State.Image.pixels.W, (c4 & 0xffU) - shared->State.Image.pixels.Wp1, shared->State.Image.plane), 16));
      pr2 = Image.Color.APM1s[1].p(pr, finalize64(hash(c0, shared->State.Image.pixels.N, (c4 & 0xffU) - shared->State.Image.pixels.Np1, shared->State.Image.plane), 16));

      pr = (pr * 2 + pr1 * 3 + pr2 * 3 + 4) >> 3U;
      pr = (pr + pr0 + 1) >> 1U;
      break;
    }
    case IMAGE8GRAY: {
      int limit = 0x3FFU >> (static_cast<int>(blockPos < 0xFFF) * 4);
      pr = Image.Gray.APMs[0].p(pr0, (c0 << 4) | (shared->State.misses & 0xFU), limit);
      pr1 = Image.Gray.APMs[1].p(pr, (c0 << 8) | shared->State.Image.ctx, limit);
      pr2 = Image.Gray.APMs[2].p(pr0, bpos | (shared->State.Image.ctx & 0xF8U) | (shared->State.Match.expectedByte << 8U), limit);

      pr0 = (2 * pr0 + pr1 + pr2 + 2) >> 2U;
      pr = (pr + pr0 + 1) >> 1U;
      break;
    }
    case IMAGE8: {
      int limit = 0x3FFU >> (static_cast<int>(blockPos < 0xFFFU) * 4);
      pr = Image.Palette.APMs[0].p(pr0, (c0 << 4U) | (shared->State.misses & 0xFU), limit);
      pr1 = Image.Palette.APMs[1].p(pr0, finalize64(hash(c0 | shared->State.Image.pixels.W << 8U | shared->State.Image.pixels.N << 16U), 16), limit);
      pr2 = Image.Palette.APMs[2].p(pr0, finalize64(hash(c0 | shared->State.Image.pixels.N << 8U | shared->State.Image.pixels.NN << 16U), 16), limit);
      pr3 = Image.Palette.APMs[3].p(pr0, finalize64(hash(c0 | shared->State.Image.pixels.W << 8U | shared->State.Image.pixels.WW << 16U), 16), limit);

      pr0 = (pr0 + pr1 + pr2 + pr3 + 2) >> 2U;
      pr1 = Image.Palette.APM1s[0].p(pr0, finalize64(hash(c0 | shared->State.Match.expectedByte << 8U | shared->State.Image.pixels.N << 16U), 16));
      pr2 = Image.Palette.APM1s[1].p(pr, finalize64(hash(c0 | shared->State.Image.pixels.W << 8U | shared->State.Image.pixels.N << 16U), 16));

      pr = (pr * 2 + pr1 + pr2 + 2) >> 2U;
      pr = (pr + pr0 + 1) >> 1U;
      break;
    }
    case JPEG: {
      pr = Jpeg.APMs[0].p(pr0, shared->State.JPEG.state, 0x3FF);
      pr = (pr + pr0 + 1) / 2;
      break;
    }
    case DEC_ALPHA: {
      int const limit = 0x3FFu >> (static_cast<int>(blockPos < 0xFFFu) * 4);
      pr = DEC.APMs[0].p(pr0, (shared->State.DEC.state * 26u) + shared->State.DEC.bcount, limit);
      pr = (pr * 4 + pr0 * 2 + 3) / 6;
      break;
    }
    case EXE: {
      int const limit = 0x3FFu >> (static_cast<int>(blockPos < 0xFFFu) * 4);
      pr = x86_64.APMs[0].p(pr0, (shared->State.x86_64.state << 3u) | bpos, limit);
      pr1 = x86_64.APMs[1].p(pr0, (c0 << 8u) | shared->State.x86_64.state, limit);
      pr2 = x86_64.APMs[2].p((pr1+pr+1)/2, finalize64(hash(c4 & 0xFFu, bpos, shared->State.misses & 0x1u, shared->State.x86_64.state >> 3), 16), limit);
      pr = (pr + pr0 + pr1 + pr2 + 2) >> 2;
      break;
    }
    default: {
      pr = Generic.APM1s[0].p(pr0, (shared->State.Match.length3) << 11U | c0 << 3U | (shared->State.misses & 0x7U));
      const uint16_t ctx1 = c0 | (c4 & 0xffU) << 8U;
      const uint16_t ctx2 = c0 ^finalize64(hash(c4 & 0xffffU), 16);
      const uint16_t ctx3 = c0 ^finalize64(hash(c4 & 0xffffffU), 16);
      pr1 = Generic.APM1s[1].p(pr0, ctx1);
      pr2 = Generic.APM1s[2].p(pr0, ctx2);
      pr3 = Generic.APM1s[3].p(pr0, ctx3);

      pr0 = (pr0 + pr1 + pr2 + pr3 + 2) >> 2U;
      pr1 = Generic.APM1s[4].p(pr, (shared->State.Match.expectedByte << 8U) | (c4 & 0xffu));
      pr2 = Generic.APM1s[5].p(pr, ctx2);
      pr3 = Generic.APM1s[6].p(pr, ctx3);

      pr = (pr + pr1 + pr2 + pr3 + 2) >> 2U;
      pr = (pr + pr0 + 1) >> 1U;
    }
  }
  return pr;
}
