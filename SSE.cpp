#include "SSE.hpp"
#include "Hash.hpp"

SSE::SSE(Shared* const sh) : shared(sh),
  Text {
    { /*APM:*/  {sh,0x10000,20}, {sh,0x10000,24}, {sh,0x10000,24}, {sh,0x10000,24}}, /* APM: contexts, steps */
    { /*APM1:*/ {sh,0x10000,7}, {sh,0x10000,6}, {sh,0x10000,6}}, /* APM1: contexts, rate */
    { /*ApmPostA: */ sh,8},
    { /*ApmPostB: */ sh,8}
  },
  Image {
    // color:
    { { /*APM:*/ {sh,0x1000,24}, {sh,0x10000,24}, {sh,0x10000,24}, {sh,0x10000,24}}, /*APM1:*/ {{sh,0x10000,7}, {sh,0x10000,7}},
      { /*ApmPostA: */ sh,8},
      { /*ApmPostB: */ sh,8}
    }, 
    // palette:
    { { /*APM:*/ {sh,0x1000,24}, {sh,0x10000,24}, {sh,0x10000,24}, {sh,0x10000,24}}, /*APM1:*/ {{sh,0x10000,5}, {sh,0x10000,6}},
      { /*ApmPostA: */ sh,1},
      { /*ApmPostB: */ sh,1}
    }, 
    // gray:
    { { /*APM:*/ {sh,0x1000,24}, {sh,0x10000,24}, {sh,0x10000,24}} ,
      { /*ApmPostA: */ sh,8},
      { /*ApmPostB: */ sh,8}
    }
  },
  Audio{
    { /*APM:*/ {sh,256*256,24} },
    { /*ApmPostA: */ sh,8},
    { /*ApmPostB: */ sh,8}
  },
  Jpeg{
    { /*APM:*/ {sh,0x1000 + 1,24} },
    { /*ApmPostA: */ sh,1},
    { /*ApmPostB: */ sh,1}
  },
  DEC{
    { /*APM:*/ {sh,25*26,20} },
    { /*ApmPostA: */ sh,8},
    { /*ApmPostB: */ sh,8}
  },
  x86_64{
    { /*APM:*/ {sh,0x10000,20}, {sh,0x10000,16}, {sh,0x10000,16} },
    { /*APM1:*/ {sh,0x10000,7}, {sh,0x10000,7}, {sh,0x10000,7} },
    { /*ApmPostA: */ sh,1},
    { /*ApmPostB: */ sh,1}
  },
  Generic {
    { /*APM:*/  {sh,1<<10,20}, {sh,0x10000,24}, {sh,0x10000,24}, {sh,0x10000,24}}, /* APM: contexts, steps */
    { /*APM1:*/ {sh,0x10000,7}, {sh,0x10000,7}, {sh,0x10000,7}},
    { /*ApmPostA: */ sh,8},
    { /*ApmPostB: */ sh,8}
  }
  {}

auto SSE::p(const int pr_orig) -> int {

  INJECT_SHARED_c0
  INJECT_SHARED_bpos
  INJECT_SHARED_c4
  INJECT_SHARED_blockPos
  INJECT_SHARED_blockType

  assert(shared->State.NormalModel.order <= 7u);
  assert(shared->State.WordModel.order <= 31u);
  assert(shared->State.Text.order <= 15u);

  assert(shared->State.x86_64.state <= 255u);
  assert(shared->State.Audio <= 255u);
  assert(shared->State.JPEG.state <= 4095u + 1u);

  uint32_t misses4 = shared->State.misses & 15u;
  constexpr int limit = 1023;

  switch(blockType) {
    case TEXT:
    case TEXT_EOL: {
      uint32_t pr0 = Text.APMs[0].p(pr_orig, static_cast<uint32_t>(c0) << 8 | (shared->State.Text.mask & 0x0F) | misses4 << 4, limit);
      uint32_t pr1 = Text.APMs[1].p(pr_orig, finalize64(hash(bpos, misses4 & 3, c4 & 0xFFFF, shared->State.Text.mask >> 4), 16), limit);
      uint32_t pr2 = Text.APMs[2].p(pr_orig, finalize64(hash(c0, shared->State.Match.expectedByte << 2 | shared->State.Match.length2), 16), limit);
      uint32_t pr3 = Text.APMs[3].p(pr_orig, finalize64(hash(c0, c4 & 0xFFFF, shared->State.Text.firstLetter), 16), limit);

      uint32_t prA = (pr_orig + pr1 + pr2 + pr3 + 2) >> 2;

      uint32_t pr4 = Text.APM1s[0].p(prA, (shared->State.WordModel.order>>2)<<13 | shared->State.Match.expectedByte << 5 | shared->State.Match.length2 << 3 | (shared->State.Text.order>>1)); //16
      uint32_t pr5 = Text.APM1s[1].p(pr0, finalize64(hash(c0, c4 & 0x00FFFFFF), 16));
      uint32_t pr6 = Text.APM1s[2].p(pr0, finalize64(hash(c0, c4), 16));

      uint32_t prB = (pr0 + pr4 + pr5 + pr6 + 2) >> 2;

      uint32_t pr = (Text.APMPostA.p(prA, bpos) + Text.APMPostB.p(prB, bpos) + 1) >> 1;
      return pr;
      break;
    }
    case IMAGE24:
    case IMAGE32: {
      uint32_t pr0 = Image.Color.APMs[0].p(pr_orig, static_cast<uint32_t>(c0) << 4 | misses4, limit);
      uint32_t pr1 = Image.Color.APMs[1].p(pr_orig, finalize64(hash(c0, shared->State.Image.pixels.W, shared->State.Image.pixels.WW), 16), limit);
      uint32_t pr2 = Image.Color.APMs[2].p(pr_orig, finalize64(hash(c0, shared->State.Image.pixels.N, shared->State.Image.pixels.NN), 16), limit);
      uint32_t pr3 = Image.Color.APMs[3].p(pr_orig, (c0 << 8) | shared->State.Image.ctx, limit);

      uint32_t prA = (pr_orig + pr1 + pr2 + pr3 + 2) >> 2;

      uint32_t pr4 = Image.Color.APM1s[0].p(pr0, finalize64(hash(c0, shared->State.Image.pixels.W, (c4 & 0xFF) - shared->State.Image.pixels.Wp1, shared->State.Image.plane), 16));
      uint32_t pr5 = Image.Color.APM1s[1].p(pr0, finalize64(hash(c0, shared->State.Image.pixels.N, (c4 & 0xFF) - shared->State.Image.pixels.Np1, shared->State.Image.plane), 16));

      uint32_t prB = (pr0 * 2 + pr4 * 3 + pr5 * 3 + 4) >> 3;
      
      uint32_t pr = (Image.Color.APMPostA.p(prA, bpos) + Image.Color.APMPostB.p(prB, bpos) + 1) >> 1;
      return pr;
      break;
    }
    case IMAGE8GRAY: {
      uint32_t pr0 = Image.Gray.APMs[0].p(pr_orig, static_cast<uint32_t>(c0) << 4 | misses4, limit);
      uint32_t pr1 = Image.Gray.APMs[1].p(pr0, (c0 << 8) | shared->State.Image.ctx, limit);
      uint32_t pr2 = Image.Gray.APMs[2].p(pr_orig, bpos | (shared->State.Image.ctx & 0xF8) | (shared->State.Match.expectedByte << 8), limit);

      int prA = (2 * pr_orig + pr1 + pr2 + 2) >> 2;

      uint32_t pr = (Image.Gray.APMPostA.p(pr0, bpos) + Image.Gray.APMPostB.p(prA, bpos) + 1) >> 1;
      return pr;
      break;
    }
    case IMAGE8: {
      uint32_t pr0 = Image.Palette.APMs[0].p(pr_orig, c0 | misses4 << 8, limit);
      uint32_t pr1 = Image.Palette.APMs[1].p(pr_orig, finalize64(hash(c0 | shared->State.Image.pixels.W << 8 | shared->State.Image.pixels.N << 16), 16), limit);
      uint32_t pr2 = Image.Palette.APMs[2].p(pr_orig, finalize64(hash(c0 | shared->State.Image.pixels.N << 8 | shared->State.Image.pixels.NN << 16), 16), limit);
      uint32_t pr3 = Image.Palette.APMs[3].p(pr_orig, finalize64(hash(c0 | shared->State.Image.pixels.W << 8 | shared->State.Image.pixels.WW << 16), 16), limit);

      uint32_t prA = (pr_orig + pr1 + pr2 + pr3 + 2) >> 2;

      uint32_t pr4 = Image.Palette.APM1s[0].p(prA, finalize64(hash(c0 | shared->State.Match.expectedByte << 8 | shared->State.Image.pixels.N << 16), 16));
      uint32_t pr5 = Image.Palette.APM1s[1].p(pr0, finalize64(hash(c0 | shared->State.Image.pixels.W << 8 | shared->State.Image.pixels.N << 16), 16));

      uint32_t prB = (pr0 * 2 + pr4 + pr5 + 2) >> 2;

      uint32_t pr = (Image.Palette.APMPostA.p(prA, 0) + Image.Palette.APMPostB.p(prB, 0) + 1) >> 1;
      return pr;
      break;
    }
    case AUDIO:
    case AUDIO_LE: {
      uint32_t pr0 = Audio.APMs[0].p(pr_orig, shared->State.Audio << 8 | static_cast<uint32_t>(bpos) << 5 | (shared->State.misses & 31), limit);
      uint32_t pr = (Audio.APMPostA.p(pr_orig, bpos) + Audio.APMPostB.p(pr0, bpos) + 1) >> 1;
      return pr;
      break;
    }
    case JPEG: {
      uint32_t pr0 = Jpeg.APMs[0].p(pr_orig, shared->State.JPEG.state, limit);
      uint32_t pr = (Jpeg.APMPostA.p(pr_orig, 0) + Jpeg.APMPostB.p(pr0, 0) + 1) >> 1;
      return pr;
      break;
    }
    case DEC_ALPHA: {
      uint32_t pr0 = DEC.APMs[0].p(pr_orig, (shared->State.DEC.state * 26u) + shared->State.DEC.bcount, limit);
      uint32_t pr = (DEC.APMPostA.p(pr_orig, 0) + DEC.APMPostB.p(pr0, 0)+1) >> 1;
      return pr;
      break;
    }
    case EXE: {
      uint32_t pr0 = x86_64.APMs[0].p(pr_orig, shared->State.x86_64.state << 7 | misses4 << 3 | bpos, limit);//15
      uint32_t pr1 = x86_64.APMs[1].p(pr_orig, shared->State.x86_64.state<<8 | c0, limit); // 16
      uint32_t pr2 = x86_64.APMs[2].p((pr0 + pr1 + 1) >> 1, finalize64(hash(c4 & 0xFF, bpos, shared->State.misses & 1, shared->State.x86_64.state >> 3), 16), limit);

      uint32_t prA = (pr_orig + pr0 + pr1 + pr2 + 2) >> 2;

      uint32_t pr4 = x86_64.APM1s[0].p(prA, shared->State.Match.expectedByte << 5 | shared->State.NormalModel.order << 2 | shared->State.Match.length2); //13
      uint32_t pr5 = x86_64.APM1s[1].p(prA, c0 | (c4 & 0xFF) << 8); //16
      uint32_t pr6 = x86_64.APM1s[1].p(pr_orig, c0 | (c4 & 0xFF) << 8); //16

      uint32_t prB = (pr0 + pr4 + pr5 + pr6 + 2) >> 2;

      uint32_t pr = (x86_64.APMPostA.p(prA, 0) + x86_64.APMPostB.p(prB, 0) + 1) >> 1;
      return pr;
      break;
    }
    default: {
      uint32_t pr0 = Generic.APMs[0].p(pr_orig, shared->State.Match.length2 << 8 | static_cast<uint32_t>(bpos) << 5 | (shared->State.misses & 31), limit); //10
      uint32_t pr1 = Generic.APMs[1].p(pr_orig, shared->State.NormalModel.order << 5 | shared->State.Match.length2 << 3 | bpos, limit); //8
      uint32_t pr2 = Generic.APMs[2].p(pr_orig, c0 | (c4 & 0xFF) << 8, limit); //16
      uint32_t pr3 = Generic.APMs[3].p(pr_orig, c0 << 8 | (c4 & 0xF0) | (c4 & 0xF000 >> 12), limit); //16

      uint32_t prA = (pr_orig + pr1 + pr2 + pr3 + 2) >> 2;
      
      uint32_t misses2 = (shared->State.misses & 3) << 14;
      uint32_t pr4 = Generic.APM1s[0].p(prA, misses2 | shared->State.Match.expectedByte << 5 | shared->State.NormalModel.order << 2 | shared->State.Match.length2); //13
      uint32_t pr5 = Generic.APM1s[1].p(pr0, misses2 | shared->State.Match.length2 << 11 | (shared->State.WordModel.order >> 2) << 8 | c0); //13
      uint32_t pr6 = Generic.APM1s[2].p(pr0, misses2 | (bpos>>1) << 12 | (c4 & 0xFF) << 4 | (static_cast<uint32_t>(shared->State.WordModel.order) >> 1)); //14

      uint32_t prB = (pr0 + pr4 + pr5 + pr6 + 2) >> 2;

      uint32_t pr = (Generic.APMPostA.p(prA, shared->State.NormalModel.order) + Generic.APMPostB.p(prB, shared->State.Match.mode3) + 1) >> 1;
      return pr;
    }
  }
  
}
