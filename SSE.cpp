#include "SSE.hpp"
#include "Hash.hpp"
#include "Utils.hpp"

SSE::SSE(Shared* const sh) : shared(sh),
Text{
    { /*APM:*/  {sh,1 << 15,20,1023}, {sh,1 << 16,24,1023}, {sh,1 << 16,24,1023}, {sh,1 << 16,24,1023}} , /* APM: contexts, steps */
    { /*APM1:*/ {sh,1 << 16,7}, {sh,1 << 16,6}, {sh,1 << 16,6}}, /* APM1: contexts, rate */
    { /*ApmPostA: */ sh,8},
    { /*ApmPostB: */ sh,8}
},
Image{
  // color:
    { { /*APM:*/ {sh,1 << 11,24,1023}, {sh,1 << 16,24,1023}, {sh,1 << 16,24,1023}, {sh,1 << 16,24,1023}},
      { /*APM1:*/ {sh,1 << 16,7}, {sh,1 << 16,7}} ,
      { /*ApmPostA: */ sh,8},
      { /*ApmPostB: */ sh,8}
    }, 
    // palette:
    { { /*APM:*/ {sh,1 << 11,24,1023}, {sh,1 << 16,24,1023}, {sh,1 << 16,24,1023}, {sh,1 << 16,24,1023}},
      { /*APM1:*/ {sh,1 << 16,5}, {sh,1 << 16,6}},
      { /*ApmPostA: */ sh,1},
      { /*ApmPostB: */ sh,1}
    }, 
    // gray:
    { { /*APM:*/ {sh,1 << 11,24,1023}, {sh,1 << 16,24,1023}, {sh,1 << 16,24,1023}} ,
      { /*ApmPostA: */ sh,8},
      { /*ApmPostB: */ sh,8}
    }
  },
  Audio{
    { /*APM:*/ {sh,1 << 14,24,1023} },
    { /*ApmPostA: */ sh,8},
    { /*ApmPostB: */ sh,8}
  },
  Jpeg{
    { /*APM:*/ {sh,0x1000 + 1,24,1023} },
    { /*ApmPostA: */ sh,1},
    { /*ApmPostB: */ sh,1}
  },
  DEC{
    { /*APM:*/ {sh,25*26,20,1023} },
    { /*ApmPostA: */ sh,8},
    { /*ApmPostB: */ sh,8}
  },
  x86_64{
    { /*APM:*/ {sh,1 << 14,20,1023}, {sh,1 << 16,16,1023}, {sh,1 << 16,16,1023} },
    { /*APM1:*/ {sh,1 << 13,7}, {sh,1 << 16,7}, {sh,1 << 16,7} },
    { /*ApmPostA: */ sh,1},
    { /*ApmPostB: */ sh,1}
  },
  Generic {
    { /*APM:*/  {sh,1<<8,20,1023}, {sh,1 << 8,24,1023}, {sh,1 << 16,24,1023}, {sh,1 << 16,24,1023}}, /* APM: contexts, steps */
    { /*APM1:*/ {sh,1 << 16,7}, {sh,1 << 16,7}, {sh,1 << 16,7}},
    { /*ApmPostA: */ sh,8},
    { /*ApmPostB: */ sh,8}
  }
  {}

uint32_t SSE::p(const uint32_t pr_orig) {

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

  //uint32_t misses4 = shared->State.misses & 15u;
  uint32_t misses = shared->State.misses << ((8 - bpos) & 7); //byte-aligned
  misses = (misses & 0xffffff00) | (misses & 0xff) >> ((8 - bpos) & 7);

  uint32_t misses3 =
    ((misses & 0x1) != 0) |
    ((misses & 0xfe) != 0) << 1 |
    ((misses & 0xff00) != 0) << 2;
  
  const BlockType normalizedBlockType =
    blockType == BlockType::JPEG && shared->State.JPEG.state == 0 ? BlockType::DEFAULT :
    blockType;

  switch(normalizedBlockType) {
    case BlockType::TEXT:
    case BlockType::TEXT_EOL: {
      uint32_t pr0 = Text.APMs[0].p(pr_orig, static_cast<uint32_t>(c0) << 7 | (shared->State.Text.mask & 0x0F) | misses3 << 4); //15
      uint32_t pr1 = Text.APMs[1].p(pr_orig, finalize64(hash(bpos, misses3 & 3, c4 & 0xFFFF, shared->State.Text.mask >> 4), 16)); //16
      uint32_t pr2 = Text.APMs[2].p(pr_orig, finalize64(hash(c0, shared->State.Match.expectedByte << 2 | shared->State.Match.length2), 16)); //16
      uint32_t pr3 = Text.APMs[3].p(pr_orig, finalize64(hash(c0, c4 & 0xFFFF, shared->State.Text.firstLetter), 16)); //16

      uint32_t prA = (pr_orig + pr1 + pr2 + pr3 + 2) >> 2;

      uint32_t pr4 = Text.APM1s[0].p(prA, (shared->State.WordModel.order>>2)<<13 | shared->State.Match.expectedByte << 5 | shared->State.Match.length2 << 3 | (shared->State.Text.order>>1)); //16
      uint32_t pr5 = Text.APM1s[1].p(pr0, finalize64(hash(c0, c4 & 0x00FFFFFF), 16)); //16
      uint32_t pr6 = Text.APM1s[2].p(pr0, finalize64(hash(c0, c4), 16)); //16

      uint32_t prB = (pr0 + pr4 + pr5 + pr6 + 2) >> 2;

      uint32_t pr = (Text.APMPostA.p(prA, bpos) + Text.APMPostB.p(prB, bpos) + 1) >> 1;
      return pr;
      break;
    }
    case BlockType::IMAGE24:
    case BlockType::IMAGE32: {
      uint32_t pr0 = Image.Color.APMs[0].p(pr_orig, static_cast<uint32_t>(c0) << 3 | (bpos==0 ? 0 : (misses3&3))); //11
      uint32_t pr1 = Image.Color.APMs[1].p(pr_orig, finalize64(hash(c0, shared->State.Image.pixels.W, shared->State.Image.pixels.WW), 16)); //16
      uint32_t pr2 = Image.Color.APMs[2].p(pr_orig, finalize64(hash(c0, shared->State.Image.pixels.N, shared->State.Image.pixels.NN), 16)); //16
      uint32_t pr3 = Image.Color.APMs[3].p(pr_orig, (c0 << 8) | shared->State.Image.ctx); //16

      uint32_t prA = (pr_orig + pr1 + pr2 + pr3 + 2) >> 2;

      uint32_t pr4 = Image.Color.APM1s[0].p(pr0, finalize64(hash(c0, shared->State.Image.pixels.W, (c4 & 0xFF) - shared->State.Image.pixels.Wp1, shared->State.Image.plane), 16)); //16
      uint32_t pr5 = Image.Color.APM1s[1].p(pr0, finalize64(hash(c0, shared->State.Image.pixels.N, (c4 & 0xFF) - shared->State.Image.pixels.Np1, shared->State.Image.plane), 16)); //16

      uint32_t prB = (pr0 * 2 + pr4 * 3 + pr5 * 3 + 4) >> 3;
      
      uint32_t pr = (Image.Color.APMPostA.p(prA, bpos) + Image.Color.APMPostB.p(prB, bpos) + 1) >> 1;
      return pr;
      break;
    }
    case BlockType::IMAGE8GRAY: {
      uint32_t pr0 = Image.Gray.APMs[0].p(pr_orig, static_cast<uint32_t>(c0) << 3 | (bpos == 0 ? 0 : (misses3 & 3))); //11
      uint32_t pr1 = Image.Gray.APMs[1].p(pr0, (c0 << 8) | shared->State.Image.ctx); //16
      uint32_t pr2 = Image.Gray.APMs[2].p(pr_orig, bpos | (shared->State.Image.ctx & 0xF8) | (shared->State.Match.expectedByte << 8)); //16

      int prA = (2 * pr_orig + pr1 + pr2 + 2) >> 2;

      uint32_t pr = (Image.Gray.APMPostA.p(pr0, bpos) + Image.Gray.APMPostB.p(prA, bpos) + 1) >> 1;
      return pr;
      break;
    }
    case BlockType::IMAGE8: {
      uint32_t pr0 = Image.Palette.APMs[0].p(pr_orig, static_cast<uint32_t>(c0) << 3 | (bpos == 0 ? 0 : (misses3 & 3))); //11
      uint32_t pr1 = Image.Palette.APMs[1].p(pr_orig, finalize64(hash(c0 | shared->State.Image.pixels.W << 8 | shared->State.Image.pixels.N << 16), 16)); //16
      uint32_t pr2 = Image.Palette.APMs[2].p(pr_orig, finalize64(hash(c0 | shared->State.Image.pixels.N << 8 | shared->State.Image.pixels.NN << 16), 16)); //16
      uint32_t pr3 = Image.Palette.APMs[3].p(pr_orig, finalize64(hash(c0 | shared->State.Image.pixels.W << 8 | shared->State.Image.pixels.WW << 16), 16)); //16

      uint32_t prA = (pr_orig + pr1 + pr2 + pr3 + 2) >> 2;

      uint32_t pr4 = Image.Palette.APM1s[0].p(prA, finalize64(hash(c0 | shared->State.Match.expectedByte << 8 | shared->State.Image.pixels.N << 16), 16)); //16
      uint32_t pr5 = Image.Palette.APM1s[1].p(pr0, finalize64(hash(c0 | shared->State.Image.pixels.W << 8 | shared->State.Image.pixels.N << 16), 16)); //16

      uint32_t prB = (pr0 * 2 + pr4 + pr5 + 2) >> 2;

      uint32_t pr = (Image.Palette.APMPostA.p(prA, 0) + Image.Palette.APMPostB.p(prB, 0) + 1) >> 1;
      return pr;
      break;
    }
    //case BlockType::IMAGE4: //TODO
    //case BlockType::IMAGE1: //TODO
    case BlockType::AUDIO:
    case BlockType::AUDIO_LE: {
      uint32_t pr0 = Audio.APMs[0].p(pr_orig, shared->State.Audio << 6 | static_cast<uint32_t>(bpos) << 3 | misses3); //14
      uint32_t pr = (Audio.APMPostA.p(pr_orig, bpos) + Audio.APMPostB.p(pr0, bpos) + 1) >> 1;
      return pr;
      break;
    }
    case BlockType::JPEG: {
      uint32_t pr0 = Jpeg.APMs[0].p(pr_orig, shared->State.JPEG.state);
      uint32_t pr = (Jpeg.APMPostA.p(pr_orig, 0) + Jpeg.APMPostB.p(pr0, 0) + 1) >> 1;
      return pr;
      break;
    }
    case BlockType::DEC_ALPHA: {
      uint32_t pr0 = DEC.APMs[0].p(pr_orig, (shared->State.DEC.state * 26u) + shared->State.DEC.bcount);
      uint32_t pr = (DEC.APMPostA.p(pr_orig, 0) + DEC.APMPostB.p(pr0, 0)+1) >> 1;
      return pr;
      break;
    }
    case BlockType::EXE: {
      uint32_t pr0 = x86_64.APMs[0].p(pr_orig, shared->State.x86_64.state << 6 | misses3 << 3 | bpos);//14
      uint32_t pr1 = x86_64.APMs[1].p(pr_orig, shared->State.x86_64.state<<8 | c0); // 16
      uint32_t pr2 = x86_64.APMs[2].p((pr0 + pr1 + 1) >> 1, finalize64(hash(c4 & 0xFF, bpos, misses3 & 1, shared->State.x86_64.state >> 3), 16)); //16

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
      uint32_t pr0 = Generic.APMs[0].p(pr_orig, shared->State.Match.length2 << 6 | static_cast<uint32_t>(bpos) << 3 | misses3); //8
      uint32_t pr1 = Generic.APMs[1].p(pr_orig, shared->State.NormalModel.order << 5 | shared->State.Match.length2 << 3 | bpos); //8
      uint32_t pr2 = Generic.APMs[2].p(pr_orig, c0 | (c4 & 0xFF) << 8); //16
      uint32_t pr3 = Generic.APMs[3].p(pr_orig, c0 << 8 | (c4 & 0xF0) | (c4 & 0xF000 >> 12)); //16

      uint32_t prA = (pr_orig + pr1 + pr2 + pr3 + 2) >> 2;
      
      uint32_t pr4 = Generic.APM1s[0].p(prA, misses3<<13 | shared->State.Match.expectedByte << 5 | shared->State.NormalModel.order << 2 | shared->State.Match.length2); //16
      uint32_t pr5 = Generic.APM1s[1].p(pr0, misses3<<13 | shared->State.Match.length2 << 11 | (shared->State.WordModel.order >> 2) << 8 | c0); //16
      uint32_t pr6 = Generic.APM1s[2].p(pr0, (misses3&3)<<14 | (bpos>>1) << 12 | (c4 & 0xFF) << 4 | (static_cast<uint32_t>(shared->State.WordModel.order) >> 1)); //16

      uint32_t prB = (pr0 + pr4 + pr5 + pr6 + 2) >> 2;

      uint32_t pr = (Generic.APMPostA.p(prA, shared->State.NormalModel.order) + Generic.APMPostB.p(prB, shared->State.Match.mode3) + 1) >> 1;
      return pr;
    }
  }
  
}
