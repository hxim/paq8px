#include "SSE.hpp"
#include "Hash.hpp"
#include "Utils.hpp"

// ---------------------------------------------------------------------------
// SSE (Secondary Symbol Estimation)
// ---------------------------------------------------------------------------
//
// WHAT IT DOES
//
// The mixer outputs a probability that the next bit is 1. That probability is
// good, but not perfectly calibrated: in some situations the mixer is
// systematically too confident, in others not confident enough.
//
// The SSE stage acts as an adaptive probability correction layer. It corrects
// the systematic bias with lookup tables called APMs (Adaptive Probability Maps). 
// 
// Each table answers the question: "when the incoming probability was p, and
// we were in context X, how often was the bit actually 1?" After each bit,
// the table moves its entries a little toward what actually happened.
//
// No single context captures every kind of miscalibration, so we use several
// tables, each keyed by a different context (the partial byte c0, previous
// bytes, match state, neighbouring pixels, ...). That gives us several
// corrected estimates of the same bit, which then have to be combined into
// one.
//
// Throughout this comment, "cost" means the number of bits the arithmetic
// coder spends on one bit: -log2(the probability we gave to the value that
// actually occurred). 50% costs 1 bit, 90% costs 0.15 bits, 1% costs 6.64
// bits.
//
// THE BUILDING BLOCKS
//
// APM and APM1 (12-bit probability in, 12-bit probability out)
//
//   For each context, the table holds a row of points spread across the
//   probability range. The points are evenly spaced on the "stretch" scale,
//   stretch(p) = ln(p/(1-p)), which puts them closer together near 0% and
//   100%, where a small change in probability makes a big difference in
//   cost. The input probability falls between two points, and the output is
//   interpolated between their stored probabilities.
//
//   Every table starts out as (roughly) the identity map: output = input.
//
//   The two types differ in how fast they learn, which matters below:
//
//   APM:  each point counts its updates, and the update step is
//         1/(count+2). A point learns quickly at first, then settles down.
//         The count stops at a limit (1023 here), so in the long run a point
//         reflects roughly its last thousand updates. Long memory: stable,
//         precise statistics, but slow to react when the data changes.
//
//   APM1: the update step is fixed at 1/2^rate (1/32 to 1/128 here), so a
//         point reflects roughly its last 2^rate updates. Short memory:
//         reacts quickly to changes, but is noisier.
//
// APMPost (12-bit probability in, 31-bit probability out)
//
//   Everything before this stage works with 12-bit probabilities, in steps
//   of 1/4096, but the arithmetic coder works with 31 bits. The difference
//   matters for highly predictable data: a 12-bit probability can never go
//   above 4095/4096, so even a bit that is certain costs at least 0.00035
//   bits. Over a megabyte of zeros (8 million bits), that adds up to about
//   350 bytes.
//
//   APMPost removes that limit. It has one cell for each of the 4096
//   possible input values (per context), and each cell simply counts the 0s
//   and 1s it has seen. Its output is the ratio of those counts, computed
//   with 31-bit precision. So the cell for input 4095/4096 can discover that
//   the bit is really 1 in 99.999% of cases, and say so.
//
//   Each cell starts with about 2048 imaginary observations that agree with
//   its input value, so APMPost begins as the identity map and moves away
//   from it only after thousands of real observations. Its context is kept
//   small (bpos, model order, match mode, or none): the more contexts, the
//   fewer observations per cell, so the context mainly sets how fast it
//   learns.
//
//   Because it counts real outcomes, APMPost also recalibrates, but slowly,
//   and only in cells that see a lot of traffic.
//
// TWO WAYS TO AVERAGE PROBABILITIES
//
//   Linear: average the probabilities themselves.
//   Logit:  average the stretch(p) values, then convert the result back to a
//           probability with squash(), the inverse of stretch().
//
// The two behave very differently when the estimates disagree. Example: three
// tables say the bit is 1 with 70% probability; a fourth is very sure it is a
// 0 and says 0.1%.
//
//                     pooled p(1)   cost if bit=1   cost if bit=0
//   Linear average       52.5%        0.93 bits       1.07 bits
//   Logit average        25.1%        1.99 bits       0.42 bits
//
// On the stretch scale, extreme probabilities lie far out: stretch(0.001) is
// about -6.9, while stretch(0.7) is only about +0.85. So one very confident
// estimate pulls a logit average strongly toward itself. A linear average
// gives each estimate exactly its share and no more.
//
// A logit average therefore wins when confident outliers are usually right,
// and a linear average wins when they are often wrong.
//
// WHY LINEAR HERE
//
// The mixer combines its inputs on the stretch scale, and that is the right
// choice there: its weights are learned bit by bit, so it discovers which
// inputs deserve trust, and its weights may sum to more than 1, which lets
// several agreeing inputs reinforce each other.
//
// Here the situation is different:
//
//  - The weights are fixed by hand. Nothing lowers the weight of a table that
//    turns out to be unreliable.
//
//  - When one table is far more confident than the others, that is often a
//    symptom of a problem rather than a real insight:
//      * a context hashed down to 16 bits, sharing its row with unrelated
//        contexts,
//      * an APM1 row that has just adapted to a short burst of unusual data,
//      * an APM row still holding long-term statistics from data that has
//        since changed.
//    (A row that has seen little data is not a problem: it starts as the
//    identity map, so it simply passes its input through.)
//
//  - The estimates are strongly correlated. They all start from the same
//    mixer output, most of their contexts include c0 or bpos, and some tables
//    are chained: they refine another table's output instead of the mixer's.
//    Their agreement is weaker evidence than it looks.
//
// Linear averaging with weights that sum to 1 also comes with a guarantee:
// the result is never much worse than the best estimate for that bit. If an
// estimate has weight w, the averaged probability of the actual bit is at
// least w times that estimate's probability. Therefore:
//
//     cost(average) <= cost(that estimate) + log2(1/w) bits
//
// With 4 equal weights, the average costs at most 2 bits more than the best
// estimate, no matter how badly the other three fail. (In the example above,
// the best estimate costs 0.51 bits and the linear average 0.93 bits.)
// A logit average has no such bound: the more extreme the wrong estimates,
// the more it costs.
//
// THE PRICE, AND HOW IT IS PAID
//
// A linear average can never be more confident than its most confident
// input, and when the inputs disagree it comes out less confident than it
// should be. This is proven: if each input is well calibrated and they are
// not all identical, any average with positive weights summing to 1 is
// miscalibrated toward underconfidence, however strongly the inputs are
// correlated. The miscalibration grows with how much the inputs disagree.
// Recalibrating the average (replacing p by the actual frequency of 1s
// observed when the average said p) gives a strictly better forecast under
// every proper scoring rule, coding cost included.
// (Ranjan & Gneiting, "Combining probability forecasts", JRSS-B 72(1):71-91,
// 2010, doi:10.1111/j.1467-9868.2009.00726.x, Theorem 1.)
//
// The remedy the paper recommends has the same shape as this stage: a
// linear average followed by a learned recalibration map. Here, in several
// paths an APM1 takes an average (prA) as its input and learns, per context,
// how that average relates to the actual outcomes, including sharpening it
// when it is too timid. APMPost adds a slow, long-term correction on top.
//
// The final 50/50 average is, by the same theorem, slightly underconfident,
// and nothing corrects it afterwards. Its inputs rarely disagree much, so
// the loss is small, and by the bound above it can never exceed 1 bit
// more than the better of the two.
//
// LAYOUT
//
// Most paths build two estimates and average them at the end:
//
//   prA = average of the mixer output and APMs refining it
//   prB = average of pr0 and APM1s, most of them chained on refined estimates
//   pr  = ( APMPostA(prA) + APMPostB(prB) ) / 2
//
// In the paths that use APM1s (text, generic, executable, 24/32-bit and
// palette images), the two branches learn at different speeds. prA comes
// only from APMs (long memory), while half to three quarters of prB's weight
// comes from APM1s (short memory). When the data changes, the fast branch
// catches up first; in stable data, the slow branch has the more precise
// statistics. Which branch is better changes over time, and the final linear
// average never loses more than 1 bit to whichever is currently better.

SSE::SSE(Shared* const sh) : shared(sh),
Text{
    { /*APM:*/  {sh,1 << 15,20,1023}, {sh,1 << 16,24,1023}, {sh,1 << 16,24,1023}, {sh,1 << 16,24,1023}} , /* APM: contexts, steps */
    { /*APM1:*/ {sh,256 * 257,7}, {sh,1 << 16,6}, {sh,1 << 16,6}}, /* APM1: contexts, rate */
    { /*ApmPostA: */ sh,8},
    { /*ApmPostB: */ sh,8}
},
Image{
  // color:
    { { /*APM:*/ {sh,1 << 7,24,1023}, {sh,1 << 16,24,1023}, {sh,1 << 16,24,1023}, {sh,1 << 16,24,1023}},
      { /*APM1:*/ {sh,1 << 15,7}, {sh,1 << 16,7}} ,
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
  { { /*APM:*/ {sh,1 << 11,24,1023}, {sh,1 << 16,24,1023}, {sh,256 * 257,24,1023}} ,
    { /*ApmPostA: */ sh,8},
    { /*ApmPostB: */ sh,8}
  },
  // bilevel:
  { { /*APM:*/ {sh,1 << 10,20,1023}, {sh,1 << 16,20,1023}, {sh,41 * 8,20,1023}, {sh,64,20,1023} },
    { /*APM1:*/ {sh,1 << 16,6}, {sh,1 << 16,6}, {sh,1 << 16,6}},
    { /*ApmPostFinal: */ sh,16}
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
  { /*APM:*/ {sh,25 * 26,20,1023} },
  { /*ApmPostA: */ sh,8},
  { /*ApmPostB: */ sh,8}
},
x86_64{
  { /*APM:*/ {sh,1 << 14,20,1023}, {sh,1 << 16,16,1023}, {sh,1 << 16,16,1023} },
  { /*APM1:*/ {sh,64 * 257,7}, {sh,1 << 16,7}, {sh,1 << 16,7} },
  { /*ApmPostA: */ sh,1},
  { /*ApmPostB: */ sh,1}
},
Generic{
  { /*APM:*/  {sh,1 << 8,20,1023}, {sh,1 << 8,24,1023}, {sh,1 << 16,24,1023}, {sh,1 << 16,24,1023}}, /* APM: contexts, steps */
  { /*APM1:*/ {sh,256 * 257,7}, {sh,1 << 16,7}, {sh,1 << 16,7}},
  { /*ApmPostA: */ sh,8},
  { /*ApmPostB: */ sh,8}
} {
}

uint32_t SSE::p(const uint32_t pr_orig) {

  INJECT_SHARED_c0
    INJECT_SHARED_bpos
    INJECT_SHARED_c4
    INJECT_SHARED_blockPos
    INJECT_SHARED_blockType

    assert(shared->State.NormalModel.order <= 7);
  assert(shared->State.WordModel.order <= 31);
  assert(shared->State.Text.order <= 15);

  assert(shared->State.x86_64.state <= 255);
  assert(shared->State.Audio <= 255);
  assert(shared->State.JPEG.state <= 4095 + 1);

  assert(shared->State.Image1.ctx12 < (1u << 12));
  assert(shared->State.Image1.shape < (1u << 22));
  assert(shared->State.Image1.ink <= 40);
  assert(shared->State.Image1.match <= 32);

  assert(shared->State.Image.plane <= 3);
  assert(shared->State.Image.lossQ <= 639);

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

  switch (normalizedBlockType) {
  case BlockType::TEXT:
  case BlockType::TEXT_EOL: {
    uint32_t pr0 = Text.APMs[0].p(pr_orig, static_cast<uint32_t>(c0) << 7 | (shared->State.Text.mask & 0x0F) | misses3 << 4); //15
    uint32_t pr1 = Text.APMs[1].p(pr_orig, finalize64(hash(bpos, misses3 & 3, c4 & 0xFFFF, shared->State.Text.mask >> 4), 16)); //16
    uint32_t pr2 = Text.APMs[2].p(pr_orig, finalize64(hash(c0, shared->State.Match.expectedByte << 2 | shared->State.Match.length2), 16)); //16
    uint32_t pr3 = Text.APMs[3].p(pr_orig, finalize64(hash(c0, c4 & 0xFFFF, shared->State.Text.firstLetter), 16)); //16

    uint32_t prA = (pr_orig + pr1 + pr2 + pr3 + 2) >> 2;

    uint32_t pr4 = Text.APM1s[0].p(prA, shared->State.Match.expectedByte + ((shared->State.WordModel.order >> 2) << 5 | shared->State.Match.length2 << 3 | (shared->State.Text.order >> 1)) * 257); //256*257
    uint32_t pr5 = Text.APM1s[1].p(pr0, finalize64(hash(c0, c4 & 0x00FFFFFF), 16)); //16
    uint32_t pr6 = Text.APM1s[2].p(pr0, finalize64(hash(c0, c4), 16)); //16

    uint32_t prB = (pr0 + pr4 + pr5 + pr6 + 2) >> 2;

    uint32_t pr = (Text.APMPostA.p(prA, bpos) + Text.APMPostB.p(prB, bpos) + 1) >> 1;
    return pr;
    break;
  }
  case BlockType::IMAGE24:
  case BlockType::IMAGE32: {
    uint32_t plane = shared->State.Image.plane; // 0..3
    uint32_t pr0 = Image.Color.APMs[0].p(pr_orig, plane << 5 | bpos << 2 | (bpos == 0 ? 0 : (misses3 & 3))); //7, even the plane could be enough
    uint32_t pr1 = Image.Color.APMs[1].p(pr_orig, finalize64(hash(c0, shared->State.Image.pixels.W, shared->State.Image.pixels.WW), 16)); //16
    uint32_t pr2 = Image.Color.APMs[2].p(pr_orig, finalize64(hash(c0, shared->State.Image.pixels.N, shared->State.Image.pixels.NN), 16)); //16
    uint32_t pr3 = Image.Color.APMs[3].p(pr_orig, (c0 << 8) | shared->State.Image.ctx); //16

    uint32_t prA = (pr_orig + pr1 + pr2 + pr3 + 2) >> 2;

    uint32_t lossQ = shared->State.Image.lossQ; // 0 .. 639
    uint32_t pr4 = Image.Color.APM1s[0].p(pr0, (lossQ >> 2) << 5 | plane << 3 | bpos); //15
    uint32_t pr5 = Image.Color.APM1s[1].p(pr0, finalize64(hash(c0, min(lossQ, 255), plane), 16)); //16

    uint32_t prB = (pr0 * 2 + pr4 * 3 + pr5 * 3 + 4) >> 3;

    uint32_t pr = (Image.Color.APMPostA.p(prA, bpos) + Image.Color.APMPostB.p(prB, bpos) + 1) >> 1;
    return pr;
    break;
  }
  case BlockType::IMAGE8GRAY: {
    uint32_t pr0 = Image.Gray.APMs[0].p(pr_orig, static_cast<uint32_t>(c0) << 3 | (bpos == 0 ? 0 : (misses3 & 3))); //11
    uint32_t pr1 = Image.Gray.APMs[1].p(pr0, (c0 << 8) | shared->State.Image.ctx); //16
    uint32_t pr2 = Image.Gray.APMs[2].p(pr_orig, (bpos | (shared->State.Image.ctx & 0xF8)) * 257 + shared->State.Match.expectedByte); //256*257

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

    uint32_t pr4 = Image.Palette.APM1s[0].p(prA, finalize64(hash(c0 | shared->State.Image.pixels.N << 8, shared->State.Match.expectedByte), 16)); //16
    uint32_t pr5 = Image.Palette.APM1s[1].p(pr0, finalize64(hash(c0 | shared->State.Image.pixels.W << 8, shared->State.NormalModel.order), 16)); //16

    uint32_t prB = (pr0 * 2 + pr4 + pr5 + 2) >> 2;

    uint32_t pr = (Image.Palette.APMPostA.p(prA, 0) + Image.Palette.APMPostB.p(prB, 0) + 1) >> 1;
    return pr;
    break;
  }
  case BlockType::IMAGE1: {
    const auto& im = shared->State.Image1;
    const uint32_t misses = shared->State.misses;
    const uint32_t misses3im1 = (misses & 1) | ((misses & 0x6) != 0) << 1 | ((misses & 0xfffffff8) != 0) << 2;

    uint32_t pr0 = Image.Bilevel.APMs[0].p(pr_orig, (im.ctx12 & 0x7f) << 3 | misses3im1); //10
    uint32_t pr1 = Image.Bilevel.APMs[1].p(pr_orig, finalize64(hash(im.rowAbove, im.column), 16)); //16
    uint32_t pr2 = Image.Bilevel.APMs[2].p(pr_orig, static_cast<uint32_t>(im.ink) << 3 | misses3im1); //41*8
    uint32_t pr3 = Image.Bilevel.APMs[3].p(pr_orig, im.ref); //64

    uint32_t pr4 = Image.Bilevel.APM1s[0].p(pr0, finalize64(hash(im.run, im.ink, im.ref, misses3im1), 16)); //16
    uint32_t pr5 = Image.Bilevel.APM1s[1].p(pr0, finalize64(hash(im.ctx12, im.shape), 16)); //16
    uint32_t pr6 = Image.Bilevel.APM1s[2].p(pr0, finalize64(hash(im.match, im.ctx12), 16)); //16

    const uint32_t prA = (pr_orig + pr1 + pr2 + pr3 + pr4 + pr5 + pr6 + 3) / 7;

    uint32_t pr = Image.Bilevel.APMPostFinal.p(prA, im.run >> 1);
    return pr;
    break;
  }
//case BlockType::IMAGE4: //TODO
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
    uint32_t pr0 = DEC.APMs[0].p(pr_orig, (shared->State.DEC.state * 26) + shared->State.DEC.bcount);
    uint32_t pr = (DEC.APMPostA.p(pr_orig, 0) + DEC.APMPostB.p(pr0, 0) + 1) >> 1;
    return pr;
    break;
  }
  case BlockType::EXE: {
    uint32_t pr0 = x86_64.APMs[0].p(pr_orig, shared->State.x86_64.state << 6 | misses3 << 3 | bpos);//14
    uint32_t pr1 = x86_64.APMs[1].p(pr_orig, shared->State.x86_64.state << 8 | c0); // 16
    uint32_t pr2 = x86_64.APMs[2].p((pr0 + pr1 + 1) >> 1, finalize64(hash(c4 & 0xFF, bpos, misses3 & 1, shared->State.x86_64.state >> 3), 16)); //16

    uint32_t prA = (pr_orig + pr0 + pr1 + pr2 + 2) >> 2;

    uint32_t pr4 = x86_64.APM1s[0].p(prA, shared->State.Match.expectedByte + (shared->State.NormalModel.order << 3 | shared->State.Match.mode3) * 257); //64*257
    uint32_t pr5 = x86_64.APM1s[1].p(prA, c0 | (c4 & 0xFF) << 8); //16
    uint32_t pr6 = x86_64.APM1s[2].p(pr_orig, c0 | (c4 & 0xFF) << 8); //16

    uint32_t prB = (pr0 + pr4 + pr5 + pr6 + 2) >> 2;

    uint32_t pr = (x86_64.APMPostA.p(prA, 0) + x86_64.APMPostB.p(prB, 0) + 1) >> 1;
    return pr;
    break;
  }
  default: {
    uint32_t pr0 = Generic.APMs[0].p(pr_orig, shared->State.Match.length2 << 6 | static_cast<uint32_t>(bpos) << 3 | misses3); //8
    uint32_t pr1 = Generic.APMs[1].p(pr_orig, shared->State.NormalModel.order << 5 | shared->State.Match.length2 << 3 | bpos); //8
    uint32_t pr2 = Generic.APMs[2].p(pr_orig, c0 | (c4 & 0xFF) << 8); //16
    uint32_t pr3 = Generic.APMs[3].p(pr_orig, c0 << 8 | (c4 & 0xF0) | ((c4 & 0xF000) >> 12)); //16

    uint32_t prA = (pr_orig + pr1 + pr2 + pr3 + 2) >> 2;

    uint32_t pr4 = Generic.APM1s[0].p(prA, shared->State.Match.expectedByte + (misses3 << 5 | shared->State.NormalModel.order << 2 | shared->State.Match.length2) * 257); //256*257
    uint32_t pr5 = Generic.APM1s[1].p(pr0, misses3 << 13 | shared->State.Match.length2 << 11 | (shared->State.WordModel.order >> 2) << 8 | c0); //16
    uint32_t pr6 = Generic.APM1s[2].p(pr0, (misses3 & 3) << 14 | (bpos >> 1) << 12 | (c4 & 0xFF) << 4 | (static_cast<uint32_t>(shared->State.WordModel.order) >> 1)); //16

    uint32_t prB = (pr0 + pr4 + pr5 + pr6 + 2) >> 2;

    uint32_t pr = (Generic.APMPostA.p(prA, shared->State.NormalModel.order) + Generic.APMPostB.p(prB, shared->State.Match.mode3) + 1) >> 1;
    return pr;
  }
  }

}
