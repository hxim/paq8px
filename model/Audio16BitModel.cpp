#include "Audio16BitModel.hpp"
#include "../BitCount.hpp"

Audio16BitModel::Audio16BitModel(Shared* const sh) : AudioModel(sh), 
  sMap1B{ /* SmallStationaryContextMap : BitsOfContext, InputBits, Rate, Scale */
    /*nOLS: 0-3*/ {{sh,17,1,7,128},{sh,17,1,10,128},{sh,17,1,6,86},{sh,17,1,6,128}}, {{sh,17,1,7,128},{sh,17,1,10,128},{sh,17,1,6,86},{sh,17,1,6,128}}, {{sh,17,1,7,128},{sh,17,1,10,128},{sh,17,1,6,86},{sh,17,1,6,128}}, {{sh,17,1,7,128},{sh,17,1,10,128},{sh,17,1,6,86},{sh,17,1,6,128}},
    /*nOLS: 4-7*/ {{sh,17,1,7,128},{sh,17,1,10,128},{sh,17,1,6,86},{sh,17,1,6,128}}, {{sh,17,1,7,128},{sh,17,1,10,128},{sh,17,1,6,86},{sh,17,1,6,128}}, {{sh,17,1,7,128},{sh,17,1,10,128},{sh,17,1,6,86},{sh,17,1,6,128}}, {{sh,17,1,7,128},{sh,17,1,10,128},{sh,17,1,6,86},{sh,17,1,6,128}},
    /*nLMS: 0-2*/ {{sh,17,1,7,86}, {sh,17,1,10,86}, {sh,17,1,6,64},{sh,17,1,6,86}},  {{sh,17,1,7,86},{sh,17,1,10,86},  {sh,17,1,6,64},{sh,17,1,6,86}},  {{sh,17,1,7,86}, {sh,17,1,10,86}, {sh,17,1,6,64},{sh,17,1,6,86}},
    /*nSSM: 0-2*/ {{sh,17,1,7,86}, {sh,17,1,10,86}, {sh,17,1,6,64},{sh,17,1,6,86}},  {{sh,17,1,7,86},{sh,17,1,10,86},  {sh,17,1,6,64},{sh,17,1,6,86}},  {{sh,17,1,7,86}, {sh,17,1,10,86}, {sh,17,1,6,64},{sh,17,1,6,86}}
  }
{
  /* s, d, sameChannelRate, otherChannelRate */
  lms[0][0] = LMS::create(sh->chosenSimd, 1280, 640, 5e-5f, 5e-5f);
  lms[0][1] = LMS::create(sh->chosenSimd, 1280, 640, 5e-5f, 5e-5f);

  lms[1][0] = LMS::create(sh->chosenSimd, 640, 64, 7e-5f, 1e-5f);
  lms[1][1] = LMS::create(sh->chosenSimd, 640, 64, 7e-5f, 1e-5f);

  lms[2][0] = LMS::create(sh->chosenSimd, 2456, 8, 2e-5f, 2e-6f);
  lms[2][1] = LMS::create(sh->chosenSimd, 2456, 8, 2e-5f, 2e-6f);

  for (int i = 0; i < nOLS; i++) {
    ols[i][0] = create_OLS_double(sh->chosenSimd, num[i], solveInterval[i], lambda[i], nu);
    ols[i][1] = create_OLS_double(sh->chosenSimd, num[i], solveInterval[i], lambda[i], nu);
  }
}

void Audio16BitModel::setParam(int info) {
  INJECT_SHARED_bpos
  INJECT_SHARED_blockPos
  if( blockPos == 0 && bpos == 0 ) {
    info |= 4; // force internal flag for Big Endian; why: EndiannessFilter already transformed the data from LE to BE
    assert((info & 2) != 0); // has to be 16-bit
    stereo = (info & 1);
    lsb = static_cast<uint32_t>(info < 4);
    mask = 0;
    wMode = info;
    for( int i = 0; i < nLMS; i++ ) {
      lms[i][0]->reset(), lms[i][1]->reset();
    }
  }
}

// Map a signed value to an N-bit unsigned integer by clamping to [0, 2^bits - 1].
// Used to pack residuals into context keys for SmallStationaryContextMap.
static int clamp_to_n_bits_unsigned(int x, int bits) {
  const int half = 1 << (bits - 1);
  const int min = -half;
  const int max = half - 1;
  if (x < min) return min - min;
  if (x > max) return max - min;
  return x - min; // positive
}

void Audio16BitModel::mix(Mixer& m) {
  INJECT_SHARED_bpos
  INJECT_SHARED_blockPos
  if (bpos == 0 && blockPos != 0) {
    ch = (stereo) != 0 ? (blockPos & 2) >> 1 : 0;
    lsb = (blockPos & 1) ^ static_cast<uint32_t>(wMode < 4);
    if ((blockPos & 1) == 0) {
      sample = (wMode < 4) ? s2(2) : t2(2);
      const int pCh = ch ^ stereo;
      int i = 0;
      for (errLog = 0; i < nOLS; i++) {
        ols[i][pCh]->update(sample);
        residuals[i][pCh] = sample - prd[i][pCh][0];
        const uint32_t absResidual = static_cast<uint32_t>(abs(residuals[i][pCh]));
        mask += mask + static_cast<uint32_t>(absResidual > 128);
        errLog += square(absResidual >> 6);
      }
      for (int j = 0; j < nLMS; j++) {
        lms[j][pCh]->update(sample);
      }
      for (; i < nSSM; i++) {
        residuals[i][pCh] = sample - prd[i][pCh][0];
      }
      errLog = min(0xF, ilog2(errLog));

      if (stereo != 0) {
        for (int i = 1; i <= 24; i++) {
          ols[0][ch]->add((double)x2(i));
        }
        for (int i = 1; i <= 104; i++) {
          ols[0][ch]->add((double)x1(i));
        }
      }
      else {
        for (int i = 1; i <= 128; i++) {
          ols[0][ch]->add((double)x1(i));
        }
      }

      int k1 = 90;
      int k2 = k1 - 12 * stereo;
      for (int j = (i = 1); j <= k1; j++, i += 1
        << (static_cast<int>(j > 16) +
          static_cast<int>(j > 32) +
          static_cast<int>(j > 64))) {
        ols[1][ch]->add((double)x1(i));
      }
      for (int j = (i = 1); j <= k2; j++, i += 1
        << (static_cast<int>(j > 5) +
          static_cast<int>(j > 10) +
          static_cast<int>(j > 17) +
          static_cast<int>(j > 26) +
          static_cast<int>(j > 37))) {
        ols[2][ch]->add((double)x1(i));
      }
      for (int j = (i = 1); j <= k2; j++, i += 1
        << (static_cast<int>(j > 3) +
          static_cast<int>(j > 7) +
          static_cast<int>(j > 14) +
          static_cast<int>(j > 20) +
          static_cast<int>(j > 33) +
          static_cast<int>(j > 49))) {
        ols[3][ch]->add((double)x1(i));
      }
      for (int j = (i = 1); j <= k2; j++, i += 1 +
        static_cast<int>(j > 4) +
        static_cast<int>(j > 8)) {
        ols[4][ch]->add((double)x1(i));
      }
      for (int j = (i = 1); j <= k1; j++, i += 2 +
        (static_cast<int>(j > 3) +
          static_cast<int>(j > 9) +
          static_cast<int>(j > 19) +
          static_cast<int>(j > 36) +
          static_cast<int>(j > 61))) {
        ols[5][ch]->add((double)x1(i));
      }

      if (stereo != 0) {
        for (i = 1; i <= k1 - k2; i++) {
          const double s = (double)x2(i);
          ols[2][ch]->add(s);
          ols[3][ch]->add(s);
          ols[4][ch]->add(s);
        }
      }

      k1 = 28, k2 = k1 - 6 * stereo;
      for (i = 1; i <= k2; i++) {
        ols[6][ch]->add((double)x1(i));
      }
      for (i = 1; i <= k1 - k2; i++) {
        ols[6][ch]->add((double)x2(i));
      }

      k1 = 32, k2 = k1 - 8 * stereo;
      for (i = 1; i <= k2; i++) {
        ols[7][ch]->add((double)x1(i));
      }
      for (i = 1; i <= k1 - k2; i++) {
        ols[7][ch]->add((double)x2(i));
      }

      for (i = 0; i < nOLS; i++) {
        double prediction = ols[i][ch]->predict();
        prd[i][ch][0] = signedClip16(static_cast<int>(round(prediction)));
      }
      for (; i < nOLS + nLMS; i++) {
        float prediction = lms[i - nOLS][ch]->predict(sample);
        prd[i][ch][0] = signedClip16(static_cast<int>(round(prediction)));
      }
      prd[i++][ch][0] = signedClip16(x1(1) * 2 - x1(2));
      prd[i++][ch][0] = signedClip16(x1(1) * 3 - x1(2) * 3 + x1(3));
      prd[i][ch][0] = signedClip16(x1(1) * 4 - x1(2) * 6 + x1(3) * 4 - x1(4));
      for (i = 0; i < nSSM; i++) {
        prd[i][ch][1] = signedClip16(prd[i][ch][0] + residuals[i][pCh]);
      }
    }
    shared->State.Audio = 0x80 | (mxCtx = ilog2(min(0x1F, bitCount(mask))) * 4 + ch * 2 + lsb);
  }

  INJECT_SHARED_c0
  INJECT_SHARED_c1
  const int b = int16_t((wMode < 4) ?
    (lsb) != 0 ?
    uint8_t(c0 << (8 - bpos)) :
    (c0 << (16 - bpos)) | c1
    :
    (lsb) != 0 ?
    (c1 << 8) | uint8_t(c0 << (8 - bpos)) :
    c0 << (16 - bpos));

  int pos = lsb * 8 + bpos; // bits processed so far: 0..15
  assert(pos >= 0 && pos <= 15);

  for (int i = 0; i < nSSM; i++) {
    const int pred0 = prd[i][ch][0];
    const int pred1 = prd[i][ch][1];

    // Signed prediction error (residual); range is [-65535, 65535]
    const int res0 = pred0 - b; // 16 bits + 1 sign = 17 bits
    const int res1 = pred1 - b;

    // Number of residual bits (including sign) used for each of the 4 context maps.
    // Wider contexts capture more precision but reduce statistical sharing.
    // Narrower contexts learn faster but have less precision.
    int N1 = 8;
    int N2 = 13;
    int N3 = 4;
    int N4 = 5;

    // Shift the residual right so that only bits at or above the current bit position `pos`
    // contribute to the context, progressively narrowing precision as decoding advances.
    int shift1 = ((17 - N1) - pos);
    int shift2 = ((17 - N2) - pos);
    int shift3 = ((17 - N3) - pos);
    int shift4 = ((17 - N4) - pos);
    if (shift1 < 0) shift1 = 0;
    if (shift2 < 0) shift2 = 0;
    if (shift3 < 0) shift3 = 0;
    if (shift4 < 0) shift4 = 0;

    sMap1B[i][0].set(clamp_to_n_bits_unsigned(res0 >> shift1, N1) << 4 | pos);
    sMap1B[i][1].set(clamp_to_n_bits_unsigned(res0 >> shift2, N2) << 4 | pos);
    sMap1B[i][2].set(clamp_to_n_bits_unsigned(res0 >> shift3, N3) << 4 | pos);
    sMap1B[i][3].set(clamp_to_n_bits_unsigned(res1 >> shift4, N4) << 4 | pos);
    sMap1B[i][0].mix(m);
    sMap1B[i][1].mix(m);
    sMap1B[i][2].mix(m);
    sMap1B[i][3].mix(m);
  }

  m.set((errLog << 9) | (lsb << 8) | c0, 8192);
  m.set((uint8_t(mask) << 4) | (ch << 3) | (lsb << 2) | (bpos >> 1), 4096);
  m.set((mxCtx << 7) | (c1 >> 1), 2560);
  m.set((errLog << 4) | (ch << 3) | (lsb << 2) | (bpos >> 1), 256);
  m.set(mxCtx, 20);
}
