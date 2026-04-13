#include "Audio8BitModel.hpp"
#include "../BitCount.hpp"

Audio8BitModel::Audio8BitModel(Shared* const sh) : AudioModel(sh),
/* ResidualMap: numContexts, histogramsPerContext, scale=64 */
  mapR1{ sh, nSSM, 1 << 6, 64 }, 
  mapR2{ sh, nSSM, 1 << 6, 64 },
  mapR3{ sh, nSSM, 1 << 6, 96 },
  mapR4{ sh, nSSM, 1 << 6, 96 }
{
  /* s, d, sameChannelRate, otherChannelRate */
  lms[0][0] = LMS::create(sh->chosenSimd, 1280, 640, 3e-5f, 2e-5f);
  lms[0][1] = LMS::create(sh->chosenSimd, 1280, 640, 3e-5f, 2e-5f);

  lms[1][0] = LMS::create(sh->chosenSimd, 640, 64, 8e-5f, 1e-5f);
  lms[1][1] = LMS::create(sh->chosenSimd, 640, 64, 8e-5f, 1e-5f);

  lms[2][0] = LMS::create(sh->chosenSimd, 2448 + 8, 8, 1.6e-5f, 1e-6f);
  lms[2][1] = LMS::create(sh->chosenSimd, 2448 + 8, 8, 1.6e-5f, 1e-6f);

  for (int i = 0; i < nOLS; i++) {
    ols[i][0] = create_OLS_float(sh->chosenSimd, num[i], solveInterval[i], lambda[i], nu);
    ols[i][1] = create_OLS_float(sh->chosenSimd, num[i], solveInterval[i], lambda[i], nu);
  }
}

void Audio8BitModel::setParam(int info) {
  INJECT_SHARED_bpos
  INJECT_SHARED_blockPos
  if(blockPos == 0 && bpos == 0 ) {
    assert((info & 2) == 0); // has to be 8-bit
    stereo = (info & 1);
    mask = 0;
    wMode = info;
    for( int i = 0; i < nLMS; i++ ) {
      lms[i][0].get()->reset(), lms[i][1].get()->reset();
    }
  }
}

void Audio8BitModel::mix(Mixer &m) {
  INJECT_SHARED_bpos
  INJECT_SHARED_blockPos
  INJECT_SHARED_c1

  loss += shared->State.loss; // += 0..255

  if( bpos == 0 ) {

    ch = (stereo) != 0 ? blockPos & 1 : 0;
    const int8_t s = static_cast<int32_t>(((wMode & 4) > 0) ? c1 ^ 128 : c1) - 128;
    const int pCh = ch ^ stereo;

    for (int i = 0; i < nSSM; i++) {
      int prediction0 = prd[i][pCh][0] + 128;
      uint8_t err0 = abs(c1 - prediction0); // 0..255
      predErrBuf0[i] = ((predErrBuf0[i] * 15) >> 4) + err0; // -> 4065

      int prediction1 = prd[i][pCh][1] + 128;
      uint8_t err1 = abs(c1 - prediction1); // 0..255
      predErrBuf1[i] = ((predErrBuf1[i] * 15) >> 4) + err1; // -> 4065
    }

    lossQ = (lossQ * 15) >> 4;
    lossQ += loss;  // +0..2040 -> max: 32625, typical: ~ 4000s
    loss = 0;

    int i = 0;
    for( errLog = 0; i < nOLS; i++ ) {
      ols[i][pCh].get()->update(s);
      residuals[i][pCh] = s - prd[i][pCh][0];
      const uint32_t absResidual = static_cast<uint32_t>(abs(residuals[i][pCh]));
      mask += mask + static_cast<uint32_t>(absResidual > 4);
      errLog += square(absResidual);
    }
    for( int j = 0; j < nLMS; j++ ) {
      lms[j][pCh].get()->update(s);
    }
    for( ; i < nSSM; i++ ) {
      residuals[i][pCh] = s - prd[i][pCh][0];
    }
    errLog = min(0xF, ilog2(errLog));
    shared->State.Audio = mxCtx = ilog2(min(0x1F, bitCount(mask))) * 2 + ch;

    int k1 = 90;
    int k2 = k1 - 12 * stereo;
    for( int j = (i = 1); j <= k1; j++, i += 1
            << (static_cast<int>(j > 8) + 
                static_cast<int>(j > 16) + 
                static_cast<int>(j > 64))) {
      ols[1][ch]->add((float)x1(i));
    }
    for( int j = (i = 1); j <= k2; j++, i += 1
            << (static_cast<int>(j > 5) + 
                static_cast<int>(j > 10) + 
                static_cast<int>(j > 17) + 
                static_cast<int>(j > 26) +
                static_cast<int>(j > 37))) {
      ols[2][ch]->add((float)x1(i));
    }
    for( int j = (i = 1); j <= k2; j++, i += 1
            << (static_cast<int>(j > 3) + 
                static_cast<int>(j > 7) + 
                static_cast<int>(j > 14) + 
                static_cast<int>(j > 20) +
                static_cast<int>(j > 33) + 
                static_cast<int>(j > 49))) {
      ols[3][ch]->add((float)x1(i));
    }
    for( int j = (i = 1); j <= k2; j++, i += 1 + 
                static_cast<int>(j > 4) + 
                static_cast<int>(j > 8)) {
      ols[4][ch]->add((float)x1(i));
    }
    for( int j = (i = 1); j <= k1; j++, i += 2 + 
               (static_cast<int>(j > 3) + 
                static_cast<int>(j > 9) + 
                static_cast<int>(j > 19) +
                static_cast<int>(j > 36) + 
                static_cast<int>(j > 61))) {
      ols[5][ch]->add((float)x1(i));
    }
    if( stereo != 0 ) {
      for( i = 1; i <= k1 - k2; i++ ) {
        const float s = (float)x2(i);
        ols[2][ch]->add(s);
        ols[3][ch]->add(s);
        ols[4][ch]->add(s);
      }
    }
    k1 = 28;
    k2 = k1 - 6 * stereo;
    for( i = 1; i <= k2; i++ ) {
      const float s = (float)x1(i);
      ols[0][ch]->add(s);
      ols[6][ch]->add(s);
      ols[7][ch]->add(s);
    }
    for( ; i <= 96; i++ ) {
      ols[0][ch]->add((float)x1(i));
    }
    if( stereo != 0 ) {
      for( i = 1; i <= k1 - k2; i++ ) {
        const float s = (float)x2(i);
        ols[0][ch]->add(s);
        ols[6][ch]->add(s);
        ols[7][ch]->add(s);
      }
      for( ; i <= 32; i++ ) {
        ols[0][ch]->add((float)x2(i));
      }
    } else {
      for( ; i <= 128; i++ ) {
        ols[0][ch]->add((float)x1(i));
      }
    }

    for( i = 0; i < nOLS; i++ ) {
      float prediction = ols[i][ch]->predict();
      prd[i][ch][0] = signedClip8(static_cast<int>(roundf(prediction)));
    }
    for( ; i < nOLS + nLMS; i++ ) {
      float prediction = lms[i - nOLS][ch]->predict(s);
      prd[i][ch][0] = signedClip8(static_cast<int>(roundf(prediction)));
    }
    prd[i++][ch][0] = signedClip8(x1(1) * 2 - x1(2));
    prd[i++][ch][0] = signedClip8(x1(1) * 3 - x1(2) * 3 + x1(3));
    prd[i][ch][0] = signedClip8(x1(1) * 4 - x1(2) * 6 + x1(3) * 4 - x1(4));
    for( i = 0; i < nSSM; i++ ) {
      prd[i][ch][1] = signedClip8(prd[i][ch][0] + residuals[i][pCh]);
    }

    auto lossQ5bit = min(lossQ / 384, 31);
    for (int i = 0; i < nSSM; i++) {
      mapR1.set(prd[i][ch][0] + 128, lossQ5bit << 1 | ch);
      mapR2.set(prd[i][ch][1] + 128, lossQ5bit << 1 | ch);
      mapR3.set(prd[i][ch][0] + 128, min(predErrBuf0[i] >> 4, 31) << 1 | ch);
      mapR4.set(prd[i][ch][1] + 128, min(predErrBuf1[i] >> 4, 31) << 1 | ch);
    }
  }

  // for every bit

  mapR1.mix(m);
  mapR2.mix(m);
  mapR3.mix(m);
  mapR4.mix(m);

  INJECT_SHARED_c0
  m.set((errLog << 8) | c0, 4096);
  m.set((uint8_t(mask) << 3) | (ch << 2) | (bpos >> 1), 2048);
  m.set((mxCtx << 7) | (c1 >> 1), 1280);
  m.set((errLog << 4) | (ch << 3) | bpos, 256);
  m.set(mxCtx, 10);
}
