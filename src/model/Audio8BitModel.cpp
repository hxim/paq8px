#include "Audio8BitModel.hpp"
#include "../BitCount.hpp"

Audio8BitModel::Audio8BitModel(Shared* const sh) : AudioModel(sh),
  sMap1B{ /* SmallStationaryContextMap : BitsOfContext, InputBits, Rate, Scale */
    /*nOLS: 0-3*/ {{sh,11,1,6,128}, {sh,11,1,9,128}, {sh,11,1,7,86}}, {{sh,11,1,6,128}, {sh,11,1,9,128}, {sh,11,1,7,86}}, {{sh,11,1,6,128}, {sh,11,1,9,128}, {sh,11,1,7,86}}, {{sh,11,1,6,86}, {sh,11,1,9,86}, {sh,11,1,7,86}},
    /*nOLS: 4-7*/ {{sh,11,1,6,128}, {sh,11,1,9,128}, {sh,11,1,7,86}}, {{sh,11,1,6,128}, {sh,11,1,9,128}, {sh,11,1,7,86}}, {{sh,11,1,6,128}, {sh,11,1,9,128}, {sh,11,1,7,86}}, {{sh,11,1,6,128}, {sh,11,1,9,128}, {sh,11,1,7,86}},
    /*nLMS: 0-2*/ {{sh,11,1,6,86}, {sh,11,1,9,86}, {sh,11,1,7,86}}, {{sh,11,1,6,86}, {sh,11,1,9,86}, {sh,11,1,7,86}}, {{sh,11,1,6,86}, {sh,11,1,9,86}, {sh,11,1,7,86}},
    /*nSSM: 0-2*/ {{sh,11,1,6,86}, {sh,11,1,9,86}, {sh,11,1,7,86}}, {{sh,11,1,6,86}, {sh,11,1,9,86}, {sh,11,1,7,86}}, {{sh,11,1,6,86}, {sh,11,1,9,86}, {sh,11,1,7,86}}
  }
{}

void Audio8BitModel::setParam(int info) {
  INJECT_SHARED_bpos
  INJECT_SHARED_blockPos
  if(blockPos == 0 && bpos == 0 ) {
    assert((info & 2) == 0);
    stereo = (info & 1);
    mask = 0;
    wMode = info;
    for( int i = 0; i < nLMS; i++ ) {
      lms[i][0].reset(), lms[i][1].reset();
    }
  }
}

void Audio8BitModel::mix(Mixer &m) {
  INJECT_SHARED_bpos
  INJECT_SHARED_blockPos
  INJECT_SHARED_c1
  if( bpos == 0 ) {
    ch = (stereo) != 0 ? blockPos & 1 : 0;
    const int8_t s = static_cast<int32_t>(((wMode & 4) > 0) ? c1 ^ 128 : c1) - 128;
    const int pCh = ch ^ stereo;
    int i = 0;
    for( errLog = 0; i < nOLS; i++ ) {
      ols[i][pCh].update(s);
      residuals[i][pCh] = s - prd[i][pCh][0];
      const uint32_t absResidual = static_cast<uint32_t>(abs(residuals[i][pCh]));
      mask += mask + static_cast<uint32_t>(absResidual > 4);
      errLog += square(absResidual);
    }
    for( int j = 0; j < nLMS; j++ ) {
      lms[j][pCh].update(s);
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
      ols[1][ch].add(x1(i));
    }
    for( int j = (i = 1); j <= k2; j++, i += 1
            << (static_cast<int>(j > 5) + 
                static_cast<int>(j > 10) + 
                static_cast<int>(j > 17) + 
                static_cast<int>(j > 26) +
                static_cast<int>(j > 37))) {
      ols[2][ch].add(x1(i));
    }
    for( int j = (i = 1); j <= k2; j++, i += 1
            << (static_cast<int>(j > 3) + 
                static_cast<int>(j > 7) + 
                static_cast<int>(j > 14) + 
                static_cast<int>(j > 20) +
                static_cast<int>(j > 33) + 
                static_cast<int>(j > 49))) {
      ols[3][ch].add(x1(i));
    }
    for( int j = (i = 1); j <= k2; j++, i += 1 + 
                static_cast<int>(j > 4) + 
                static_cast<int>(j > 8)) {
      ols[4][ch].add(x1(i));
    }
    for( int j = (i = 1); j <= k1; j++, i += 2 + 
               (static_cast<int>(j > 3) + 
                static_cast<int>(j > 9) + 
                static_cast<int>(j > 19) +
                static_cast<int>(j > 36) + 
                static_cast<int>(j > 61))) {
      ols[5][ch].add(x1(i));
    }
    if( stereo != 0 ) {
      for( i = 1; i <= k1 - k2; i++ ) {
        const double s = static_cast<double>(x2(i));
        ols[2][ch].addFloat(s);
        ols[3][ch].addFloat(s);
        ols[4][ch].addFloat(s);
      }
    }
    k1 = 28;
    k2 = k1 - 6 * stereo;
    for( i = 1; i <= k2; i++ ) {
      const double s = static_cast<double>(x1(i));
      ols[0][ch].addFloat(s);
      ols[6][ch].addFloat(s);
      ols[7][ch].addFloat(s);
    }
    for( ; i <= 96; i++ ) {
      ols[0][ch].add(x1(i));
    }
    if( stereo != 0 ) {
      for( i = 1; i <= k1 - k2; i++ ) {
        const double s = static_cast<double>(x2(i));
        ols[0][ch].addFloat(s);
        ols[6][ch].addFloat(s);
        ols[7][ch].addFloat(s);
      }
      for( ; i <= 32; i++ ) {
        ols[0][ch].add(x2(i));
      }
    } else {
      for( ; i <= 128; i++ ) {
        ols[0][ch].add(x1(i));
      }
    }

    for( i = 0; i < nOLS; i++ ) {
      prd[i][ch][0] = signedClip8(static_cast<int>(floor(ols[i][ch].predict())));
    }
    for( ; i < nOLS + nLMS; i++ ) {
      prd[i][ch][0] = signedClip8(static_cast<int>(floor(lms[i - nOLS][ch].predict(s))));
    }
    prd[i++][ch][0] = signedClip8(x1(1) * 2 - x1(2));
    prd[i++][ch][0] = signedClip8(x1(1) * 3 - x1(2) * 3 + x1(3));
    prd[i][ch][0] = signedClip8(x1(1) * 4 - x1(2) * 6 + x1(3) * 4 - x1(4));
    for( i = 0; i < nSSM; i++ ) {
      prd[i][ch][1] = signedClip8(prd[i][ch][0] + residuals[i][pCh]);
    }
  }
  INJECT_SHARED_c0
  const int8_t b = c0 << (8 - bpos);
  for( int i = 0; i < nSSM; i++ ) {
    const uint32_t ctx = (prd[i][ch][0] - b) * 8 + bpos;
    sMap1B[i][0].set(ctx);
    sMap1B[i][1].set(ctx);
    sMap1B[i][2].set((prd[i][ch][1] - b) * 8 + bpos);
    sMap1B[i][0].mix(m);
    sMap1B[i][1].mix(m);
    sMap1B[i][2].mix(m);
  }
  m.set((errLog << 8) | c0, 4096);
  m.set((uint8_t(mask) << 3) | (ch << 2) | (bpos >> 1), 2048);
  m.set((mxCtx << 7) | (c1 >> 1), 1280);
  m.set((errLog << 4) | (ch << 3) | bpos, 256);
  m.set(mxCtx, 10);
}
