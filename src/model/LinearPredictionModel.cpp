#include "LinearPredictionModel.hpp"

LinearPredictionModel::LinearPredictionModel(const Shared* const sh) : shared(sh),
  sMap {{sh, 11, 1, 6, 128},
        {sh, 11, 1, 6, 128},
        {sh, 11, 1, 6, 128},
        {sh, 11, 1, 6, 128},
        {sh, 11, 1, 6, 128}
  }
{}

void LinearPredictionModel::mix(Mixer &m) {
  INJECT_SHARED_bpos
  if( bpos == 0 ) {
    INJECT_SHARED_buf
    const uint8_t W = buf(1);
    const uint8_t WW = buf(2);
    const uint8_t WWW = buf(3);
    int i = 0;
    for( ; i < nOLS; i++ ) {
      ols[i].update(W);
    }
    for( i = 1; i <= 32; i++ ) {
      ols[0].add(buf(i));
      ols[1].add(buf(i * 2 - 1));
      ols[2].add(buf(i * 2));
    }
    for( i = 0; i < nOLS; i++ ) {
      prd[i] = clip(static_cast<int>(floor(ols[i].predict())));
    }
    prd[i++] = clip(W * 2 - WW);
    prd[i] = clip(W * 3 - WW * 3 + WWW);
  }
  INJECT_SHARED_c0
  const uint8_t b = c0 << (8 - bpos);
  for( int i = 0; i < nSSM; i++ ) {
    sMap[i].set((prd[i] - b) * 8 + bpos);
    sMap[i].mix(m);
  }
}
