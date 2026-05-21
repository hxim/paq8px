#include "LinearPredictionModel.hpp"

LinearPredictionModel::LinearPredictionModel(const Shared* const sh) : shared(sh),
  mapR { sh, nRM, 32, 128 } /* ResidualMap: numContexts, histogramsPerContext, scale=64 */
{
  for (int i = 0; i < nOLS; i++) {
    ols[i] = create_OLS_float(sh->chosenSimd, num[i], solveInterval[i], lambda[i], nu);
  }
}

void LinearPredictionModel::mix(Mixer &m) {
  INJECT_SHARED_bpos
  if( bpos == 0 ) {

    //for every byte

    INJECT_SHARED_c1
    for (int i = 0; i < nRM; i++) {
      int prediction = prd[i];
      uint8_t err = rabs(c1, prediction); // 0..128
      predErrBuf[i] = ((predErrBuf[i] * 15) >> 4) + err; // -> 2033 absolute max
    }

    INJECT_SHARED_buf
    const uint8_t W = buf(1);
    const uint8_t WW = buf(2);
    const uint8_t WWW = buf(3);

    int i = 0;
    for( ; i < nOLS; i++ ) {
      ols[i]->update(W);
    }
    for( i = 1; i <= 32; i++ ) {
      ols[0]->add(buf(i)); // for 8-bit values
      ols[1]->add(buf(i * 2)); // for 16-bit values (hi) and 8-bit values with gap
      ols[2]->add(buf(i * 3)); // for rgb images and 8-bit values with gap
    }
    for( i = 0; i < nOLS; i++ ) {
      float prediction = ols[i]->predict();
      prd[i] = (short)roundf(prediction);
    }

    prd[i++] = W * 2 - WW; // for 8-bit values
    prd[i++] = W * 3 - WW * 3 + WWW; // for 8-bit values
    prd[i++] = WW * 2 - buf(4); // for 16-bit values (hi)
    prd[i++] = WWW * 2 - buf(6); // for rgb images

    INJECT_SHARED_blockPos
    for (int i = 0; i < nRM; i++) {
      mapR.set(prd[i], min(predErrBuf[i] >> 4, 15) << 1 | (blockPos & 1));
    }
  }

  // for every bit

  mapR.mix(m);
}
