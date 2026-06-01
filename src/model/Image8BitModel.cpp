#include <cmath> // round

#include "Image8BitModel.hpp"

Image8BitModel::Image8BitModel(Shared* const sh, const uint64_t size) :
  shared(sh),
  cm(sh, size, nCM, 64),
  map{ /* StationaryMap : BitsOfContext, InputBits, Scale=64, Rate=16  */
    /*nSM0: 0- 1*/ {sh, 0,8}, {sh,15,1},
  },
  mapR1{ sh, nRM, 1 << 5, 74 }, /* ResidualMap: numContexts, histogramsPerContext, scale=64 */
  mapR2{ sh, nRM, 1 << 3, 74 }, /* ResidualMap: numContexts, histogramsPerContext, scale=64 */
  mapR3{ sh, nRM, 1 << 5, 74 }, /* ResidualMap: numContexts, histogramsPerContext, scale=64 */
  mapOLS1{ sh, nOLS, 1 << 5, 74 },  /* ResidualMap: numContexts, histogramsPerContext, scale=64 */
  mapOLS2{ sh, nOLS, 1 << 3, 74 },  /* ResidualMap: numContexts, histogramsPerContext, scale=64 */
  pltMap{   /* SmallStationaryContextMap: BitsOfContext, InputBits, Rate, Scale */
    {sh,11,1,7,64}, {sh,11,1,7,64}, {sh,11,1,7,64}, {sh,11,1,7,64}
  },
  sceneMap{ /* IndirectMap: BitsOfContext, InputBits, Scale, Limit */
    {sh,8,8,64,255}, {sh,8,8,64,255}, {sh,22,1,64,255}, {sh,11,1,64,255}, {sh,11,1,64,255}
  },
  iCtx{     /* IndirectContext<U8>: BitsPerContext, InputBits */
    {16,8}, {16,8}, {16,8}, {16,8}
  } {
  for (int i = 0; i < nOLS; i++) {
    ols[i] = create_OLS_float(sh->chosenSimd, num[i], 1, lambda[i], nu);
  }

  sceneOls = create_OLS_float(sh->chosenSimd, 13, 1, 0.994f, nu);
}

ALWAYS_INLINE uint8_t Image8BitModel::Ls(int relX, int relY) const {
  if (line - relY < 0)  //the lossBuf buffer is pre-filled with 255 initially, so this check is not really necessary
    return 255;
  if (x - relX < 0)
    return 255;
  if (x - relX >= w)
    return 255;
  int offset = relY * w + relX;
  const uint32_t valuesPerByte = 1;
  return lossBuf((offset - 1) * valuesPerByte + 1);
}

ALWAYS_INLINE uint8_t Image8BitModel::GetPredErr(const uint32_t ctxIndex, int relX, int relY) const {
  if (line - relY < 0)  //the predErrBuf buffer is pre-filled with 255 initially, so this check is not really necessary
    return 255;
  if (x - relX < 0)
    return 255;
  if (x - relX >= w)
    return 255;
  int offset = relY * w + relX;
  const uint32_t valuesPerByte = (nRM + nOLS);
  return predErrBuf((offset - 1) * valuesPerByte + ctxIndex + 1);
}

ALWAYS_INLINE uint32_t Image8BitModel::GetPredErrAvg(const uint32_t predictorIndex) const {
  // Current pixel's prediction confidence is based on the already known error at neighboring pixel predictions
  uint32_t predErrW = GetPredErr(predictorIndex, 1, 0);
  uint32_t predErrN = GetPredErr(predictorIndex, 0, 1);
  uint32_t predErrNW = GetPredErr(predictorIndex, 1, 1);
  uint32_t predErrNE = GetPredErr(predictorIndex, -1, 1);
  uint32_t predErrWW = GetPredErr(predictorIndex, 2, 0);
  uint32_t predErrNN = GetPredErr(predictorIndex, 0, 2);
  uint32_t predErrAvg = (2 * predErrW + 2 * predErrN + predErrNE + predErrNW + predErrWW + predErrNN) >> 3; // 0..255
  return predErrAvg;
}

ALWAYS_INLINE static int avg(int x, int y) {
  return (x + y + 1) >> 1;  //note: rounding here works properly only when x+y is non-negative, but we don't really need the function to be aware of negative values as they are rare
}

// Stores a prediction with a spread signal based on the distance between two reference pixels.
// ref1 and ref2 are spatial reference pixels; their difference measures local variation -
// higher difference = less confident = used as a context component in the probability map.
// Use when the prediction value is computed externally (e.g. a complex algebraic expression)
// but a natural pair of reference pixels still exists to supply the spread signal.
ALWAYS_INLINE void Image8BitModel::MakePrediction(int i, uint8_t ref1, uint8_t ref2, int prediction) {
  uint32_t absdiff = rabs(ref1, ref2);
  predictions[i] = absdiff << 16 | ((prediction) & 65535);
}

// Trend extrapolation: continues the gradient observed from pxFar to px,
// starting from the spatial origin.
// prediction = origin + (px - pxFar)
// spread = rabs(px - pxFar): larger gradient = more uncertain prediction.
// Use when pixels are expected to follow a consistent directional trend
// (e.g. continuing a horizontal, vertical, or diagonal gradient).
ALWAYS_INLINE void Image8BitModel::MakePredictionTrend(int i, int px, int pxFar, int origin) {
  uint32_t absdiff = rabs(px, pxFar);
  int prediction = origin + px - pxFar;
  predictions[i] = absdiff << 16 | (prediction & 65535);
}

// Smoothed trend: applies the gradient at half strength instead of full extrapolation.
// prediction = avg(origin, origin + (px - pxFar))
// spread = rabs(px - pxFar): same signal as Trend.
// Use in smoother regions where a full trend extrapolation would overshoot,
// or when the gradient is expected to taper rather than continue linearly.
ALWAYS_INLINE void Image8BitModel::MakePredictionSmooth(int i, int px, int pxFar, int origin) {
  uint32_t absdiff = rabs(px, pxFar);
  int prediction = avg(origin, origin + px - pxFar);
  predictions[i] = absdiff << 16 | (prediction & 65535);
}

ALWAYS_INLINE void Image8BitModel::MakePredictionAvg(int i, int px1, int px2) {
  uint32_t absdiff = rabs(px1, px2);
  int prediction = avg(px1, px2);
  predictions[i] = absdiff << 16 | (prediction & 65535);
}

// Stores only a prediction value; the spread signal is absent (upper 16 bits are zero).
// Use for complex algebraic combinations of multiple pixels where no single
// natural pair of reference pixels exists to derive a meaningful spread signal.
ALWAYS_INLINE void Image8BitModel::MakePredictionC(int i, int prediction) {
  predictions[i] = prediction & 65535;
}

void Image8BitModel::init(int pos) {
  x = line = jump = 0;
  columns[0] = max(1, w / max(1, ilog2(w) * 2));
  columns[1] = max(1, columns[0] / max(1, ilog2(columns[0])));
  if (isGray) {
    if (lastPos != 0 && false) { // todo: when shall we reset ?
      for (int i = 0; i < nSM; i++) {
        map[i].reset();
      }
    }
  }
  else if (frameWidth != w) {
    for (int i = 0; i < nPltMaps; i++) {
      iCtx[i].reset();
      pltMap[i].reset();
    }
  }
  prevFramePos = framePos;
  framePos = pos;
  prevFrameWidth = frameWidth;
  frameWidth = w;

  lossBuf.setSize(nextPowerOf2(LOSS_BUF_ROWS * w));
  lossBuf.fill(255);

  predErrBuf.setSize(nextPowerOf2(PRED_ERR_BUF_ROWS * w * (nRM + nOLS)));
  predErrBuf.fill(255);
}

void Image8BitModel::setParam(int width, uint32_t isGray) {
  this->w = width;
  this->isGray = isGray;
}

void Image8BitModel::mix(Mixer& m) {

  loss += shared->State.loss; // += 0..1023

  INJECT_SHARED_bpos
  if (bpos == 0) {
    INJECT_SHARED_pos
    if (pos != lastPos + 1) {
      init(pos);
    }
    else {
      x++;
      if (x >= w) {
        x = 0;
        line++;
      }
    }
    lastPos = pos;

    INJECT_SHARED_buf
    INJECT_SHARED_c1
    if (x == 0) {
      memset(&jumps[0], 0, sizeof(short) * jumps.size());
      if (line > 0 && w > 8) {
        uint8_t bMask = 0xFF - ((1 << isGray) - 1);
        uint32_t pMask = bMask * 0x01010101;
        uint32_t left = 0;
        uint32_t right = 0;
        int l = min(w, static_cast<int>(jumps.size()));
        int end = l - 4;
        do {
          left = ((buf(l - x) << 24) | (buf(l - x - 1) << 16) | (buf(l - x - 2) << 8) | buf(l - x - 3)) & pMask;
          int i = end;
          while (i >= x + 4) {
            right = ((buf(l - i - 3) << 24) | (buf(l - i - 2) << 16) | (buf(l - i - 1) << 8) | buf(l - i)) & pMask;
            if (left == right) {
              int j = (i + 3 - x - 1) / 2;
              int k = 0;
              for (; k <= j; k++) {
                if (k < 4 || (buf(l - x - k) & bMask) == (buf(l - i - 3 + k) & bMask)) {
                  jumps[x + k] = -(x + (l - i - 3) + 2 * k);
                  jumps[i + 3 - k] = i + 3 - x - 2 * k;
                }
                else {
                  break;
                }
              }
              x += k;
              end -= k;
              break;
            }
            i--;
          }
          x++;
          if (x > end) {
            break;
          }
        } while (x + 4 < l);
        x = 0;
      }
    }

    column[0] = x / columns[0];
    column[1] = x / columns[1];

    WWWWW = buf(5);
    WWWW = buf(4);
    WWW = buf(3);
    WW = buf(2);
    W = buf(1);
    NWWWW = buf(w + 4);
    NWWW = buf(w + 3);
    NWW = buf(w + 2);
    NW = buf(w + 1);
    N = buf(w);
    NE = buf(w - 1);
    NEE = buf(w - 2);
    NEEE = buf(w - 3);
    NEEEE = buf(w - 4);
    NNWWW = buf(w * 2 + 3);
    NNWW = buf(w * 2 + 2);
    NNW = buf(w * 2 + 1);
    NN = buf(w * 2);
    NNE = buf(w * 2 - 1);
    NNEE = buf(w * 2 - 2);
    NNEEE = buf(w * 2 - 3);
    NNNWW = buf(w * 3 + 2);
    NNNW = buf(w * 3 + 1);
    NNN = buf(w * 3);
    NNNE = buf(w * 3 - 1);
    NNNEE = buf(w * 3 - 2);
    NNNNW = buf(w * 4 + 1);
    NNNN = buf(w * 4);
    NNNNE = buf(w * 4 - 1);
    NNNNN = buf(w * 5);
    NNNNNN = buf(w * 6);

    uint8_t WWWWWW = buf(6);
    uint8_t NNWWWW = buf(2 * w + 4);
    uint8_t NNNWWW = buf(3 * w + 3);
    uint8_t NNNWWWW = buf(3 * w + 4);
    uint8_t NNNNWW = buf(4 * w + 2);
    uint8_t NNNNWWW = buf(4 * w + 3);
    uint8_t NNNNWWWW = buf(4 * w + 4);
    uint8_t NEEEEE = buf(1 * w - 5);
    uint8_t NEEEEEE = buf(1 * w - 6);
    uint8_t NNEEEE = buf(2 * w - 4);
    uint8_t NNNEEE = buf(3 * w - 3);
    uint8_t NNNEEEE = buf(3 * w - 4);
    uint8_t NNNNEEE = buf(4 * w - 3);
    uint8_t NNNNEE = buf(4 * w - 2);
    uint8_t NNNNEEEE = buf(4 * w - 4);
    uint8_t NNNNNNEE = buf(6 * w - 2);
    uint8_t NNNWWWWW = buf(3 * w + 5);
    uint8_t NEEEEEEE = buf(1 * w - 7);

    if (prevFramePos > 0 && prevFrameWidth == w) {
      int offset = prevFramePos + line * w + x;
      prvFrmPx = buf[offset];
      if (isGray != 0) {
        sceneOls->update((float)W);
        sceneOls->add((float)W);
        sceneOls->add((float)NW);
        sceneOls->add((float)N);
        sceneOls->add((float)NE);
        for (int i = -1; i < 2; i++) {
          for (int j = -1; j < 2; j++) {
            sceneOls->add((float)buf[offset + i * w + j]);
          }
        }
        float prediction = sceneOls->predict();
        prvFrmPrediction = clip(int(roundf(prediction)));
      }
      else {
        prvFrmPrediction = W;
      }
    }
    else {
      prvFrmPx = prvFrmPrediction = W;
    }
    sceneMap[0].setDirect(prvFrmPx);
    sceneMap[1].setDirect(prvFrmPrediction);

    jump = jumps[min(x, static_cast<int>(jumps.size()) - 1)];
    uint64_t i = isGray * 1024;
    const uint8_t R_ = CM_USE_RUN_STATS;
    cm.set(R_, hash(++i, (jump != 0) ? (0x100 | buf(abs(jump))) * (1 - 2 * static_cast<int>(jump < 0)) : N, line & 3));
    if (!isGray) {
      for (int j = 0; j < nPltMaps; j++) {
        iCtx[j] += W;
      }
      iCtx[0] = W | (NE << 8);
      iCtx[1] = W | (N << 8);
      iCtx[2] = W | (WW << 8);
      iCtx[3] = N | (NN << 8);
//    cm.set(R_, hash(++i, W)); // ineffective, covered by NormalModel
      cm.set(R_, hash(++i, W, column[0]));
      cm.set(R_, hash(++i, N));
      cm.set(R_, hash(++i, N, column[0]));
      cm.set(R_, hash(++i, NW));
      cm.set(R_, hash(++i, NW, column[0]));
      cm.set(R_, hash(++i, NE));
      cm.set(R_, hash(++i, NE, column[0]));
      cm.set(R_, hash(++i, NWW));
      cm.set(R_, hash(++i, NEE));
      cm.set(R_, hash(++i, WW));
      cm.set(R_, hash(++i, NN));
      cm.set(R_, hash(++i, W, N));
      cm.set(R_, hash(++i, W, NW));
      cm.set(R_, hash(++i, W, NE));
      cm.set(R_, hash(++i, W, NEE));
      cm.set(R_, hash(++i, W, NWW));
      cm.set(R_, hash(++i, N, NW));
      cm.set(R_, hash(++i, N, NE));
      cm.set(R_, hash(++i, NW, NE));
//    cm.set(R_, hash(++i, W, WW)); // ineffective, covered by NormalModel
      cm.set(R_, hash(++i, N, NN));
      cm.set(R_, hash(++i, NW, NNWW));
      cm.set(R_, hash(++i, NE, NNEE));
      cm.set(R_, hash(++i, NW, NWW));
      cm.set(R_, hash(++i, NW, NNW));
      cm.set(R_, hash(++i, NE, NEE));
      cm.set(R_, hash(++i, NE, NNE));
      cm.set(R_, hash(++i, N, NNW));
      cm.set(R_, hash(++i, N, NNE));
      cm.set(R_, hash(++i, N, NNN));
      cm.set(R_, hash(++i, W, WWW));
      cm.set(R_, hash(++i, WW, NEE));
      cm.set(R_, hash(++i, WW, NN));
      cm.set(R_, hash(++i, W, NEEE));
      cm.set(R_, hash(++i, W, NEEEE));
      cm.set(R_, hash(++i, W, N, NW));
      cm.set(R_, hash(++i, N, NN, NNN));
      cm.set(R_, hash(++i, W, NE, NEE));
      cm.set(R_, hash(++i, W, NW, N, NE));
      cm.set(R_, hash(++i, N, NE, NN, NNE));
      cm.set(R_, hash(++i, N, NW, NNW, NN));
      cm.set(R_, hash(++i, W, WW, NWW, NW));
      cm.set(R_, hash(++i, W, NW << 8 | N, WW << 8 | NWW));
      cm.set(R_, hash(++i, column[0]));
      cm.set(R_, hash(++i, N, column[1]));
      cm.set(R_, hash(++i, W, column[1]));
      for (int j = 0; j < nPltMaps; j++) {
        cm.set(R_, hash(++i, iCtx[j]()));
      }

      ctx = min(0x1F, x / min(0x20, columns[0]));
      res = W;
    }
    else { // gray

      lossBuf.add(static_cast<uint8_t>(min(loss >> 2, 255)));  // 0..255
      loss = 0;

      //what was the total cost at the neighboring pixes
      lossQ = //0 x 6 .. 255 x 6 = 0 .. 1530
        Ls(1, 0) + // W
        Ls(0, 1) + // N
        Ls(2, 0) + // WW
        Ls(0, 2) + // NN
        Ls(1, 1) + // NW
        Ls(-1, 1); // NE

      // let's trim the higher part (640-1530) - it is almost always completely empty OR such high values indicate that we are off frame
      // cap at 639 = 16*40 - 1, so lossQ4 = lossQ/40 fits in [0..15]
      lossQ = min(lossQ, 639);
      shared->State.Image.lossQ = lossQ; //0..639

      for (int i = nRM + nOLS - 1; i >= 0; i--) {
        short prediction = predictions[i] & 65535;
        uint8_t err;
        if (prediction == INT16_MAX) // currently never happens, todo
          err = 255;
        else
          err = rabs(c1, prediction); // 0..128
        predErrBuf.add(err); //0..63 or 255
      }
      int contextIdx = 0;



      MakePredictionC(contextIdx++, (N * 3 + W * 3 - NN - WW + 2) >> 2);
      MakePredictionAvg(contextIdx++, W, NEE);

      MakePredictionTrend(contextIdx++, W, NW, N); // very strong
      MakePredictionTrend(contextIdx++, WW, NNWW, NN); //strong?
      MakePredictionTrend(contextIdx++, WWW, NNNWWW, NNN); //strong?
      MakePredictionTrend(contextIdx++, W, N, NE); // strong
      MakePredictionTrend(contextIdx++, N, NNW, NW);
      MakePredictionTrend(contextIdx++, N, NNE, NE);
      MakePredictionTrend(contextIdx++, NN, NNNNEE, NNEE);
      MakePredictionTrend(contextIdx++, N, NNN, NN);
      MakePredictionTrend(contextIdx++, NN, NNNNNN, NNNN);
      MakePredictionTrend(contextIdx++, W, WWW, WW); // strong
      MakePredictionTrend(contextIdx++, WW, WWWWWW, WWWW);
      MakePredictionTrend(contextIdx++, W, NE, NEE);
      MakePredictionTrend(contextIdx++, WW, NNEE, NNEEEE); // strong
      MakePredictionTrend(contextIdx++, NW, NWW, W); // very strong
      MakePredictionTrend(contextIdx++, NNWW, NNWWWW, WW);
      MakePredictionTrend(contextIdx++, NN, NNW, W);
      MakePredictionTrend(contextIdx++, NNNN, NNNNNNEE, NNEE);
      MakePredictionTrend(contextIdx++, NE, NN, NW); // strong

      MakePredictionSmooth(contextIdx++, W, NNE, NE); // very strong
      MakePredictionTrend(contextIdx++, NNE, NNNE, NE);
      MakePredictionTrend(contextIdx++, NEE, NNEEE, NEE); // strong
      MakePredictionTrend(contextIdx++, NEEE, NNNEEE, NEEE);
      MakePredictionTrend(contextIdx++, NNE, NN, W);  // strong
      MakePredictionTrend(contextIdx++, NNW, NNWW, W);  // strong

      MakePredictionC(contextIdx++, (N * 3 - NN * 3 + NNN));
      MakePredictionC(contextIdx++, (W * 3 - WW * 3 + WWW));
      MakePredictionC(contextIdx++, clamp4(N * 3 - NN * 3 + NNN, N, W, NE, NW)); // very strong
      MakePredictionC(contextIdx++, clamp4(W * 3 - WW * 3 + WWW, N, W, NE, NW)); // strong

      MakePredictionC(contextIdx++, ((15 * N - 20 * NN + 15 * NNN - 6 * NNNN + NNNNN + clamp4(4 * W - 6 * NWW + 4 * NNWWW - NNNWWWW, W, NW, N, NN)) / 6)); // very strong
      MakePredictionC(contextIdx++, ((6 * NE - 4 * NNEE + NNNEEE + (4 * W - 6 * NW + 4 * NNW - NNNW)) / 4));
      MakePredictionC(contextIdx++, (((N + 3 * NW) / 4) * 3 - avg(NNW, NNWW) * 3 + (NNNWW * 3 + NNNWWW) / 4));
      MakePredictionC(contextIdx++, ((W * 2 + NW) - (WW + 2 * NWW) + NWWW)); // strong
      MakePredictionAvg(contextIdx++, NEEEE, NEEEEEE); // strong
      MakePredictionAvg(contextIdx++, WWWW, WWWWWW);
      MakePredictionAvg(contextIdx++, NNNN, NNNNNN);// strong
      MakePredictionAvg(contextIdx++, NNNNWW, NNWW);
      MakePredictionAvg(contextIdx++, NNNNEE, NNEE);
      MakePredictionAvg(contextIdx++, WWW, WWWWWW);
      MakePredictionC(contextIdx++, W);
      MakePredictionC(contextIdx++, N);  // strong
      MakePredictionC(contextIdx++, NN);

      MakePredictionTrend(contextIdx++, W, WW, W); // strong
      MakePredictionTrend(contextIdx++, WW, WWWW, WW); // strong?
      MakePredictionTrend(contextIdx++, N, NN, N); // strong
      MakePredictionTrend(contextIdx++, NN, NNNN, NN);
      MakePredictionTrend(contextIdx++, NW, NNWW, NW);
      MakePredictionTrend(contextIdx++, NNWW, NNNNWWWW, NNWW);
      MakePredictionTrend(contextIdx++, NE, NNEE, NE);
      MakePredictionTrend(contextIdx++, NNEEE, NNNNEEEE, NNEEE);

      //MakePredictionTrend(contextIdx++, NN, NNNW, NW);
      MakePredictionTrend(contextIdx++, W, NEE, NEEE);
      //MakePredictionTrend(contextIdx++, NEE, NNEEE, NE);
      MakePredictionTrend(contextIdx++, NWW, NWWWW, WW);
      //MakePredictionC(contextIdx++, ((W + NW) * 3 - NWW * 6 + NWWW + NNWWW) / 2);
      //MakePredictionTrend(contextIdx++, (NE * 2 + NNE), (NNEE + NNNEE * 2), NNNNEEE);
      MakePredictionC(contextIdx++, (W + N + NEEEEE + NEEEEEEE) / 4);
      MakePredictionTrend(contextIdx++, WW, N, NEE);
      //MakePredictionAvg(contextIdx++, N, NNN);
      MakePredictionTrend(contextIdx++, N, NNWW, NWW); //strong
      //MakePredictionTrend(contextIdx++, (N * 2 + NE), (NN + 2 * NNE), NNNE);
      //MakePredictionTrend(contextIdx++, (NW * 2 + NNW), (NNWW + NNNWW * 2), NNNNWWW);
      //MakePredictionTrend(contextIdx++, (N * 2 + NW), (NN + 2 * NNW), NNNW);

      MakePredictionAvg(contextIdx++, 2 * N - NN, 2 * W - WW);
      MakePredictionAvg(contextIdx++, 2 * W - WW, 2 * NW - NNWW);

      MakePredictionC(contextIdx++, paeth(W, N, NW));
      MakePredictionC(contextIdx++, gap(W, N, NW, NE, WW, NNE, NN));

      // 3rd-order horizontal + 4th-order NE diagonal
      MakePredictionC(contextIdx++, (6 * W - 4 * WW + WWW + 4 * NE - 6 * NNE + 4 * NNNE - NNNNE) / 4);
      // average of 1st- through 4th-order horizontal + 4th-order NE diagonal
      MakePredictionC(contextIdx++, (10 * W - 10 * WW + 5 * WWW - WWWW + 4 * NE - 6 * NNE + 4 * NNNE - NNNNE) / 5);
      // 2nd-order H + weighted blend of 3rd-order NE and 3rd-order NEEE diagonals
      MakePredictionC(contextIdx++, (15 * W - 4 * WW + 10 * (3 * NE - 3 * NNE + NNNE) - (3 * NEEE - 3 * NNEEE + NNNEEE)) / 20); //strong
      // 2× 1st-order + 3× 2nd-order horizontal + 3rd-order NEE diagonal
      MakePredictionC(contextIdx++, (8 * W - 3 * WW + (3 * NEE - 3 * NNEE + NNNEE)) / 6); // very strong

      // 2nd-order H + 2nd-order V, interaction-corrected
      MakePredictionC(contextIdx++, (2 * W - WW) + (2 * N - NN) - (2 * NW - NNWW));

      // 3rd-order vertical + 4th-order NE diagonal
      MakePredictionC(contextIdx++, (6 * N - 4 * NN + NNN + 4 * NE - 6 * NNE + 4 * NNNE - NNNNE) / 4);
      // 4th-order vertical + 4th-order NE diagonal
      MakePredictionC(contextIdx++, ((10 * N - 10 * NN + 5 * NNN - NNNN) + (4 * NE - 6 * NNE + 4 * NNNE - NNNNE)) / 5);

      // 3rd-order vertical + 4th-order NW diagonal
      MakePredictionC(contextIdx++, (6 * N - 4 * NN + NNN + 4 * NW - 6 * NNW + 4 * NNNW - NNNNW) / 4);
      // 4th-order vertical + 4th-order NW diagonal
      MakePredictionC(contextIdx++, ((10 * N - 10 * NN + 5 * NNN - NNNN) + (4 * NW - 6 * NNW + 4 * NNNW - NNNNW)) / 5);

      // 3rd-order NE diagonal alone
      MakePredictionC(contextIdx++, (6 * NE - 4 * NNEE + NNNEEE) / 3);

      // 3rd-order H + 3rd-order V, interaction-corrected
      MakePredictionC(contextIdx++, ((6 * W - 4 * WW + WWW) + (6 * N - 4 * NN + NNN) - (6 * NW - 4 * NNWW + NNNWWW)) / 3);

      // 5th-order horizontal
      MakePredictionC(contextIdx++, (15 * W - 20 * WW + 15 * WWW - 6 * WWWW + WWWWW) / 5); // strong

      // symmetric NE+NW blend, N-corrected
      MakePredictionC(contextIdx++, (2 * NE - NNEE) + (2 * NW - NNWW) - (2 * N - NN));

      MakePredictionC(contextIdx++, 0);

      assert(contextIdx == nRM);

      //quality metric: quantized past loss in the pixel neighborhood 
      lossQ4 = min(lossQ / 40u, 7); //0..7 (3 bits)

      for (int i = 0; i < nRM; i++) {
        uint32_t spread = predictions[i] >> 16;
        short prediction = predictions[i] & 65535;
        if (prediction == INT16_MAX) { // currently never happens, todo
          mapR1.skip();
          mapR2.skip();
          mapR3.skip();
        }
        else {
          uint32_t predErrAvg = GetPredErrAvg(i);
          mapR1.set(prediction, min(predErrAvg, 31)); // 0..31 (5 bits)
          mapR2.set(prediction, lossQ4); // 0..7 (3 bits)
          mapR3.set(prediction, min(spread, 31)); // 5 bits
        }
      }

      for (int j = 0; j < nOLS; j++) {
        auto ols_j = ols[j].get();
        ols_j->update((float)W);
        auto ols_ctx_j = olsCtxs[j];
        for (int ctx_idx = 0; ctx_idx < num[j]; ctx_idx++) {
          float val = *ols_ctx_j[ctx_idx];
          ols_j->add(val);
        }

        float pred = ols_j->predict();
        short prediction = short(roundf(pred));
        predictions[nRM + j] = clip(prediction);
        uint32_t predErrAvg = GetPredErrAvg(nRM + j);
        mapOLS1.set(prediction, min(predErrAvg, 31));
        mapOLS2.set(prediction, lossQ4);
      }

      //cm.set(R_, hash(++i, W)); // ineffective, covered by NormalModel
      cm.set(R_, hash(++i, N)); 
      cm.set(R_, hash(++i, NW));
      cm.set(R_, hash(++i, NE));

      cm.set(R_, hash(++i, N, NN));
      cm.set(R_, hash(++i, NE, NNEE));
      cm.set(R_, hash(++i, NW, NNWW));
      cm.set(R_, hash(++i, W, NEE));

      cm.set(R_, hash(++i, N, NN, NNN));
      //cm.set(R_, hash(++i, W, WW, WWW)); // ineffective, covered by NormalModel

      cm.set(R_, hash(++i, W, WW, N, NN));
      cm.set(R_, hash(++i, W, N, NE, NW));

      cm.set(R_, hash(++i, (NNN + N + 4) >> 3, (N * 3 - NN * 3 + NNN) >> 1));

      cm.set(R_, hash(++i, (N * 2 - NN) >> 1, DiffQt(N, (NN * 2 - NNN))));
      cm.set(R_, hash(++i, (W * 2 - WW) >> 1, DiffQt(W, (WW * 2 - WWW))));
      cm.set(R_, hash(++i, (N * 2 - NN), DiffQt(W, (NW * 2 - NNW))));
      cm.set(R_, hash(++i, (W * 2 - WW), DiffQt(N, (NW * 2 - NWW))));
      cm.set(R_, hash(++i, (W + NEE + 1) >> 1, DiffQt(W, (WW + NE + 1) >> 1)));
      cm.set(R_, hash(++i, (W + NEE - NE), DiffQt(W, (WW + NE - N))));

      int div7 = max((x) >> 10, 7);
      int div17 = max((x) >> 9, 17);
      int div29 = max((x) >> 8, 29);

      cm.set(R_, hash(++i, w, x / div7));
      cm.set(R_, hash(++i, w, x / div17, line / 17));
      cm.set(R_, hash(++i, w, x / div29, uint8_t(W + N - NW) >> 1));

      cm.set(R_, hash(++i, w, W >> 2, NE >> 2, x / div29));

      ctx = min(0x1F, x / max(1, w / min(32, columns[0]))) |
        (((static_cast<int>(abs(W - N) * 16 > W + N) << 1) | static_cast<int>(abs(N - NW) > 8)) << 5) | ((W + N) & 0x180);

      res = clamp4(W + N - NW, W, NW, N, NE);
    }

    assert(i - isGray * 1024 <= nCM);

    shared->State.Image.pixels.W = W;
    shared->State.Image.pixels.N = N;
    shared->State.Image.pixels.NN = NN;
    shared->State.Image.pixels.WW = WW;
    shared->State.Image.ctx = ctx >> isGray;
  }
  INJECT_SHARED_c0
  uint8_t b = (c0 << (8 - bpos));
  if (isGray) {
    int i = 0;
    map[i++].set(0);
    map[i++].set(((static_cast<uint8_t>(clip(W + N - NW) - b)) * 8 + bpos) |
      (DiffQt(clip(N + NE - NNE), clip(N + NW - NNW)) << 11));
  }
  sceneMap[2].setDirect(finalize64(hash(x, line), 19) * 8 + bpos);
  sceneMap[3].setDirect((prvFrmPx - b) * 8 + bpos);
  sceneMap[4].setDirect((prvFrmPrediction - b) * 8 + bpos);

  // predict next bit
  cm.mix(m);
  if (isGray) {
    for (int i = 0; i < nSM; i++) {
      map[i].mix(m);
    }
    mapR1.mix(m);
    mapR2.mix(m);
    mapR3.mix(m);
    mapOLS1.mix(m);
    mapOLS2.mix(m);
  }
  else {
    for (int i = 0; i < nPltMaps; i++) {
      pltMap[i].set((bpos << 8) | iCtx[i]());
      pltMap[i].mix(m);
    }
  }
  for (int i = 0; i < nIM; i++) {
    const int scale = (prevFramePos > 0 && prevFrameWidth == w) ? 64 : 0;
    sceneMap[i].setScale(scale);
    sceneMap[i].mix(m);
  }

  col = (col + 1) & 7;
  m.set(ctx, 512);
  m.set(col << 1 | static_cast<int>(c0 == ((0x100 | res) >> (8 - bpos))), 16);
  m.set((N + W) >> 4, 32);
  m.set(c0 - 1, 255);
  m.set((static_cast<int>(abs(W - N) > 4) << 9) | (static_cast<int>(abs(N - NE) > 4) << 8) |
    (static_cast<int>(abs(W - NW) > 4) << 7) | (static_cast<int>(W > N) << 6) | (static_cast<int>(N > NE) << 5) |
    (static_cast<int>(W > NW) << 4) | (static_cast<int>(W > WW) << 3) | (static_cast<int>(N > NN) << 2) |
    (static_cast<int>(NW > NNWW) << 1) | static_cast<int>(NE > NNEE), 1024);
  m.set(min(63, column[0]), 64);
  m.set(min(127, column[1]), 128);
  m.set(min(255, (x + line) / 32), 256);
}
