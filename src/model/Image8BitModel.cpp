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
      cm.set(R_, hash(++i, W));
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
      cm.set(R_, hash(++i, W, WW));
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

      predictions[contextIdx++] = clamp4(W + N - NW, W, NW, N, NE);
      predictions[contextIdx++] = clip(W + N - NW);
      predictions[contextIdx++] = clamp4(W + NE - N, W, NW, N, NE);
      predictions[contextIdx++] = clip(W + NE - N);
      predictions[contextIdx++] = clamp4(N + NW - NNW, W, NW, N, NE);
      predictions[contextIdx++] = clip(N + NW - NNW);
      predictions[contextIdx++] = clamp4(N + NE - NNE, W, N, NE, NEE);
      predictions[contextIdx++] = clip(N + NE - NNE);
      predictions[contextIdx++] = (W + NEE) / 2;
      predictions[contextIdx++] = clip(N * 3 - NN * 3 + NNN);
      predictions[contextIdx++] = clip(W * 3 - WW * 3 + WWW);
      predictions[contextIdx++] = (W + clip(NE * 3 - NNE * 3 + NNNE)) / 2;
      predictions[contextIdx++] = (W + clip(NEE * 3 - NNEEE * 3 + NNNEEEE)) / 2;
      predictions[contextIdx++] = clip(NN + NNNN - NNNNNN);
      predictions[contextIdx++] = clip(WW + WWWW - WWWWWW);
      predictions[contextIdx++] = clip((NNNNN - 6 * NNNN + 15 * NNN - 20 * NN + 15 * N + clamp4(W * 2 - NWW, W, NW, N, NN)) / 6);
      predictions[contextIdx++] = clip((-3 * WW + 8 * W + clamp4(NEE * 3 - NNEE * 3 + NNNEE, NE, NEE, NEEE, NEEEE)) / 6);
      predictions[contextIdx++] = clip(NN + NW - NNNW);
      predictions[contextIdx++] = clip(NN + NE - NNNE);
      predictions[contextIdx++] = clip((W * 2 + NW) - (WW + 2 * NWW) + NWWW);
      predictions[contextIdx++] = clip(((NW + NWW) / 2) * 3 - NNWWW * 3 + (NNNWWWW + NNNWWWWW) / 2);
      predictions[contextIdx++] = clip(NEE + NE - NNEEE);
      predictions[contextIdx++] = clip(NWW + WW - NWWWW);
      predictions[contextIdx++] = clip(((W + NW) * 3 - NWW * 6 + NWWW + NNWWW) / 2);
      predictions[contextIdx++] = clip((NE * 2 + NNE) - (NNEE + NNNEE * 2) + NNNNEEE);
      predictions[contextIdx++] = NNNNNN;
      predictions[contextIdx++] = (NEEEE + NEEEEEE) / 2;
      predictions[contextIdx++] = (WWWW + WWWWWW) / 2;
      predictions[contextIdx++] = (W + N + NEEEEE + NEEEEEEE) / 4;
      predictions[contextIdx++] = clip(NEEE + W - NEE);
      predictions[contextIdx++] = clip(4 * NNN - 3 * NNNN);
      predictions[contextIdx++] = clip(N + NN - NNN);
      predictions[contextIdx++] = clip(W + WW - WWW);
      predictions[contextIdx++] = clip(W + NEE - NE);
      predictions[contextIdx++] = clip(WW + NEE - N);
      predictions[contextIdx++] = (clip(W * 2 - NW) + clip(W * 2 - NWW) + N + NE) / 4;
      predictions[contextIdx++] = clamp4(N * 2 - NN, W, N, NE, NEE);
      predictions[contextIdx++] = (N + NNN) / 2;
      predictions[contextIdx++] = clip(NN + W - NNW);
      predictions[contextIdx++] = clip(NWW + N - NNWW);
      predictions[contextIdx++] = clip((4 * WWW - 15 * WW + 20 * W + clip(NEE * 2 - NNEE)) / 10);
      predictions[contextIdx++] = clip((NNNEEE - 4 * NNEE + 6 * NE + clip(W * 3 - NW * 3 + NNW)) / 4);
      predictions[contextIdx++] = clip((N * 2 + NE) - (NN + 2 * NNE) + NNNE);
      predictions[contextIdx++] = clip((NW * 2 + NNW) - (NNWW + NNNWW * 2) + NNNNWWW);
      predictions[contextIdx++] = clip(NNWW + W - NNWWW);
      predictions[contextIdx++] = clip((-NNNN + 5 * NNN - 10 * NN + 10 * N + clip(W * 4 - NWW * 6 + NNWWW * 4 - NNNWWWW)) / 5);
      predictions[contextIdx++] = clip(NEE + clip(NEEE * 2 - NNEEEE) - NEEEE);
      predictions[contextIdx++] = clip(NW + W - NWW);
      predictions[contextIdx++] = clip((N * 2 + NW) - (NN + 2 * NNW) + NNNW);
      predictions[contextIdx++] = clip(NN + clip(NEE * 2 - NNEEE) - NNE);
      predictions[contextIdx++] = clip((-WWWW + 5 * WWW - 10 * WW + 10 * W + clip(NE * 2 - NNE)) / 5);
      predictions[contextIdx++] = clip((-WWWWW + 4 * WWWW - 5 * WWW + 5 * W + clip(NE * 2 - NNE)) / 4);
      predictions[contextIdx++] = clip((WWW - 4 * WW + 6 * W + clip(NE * 3 - NNE * 3 + NNNE)) / 4);
      predictions[contextIdx++] = clip((-NNEE + 3 * NE + clip(W * 4 - NW * 6 + NNW * 4 - NNNW)) / 3);
      predictions[contextIdx++] = ((W + N) * 3 - NW * 2) / 4;

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

      cm.set(R_, 0);
      cm.set(R_, hash(++i, N));
      cm.set(R_, hash(++i, W));
      cm.set(R_, hash(++i, NW));
      cm.set(R_, hash(++i, NE));
      cm.set(R_, hash(++i, N, NN));
      cm.set(R_, hash(++i, W, WW));
      cm.set(R_, hash(++i, NE, NNEE));
      cm.set(R_, hash(++i, NW, NNWW));
      cm.set(R_, hash(++i, W, NEE));
      cm.set(R_, hash(++i, (clamp4(W + N - NW, W, NW, N, NE)) / 2, DiffQt(clip(N + NE - NNE), clip(N + NW - NNW))));
      cm.set(R_, hash(++i, W / 4, NE / 4, column[0]));
      cm.set(R_, hash(++i, (clip(W * 2 - WW)) / 4, (clip(N * 2 - NN)) / 4));
      cm.set(R_, hash(++i, (clamp4(N + NE - NNE, W, N, NE, NEE)) / 4, column[0]));
      cm.set(R_, hash(++i, (clamp4(N + NW - NNW, W, NW, N, NE)) / 4, column[0]));
      cm.set(R_, hash(++i, (W + NEE) / 4, column[0]));
      cm.set(R_, hash(++i, clip(W + N - NW), column[0]));
      cm.set(R_, hash(++i, clamp4(N * 3 - NN * 3 + NNN, W, N, NN, NE), DiffQt(W, clip(NW * 2 - NNW))));
      cm.set(R_, hash(++i, clamp4(W * 3 - WW * 3 + WWW, W, N, NE, NEE), DiffQt(N, clip(NW * 2 - NWW))));
      cm.set(R_, hash(++i, (W + clamp4(NE * 3 - NNE * 3 + NNNE, W, N, NE, NEE)) / 2, DiffQt(N, (NW + NE) / 2)));
      cm.set(R_, hash(++i, (N + NNN) / 8, clip(N * 3 - NN * 3 + NNN) / 4));
      cm.set(R_, hash(++i, (W + WWW) / 8, clip(W * 3 - WW * 3 + WWW) / 4));
      cm.set(R_, hash(++i, clip((-WWWW + 5 * WWW - 10 * WW + 10 * W + clamp4(NE * 4 - NNE * 6 + NNNE * 4 - NNNNE, N, NE, NEE, NEEE)) / 5)));
      cm.set(R_, hash(++i, clip(N * 2 - NN), DiffQt(N, clip(NN * 2 - NNN))));
      cm.set(R_, hash(++i, clip(W * 2 - WW), DiffQt(NE, clip(N * 2 - NW))));

      ctx = min(0x1F, x / max(1, w / min(32, columns[0]))) |
        (((static_cast<int>(abs(W - N) * 16 > W + N) << 1) | static_cast<int>(abs(N - NW) > 8)) << 5) | ((W + N) & 0x180);

      res = clamp4(W + N - NW, W, NW, N, NE);
    }

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
