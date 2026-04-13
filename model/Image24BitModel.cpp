#include "Image24BitModel.hpp"

Image24BitModel::Image24BitModel(Shared* const sh, const uint64_t size) :
  shared(sh),
  cm(sh, size, nCM, 64),
  mapL{ sh, nLSM, 23, 74 },     /* LargeStationaryMap : Contexts, HashBits, Scale=64, Rate=16 */
  mapR1{ sh, nDM, 1 << 7, 74 }, /* ResidualMap: numContexts, histogramsPerContext, scale=64 */
  mapR2{ sh, nDM, 1 << 5, 74 }, /* ResidualMap: numContexts, histogramsPerContext, scale=64 */
  mapR3{ sh, nDM, 1 << 7, 74 }, /* ResidualMap: numContexts, histogramsPerContext, scale=64 */
  mapOLS1{ sh, nOLS, 1 << 7, 74 },  /* ResidualMap: numContexts, histogramsPerContext, scale=64 */
  mapOLS2{ sh, nOLS, 1 << 5, 74 }   /* ResidualMap: numContexts, histogramsPerContext, scale=64 */
{
  for (int i = 0; i < nOLS; i++) {
    for (int j = 0; j < 4; j++) { // RGBA color components
      ols[i][j] = create_OLS_float(sh->chosenSimd, num[i], 1, lambda[i], nu);
    }
  }
}

// Get pixel value
// Clamp to nearest valid pixel in the same color plane, when none exists, return a fixed fallback
ALWAYS_INLINE uint8_t Image24BitModel::Px(int relX, int relY, int colorShift) const {
  relX *= stride;
  relX += colorShift;
  int x0 = x - relX;
  while (x0 < 0) { //we are over the left side, we need to go right until we arrive to the pixel area
    relX -= stride;
    x0 += stride;
  }
  while (x0 >= w) { //we are over the right side, we need to go left until we arrive to the pixel area
    relX += stride;
    x0 -= stride;
  }
  if (line - relY < 0) //we are over the top, let's navigate to the first row
    relY = line;
  int offset = relY * w + relX;
  if (offset <= 0) {
    if (relY < line) //no left (or right) pixel, but there is still room above
      offset += w; //go above
    else
      return 127; //no valid neighbor pixel found
  }
  INJECT_SHARED_buf
  return buf(offset);
}

ALWAYS_INLINE uint8_t Image24BitModel::Ls(int relX, int relY) const {
  if (line - relY < 0)  //the lossBuf buffer is pre-filled with 255 initially, so this check is not really necessary
    return 255;
  relX *= stride;
  if (x - relX < 0)
    return 255;
  if (x - relX >= w)
    return 255;
  int offset = relY * w + relX;
  const uint32_t valuesPerByte = 1;
  return lossBuf((offset - 1) * valuesPerByte + 1);
}

ALWAYS_INLINE uint8_t Image24BitModel::GetPredErr(const uint32_t ctxIndex, int relX, int relY) const {
  if (line - relY < 0)  //the predErrBuf buffer is pre-filled with 255 initially, so this check is not really necessary
    return 255;
  relX *= stride;
  if (x - relX < 0)
    return 255;
  if (x - relX >= w)
    return 255;
  int offset = relY * w + relX;
  const uint32_t valuesPerByte = (nDM + nOLS);
  return predErrBuf((offset - 1) * valuesPerByte + ctxIndex + 1);
}

ALWAYS_INLINE uint32_t Image24BitModel::GetPredErrAvg(const uint32_t predictorIndex) const {
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

// abs(int8_t((c1 - prediction) & 255)): circular/wraparound distance on a 256-value ring
// int8_t cast recovers the sign from the modular difference, and abs then makes it symmetric.
// that is:
// (10 - 3) & 255 = 7   → int8_t(7)   = 7  → abs = 7
// (3 - 10) & 255 = 249 → int8_t(249) = -7 → abs = 7
// however with a large distance:
// (2 - 200) & 255 = 58  → int8_t(58)  = 58  → abs = 58
// (200 - 2) & 255 = 198 → int8_t(198) = -58 → abs = 58
// ... but the true linear distance is 198. Distances above 128 are thus
// reflected back — (200-2) and (2-200) both return 58 instead of 58 and 198.
// This ambiguity for large differences has negligible impact compared to
// the cost of using uint16_t to store the full 0..255 range.
ALWAYS_INLINE static int rabs(int x1, int x2) {
  return abs(int8_t((x1 - x2) & 255)); // 0..128
}

// Stores a prediction with a spread signal based on the distance between two reference pixels.
// ref1 and ref2 are spatial reference pixels; their difference measures local variation —
// higher difference = less confident = used as a context component in the probability map.
// Use when the prediction value is computed externally (e.g. a complex algebraic expression)
// but a natural pair of reference pixels still exists to supply the spread signal.
ALWAYS_INLINE void Image24BitModel::MakePrediction(int i, uint8_t ref1, uint8_t ref2, int prediction) {
  uint32_t absdiff = rabs(ref1, ref2);
  predictions[i] = absdiff << 16 | ((prediction) & 65535);
}

// Trend extrapolation: continues the gradient observed from pxFar to px,
// starting from the spatial origin.
// prediction = origin + (px - pxFar)
// spread = rabs(px - pxFar): larger gradient = more uncertain prediction.
// Use when pixels are expected to follow a consistent directional trend
// (e.g. continuing a horizontal, vertical, or diagonal gradient).
ALWAYS_INLINE void Image24BitModel::MakePredictionTrend(int i, int px, int pxFar, int origin) {
  uint32_t absdiff = rabs(px, pxFar);
  int prediction = origin + px - pxFar;
  predictions[i] = absdiff << 16 | (prediction & 65535);
}

// Smoothed trend: applies the gradient at half strength instead of full extrapolation.
// prediction = avg(origin, origin + (px - pxFar))
// spread = rabs(px - pxFar): same signal as Trend.
// Use in smoother regions where a full trend extrapolation would overshoot,
// or when the gradient is expected to taper rather than continue linearly.
ALWAYS_INLINE void Image24BitModel::MakePredictionSmooth(int i, int px, int pxFar, int origin) {
  uint32_t absdiff = rabs(px, pxFar);
  int prediction = avg(origin, origin + px - pxFar);
  predictions[i] = absdiff << 16 | (prediction & 65535);
}

ALWAYS_INLINE void Image24BitModel::MakePredictionAvg(int i, int px1, int px2) {
  uint32_t absdiff = rabs(px1, px2);
  int prediction = avg(px1, px2);
  predictions[i] = absdiff << 16 | (prediction & 65535);
}

// Stores only a prediction value; the spread signal is absent (upper 16 bits are zero).
// Use for complex algebraic combinations of multiple pixels where no single
// natural pair of reference pixels exists to derive a meaningful spread signal.
ALWAYS_INLINE void Image24BitModel::MakePredictionC(int i, int prediction) {
  predictions[i] = prediction & 65535;
}

void Image24BitModel::update() {
  INJECT_SHARED_bpos
  INJECT_SHARED_c1

  if (color < 4) // no need to accumulate loss from the padding zone
    loss += shared->State.loss; // += 0..1023

  // for every byte
  if (bpos == 0) {

    INJECT_SHARED_pos
    if (pos - lastPos != 1) {
      init();
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
    if (x == 0) {
      color = 0;
    }
    else if( x < w - padding ) {
      color++;
      if( color >= stride ) {
        color = 0;
      }
    } else {
      color = 4; // flag for padding zone
    }

    if (color < 4) { // we are in the pixel area

      lossBuf.add(static_cast<uint8_t>(min(loss >> 2, 255)));  // 0..255
      loss = 0;

      WWWWWW = Px(6, 0, 0); //buf(6*stride)
      WWWWW = Px(5, 0, 0); //buf(5*stride)
      WWWW = Px(4, 0, 0); //buf(4*stride)
      WWW = Px(3, 0, 0); //buf(3*stride)
      WW = Px(2, 0, 0); //buf(2*stride)
      W = Px(1, 0, 0); //buf(1*stride)

      NWWWW = Px(4, 1, 0); //buf(4*stride + 1*w)
      NWWW = Px(3, 1, 0); //buf(3*stride + 1*w)
      NWW = Px(2, 1, 0); //buf(2*stride + 1*w)
      NW = Px(1, 1, 0); //buf(1*stride + 1*w)
      N = Px(0, 1, 0); //buf(1*w)

      NE = Px(-1, 1, 0); //buf(-1*stride + 1*w)
      NEE = Px(-2, 1, 0); //buf(-2*stride + 1*w)
      NEEE = Px(-3, 1, 0); //buf(-3*stride + 1*w)
      NEEEE = Px(-4, 1, 0); //buf(-4*stride + 1*w)

      NNNWWW = Px(3, 3, 0); //buf(3*stride + 3*w)
      NNWWW = Px(3, 2, 0); //buf(3*stride + 2*w)
      NNWW = Px(2, 2, 0); //buf(2*stride + 2*w)
      NNW = Px(1, 2, 0); //buf(1*stride + 2*w)
      NN = Px(0, 2, 0); //buf(2*w)

      NNE = Px(-1, 2, 0); //buf(-1*stride + 2*w)
      NNEE = Px(-2, 2, 0); //buf(-2*stride + 2*w)
      NNEEE = Px(-3, 2, 0); //buf(-3*stride + 2*w)

      NNNWW = Px(2, 3, 0); //buf(2*stride + 3*w)
      NNNW = Px(1, 3, 0); //buf(1*stride + 3*w)
      NNN = Px(0, 3, 0); //buf(3*w)

      NNNE = Px(-1, 3, 0); //buf(-1*stride + 3*w)
      NNNEE = Px(-2, 3, 0); //buf(-2*stride + 3*w)
      NNNEEE = Px(-3, 3, 0); //buf(-3*stride + 3*w)

      NNNNW = Px(1, 4, 0); //buf(1*stride + 4*w)
      NNNN = Px(0, 4, 0); //buf(4*w)
      NNNNE = Px(-1, 4, 0); //buf(-1*stride + 4*w)
      NNNNN = Px(0, 5, 0); //buf(5*w)
      NNNNNN = Px(0, 6, 0); //buf(6*w)

      WWp1 = Px(2, 0, 1); //buf(2*stride + 1)
      Wp1 = Px(1, 0, 1); //buf(1*stride + 1)
      p1 = Px(0, 0, 1); //buf(1)
      NWp1 = Px(1, 1, 1); //buf(1*stride + 1*w + 1)
      Np1 = Px(0, 1, 1); //buf(1*w + 1)
      NEp1 = Px(-1, 1, 1); //buf(-1*stride + 1*w + 1)
      NNp1 = Px(0, 2, 1); //buf(2*w + 1)

      uint8_t NNNp1 = Px(0, 3, 1); //buf(3*w + 1)
      uint8_t WWWp1 = Px(3, 0, 1); //buf(3*stride + 1)
      uint8_t NNWp1 = Px(1, 2, 1); //buf(1*stride + 2*w + 1)
      uint8_t NNEp1 = Px(-1, 2, 1); //buf(-1*stride + 2*w + 1)
      uint8_t NWWp1 = Px(2, 1, 1); //buf(2*stride + 1*w + 1)
      uint8_t NEEp1 = Px(-2, 1, 1); //buf(-2*stride + 1*w + 1)
      uint8_t NNWWp1 = Px(2, 2, 1); //buf(2*stride + 2*w + 1)
      uint8_t NNEEp1 = Px(-2, 2, 1); //buf(-2*stride + 2*w + 1)
      uint8_t NNNNp1 = Px(0, 4, 1); //buf(4*w + 1)
      uint8_t NNNNNNp1 = Px(0, 6, 1); //buf(6*w + 1)
      uint8_t WWWWp1 = Px(4, 0, 1); //buf(4*stride + 1)
      uint8_t WWWWWWp1 = Px(6, 0, 1); //buf(6*stride + 1)
      uint8_t NNNWp1 = Px(1, 3, 1); //buf(1*stride + 3*w + 1)
      uint8_t NNNEp1 = Px(-1, 3, 1); //buf(-1*stride + 3*w + 1)

      WWp2 = Px(2, 0, 2); //buf(2*stride + 2)
      Wp2 = Px(1, 0, 2); //buf(1*stride + 2)
      p2 = Px(0, 0, 2); //buf(2)
      NWp2 = Px(1, 1, 2); //buf(1*stride + 1*w + 2)
      Np2 = Px(0, 1, 2); //buf(1*w + 2)
      NEp2 = Px(-1, 1, 2); //buf(-1*stride + 1*w + 2)
      NNp2 = Px(0, 2, 2); //buf(2*w + 2)

      uint8_t NNNp2 = Px(0, 3, 2); //buf(3*w + 2)
      uint8_t WWWp2 = Px(3, 0, 2); //buf(3*stride + 2)
      uint8_t NNWp2 = Px(1, 2, 2); //buf(1*stride + 2*w + 2)
      uint8_t NNEp2 = Px(-1, 2, 2); //buf(-1*stride + 2*w + 2)
      uint8_t NWWp2 = Px(2, 1, 2); //buf(2*stride + 1*w + 2)
      uint8_t NEEp2 = Px(-2, 1, 2); //buf(-2*stride + 1*w + 2)
      uint8_t NNWWp2 = Px(2, 2, 2); //buf(2*stride + 2*w + 2)
      uint8_t NNEEp2 = Px(-2, 2, 2); //buf(-2*stride + 2*w + 2)
      uint8_t NNNNp2 = Px(0, 4, 2); //buf(4*w + 2)
      uint8_t NNNNNNp2 = Px(0, 6, 2); //buf(6*w + 2)
      uint8_t WWWWp2 = Px(4, 0, 2); //buf(4*stride + 2)
      uint8_t WWWWWWp2 = Px(6, 0, 2); //buf(6*stride + 2)
      uint8_t NNNWp2 = Px(1, 3, 2); //buf(1*stride + 3*w + 2)
      uint8_t NNNEp2 = Px(-1, 3, 2); //buf(-1*stride + 3*w + 2)

      uint8_t NNNNWW = Px(2, 4, 0); //buf(2*stride + 4*w)
      uint8_t NNWWWW = Px(4, 2, 0); //buf(4*stride + 2*w)
      uint8_t NNNNEE = Px(-2, 4, 0); //buf(-2*stride + 4*w)
      uint8_t NNEEEE = Px(-4, 2, 0); //buf(-4*stride + 2*w)
      uint8_t NNNNNNWW = Px(2, 6, 0); //buf(2*stride + 6*w)
      uint8_t NNNNNNEE = Px(-2, 6, 0); //buf(-2*stride + 6*w)
      uint8_t NNNNWWWW = Px(4, 4, 0); //buf(4*stride + 4*w)
      uint8_t NNNNEEEE = Px(-4, 4, 0); //buf(-4*stride + 4*w)
      uint8_t NEEEEEE = Px(-6, 1, 0); //buf(-6*stride + 1*w)
      uint8_t NNNWWWW = Px(4, 3, 0); //buf(4*stride + 3*w)
      uint8_t NNNEEEE = Px(-4, 3, 0); //buf(-4*stride + 3*w)

      // mixer context: edge detection

      // Vertical ↓ (0)
      int r_N_vert = rabs(N, (NN * 2 - NNN));
      int r_W_vert = rabs(W, (NW * 2 - NNW));
      int r_NE_vert = rabs(NE, (NNE * 2 - NNNE));
      int score_vert = (2 * r_N_vert + r_W_vert + r_NE_vert) / 4;

      // Horizontal → (1)
      int r_W_horiz = rabs(W, (WW * 2 - WWW));
      int r_N_horiz = rabs(N, (NW * 2 - NWW));
      int score_horiz = (2 * r_W_horiz + r_N_horiz) / 3;

      // Diagonal ↘ (2)
      int r_NW_diag1 = rabs(NW, (NNWW * 2 - NNNWWW));
      int r_N_diag1 = rabs(N, (NNW * 2 - NNNWW));
      int r_W_diag1 = rabs(W, (NWW * 2 - NNWWW));
      int score_diag1 = (2 * r_NW_diag1 + r_N_diag1 + r_W_diag1) / 4;

      // Diagonal ↙ (3)
      int r_NE_diag2 = rabs(NE, (NNEE * 2 - NNNEEE));
      int r_N_diag2 = rabs(N, (NNE * 2 - NNNEE));
      int r_NEE_diag2 = rabs(NEE, (NNEEE * 2 - NNNEEEE));
      int score_diag2 = (2 * r_NE_diag2 + r_N_diag2 + r_NEE_diag2) / 4;

      ctx_best_direction = 0; // 0..3

      int ctx_best_score = score_vert;
      if (score_horiz < ctx_best_score) { ctx_best_score = score_horiz; ctx_best_direction = 1; }
      if (score_diag1 < ctx_best_score) { ctx_best_score = score_diag1; ctx_best_direction = 2; }
      if (score_diag2 < ctx_best_score) { ctx_best_score = score_diag2; ctx_best_direction = 3; }

      ctx_best_residual = DiffQt(0, ctx_best_score); // 0..7

      //what was the total cost at the neighboring pixes of this same channel
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

      // Written in reverse order (nDM+nOLS-1 down to 0) so that GetPredErr()'s
      // read formula  predErr((offset-1) * (nDM+nOLS) + ctxIndex + 1)  resolves correctly:
      // after writing nDM values, predictor ctxIndex sits at ring offset -(ctxIndex+1).


      uint32_t lowestErr = 255;
      uint8_t bestPredictorIndex = 0;
      static_assert(nDM + nOLS <= 255); // the index need to fit to a byte
      for (int i = nDM + nOLS - 1; i >= 0; i--) {
        short prediction = predictions[i] & 65535;
        uint8_t err;
        if (prediction == INT16_MAX) // currently never happens, todo
          err = 255;
        else
          err = rabs(c1, prediction); // 0..128
        predErrBuf.add(err); //0..63 or 255

        uint32_t linearError = abs(c1 - prediction);
        if (linearError < lowestErr) {
          lowestErr = linearError;
          bestPredictorIndex = i;
        }
      }
      bestPredictorIndexes.add(bestPredictorIndex);

      // these contexts are best for photographic images

      int contextIdx = 0;

      //for tuning:
      //int z = 0;
      //int toskip = shared->tuning_param + z;
      //z++; if (toskip != z) MakePredictionC(contextIdx++, p1);

      // note about p1 and p2:
      // their semantics change depending on 'color':
      // if the image is RGB:
      //  color = 0 (R) -> p1 = previous pixel's B, p2 = previous pixel's G
      //  color = 1 (G) -> p1 =  current pixel's R, p2 = previous pixel's B
      //  color = 2 (B) -> p1 =  current pixel's G, p2 =  current pixel's R <- this is the intention

      // p1-based predictors

      MakePredictionC(contextIdx++, p1); // very strong

      MakePredictionTrend(contextIdx++, p1, Np1, N); // strong
      MakePredictionSmooth(contextIdx++, p1, Np1, N); // strong
      MakePrediction(contextIdx++, p1, Np1, N);

      MakePredictionTrend(contextIdx++, p1, Wp1, W);
      MakePredictionSmooth(contextIdx++, p1, Wp1, W);
      MakePrediction(contextIdx++, p1, Wp1, W);

      MakePredictionTrend(contextIdx++, p1, NEp1, NE); // very strong
      MakePrediction(contextIdx++, p1, NEp1, NE);
      MakePredictionTrend(contextIdx++, p1, NNp1, NN); // very strong
      MakePrediction(contextIdx++, p1, NNp1, NN);

      MakePredictionTrend(contextIdx++, p1, NWp1, NW);
      MakePredictionSmooth(contextIdx++, p1, NWp1, NW);
      MakePrediction(contextIdx++, p1, NWp1, NW);

      MakePredictionTrend(contextIdx++, p1, WWp1, WW);
      MakePrediction(contextIdx++, p1, WWp1, WW);
//    MakePredictionSmooth(contextIdx++, p1, WWp1, WW); // weak

      MakePredictionSmooth(contextIdx++, p1, NNEEp1, NE);

      MakePredictionTrend(contextIdx++, p1, (-3 * WWp1 + 8 * Wp1 + (NEEp1 * 2 - NNEEp1)) / 6, (-3 * WW + 8 * W + (NEE * 2 - NNEE)) / 6);
      MakePredictionTrend(contextIdx++, p1, avg(WWp1, (NEEp1 * 2 - NNEEp1)), avg(WW, (NEE * 2 - NNEE)));

      MakePredictionTrend(contextIdx++, p1, Wp1 + Np1 - NWp1, W + N - NW); // strong
      MakePredictionTrend(contextIdx++, p1, Np1 + NWp1 - NNWp1, N + NW - NNW);
      MakePredictionTrend(contextIdx++, p1, Np1 + NEp1 - NNEp1, N + NE - NNE); //strong
      MakePredictionTrend(contextIdx++, p1, Np1 + NNp1 - NNNp1, N + NN - NNN);
      MakePredictionTrend(contextIdx++, p1, Wp1 + WWp1 - WWWp1, W + WW - WWW);
      MakePredictionTrend(contextIdx++, p1, Wp1 + NEEp1 - NEp1, W + NEE - NE);

//    MakePredictionTrend(contextIdx++, p1, NNp1 + Wp1 - NNWp1, NN + W - NNW); // weak
      MakePredictionTrend(contextIdx++, p1, Np1 * 2 - NNp1, N * 2 - NN); // strong
      MakePredictionTrend(contextIdx++, p1, Wp1 * 2 - WWp1, W * 2 - WW); // very strong

      MakePredictionTrend(contextIdx++, p1, Wp1 + NEp1 - Np1, W + NE - N);
      MakePredictionTrend(contextIdx++, p1, NEp1 + NWp1 - NNp1, NE + NW - NN);
      MakePredictionTrend(contextIdx++, p1, NWp1 + Wp1 - NWWp1, NW + W - NWW);

//    MakePredictionTrend(contextIdx++, p1, NEp1 * 2 - NNEEp1, NE * 2 - NNEE); // weak
//    MakePredictionTrend(contextIdx++, p1, Np1 * 3 - NNp1 * 3 + NNNp1, N * 3 - NN * 3 + NNN); // weak

      // p2-based predictors

      MakePredictionC(contextIdx++, p2); // very strong

      MakePredictionTrend(contextIdx++, p2, Np2, N);
      MakePredictionSmooth(contextIdx++, p2, Np2, N);

      MakePredictionTrend(contextIdx++, p2, Wp2, W);
      MakePredictionSmooth(contextIdx++, p2, Wp2, W);

      MakePredictionTrend(contextIdx++, p2, NEp2, NE);
      MakePredictionTrend(contextIdx++, p2, NNp2, NN);
      MakePredictionSmooth(contextIdx++, p2, NWp2, NW); 
      
      MakePredictionTrend(contextIdx++, p2, WWp2, WW);
//    MakePredictionSmooth(contextIdx++, p2, WWp2, WW); // weak

      MakePredictionSmooth(contextIdx++, p2, NNEEp2, NE); // very strong

      MakePredictionTrend(contextIdx++, p2, Wp2 + Np2 - NWp2, W + N - NW);
      MakePredictionTrend(contextIdx++, p2, Np2 + NWp2 - NNWp2, N + NW - NNW);
      MakePredictionTrend(contextIdx++, p2, Np2 + NEp2 - NNEp2, N + NE - NNE); // strong
//    MakePredictionTrend(contextIdx++, p2, Np2 + NNp2 - NNNp2, N + NN - NNN); // weak
      MakePredictionTrend(contextIdx++, p2, Wp2 + WWp2 - WWWp2, W + WW - WWW);
//    MakePredictionTrend(contextIdx++, p2, Wp2 + NEEp2 - NEp2, W + NEE - NE); // weak

//    MakePredictionTrend(contextIdx++, p2, NNp2 + Wp2 - NNWp2, NN + W - NNW); // weak
      MakePredictionTrend(contextIdx++, p2, Np2 * 2 - NNp2, N * 2 - NN);
      MakePredictionTrend(contextIdx++, p2, Wp2 * 2 - WWp2, W * 2 - WW);

//    MakePredictionTrend(contextIdx++, p2, Wp2 + NEp2 - Np2, W + NE - N);  // weak
//    MakePredictionTrend(contextIdx++, p2, NEp2 + NWp2 - NNp2, NE + NW - NN);  // weak
//    MakePredictionTrend(contextIdx++, p2, NWp2 + Wp2 - NWWp2, NW + W - NWW);  // weak

      MakePredictionTrend(contextIdx++, p2, NWp2 * 2 - NNWWp2, NW * 2 - NNWW);
      MakePredictionTrend(contextIdx++, p2, NEp2 * 2 - NNEEp2, NE * 2 - NNEE);
      
      // predictors using only the current color plane
      
      MakePredictionC(contextIdx++, (N * 3 + W * 3 - NN - WW + 2) >> 2);
      MakePredictionAvg(contextIdx++, W, NEE);


      MakePredictionTrend(contextIdx++, W, NW, N); // very strong
      MakePredictionTrend(contextIdx++, WW, NNWW, NN); //strong?
      MakePredictionTrend(contextIdx++, WWW, NNNWWW, NNN); //strong?
      MakePredictionTrend(contextIdx++, W, N, NE); // strong
//    MakePredictionTrend(contextIdx++, WW, NN, NNEE);  // weak
//    MakePredictionTrend(contextIdx++, WWW, NNN, NNNEEE);  // weak
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
//    MakePredictionTrend(contextIdx++, NNNN, NNNNNNWW, NNWW);  // weak
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
//    MakePredictionC(contextIdx++, ((W * 2 - NW) + (W * 2 - NWW) + N + NE) / 4);  // weak
//    MakePredictionSmooth(contextIdx++, avg(N, W), N, W);  // weak
      MakePredictionAvg(contextIdx++, NEEEE, NEEEEEE); // strong
      MakePredictionAvg(contextIdx++, WWWW, WWWWWW);
      MakePredictionAvg(contextIdx++, NNNN, NNNNNN);// strong
//    MakePredictionTrend(contextIdx++, NNNN, NNNNNN, NN); // weak
//    MakePredictionC(contextIdx++, avg(NNNNWWWW, NNWW)); // weak
//    MakePredictionC(contextIdx++, avg(NNNNEEEE, NNEE)); // weak
      MakePredictionAvg(contextIdx++, NNNNWW, NNWW);
      MakePredictionAvg(contextIdx++, NNNNEE, NNEE);
//    MakePredictionTrend(contextIdx++, NNEE, NNNNEEEE, NNEE); // weak
//    MakePredictionTrend(contextIdx++, NNEE, NNNNEE, NE); // weak
      MakePredictionAvg(contextIdx++, WWW, WWWWWW);
//    MakePredictionTrend(contextIdx++, WWW, WWWWWW, W);
//    MakePredictionTrend(contextIdx++, WWWW, WWWWWW, WW); // weak
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

      // 3rd-order horizontal + 4th-order NW diagonal
//    MakePredictionC(contextIdx++, (6 * W - 4 * WW + WWW + 4 * NW - 6 * NNW + 4 * NNNW - NNNNW) / 4); // weak
      // 4th-order horizontal + 4th-order NW diagonal
//    MakePredictionC(contextIdx++, (10 * W - 10 * WW + 5 * WWW - WWWW + 4 * NW - 6 * NNW + 4 * NNNW - NNNNW) / 5); // weak

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
      // 3rd-order NW diagonal alone
//    MakePredictionC(contextIdx++, (6 * NW - 4 * NNWW + NNNWWW) / 3); // weak

      // 3rd-order H + 3rd-order V, interaction-corrected
      MakePredictionC(contextIdx++, ((6 * W - 4 * WW + WWW) + (6 * N - 4 * NN + NNN) - (6 * NW - 4 * NNWW + NNNWWW)) / 3);

      // 5th-order horizontal
      MakePredictionC(contextIdx++, (15 * W - 20 * WW + 15 * WWW - 6 * WWWW + WWWWW) / 5); // strong

      // symmetric NE+NW blend, N-corrected
      MakePredictionC(contextIdx++, (2 * NE - NNEE) + (2 * NW - NNWW) - (2 * N - NN));

      // 2nd-order H + 2nd-order V + 2nd-order NE + 2nd-order NW, fully interaction-corrected
//    MakePredictionC(contextIdx++, (2 * W - WW) + (2 * N - NN) - (2 * NE - NNEE));

      // 4th-order horizontal
//    MakePredictionC(contextIdx++, (10 * W - 10 * WW + 5 * WWW - WWWW) / 4);

      // 3rd-order vertical
//    MakePredictionC(contextIdx++, (6 * N - 4 * NN + NNN) / 3);

      // 4th-order vertical
//    MakePredictionC(contextIdx++, (10 * N - 10 * NN + 5 * NNN - NNNN) / 4);
      // 5th-order vertical
//    MakePredictionC(contextIdx++, (15 * N - 20 * NN + 15 * NNN - 6 * NNNN + NNNNN) / 5);

      // 2nd-order NE diagonal
//    MakePredictionC(contextIdx++, (2 * NE - NNEE));

      // 2nd-order NW diagonal
//    MakePredictionC(contextIdx++, (2 * NW - NNWW));


      MakePredictionC(contextIdx++, 0);

      assert(contextIdx == nDM);

      //quality metric: quantized past loss in the pixel neighborhood 
      lossQ4 =
        color == 2 ? (lossQ < 1 ? lossQ : min(1 + ((lossQ - 1) / 40u), 7)) :
        color == 1 ? (lossQ < 8 ? (lossQ >> 2) : min(2 + ((lossQ - 8) / 40u), 7)) :
        min(lossQ / 40u, 7); //0..7 (3 bits)

      // map the predictions to histograms

      for (int i = 0; i < nDM; i++) {
        uint32_t spread = predictions[i] >> 16;
        short prediction = predictions[i] & 65535;
        if (prediction == INT16_MAX) { // currently never happens, todo
          mapR1.skip();
          mapR2.skip();
          mapR3.skip();
        }
        else {
          uint32_t predErrAvg = GetPredErrAvg(i);
          mapR1.set(prediction, min(predErrAvg, 31) << 2 | color); // 0..31, 0..3 (5+2 bits)
          mapR2.set(prediction, lossQ4 << 2 | color); // 0..7, 0..3 (3+2 bits)
          mapR3.set(prediction, min(spread, 31) << 2 | color); // 5+2 bits
        }
      }

      // OLS predictors

      int k = (color > 0) ? color - 1 : stride - 1; // previous color index
      for (int j = 0; j < nOLS; j++) {
        ols[j][k]->update(p1);
        auto ols_j_color = ols[j][color].get();
        auto ols_ctx_j = olsCtxs[j];
        for (int ctx_idx = 0; ctx_idx < num[j]; ctx_idx++) {
          float val = *ols_ctx_j[ctx_idx];
          ols_j_color->add(val);
        }
        float pred = ols_j_color->predict();
        short prediction = short(roundf(pred));
        MakePredictionC(contextIdx++, prediction);

        uint32_t predErrAvg = GetPredErrAvg(j + nDM);
        mapOLS1.set(prediction, min(predErrAvg, 31) << 2 | color); // 0..31, 0..3 (5+2 bits)
        mapOLS2.set(prediction, lossQ4 << 2 | color); // 0..7, 0..3 (3+2 bits)
      }

      assert(contextIdx == nDM + nOLS);

      // these contexts are best for non-photographic images (logos, icons, screenshots, infographics)
      // todo: review and optimize these contexts

      uint64_t i = color * 1024;
      const uint8_t R_ = CM_USE_RUN_STATS;

      //for tuning:
      //int toskip = shared->tuning_param - 1 + i;
      //if (toskip == i) cm.skip(__), i++; else ...
      
      cm.set(R_, hash(++i, W, p1));
      cm.set(R_, hash(++i, W, p2));
      cm.set(R_, hash(++i, N, p1));
      cm.set(R_, hash(++i, N, p2));
      cm.set(R_, hash(++i, p1, p2));
      cm.set(R_, hash(++i, N, NN, p1));
      cm.set(R_, hash(++i, N, NN, p2));
      cm.set(R_, hash(++i, W, WW, p1));
      cm.set(R_, hash(++i, W, WW, p2));
      cm.set(R_, hash(++i, N, W, p1, p2));

      cm.set(R_, hash(++i, W, WW, N, NN));
      cm.set(R_, hash(++i, W, N, NE, NW));
      
      cm.set(R_, hash(++i, (NNN + N + 4) >> 3, (N * 3 - NN * 3 + NNN) >> 1));
      cm.set(R_, hash(++i, (W + N - NW) >>1, (W + p1 - Wp1) >>1));
      cm.set(R_, hash(++i, W >> 2, DiffQt(W, p1), DiffQt(W, p2)));
      cm.set(R_, hash(++i, N >> 2, DiffQt(N, p1), DiffQt(N, p2)));
      cm.set(R_, hash(++i, (W + N + 4) >> 3, p1 >> 4, p2 >> 4)); // strong
      cm.set(R_, hash(++i, W, p1 - Wp1)); // strong
      cm.set(R_, hash(++i, N, p1 - Np1)); // very strong

      cm.set(R_, hash(++i, (N * 2 - NN) >>1, DiffQt(N, (NN * 2 - NNN))));
      cm.set(R_, hash(++i, (W * 2 - WW) >>1, DiffQt(W, (WW * 2 - WWW))));
      cm.set(R_, hash(++i, (N * 2 - NN), DiffQt(W, (NW * 2 - NNW))));
      cm.set(R_, hash(++i, (W * 2 - WW), DiffQt(N, (NW * 2 - NWW))));
      cm.set(R_, hash(++i, (W + NEE + 1) >> 1, DiffQt(W, (WW + NE + 1) >> 1)));
      cm.set(R_, hash(++i, (W + NEE - NE), DiffQt(W, (WW + NE - N))));
      cm.set(R_, hash(++i, N, NN, NNN));
      cm.set(R_, hash(++i, W, WW, WWW));

      int div7 = max((x / stride) >> 10, 7);
      int div17 = max((x / stride) >> 9, 17);
      int div29 = max((x / stride) >> 8, 29);

      cm.set(R_, hash(++i, w, x / stride / div7));
      cm.set(R_, hash(++i, w, x / stride / div17, line / 17));
      cm.set(R_, hash(++i, w, x / stride / div29, uint8_t(W + N - NW) >> 1));

      assert(i - color * 1024 == nCM);

      ctx[0] =  // 9 bits
        (static_cast<int>(rabs(W, NW) > 3) << 8) |
        (static_cast<int>(rabs(NW, N) > 3) << 7) |
        (static_cast<int>(rabs(N, NE) > 3) << 6) |
        (static_cast<int>(N > NW) << 5) |
        (static_cast<int>(N > NE) << 4) |
        (static_cast<int>(N > NN) >> 3) |
        (static_cast<int>(W > N) << 2) |
        (static_cast<int>(W > NW) << 1) |
        (static_cast<int>(W > WW) << 0);


      ctx[1] =  // 8 bits
        (DiffQt(p1, (Np1 + NEp1 - NNEp1))) << 4 |
        (DiffQt((N + NE - NNE), (N + NW - NNW)));

      shared->State.Image.plane = color;
      shared->State.Image.pixels.W = W;
      shared->State.Image.pixels.N = N;
      shared->State.Image.pixels.NN = NN;
      shared->State.Image.pixels.WW = WW;
      shared->State.Image.ctx = ((color << 9) | ctx[0]) >> 3; // 8 bits
    }
  }

  //for every bit

  if (color != 4) {
    INJECT_SHARED_c0

      //these contexts are best for non-photographic images (logos, icons, screenshots, infographics)
      //but they also work well for modelling with the direct pixel neigborhood in not-too-noisy photographic images

      int i = (c0 << 2 | color) * 256;

    //for tuning:
    //int toskip = shared->tuning_param - 1 + i;
    //if (toskip == i) mapL.skip(), i++; else 

    mapL.set(hash(++i, p2)); // very strong
    mapL.set(hash(++i, p1, p2)); // strong

    // W
//  mapL.set(hash(++i, W, p1)); // weak
    mapL.set(hash(++i, W, p2)); // strong
    mapL.set(hash(++i, W, p1, p2));

    // N
//  mapL.set(hash(++i, N, p1)); // weak
//  mapL.set(hash(++i, N, p2)); // weak
    mapL.set(hash(++i, N, Np1, Np2));

    // W + N
//  mapL.set(hash(++i, W, N, p1, p2));  // weak
    mapL.set(hash(++i, W, p1, p2, N, Np1));
    mapL.set(hash(++i, W, p1, p2, N, Np1, Np2));

    // W + NE
    mapL.set(hash(++i, W, p1, p2, NE));
    mapL.set(hash(++i, W, p1, p2, NE, NEp1));
    mapL.set(hash(++i, W, p1, p2, NE, NEp1, NEp2));


    // N + NW
//  mapL.set(hash(++i, N, NW, p2)); // weak

    // N + NE
    mapL.set(hash(++i, N, NE, p1));
//  mapL.set(hash(++i, N, NE, p2)); // weak

    //N + NN
//  mapL.set(hash(++i, N, NN, p1));  // weak
//  mapL.set(hash(++i, N, NN, p2)); // weak

    // W + WW
//  mapL.set(hash(++i, W, WW, p1)); // weak
//  mapL.set(hash(++i, W, WW, p2)); // weak

    // NW + NE (cross-diagonal)
    mapL.set(hash(++i, NW, NE, p1));
    mapL.set(hash(++i, NW, NE, p2));

    // NW + NN
    mapL.set(hash(++i, NW, NN, p1));

    // W + N + NE
    mapL.set(hash(++i, W, N, NE, p1));

    // W + N + NW
    mapL.set(hash(++i, W, N, NW, p1));
//  mapL.set(hash(++i, W, p1, p2, N, NW, NWp1)); //weak

    // N + NE + NW
    mapL.set(hash(++i, N, Np1, NE, NW, p1));

    mapL.set(hash(++i, NE, NEE, p1)); // strong
    mapL.set(hash(++i, NE, NEE, p2));
    mapL.set(hash(++i, NE, NEp1, NEp2, NEE)); // strong

    mapL.set(hash(++i, NN, NNN, p1));
    mapL.set(hash(++i, NN, NNN, p2));
    mapL.set(hash(++i, NN, NNp1, NNp2, NNN));

//  mapL.set(hash(++i, WW, WWW, p1)); // weak
//  mapL.set(hash(++i, WW, WWW, p2)); // weak
    mapL.set(hash(++i, WW, WWp1, WWp2, WWW));

    mapL.set(hash(++i, Np1, Np2, Wp1, Wp2));
    mapL.set(hash(++i, NWp1, NWp2, NEp1, NEp2));

    mapL.set(hash(++i, (W + N - NW) >> 1, p1));  // very strong
    mapL.set(hash(++i, (W + N - NW) >> 1, p1, p2)); // very strong
    mapL.set(hash(++i, (N + NE - NNE) >> 1, p1)); // strong
    mapL.set(hash(++i, (N + NE - NNE) >> 1, p1, p2));
    mapL.set(hash(++i, (W * 2 - WW) >> 1, p1));
    mapL.set(hash(++i, (N * 2 - NN) >> 1, p1));
    mapL.set(hash(++i, (W * 2 - WW) >> 1, p1, p2));
    mapL.set(hash(++i, (N * 2 - NN) >> 1, p1, p2));

    mapL.set(hash(++i, (NW + W - NWW) >> 1, p1));
    mapL.set(hash(++i, (NW + W - NWW) >> 1, p1, p2));
    mapL.set(hash(++i, (NW + N - NNW) >> 1, p1));
    mapL.set(hash(++i, (NW + N - NNW) >> 1, p1, p2));

//  mapL.set(hash(++i, (NE * 2 - NNEE) >> 1, p1)); 
//  mapL.set(hash(++i, (NE * 2 - NNEE) >> 1, p1, p2));
//  mapL.set(hash(++i, (NW * 2 - NNWW) >> 1, p1));
//  mapL.set(hash(++i, (NW * 2 - NNWW) >> 1, p1, p2)); // weak

//  mapL.set(hash(++i, ((W * 2 - WW) + (N * 2 - NN) - (NW * 2 - NNWW)) >> 1, p1)); // weak
//  mapL.set(hash(++i, ((W * 2 - WW) + (N * 2 - NN) - (NW * 2 - NNWW)) >> 1, p1, p2)); // weak

//  mapL.set(hash(++i, (N * 3 - NN * 3 + NNN) >> 1, p1));
//  mapL.set(hash(++i, (W * 3 - WW * 3 + WWW) >> 1, p1));
//  mapL.set(hash(++i, (N * 3 - NN * 3 + NNN) >> 1, p1, p2)); // weak
//  mapL.set(hash(++i, (W * 3 - WW * 3 + WWW) >> 1, p1, p2)); // weak

//  mapL.set(hash(++i, ((NE * 2 - NNEE) + (NW * 2 - NNWW) - (N * 2 - NN)) >> 1, p1)); // weak
//  mapL.set(hash(++i, ((NE * 2 - NNEE) + (NW * 2 - NNWW) - (N * 2 - NN)) >> 1, p1, p2));

    mapL.set(hash(++i, paeth(W, N, NW) >> 1, p1));
    mapL.set(hash(++i, paeth(W, N, NW) >> 1, p1, p2)); // strong
//  mapL.set(hash(++i, gap(W, N, NW, NE, WW, NNE, NN) >> 1, p1)); // weak

    assert(i - ((c0 << 2 | color) * 256) == nLSM);


  }
}

void Image24BitModel::init() {
  stride = 3 + alpha;
  padding = w % stride;
  x = color = line = 0;
  if( lastPos > 0 && false ) { // todo: when shall we reset?
    for (int i = 0; i < nLSM; i++) {
      mapL.reset();
    }
  }
  lossBuf.setSize(nextPowerOf2(LOSS_BUF_ROWS * w));
  lossBuf.fill(255);

  predErrBuf.setSize(nextPowerOf2(PRED_ERR_BUF_ROWS * w * (nDM + nOLS)));
  predErrBuf.fill(255);

  bestPredictorIndexes.setSize(nextPowerOf2(BEST_PRED_ROWS * w));
}

void Image24BitModel::setParam(int width, uint32_t alpha0) {
  w = width;
  alpha = alpha0;
}

void Image24BitModel::mix(Mixer &m) {

  // predict next bit
  if (color < 4) { // pixel area
    cm.mix(m);

    mapR1.mix(m);
    mapR2.mix(m);
    mapR3.mix(m);
    mapOLS1.mix(m);
    mapOLS2.mix(m);
    mapL.mix(m);

    INJECT_SHARED_bpos
    INJECT_SHARED_c0
    assert(color < 4);

    const int bp = (UINT32_C(0x33322210) >> (bpos << 2)) & 0xF; // {bpos:0}->0  {bpos:1}->1  {bpos:2,3,4}->2  {bpos:5,6,7}->3

    m.set(1 + bpos, 1 + 8);   // 1: account for padding zone

    m.set(lossQ4 /*0..7*/, 8);
    m.set(((line & 7) << 3) | bpos, 8 * 8);

    m.set(((x / stride) & 7) << 3 | (line & 7), 8 * 8);
    m.set(((x / stride) & 15), 16);

    int div3 = max((x / stride) >> 9, 3);
    int div19 = max((x / stride) >> 8, 19);
    m.set((x / stride / div3) & 511, 512);
    m.set((x / stride / div19) & 255, 256);

    int pred1 = 0x100 | ((N + W + 1) >> 1);
    int pred2 = 0x100 | int8_t(N + W - NW);
    int pred3 = 0x100 | int8_t(2 * N - NN);
    int pred4 = 0x100 | int8_t(2 * W - WW);
    m.set(
      (c0 == (pred1 >> (8 - bpos))) << 6 |
      (c0 == (pred2 >> (8 - bpos))) << 5 |
      (c0 == (pred3 >> (8 - bpos))) << 4 |
      (c0 == (pred4 >> (8 - bpos))) << 3 |
      bpos, 16 * 8);
    m.set(lossQ4 << 2 | bp, 8 * 4);
    m.set((ctx[1] << 2) | bp, 1024);
    m.set(ctx_best_direction << 3 | ctx_best_residual, 4 * 8);

    m.set(((x / stride + line) >> 5) & 255, 256);

    auto bestPredictorIndexW =
      color == 0 ? bestPredictorIndexes(stride) : // W, same channel
      bestPredictorIndexes(1);                    // current pixel, prev channel (co-located i.e. within the same pixel as what we predict)
    auto bestPredictorIndexN = bestPredictorIndexes(w);                    // N, same channel

    m.set(bestPredictorIndexW, (nDM + nOLS));
    m.set(bestPredictorIndexN, (nDM + nOLS));

    int bestErrN = GetPredErrAvg(bestPredictorIndexN);
    int bestErrW = GetPredErrAvg(bestPredictorIndexW);
    m.set(min(31, bestErrW) << 5 | min(31, bestErrN), 32 * 32);

    auto bestPredictorIndexNE = bestPredictorIndexes(w + stride); // NW, same channel
    auto bestPredictorIndexNW = bestPredictorIndexes(w - stride); // NE, same channel

    int bestErrNE = GetPredErrAvg(bestPredictorIndexNE);
    int bestErrNW = GetPredErrAvg(bestPredictorIndexNW);
    m.set(min(31, bestErrNE) << 5 | min(31, bestErrNW), 32 * 32);

    uint32_t rabsBits3 = ctx[0] >> 6; // the top 3 rabs bits
    m.set(rabsBits3, 8);
    uint32_t cmpBits6 = ctx[0] & 0x3F; // the 6 comparison bits
    m.set(cmpBits6 << 3 | lossQ4, 64 * 8);

    int trendN = 2 * N - NN;
    int trendW = 2 * W - WW;
    int trendNE = 2 * NE - NNEE;
    int trendNW = 2 * NW - NNWW;

    int trendMin = min(min(min(trendN, trendW), trendNE), trendNW);
    int trendMax = max(max(max(trendN, trendW), trendNE), trendNW);

    m.set((min(trendMax - trendMin, 255) >> 2) << 2 | (bpos >> 1), 64 * 4);

    m.set(((p1 >> 5) << 5) | ((p2 >> 5) << 2) | (bpos >> 1), 8 * 8 * 4);


  }
  else {
    // padding zone
    m.add(-2047);  //predict 0
    m.set(0, MIXERCONTEXTS);
  }
}
