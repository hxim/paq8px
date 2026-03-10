#include "Image24BitModel.hpp"

Image24BitModel::Image24BitModel(Shared* const sh, const uint64_t size) :
  shared(sh),
  cm(sh, size, nCM, 64),
  mapL{ sh, nLSM, 23, 74 },     /* LargeStationaryMap : Contexts, HashBits, Scale=64, Rate=16 */
  mapR1{ sh, nDM, 1 << 7, 74 }, /* ResidualMap: numContexts, histogramsPerContext, scale=64 */
  mapR2{ sh, nDM, 1 << 7, 74 }, /* ResidualMap: numContexts, histogramsPerContext, scale=64 */
  mapR3{ sh, nDM, 1 << 7, 74 }, /* ResidualMap: numContexts, histogramsPerContext, scale=64 */
  map{ /* StationaryMap : BitsOfContext, InputBits, Scale=64, Rate=16  */
    /*nOLS: 0- 5*/ {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}
  }
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
  return lossBuf(offset);
}

ALWAYS_INLINE uint8_t Image24BitModel::GetPredErr(const uint32_t ctxIndex, int relX, int relY) const {
  if (line - relY < 0)  //the predErrBuf buffer is pre-filled with 255 initially, so this check is not really necessary
    return 255;
  relX *= stride;
  if (x - relX < 0)
    return 255;
  if (x - relX >= w)
    return 255;
  int offset = relY * w + relX - 1;
  return predErrBuf(offset * nDM + ctxIndex + 1);
}

ALWAYS_INLINE int avg(int x, int y) {
  return (x + y + 1) >> 1;  //note: rounding here works properly only when x+y is non-negative, but we don't really need the function to be aware of negative values as they are rare
}

// Stores a prediction with a spread signal based on the distance between two reference pixels.
// ref1 and ref2 are spatial reference pixels; their difference measures local variation —
// higher difference = less confident = used as a context component in the probability map.
// Use when the prediction value is computed externally (e.g. a complex algebraic expression)
// but a natural pair of reference pixels still exists to supply the spread signal.
ALWAYS_INLINE void Image24BitModel::MakePrediction(int i, uint8_t ref1, uint8_t ref2, int prediction) {
  uint32_t absdiff = abs(ref1 - ref2);
  predictions[i] = absdiff << 16 | ((prediction) & 65535);
}

// Trend extrapolation: continues the gradient observed from pxFar to px,
// starting from the spatial origin.
// prediction = origin + (px - pxFar)
// spread = abs(px - pxFar): larger gradient = more uncertain prediction.
// Use when pixels are expected to follow a consistent directional trend
// (e.g. continuing a horizontal, vertical, or diagonal gradient).
ALWAYS_INLINE void Image24BitModel::MakePredictionTrend(int i, int px, int pxFar, int origin) {
  uint32_t absdiff = abs(px - pxFar);
  int prediction = origin + px - pxFar;
  predictions[i] = absdiff << 16 | (prediction & 65535);
}

// Smoothed trend: applies the gradient at half strength instead of full extrapolation.
// prediction = avg(origin, origin + (px - pxFar))
// spread = abs(px - pxFar): same signal as Trend.
// Use in smoother regions where a full trend extrapolation would overshoot,
// or when the gradient is expected to taper rather than continue linearly.
ALWAYS_INLINE void Image24BitModel::MakePredictionSmooth(int i, int px, int pxFar, int origin) {
  uint32_t absdiff = abs(px - pxFar);
  int prediction = avg(origin, origin + px - pxFar);
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

  if (color != 4) // no need to accumulate loss from the padding zone
    loss += shared->State.loss; // += 0..255

  // for every byte
  if (bpos == 0) {

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

    if (color != 4) { // we are in the pixel area

      assert((loss >> 3) <= 255);
      lossBuf.add(static_cast<uint8_t>(loss >> 3));  // 0..31
      loss = 0;

      column[0] = x / columns[0];
      column[1] = x / columns[1];

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
      uint8_t NNNNWWW = Px(4, 3, 0); //buf(4*stride + 3*w)

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

      // Written in reverse order (nDM-1 down to 0) so that GetPredErr()'s
      // read formula  predErr(offset * nDM + ctxIndex + 1)  resolves correctly:
      // after writing nDM values, predictor ctxIndex sits at ring offset -(ctxIndex+1).
      for (int i = nDM - 1; i >= 0; i--) {
        short prediction = predictions[i] & 65535;
        uint8_t err;
        if (prediction == INT16_MAX) // currently never happens, todo
          err = 255;
        else
          err = abs(int8_t((c1 - prediction) & 255));
        predErrBuf.add(err); //0..63 or 255
      }

      // these contexts are best for photographic images

      int contextIdx = 0;

      //for tuning:
      //int z = 0;
      //int toadd = shared->tuning_param - 1 + z;
      //if (toskip != z) MakePredictionC(contextIdx++, p1); z++;
      //if (toadd == z) MakePredictionC(contextIdx++, p1); z++;


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
      MakePredictionTrend(contextIdx++, p1, Wp1, W);
      MakePredictionSmooth(contextIdx++, p1, Wp1, W);
      MakePredictionTrend(contextIdx++, p1, NEp1, NE); // very strong
      MakePredictionTrend(contextIdx++, p1, NNp1, NN); // very strong

      MakePredictionTrend(contextIdx++, p1, NWp1, NW);
      MakePredictionSmooth(contextIdx++, p1, NWp1, NW);
      MakePredictionTrend(contextIdx++, p1, WWp1, WW);
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
      
      MakePredictionC(contextIdx++,(N * 3 + W * 3 - NN - WW + 2) >> 2);
      MakePrediction(contextIdx++, W, NEE, avg(W, NEE));
      MakePredictionC(contextIdx++, ((WWW - 4 * WW + 6 * W + (NE * 4 - NNE * 6 + NNNE * 4 - NNNNE)) / 4));
      MakePredictionC(contextIdx++, ((-WWWW + 5 * WWW - 10 * WW + 10 * W + (NE * 4 - NNE * 6 + NNNE * 4 - NNNNE)) / 5));
      MakePredictionC(contextIdx++, ((-4 * WW + 15 * W + 10 * (NE * 3 - NNE * 3 + NNNE) - (NEEE * 3 - NNEEE * 3 + NNNEEE)) / 20));
      MakePredictionC(contextIdx++, ((-3 * WW + 8 * W + (NEE * 3 - NNEE * 3 + NNNEE)) / 6));
      
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
      
      MakePredictionC(contextIdx++, ((NNNNN - 6 * NNNN + 15 * NNN - 20 * NN + 15 * N + clamp4(W * 4 - NWW * 6 + NNWWW * 4 - NNNNWWW, W, NW, N, NN)) / 6)); // very strong
      MakePredictionC(contextIdx++, ((NNNEEE - 4 * NNEE + 6 * NE + (W * 4 - NW * 6 + NNW * 4 - NNNW)) / 4));
      MakePredictionC(contextIdx++, (((N + 3 * NW) / 4) * 3 - avg(NNW, NNWW) * 3 + (NNNWW * 3 + NNNWWW) / 4));
      MakePredictionC(contextIdx++, ((W * 2 + NW) - (WW + 2 * NWW) + NWWW)); // strong
//    MakePredictionC(contextIdx++, ((W * 2 - NW) + (W * 2 - NWW) + N + NE) / 4);  // weak
//    MakePredictionSmooth(contextIdx++, avg(N, W), N, W);  // weak
      MakePredictionC(contextIdx++, avg(NEEEE, NEEEEEE)); // strong
      MakePredictionC(contextIdx++, avg(WWWWWW, WWWW));
      MakePredictionC(contextIdx++, avg(NNNN, NNNNNN));// strong
//    MakePredictionTrend(contextIdx++, NNNN, NNNNNN, NN); // weak
//    MakePredictionC(contextIdx++, avg(NNNNWWWW, NNWW)); // weak
//    MakePredictionC(contextIdx++, avg(NNNNEEEE, NNEE)); // weak
      MakePredictionC(contextIdx++, avg(NNNNWW, NNWW));
      MakePredictionC(contextIdx++, avg(NNNNEE, NNEE));
//    MakePredictionTrend(contextIdx++, NNEE, NNNNEEEE, NNEE); // weak
//    MakePredictionTrend(contextIdx++, NNEE, NNNNEE, NE); // weak
      MakePredictionC(contextIdx++, avg(WWWWWW, WWW));
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

      MakePredictionC(contextIdx++, avg(2 * N - NN, 2 * W - WW));
      MakePredictionC(contextIdx++, avg(2 * W - WW, 2 * NW - NNWW));

      MakePredictionC(contextIdx++, paeth(W, N, NW));
      MakePredictionC(contextIdx++, gap(W, N, NW, NE, WW, NNE, NN));

      MakePredictionC(contextIdx++, 0);

      assert(contextIdx == nDM);

      uint32_t lossQ4 = (lossQ / 40u);  //0..15 (4 bits)
      for (int i = 0; i < nDM; i++) {
        uint32_t spread = predictions[i] >> 16;
        short prediction = predictions[i] & 65535;
        if (prediction == INT16_MAX) { // currently never happens, todo
          mapR1.skip();
          mapR2.skip();
          mapR3.skip();
        }
        else {
          // Curent pixel's prediction confidence is based on the already known error at neighboring pixel predictions
          uint8_t predErrW = GetPredErr(i, 1, 0); //W
          uint8_t predErrN = GetPredErr(i, 0, 1); //N
          uint8_t predErrNW = GetPredErr(i, 1, 1); //NW
          uint8_t predErrNE = GetPredErr(i, -1, 1); //NE
          uint32_t predErrAvg = (predErrW + predErrN + predErrNE + predErrNW + 2) >> 2; //0..255
          mapR1.set(prediction, min(predErrAvg, 31) << 2 | color); //5+2 bits
          mapR2.set(prediction, lossQ4 << 2 | color); //0..31, 0..3 (5+2 bits)
          mapR3.set(prediction, min(spread, 31) << 2 | color); //5+2 bits
        }
      }

      int k = (color > 0) ? color - 1 : stride - 1; //previous color index
      for (int j = 0; j < nOLS; j++) {
        auto ols_j_color = ols[j][color].get();
        auto ols_ctx_j = olsCtxs[j];
        for (int ctx_idx = 0; ctx_idx < num[j]; ctx_idx++) {
          float val = *ols_ctx_j[ctx_idx];
          ols_j_color->add(val);
        }
        float prediction = ols_j_color->predict();
        pOLS[j] = clip(int(roundf(prediction)));
        ols[j][k]->update(p1);
      }

      int mean = (W + NW + N + NE + 2) >> 2;
      int diff4 =
        DiffQt(W, N, 4) << 12 |
        DiffQt(NW, NE, 4) << 8 |
        DiffQt(NW, N, 4) << 4 |
        DiffQt(W, NE, 4);

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
      cm.set(R_, hash(++i, N, NN, p1));
      cm.set(R_, hash(++i, N, NN, p2));
      cm.set(R_, hash(++i, W, WW, p1));
      cm.set(R_, hash(++i, W, WW, p2));
      cm.set(R_, hash(++i, N, W, p1, p2));
      
      cm.set(R_, hash(++i, (NNN + N + 4) >> 3, (N * 3 - NN * 3 + NNN) >> 1));
      cm.set(R_, hash(++i, ((-WWWW + 5 * WWW - 10 * WW + 10 * W + clamp4(NE * 4 - NNE * 6 + NNNE * 4 - NNNNE, N, NE, NEE, NEEE)) / 5) / 4));
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
      cm.set(R_, hash(++i, (clamp4((W * 2 - WW) + (N * 2 - NN) - (NW * 2 - NNWW), W, NW, N, NE))));
      cm.set(R_, hash(++i, (W + N - NW), column[0]));
      cm.set(R_, hash(++i, N, NN, NNN));
      cm.set(R_, hash(++i, W, WW, WWW));
      cm.set(R_, hash(++i, N, column[0]));
      cm.set(R_, hash(++i, column[1]));
      cm.set(R_, hash(++i, mean, diff4));

      assert(i - color * 1024 == nCM);

      // todo: review and optimize these contexts

      ctx[0] = (min(color,stride - 1) << 9) |
        (static_cast<int>(abs(W - N) > 3) << 8) |
        (static_cast<int>(W > N) << 7) |
        (static_cast<int>(W > NW) << 6) |
        (static_cast<int>(abs(N - NW) > 3) << 5) |
        (static_cast<int>(N > NW) << 4) |
        (static_cast<int>(abs(N - NE) > 3) << 3) |
        (static_cast<int>(N > NE) << 2) |
        (static_cast<int>(W > WW) << 1) |
        static_cast<int>(N > NN);
      ctx[1] = ((DiffQt(p1, (Np1 + NEp1 - buf(w * 2 - stride + 1))) >> 1) << 5) |
        ((DiffQt((N + NE - NNE), (N + NW - NNW)) >> 1) << 2) |
        min(color, stride - 1);

      shared->State.Image.plane = min(color, stride - 1);
      shared->State.Image.pixels.W = W;
      shared->State.Image.pixels.N = N;
      shared->State.Image.pixels.NN = NN;
      shared->State.Image.pixels.WW = WW;
      shared->State.Image.ctx = ctx[0] >> 3;
    }
  }

  //for every bit

  if (color != 4) {
    INJECT_SHARED_c0

    //these contexts are best for non-photographic images (logos, icons, screenshots, infographics)
    //but they also work well for modelling with the direct pixel neigborhood in not-too-noisy photographic images

    int i = (c0 << 2 | color) * 256;

    mapL.set(hash(++i, p2)); // strong
    mapL.set(hash(++i, p1, p2)); // strong

    // W
    mapL.set(hash(++i, W, p1));
    mapL.set(hash(++i, W, p2)); // strong
    mapL.set(hash(++i, W, p1, p2));

    // N
    mapL.set(hash(++i, N, p1)); // strong
    mapL.set(hash(++i, N, p2));
    mapL.set(hash(++i, N, Np1, Np2));

    // W + N
    mapL.set(hash(++i, W, N, p1, p2));
    mapL.set(hash(++i, W, p1, p2, N, Np1));
    mapL.set(hash(++i, W, p1, p2, N, Np1, Np2));

    // W + NE
    mapL.set(hash(++i, W, p1, p2, NE));
    mapL.set(hash(++i, W, p1, p2, NE, NEp1));
    mapL.set(hash(++i, W, p1, p2, NE, NEp1, NEp2));


    // N + NW
    mapL.set(hash(++i, N, NW, p2));

    // N + NE
    mapL.set(hash(++i, N, NE, p1));
    mapL.set(hash(++i, N, NE, p2));

    //N + NN
    mapL.set(hash(++i, N, NN, p1));

    // W + WW
    mapL.set(hash(++i, W, WW, p1));
    mapL.set(hash(++i, W, WW, p2));

    // NW + NE (cross-diagonal)
    mapL.set(hash(++i, NW, NE, p1));
    mapL.set(hash(++i, NW, NE, p2));

    // NW + NN
    mapL.set(hash(++i, NW, NN, p1));

    // W + N + NE
    mapL.set(hash(++i, W, N, NE, p1));

    // W + N + NW
    mapL.set(hash(++i, W, N, NW, p1));
    mapL.set(hash(++i, W, p1, p2, N, NW, NWp1));

    // N + NE + NW
    mapL.set(hash(++i, N, Np1, NE, NW, p1));

    mapL.set(hash(++i, NE, NEE, p1));
    mapL.set(hash(++i, NE, NEE, p2));
    mapL.set(hash(++i, NE, NEp1, NEp2, NEE));

    mapL.set(hash(++i, NN, NNN, p1));
    mapL.set(hash(++i, NN, NNN, p2));
    mapL.set(hash(++i, NN, NNp1, NNp2, NNN));

    mapL.set(hash(++i, WW, WWW, p1));
    mapL.set(hash(++i, WW, WWW, p2));
    mapL.set(hash(++i, WW, WWp1, WWp2, WWW));

    mapL.set(hash(++i, Np1, Np2, Wp1, Wp2));  // neighbor cross-plane
    mapL.set(hash(++i, NWp1, NWp2, NEp1, NEp2));

    mapL.set(hash(++i, (W + N - NW) >> 1, p1));  // strong         // gradient corrector
    mapL.set(hash(++i, (W + N - NW) >> 1, p1, p2)); // strong
    mapL.set(hash(++i, (N + NE - NNE) >> 1, p1)); // very strong    horizontal extrapolation error 
    mapL.set(hash(++i, (W * 2 - WW) >> 1, p1)); // W trend
    mapL.set(hash(++i, (N * 2 - NN) >> 1, p1)); // N trend


    assert(i - ((c0 << 2 | color) * 256) == nLSM);


    uint8_t b = (c0 << (8 - bpos));
    for (int i = 0; i < nSM; i++) {
      map[i].set(static_cast<uint8_t>(pOLS[i] - b) << 3 | bpos);
    }
  }
}

void Image24BitModel::init() {
  stride = 3 + alpha;
  padding = w % stride;
  x = color = line = 0;
  columns[0] = max(1, w / max(1, ilog2(w) * 3));
  columns[1] = max(1, columns[0] / max(1, ilog2(columns[0])));
  if( lastPos > 0 && false ) { // todo: when shall we reset?
    for (int i = 0; i < nLSM; i++) {
      mapL.reset();
    }
    for( int i = 0; i < nSM; i++ ) {
      map[i].reset();
    }
  }
  lossBuf.setSize(nextPowerOf2(LOSS_BUF_ROWS * w));
  lossBuf.fill(255);

  predErrBuf.setSize(nextPowerOf2(PRED_ERR_BUF_ROWS * w * nDM));
  predErrBuf.fill(255);
}

void Image24BitModel::setParam(int width, uint32_t alpha0) {
  w = width;
  alpha = alpha0;
}

void Image24BitModel::mix(Mixer &m) {
  INJECT_SHARED_bpos
  if( bpos == 0 ) {
    INJECT_SHARED_pos
    if(pos - lastPos != 1) {
      init();
    } else {
      x++;
      if( x >= w ) {
        x = 0;
        line++;
      }
    }
    lastPos = pos;
  }

  update();

  // predict next bit
  if (color != 4) {
    cm.mix(m);

    mapR1.mix(m);
    mapR2.mix(m);
    mapR3.mix(m);

    mapL.mix(m);

    for( int i = 0; i < nSM; i++ ) {
      map[i].mix(m);
    }

    // todo: use 3 separate mixers instead

    if (color == 0)
      m.setScaleFactor(670, 100);  // 650-680
    else if (color == 1)
      m.setScaleFactor(820, 100); // 760-880
    else if (color == 2)
      m.setScaleFactor(920, 100); // 650-690 for others 880-960


    // todo: review and optimize these mixer contexts

    INJECT_SHARED_bpos
    INJECT_SHARED_c0
    assert(color < 4);
    uint32_t colorbpos = color << 3 | bpos; // 0..23 or 0..31
    m.set(1 + colorbpos, 1 + 4 * 8);   // 1: account for padding zone

    m.set(((lossQ / 40u) /*0..15*/) << 2 | color, 16 * 4);
    m.set(((line & 7) << 5) | colorbpos, 256);
    m.set(min(63, column[0]) + ((ctx[0] >> 3) & 0xC0), 256);
    m.set(min(127, column[1]) + ((ctx[0] >> 2) & 0x180), 512);
    m.set((ctx[0] & 0x7FC) | (bpos >> 1), 2048);
    m.set(colorbpos + (static_cast<int>(c0 == ((0x100 | ((N + W + 1) >> 1)) >> (8 - bpos)))) * 32, 8 * 32);
    m.set(min(color, stride - 1) * 64 + (x % stride) * 16 + (bpos >> 1), 6 * 64);
    m.set((ctx[1] << 2) | (bpos >> 1), 1024);

    m.set(finalize64(hash(ctx[0], column[0] >> 3), 13), 8192);
    m.set(min(255, (x + line) >> 5), 256);
  }
  else {
    // padding zone
    m.add(-2047);  //predict 0
    m.set(0, MIXERCONTEXTS);
  }
}
