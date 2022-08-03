#include "Image24BitModel.hpp"

Image24BitModel::Image24BitModel(Shared* const sh, const uint64_t size) : 
  shared(sh), cm(sh, size, nCM, 64),
  SCMap { /* SmallStationaryContextMap : BitsOfContext, InputBits, Rate, Scale */
    {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74},
    {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74},
    {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74},
    {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74},
    {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74},
    {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74},
    {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74}, {sh,11,1,9,74},
    {sh,11,1,9,74}, {sh,11,1,9,74}
  },
  mapL {sh,nLSM,19,74}, /* LargeStationaryMap : Contexts, HashBits, Scale=64, Rate=16 */
  map{ /* StationaryMap : BitsOfContext, InputBits, Scale=64, Rate=16  */
    /*nSM0: 0- 7*/ {sh,8,8,74},  {sh,8,8,74},  {sh,8,8,74},  {sh,2,8,74},  {sh,15,1,74}, {sh,15,1,74}, {sh,15,1,74}, {sh,15,1,74},
    /*nSM0: 8-12*/ {sh,15,1,74}, {sh,13,1,74}, {sh,13,1,74}, {sh,13,1,74}, {sh,13,1,74},
    /*nSM1: 0- 8*/ {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74},
    /*nSM1: 9-17*/ {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74},
    /*nSM1:18-26*/ {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74},
    /*nSM1:27-35*/ {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74},
    /*nSM1:36-44*/ {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74},
    /*nSM1:45-53*/ {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74},
    /*nSM1:54-62*/ {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74},
    /*nSM1:63-71*/ {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74},
    /*nSM1:72-75*/ {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74},
    /*nOLS: 0- 5*/ {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}, {sh,11,1,74}
  }
  {}

void Image24BitModel::update() {
  INJECT_SHARED_bpos
  INJECT_SHARED_c1
  if( bpos == 0 ) {
    INJECT_SHARED_buf
    if( x + padding < w ) {
      color++;
      if( color >= stride ) {
        color = 0;
      }
    } else {
      if( padding > 0 ) {
        color = stride;
      } else {
        color = 0;
      }
    }

    column[0] = x / columns[0];
    column[1] = x / columns[1];
    WWWWWW = buf(6 * stride), WWWWW = buf(5 * stride), WWWW = buf(4 * stride), WWW = buf(3 * stride), WW = buf(2 * stride), W = buf(stride);
    NWWWW = buf(w + 4 * stride), NWWW = buf(w + 3 * stride), NWW = buf(w + 2 * stride), NW = buf(w + stride), N = buf(w), NE = buf(w - stride), NEE = buf(w - 2 * stride), NEEE = buf(w - 3 * stride), NEEEE = buf(w - 4 * stride);
    NNWWW = buf(w * 2 + stride * 3), NNWW = buf((w + stride) * 2), NNW = buf(w * 2 + stride), NN = buf(w * 2), NNE = buf(w * 2 - stride), NNEE = buf((w - stride) * 2), NNEEE = buf(w * 2 - stride * 3);
    NNNWW = buf(w * 3 + stride * 2), NNNW = buf(w * 3 + stride), NNN = buf(w * 3), NNNE = buf(w * 3 - stride), NNNEE = buf(w * 3 - stride * 2);
    NNNNW = buf(w * 4 + stride), NNNN = buf(w * 4), NNNNE = buf(w * 4 - stride);
    NNNNN = buf(w * 5);
    NNNNNN = buf(w * 6);
    WWp1 = buf(stride * 2 + 1), Wp1 = buf(stride + 1), p1 = buf(1), NWp1 = buf(w + stride + 1), Np1 = buf(w + 1), NEp1 = buf(w - stride + 1), NNp1 = buf(w * 2 + 1);
    WWp2 = buf(stride * 2 + 2), Wp2 = buf(stride + 2), p2 = buf(2), NWp2 = buf(w + stride + 2), Np2 = buf(w + 2), NEp2 = buf(w - stride + 2), NNp2 = buf(w * 2 + 2);

    int j = -1;
    mapContexts[++j] = clamp4(N + p1 - Np1, W, NW, N, NE);
    mapContexts[++j] = clamp4(N + p2 - Np2, W, NW, N, NE);
    mapContexts[++j] = (W + clamp4(NE * 3 - NNE * 3 + NNNE, W, N, NE, NEE)) / 2;
    mapContexts[++j] = clamp4((W + clip(NE * 2 - NNE)) / 2, W, NW, N, NE);
    mapContexts[++j] = (W + NEE) / 2;
    mapContexts[++j] = ((WWW - 4 * WW + 6 * W + (NE * 4 - NNE * 6 + NNNE * 4 - NNNNE)) / 4);
    mapContexts[++j] = ((-WWWW + 5 * WWW - 10 * WW + 10 * W + clamp4(NE * 4 - NNE * 6 + NNNE * 4 - NNNNE, N, NE, NEE, NEEE)) / 5);
    mapContexts[++j] = ((-4 * WW + 15 * W + 10 * (NE * 3 - NNE * 3 + NNNE) - (NEEE * 3 - NNEEE * 3 + buf(w * 3 - 3 * stride))) / 20);
    mapContexts[++j] = ((-3 * WW + 8 * W + clamp4(NEE * 3 - NNEE * 3 + NNNEE, NE, NEE, NEEE, NEEEE)) / 6);
    mapContexts[++j] = ((W + (NE * 2 - NNE)) / 2 + p1 - (Wp1 + (NEp1 * 2 - buf(w * 2 - stride + 1))) / 2);
    mapContexts[++j] = ((W + (NE * 2 - NNE)) / 2 + p2 - (Wp2 + (NEp2 * 2 - buf(w * 2 - stride + 2))) / 2);
    mapContexts[++j] = ((-3 * WW + 8 * W + (NEE * 2 - NNEE)) / 6 + p1 -(-3 * WWp1 + 8 * Wp1 + (buf(w - stride * 2 + 1) * 2 - buf(w * 2 - stride * 2 + 1))) / 6);
    mapContexts[++j] = ((-3 * WW + 8 * W + (NEE * 2 - NNEE)) / 6 + p2 -(-3 * WWp2 + 8 * Wp2 + (buf(w - stride * 2 + 2) * 2 - buf(w * 2 - stride * 2 + 2))) / 6);
    mapContexts[++j] = ((W + NEE) / 2 + p1 - (Wp1 + buf(w - stride * 2 + 1)) / 2);
    mapContexts[++j] = ((W + NEE) / 2 + p2 - (Wp2 + buf(w - stride * 2 + 2)) / 2);
    mapContexts[++j] = ((WW + (NEE * 2 - NNEE)) / 2 + p1 - (WWp1 + (buf(w - stride * 2 + 1) * 2 - buf(w * 2 - stride * 2 + 1))) / 2);
    mapContexts[++j] = ((WW + (NEE * 2 - NNEE)) / 2 + p2 - (WWp2 + (buf(w - stride * 2 + 2) * 2 - buf(w * 2 - stride * 2 + 2))) / 2);
    mapContexts[++j] = (WW + NEE - N + p1 - (WWp1 + buf(w - stride * 2 + 1) - Np1));
    mapContexts[++j] = (WW + NEE - N + p2 - (WWp2 + buf(w - stride * 2 + 2) - Np2));
    mapContexts[++j] = (W + N - NW);
    mapContexts[++j] = (W + N - NW + p1 - (Wp1 + Np1 - NWp1));
    mapContexts[++j] = (W + N - NW + p2 - (Wp2 + Np2 - NWp2));
    mapContexts[++j] = (W + NE - N);
    mapContexts[++j] = (N + NW - NNW);
    mapContexts[++j] = (N + NW - NNW + p1 - (Np1 + NWp1 - buf(w * 2 + stride + 1)));
    mapContexts[++j] = (N + NW - NNW + p2 - (Np2 + NWp2 - buf(w * 2 + stride + 2)));
    mapContexts[++j] = (N + NE - NNE);
    mapContexts[++j] = (N + NE - NNE + p1 - (Np1 + NEp1 - buf(w * 2 - stride + 1)));
    mapContexts[++j] = (N + NE - NNE + p2 - (Np2 + NEp2 - buf(w * 2 - stride + 2)));
    mapContexts[++j] = (N + NN - NNN);
    mapContexts[++j] = (N + NN - NNN + p1 - (Np1 + NNp1 - buf(w * 3 + 1)));
    mapContexts[++j] = (N + NN - NNN + p2 - (Np2 + NNp2 - buf(w * 3 + 2)));
    mapContexts[++j] = (W + WW - WWW);
    mapContexts[++j] = (W + WW - WWW + p1 - (Wp1 + WWp1 - buf(stride * 3 + 1)));
    mapContexts[++j] = (W + WW - WWW + p2 - (Wp2 + WWp2 - buf(stride * 3 + 2)));
    mapContexts[++j] = (W + NEE - NE);
    mapContexts[++j] = (W + NEE - NE + p1 - (Wp1 + buf(w - stride * 2 + 1) - NEp1));
    mapContexts[++j] = (W + NEE - NE + p2 - (Wp2 + buf(w - stride * 2 + 2) - NEp2));
    mapContexts[++j] = (NN + p1 - NNp1);
    mapContexts[++j] = (NN + p2 - NNp2);
    mapContexts[++j] = (NN + W - NNW);
    mapContexts[++j] = (NN + W - NNW + p1 - (NNp1 + Wp1 - buf(w * 2 + stride + 1)));
    mapContexts[++j] = (NN + W - NNW + p2 - (NNp2 + Wp2 - buf(w * 2 + stride + 2)));
    mapContexts[++j] = (NN + NW - NNNW);
    mapContexts[++j] = (NN + NW - NNNW + p1 - (NNp1 + NWp1 - buf(w * 3 + stride + 1)));
    mapContexts[++j] = (NN + NW - NNNW + p2 - (NNp2 + NWp2 - buf(w * 3 + stride + 2)));
    mapContexts[++j] = (NN + NE - NNNE);
    mapContexts[++j] = (NN + NE - NNNE + p1 - (NNp1 + NEp1 - buf(w * 3 - stride + 1)));
    mapContexts[++j] = (NN + NE - NNNE + p2 - (NNp2 + NEp2 - buf(w * 3 - stride + 2)));
    mapContexts[++j] = (NN + NNNN - NNNNNN);
    mapContexts[++j] = (NN + NNNN - NNNNNN + p1 - (NNp1 + buf(w * 4 + 1) - buf(w * 6 + 1)));
    mapContexts[++j] = (NN + NNNN - NNNNNN + p2 - (NNp2 + buf(w * 4 + 2) - buf(w * 6 + 2)));
    mapContexts[++j] = (WW + p1 - WWp1);
    mapContexts[++j] = (WW + p2 - WWp2);
    mapContexts[++j] = (WW + WWWW - WWWWWW);
    mapContexts[++j] = (WW + WWWW - WWWWWW + p1 - (WWp1 + buf(stride * 4 + 1) - buf(stride * 6 + 1)));
    mapContexts[++j] = (WW + WWWW - WWWWWW + p2 - (WWp2 + buf(stride * 4 + 2) - buf(stride * 6 + 2)));
    mapContexts[++j] = (N * 2 - NN + p1 - (Np1 * 2 - NNp1));
    mapContexts[++j] = (N * 2 - NN + p2 - (Np2 * 2 - NNp2));
    mapContexts[++j] = (W * 2 - WW + p1 - (Wp1 * 2 - WWp1));
    mapContexts[++j] = (W * 2 - WW + p2 - (Wp2 * 2 - WWp2));
    mapContexts[++j] = (N * 3 - NN * 3 + NNN);
    mapContexts[++j] = clamp4(N * 3 - NN * 3 + NNN, W, NW, N, NE);
    mapContexts[++j] = clamp4(W * 3 - WW * 3 + WWW, W, NW, N, NE);
    mapContexts[++j] = clamp4(N * 2 - NN, W, NW, N, NE);
    mapContexts[++j] = ((NNNNN - 6 * NNNN + 15 * NNN - 20 * NN + 15 * N +
                             clamp4(W * 4 - NWW * 6 + NNWWW * 4 - buf(w * 3 + 4 * stride), W, NW, N, NN)) / 6);
    mapContexts[++j] = ((buf(w * 3 - 3 * stride) - 4 * NNEE + 6 * NE + (W * 4 - NW * 6 + NNW * 4 - NNNW)) / 4);
    mapContexts[++j] = (((N + 3 * NW) / 4) * 3 - ((NNW + NNWW) / 2) * 3 + (NNNWW * 3 + buf(w * 3 + 3 * stride)) / 4);
    mapContexts[++j] = ((W * 2 + NW) - (WW + 2 * NWW) + NWWW);
    mapContexts[++j] = ((W * 2 - NW) + (W * 2 - NWW) + N + NE) / 4;
    mapContexts[++j] = (N + W + 1) >> 1;
    mapContexts[++j] = (NEEEE + buf(w - 6 * stride) + 1) >> 1;
    mapContexts[++j] = (WWWWWW + WWWW + 1) >> 1;
    mapContexts[++j] = ((W + N) * 3 - NW * 2) >> 2;
    mapContexts[++j] = N;
    mapContexts[++j] = NN;
    assert(++j == nSM1);
    j = -1;
    scMapContexts[++j] = N + p1 - Np1;
    scMapContexts[++j] = N + p2 - Np2;
    scMapContexts[++j] = W + p1 - Wp1;
    scMapContexts[++j] = W + p2 - Wp2;
    scMapContexts[++j] = NW + p1 - NWp1;
    scMapContexts[++j] = NW + p2 - NWp2;
    scMapContexts[++j] = NE + p1 - NEp1;
    scMapContexts[++j] = NE + p2 - NEp2;
    scMapContexts[++j] = NN + p1 - NNp1;
    scMapContexts[++j] = NN + p2 - NNp2;
    scMapContexts[++j] = WW + p1 - WWp1;
    scMapContexts[++j] = WW + p2 - WWp2;
    scMapContexts[++j] = W + N - NW;
    scMapContexts[++j] = W + N - NW + p1 - Wp1 - Np1 + NWp1;
    scMapContexts[++j] = W + N - NW + p2 - Wp2 - Np2 + NWp2;
    scMapContexts[++j] = W + NE - N;
    scMapContexts[++j] = W + NE - N + p1 - Wp1 - NEp1 + Np1;
    scMapContexts[++j] = W + NE - N + p2 - Wp2 - NEp2 + Np2;
    scMapContexts[++j] = W + NEE - NE;
    scMapContexts[++j] = W + NEE - NE + p1 - Wp1 - buf(w - stride * 2 + 1) + NEp1;
    scMapContexts[++j] = W + NEE - NE + p2 - Wp2 - buf(w - stride * 2 + 2) + NEp2;
    scMapContexts[++j] = N + NN - NNN;
    scMapContexts[++j] = N + NN - NNN + p1 - Np1 - NNp1 + buf(w * 3 + 1);
    scMapContexts[++j] = N + NN - NNN + p2 - Np2 - NNp2 + buf(w * 3 + 2);
    scMapContexts[++j] = N + NE - NNE;
    scMapContexts[++j] = N + NE - NNE + p1 - Np1 - NEp1 + buf(w * 2 - stride + 1);
    scMapContexts[++j] = N + NE - NNE + p2 - Np2 - NEp2 + buf(w * 2 - stride + 2);
    scMapContexts[++j] = N + NW - NNW;
    scMapContexts[++j] = N + NW - NNW + p1 - Np1 - NWp1 + buf(w * 2 + stride + 1);
    scMapContexts[++j] = N + NW - NNW + p2 - Np2 - NWp2 + buf(w * 2 + stride + 2);
    scMapContexts[++j] = NE + NW - NN;
    scMapContexts[++j] = NE + NW - NN + p1 - NEp1 - NWp1 + NNp1;
    scMapContexts[++j] = NE + NW - NN + p2 - NEp2 - NWp2 + NNp2;
    scMapContexts[++j] = NW + W - NWW;
    scMapContexts[++j] = NW + W - NWW + p1 - NWp1 - Wp1 + buf(w + stride * 2 + 1);
    scMapContexts[++j] = NW + W - NWW + p2 - NWp2 - Wp2 + buf(w + stride * 2 + 2);
    scMapContexts[++j] = W * 2 - WW;
    scMapContexts[++j] = W * 2 - WW + p1 - Wp1 * 2 + WWp1;
    scMapContexts[++j] = W * 2 - WW + p2 - Wp2 * 2 + WWp2;
    scMapContexts[++j] = N * 2 - NN;
    scMapContexts[++j] = N * 2 - NN + p1 - Np1 * 2 + NNp1;
    scMapContexts[++j] = N * 2 - NN + p2 - Np2 * 2 + NNp2;
    scMapContexts[++j] = NW * 2 - NNWW;
    scMapContexts[++j] = NW * 2 - NNWW + p1 - NWp1 * 2 + buf(w * 2 + stride * 2 + 1);
    scMapContexts[++j] = NW * 2 - NNWW + p2 - NWp2 * 2 + buf(w * 2 + stride * 2 + 2);
    scMapContexts[++j] = NE * 2 - NNEE;
    scMapContexts[++j] = NE * 2 - NNEE + p1 - NEp1 * 2 + buf(w * 2 - stride * 2 + 1);
    scMapContexts[++j] = NE * 2 - NNEE + p2 - NEp2 * 2 + buf(w * 2 - stride * 2 + 2);
    scMapContexts[++j] = N * 3 - NN * 3 + NNN + p1 - Np1 * 3 + NNp1 * 3 - buf(w * 3 + 1);
    scMapContexts[++j] = N * 3 - NN * 3 + NNN + p2 - Np2 * 3 + NNp2 * 3 - buf(w * 3 + 2);
    scMapContexts[++j] = N * 3 - NN * 3 + NNN;
    scMapContexts[++j] = (W + NE * 2 - NNE + 1) >> 1;
    scMapContexts[++j] = (W + NE * 3 - NNE * 3 + NNNE+1) >> 1;
    scMapContexts[++j] = (W + NE * 2 - NNE) / 2 + p1 - (Wp1 + NEp1 * 2 - buf(w * 2 - stride + 1)) / 2;
    scMapContexts[++j] = (W + NE * 2 - NNE) / 2 + p2 - (Wp2 + NEp2 * 2 - buf(w * 2 - stride + 2)) / 2;
    scMapContexts[++j] = NNE + NE - NNNE;
    scMapContexts[++j] = NNE + W - NN;
    scMapContexts[++j] = NNW + W - NNWW;
    assert(++j == nSSM);
    j = 0;
    for( int k = (color > 0) ? color - 1 : stride - 1; j < nOLS; j++ ) {
      pOLS[j] = clip(int(floor(ols[j][color].predict(olsCtxs[j]))));
      ols[j][k].update(p1);
    }

    int mean = (W + NW + N + NE + 2) >> 2;
    int diff4 =
      DiffQt(W, N, 4) << 12 | 
      DiffQt(NW, NE, 4) << 8 |
      DiffQt(NW, N, 4) << 4 |
      DiffQt(W, NE, 4);

    uint64_t i = color * 1024;
    const uint8_t __ = 0;
    cm.set(__, hash(++i, (N + 1) >> 1, DiffQt(N, (NN * 2 - NNN))));
    cm.set(__, hash(++i, (W + 1) >> 1, DiffQt(W, (WW * 2 - WWW))));
    cm.set(__, hash(++i, clamp4(W + N - NW, W, NW, N, NE), DiffQt((N + NE - NNE), (N + NW - NNW))));
    cm.set(__, hash(++i, (NNN + N + 4) >> 3, (N * 3 - NN * 3 + NNN) >> 1));
    cm.set(__, hash(++i, (WWW + W + 4) >> 3, (W * 3 - WW * 3 + WWW) >> 1));
    cm.set(__, hash(++i, (W + (NE * 3 - NNE * 3 + NNNE)) >> 2, DiffQt(N, (NW + NE) >>1)));
    cm.set(__, hash(++i, ((-WWWW + 5 * WWW - 10 * WW + 10 * W + clamp4(NE * 4 - NNE * 6 + NNNE * 4 - NNNNE, N, NE, NEE, NEEE)) / 5) / 4));
    cm.set(__, hash(++i, (NEE + N - NNEE), DiffQt(W, (NW + NE - NNE))));
    cm.set(__, hash(++i, (NN + W - NNW), DiffQt(W, (NNW + WW - NNWW))));
    cm.set(__, hash(++i, p1));
    cm.set(__, hash(++i, p2));
    cm.set(__, hash(++i, (W + N - NW) >>1, (W + p1 - Wp1) >>1));
    cm.set(__, hash(++i, (N * 2 - NN) >>1, DiffQt(N, (NN * 2 - NNN))));
    cm.set(__, hash(++i, (W * 2 - WW) >>1, DiffQt(W, (WW * 2 - WWW))));
    cm.set(__, hash(++i, clamp4(N * 3 - NN * 3 + NNN, W, NW, N, NE) >> 1));
    cm.set(__, hash(++i, clamp4(W * 3 - WW * 3 + WWW, W, N, NE, NEE) >> 1));
    cm.set(__, hash(++i, DiffQt(W, Wp1), clamp4((p1 * W) / (Wp1 < 1 ? 1 : Wp1), W, N, NE, NEE))); //using max(1,Wp1) results in division by zero in VC2015
    cm.set(__, hash(++i, clamp4(N + p2 - Np2, W, NW, N, NE)));
    cm.set(__, hash(++i, (W + N - NW), column[0]));
    cm.set(__, hash(++i, (N * 2 - NN), DiffQt(W, (NW * 2 - NNW))));
    cm.set(__, hash(++i, (W * 2 - WW), DiffQt(N, (NW * 2 - NWW))));
    cm.set(__, hash(++i, (W + NEE + 1) >> 1, DiffQt(W, (WW + NE + 1) >> 1)));
    cm.set(__, hash(++i, (clamp4((W * 2 - WW) + (N * 2 - NN) - (NW * 2 - NNWW), W, NW, N, NE))));
    cm.set(__, hash(++i, W, p2));
    cm.set(__, hash(++i, N, NN, NNN));
    cm.set(__, hash(++i, W, WW, WWW));
    cm.set(__, hash(++i, N, column[0]));
    cm.set(__, hash(++i, (W + NEE - NE), DiffQt(W, (WW + NE - N))));
    cm.set(__, hash(++i, NN, NNNN, NNNNNN, column[1]));
    cm.set(__, hash(++i, WW, WWWW, WWWWWW, column[1]));
    cm.set(__, hash(++i, NNN, NNNNNN, buf(w * 9), column[1]));
    cm.set(__, hash(++i, column[1]));

    cm.set(__, hash(++i, W, DiffQt(W, WW)));
    cm.set(__, hash(++i, W, p1));
    cm.set(__, hash(++i, W >> 2, DiffQt(W, p1), DiffQt(W, p2)));
    cm.set(__, hash(++i, N, DiffQt(N, NN)));
    cm.set(__, hash(++i, N, p1));
    cm.set(__, hash(++i, N >> 2, DiffQt(N, p1), DiffQt(N, p2)));
    cm.set(__, hash(++i, (W + N + 4) >> 3, p1 >> 4, p2 >> 4));
    cm.set(__, hash(++i, p1 >> 2, p2 >> 2));
    cm.set(__, hash(++i, W, p1 - Wp1));
    cm.set(__, hash(++i, W + p1 - Wp1));
    cm.set(__, hash(++i, N, p1 - Np1));
    cm.set(__, hash(++i, N + p1 - Np1));
    cm.set(__, hash(++i, mean, diff4));

    ctx[0] = (min(color, 
      stride - 1) << 9) | 
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
      

    j = -1;
    map[++j].set((W & 0xC0) | ((N & 0xC0) >> 2) | ((WW & 0xC0) >> 4) | (NN >> 6));
    map[++j].set((N & 0xC0) | ((NN & 0xC0) >> 2) | ((NE & 0xC0) >> 4) | (NEE >> 6));
    map[++j].set(buf(1));
    map[++j].set(min(color, stride - 1));
    shared->State.Image.plane = min(color, stride - 1);
    shared->State.Image.pixels.W = W;
    shared->State.Image.pixels.N = N;
    shared->State.Image.pixels.NN = NN;
    shared->State.Image.pixels.WW = WW;
    shared->State.Image.pixels.Wp1 = Wp1;
    shared->State.Image.pixels.Np1 = Np1;
    shared->State.Image.ctx = ctx[0] >> 3;
  }
  
  INJECT_SHARED_c0
  uint8_t b = (c0 << (8 - bpos));
  int i = 3;

  map[++i].set(((static_cast<uint8_t>((W + N - NW) - b)) << 3 | bpos) | 
                     (DiffQt((N + NE - NNE), (N + NW - NNW)) << 11));
  map[++i].set(((static_cast<uint8_t>((N * 2 - NN) - b)) << 3 | bpos) | 
                     (DiffQt(W, (NW * 2 - NNW)) << 11));
  map[++i].set(((static_cast<uint8_t>((W * 2 - WW) - b)) << 3 | bpos) |
                     (DiffQt(N, (NW * 2 - NWW)) << 11));
  map[++i].set(((static_cast<uint8_t>((W + N - NW) - b)) << 3 | bpos) |
                     (DiffQt(p1, (Wp1 + Np1 - NWp1)) << 11));
  map[++i].set(((static_cast<uint8_t>((W + N - NW) - b)) << 3 | bpos) |
                     (DiffQt(p2, (Wp2 + Np2 - NWp2)) << 11));
  map[++i].set((min(color, stride - 1) << 11) | ((static_cast<uint8_t>((N + p1 - Np1) - b)) << 3 | bpos));
  map[++i].set((min(color, stride - 1) << 11) | ((static_cast<uint8_t>((N + p2 - Np2) - b)) << 3 | bpos));
  map[++i].set((min(color, stride - 1) << 11) | ((static_cast<uint8_t>((W + p1 - Wp1) - b)) << 3 | bpos));
  map[++i].set((min(color, stride - 1) << 11) | ((static_cast<uint8_t>((W + p2 - Wp2) - b)) << 3 | bpos));
  mapL.set(hash(0 << 3 | bpos, static_cast<uint8_t>(W - b), static_cast<uint8_t>(N - b)));
  mapL.set(hash(1 << 3 | bpos, static_cast<uint8_t>(W - b), static_cast<uint8_t>(WW - b)));
  mapL.set(hash(2 << 3 | bpos, static_cast<uint8_t>(N - b), static_cast<uint8_t>(NN - b)));
  mapL.set(hash(3 << 3 | bpos, static_cast<uint8_t>((N + NE - NNE) - b), static_cast<uint8_t>((N + NW - NNW) - b)));
  ++i;
  assert(i == nSM0);

  for( int j = 0; j < nSM1; i++, j++ ) {
    map[i].set(static_cast<uint8_t>(mapContexts[j] - b) <<3 | bpos);
  }

  for( int j = 0; i < nSM; i++, j++ ) {
    map[i].set(static_cast<uint8_t>(pOLS[j] - b) << 3 | bpos);
  }

  for( int i = 0; i < nSSM; i++ ) {
    SCMap[i].set(static_cast<uint8_t>(scMapContexts[i] - b) << 3 | bpos);
  }
}

void Image24BitModel::init() {
  stride = 3 + alpha;
  padding = w % stride;
  x = color = line = 0;
  columns[0] = max(1, w / max(1, ilog2(w) * 3));
  columns[1] = max(1, columns[0] / max(1, ilog2(columns[0])));
  if( lastPos > 0 && false ) { // todo: when shall we reset ?
    for (int i = 0; i < nLSM; i++) {
      mapL.reset();
    }
    for( int i = 0; i < nSM; i++ ) {
      map[i].reset();
    }
  }
}

void Image24BitModel::setParam(int width, uint32_t alpha0) {
  w = width;
  alpha = alpha0;
}

void Image24BitModel::mix(Mixer &m) {
  INJECT_SHARED_bpos
  if( bpos == 0 ) {
    INJECT_SHARED_pos
    if((color < 0) || (pos - lastPos != 1)) {
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
  cm.mix(m);

  const int order = max(0, cm.order - (nCM - 31)); //0-31
  assert(0 <= order && order <= 31);
  m.set((order >> 1) << 3 | bpos, 16 * 8);

  mapL.mix(m);

  for( int i = 0; i < nSM; i++ ) {
    map[i].mix(m);
  }

  for( int i = 0; i < nSSM; i++ ) {
    SCMap[i].mix(m);
  }

  if( ++col >= stride * 8 ) {
    col = 0;
  }
  m.set((((line & 0x7) << 5) | col), 256);
  m.set(min(63, column[0]) + ((ctx[0] >> 3) & 0xC0), 256);
  m.set(min(127, column[1]) + ((ctx[0] >> 2) & 0x180), 512);
  m.set((ctx[0] & 0x7FC) | (bpos >> 1), 2048);
  INJECT_SHARED_c0
  m.set(col + (static_cast<int>(c0 == ((0x100 | ((N + W + 1) >> 1)) >> (8 - bpos)))) * 32, 8 * 32);
  m.set(min(color, stride - 1), 4);
  m.set((c0 - 1) << 2 | min(color, stride - 1), 255 * 4);
  m.set((ctx[1] << 2) | (bpos >> 1), 1024);

  int trendN = (N >= NN && NN >= NNN) || (N <= NN && NN <= NNN);
  int trendW = (W >= WW && WW >= WWW) || (W <= WW && WW <= WWW);
  int trend = trendN << 1 | trendW;
  m.set(finalize64(hash(DiffQt(W, WW, 5), DiffQt(N, NN, 5), DiffQt(W, N, 5), trend, color), 13), 8192);
  m.set(finalize64(hash(ctx[0], column[0] >> 3), 13), 8192);
  m.set(finalize64(hash(logQt(N, 5), DiffQt(N, NN) >> 1, c0), 13), 8192);
  m.set(finalize64(hash(logQt(W, 5), DiffQt(W, WW) >> 1, c0), 13), 8192);
  m.set(min(255, (x + line) >> 5), 256);
}
