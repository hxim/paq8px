#include "Image24BitModel.hpp"

Image24BitModel::Image24BitModel(Shared* const sh, const uint64_t size) : 
  shared(sh), cm(sh, size, nCM, 64, CM_USE_RUN_STATS),
  SCMap { /* SmallStationaryContextMap : BitsOfContext, InputBits, Rate, Scale */
    {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86},
    {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86},
    {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86},
    {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86},
    {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86},
    {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86},
    {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86}, {sh,11,1,9,86},
    {sh,11,1,9,86}, {sh,11,1,9,86}, {sh, 0,8,9,86}
  },
  map{ /* StationaryMap: BitsOfContext, InputBits, Scale, Limit  */
    /*nSM0: 0- 8*/  {sh,8,8,86,1023},  {sh,8,8,86,1023},  {sh,8,8,86,1023},  {sh,2,8,86,1023},  {sh,0,8,86,1023}, {sh,15,1,86,1023}, {sh,15,1,86,1023}, {sh,15,1,86,1023}, {sh,15,1,86,1023},
    /*nSM0: 9-17*/ {sh,15,1,86,1023}, {sh,17,1,86,1023}, {sh,17,1,86,1023}, {sh,17,1,86,1023}, {sh,17,1,86,1023}, {sh,13,1,86,1023}, {sh,13,1,86,1023}, {sh,13,1,86,1023}, {sh,13,1,86,1023},
    /*nSM1: 0- 8*/ {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023},
    /*nSM1: 9-17*/ {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023},
    /*nSM1:18-26*/ {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023},
    /*nSM1:27-35*/ {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023},
    /*nSM1:36-44*/ {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023},
    /*nSM1:45-53*/ {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023},
    /*nSM1:54-62*/ {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023},
    /*nSM1:63-71*/ {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023},
    /*nSM1:72-75*/ {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023},
    /*nOLS: 0- 5*/ {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}, {sh,11,1,86,1023}
  }
{}

void Image24BitModel::update() {
  INJECT_SHARED_bpos
  INJECT_SHARED_c1
  if( bpos == 0 ) {
    INJECT_SHARED_buf
    if( x == 1 && (isPNG != 0u)) {
      filter = c1;
    } else {
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
      if( isPNG != 0u ) {
        switch( filter ) {
          case 1: {
            buffer.add(static_cast<uint8_t>(c1 + buffer(stride) * static_cast<int>(x > stride + 1 || (x == 0))));
            filterOn = x > stride;
            px = buffer(stride);
            break;
          }
          case 2: {
            buffer.add(static_cast<uint8_t>(c1 + buffer(w) * static_cast<int>((filterOn = (line > 0)))));
            px = buffer(w);
            break;
          }
          case 3: {
            buffer.add(static_cast<uint8_t>(c1 + (buffer(w) * static_cast<int>(line > 0) +
                                                          buffer(stride) * static_cast<int>(x > stride + 1 || (x == 0))) / 2));
            filterOn = (x > stride || line > 0);
            px = (buffer(stride) * static_cast<int>(x > stride) + buffer(w) * static_cast<int>(line > 0)) / 2;
            break;
          }
          case 4: {
            buffer.add(static_cast<uint8_t>(c1 + paeth(buffer(stride) * static_cast<int>(x > stride + 1 || (x == 0)),
                                                               buffer(w) * static_cast<int>(line > 0), buffer(w + stride) *
                                                                                                       static_cast<int>(line > 0 &&
                                                                                                                        (x > stride + 1 ||
                                                                                                                         (x == 0))))));
            filterOn = (x > stride || line > 0);
            px = paeth(buffer(stride) * static_cast<int>(x > stride), buffer(w) * static_cast<int>(line > 0),
                       buffer(w + stride) * static_cast<int>(x > stride && line > 0));
            break;
          }
          default:
            buffer.add(c1);
            filterOn = false;
            px = 0;
        }
        if( !filterOn ) {
          px = 0;
        }
      } else {
        buffer.add(c1);
      }
    }

    if( x > 0 || (isPNG == 0u)) {
      column[0] = (x - isPNG) / columns[0];
      column[1] = (x - isPNG) / columns[1];
      WWWWWW = buffer(6 * stride), WWWWW = buffer(5 * stride), WWWW = buffer(4 * stride), WWW = buffer(3 * stride), WW = buffer(2 * stride), W = buffer(stride);
      NWWWW = buffer(w + 4 * stride), NWWW = buffer(w + 3 * stride), NWW = buffer(w + 2 * stride), NW = buffer(w + stride), N = buffer(w), NE = buffer(w - stride), NEE = buffer(w - 2 * stride), NEEE = buffer(w - 3 * stride), NEEEE = buffer(w - 4 * stride);
      NNWWW = buffer(w * 2 + stride * 3), NNWW = buffer((w + stride) * 2), NNW = buffer(w * 2 + stride), NN = buffer(w * 2), NNE = buffer(w * 2 - stride), NNEE = buffer((w - stride) * 2), NNEEE = buffer(w * 2 - stride * 3);
      NNNWW = buffer(w * 3 + stride * 2), NNNW = buffer(w * 3 + stride), NNN = buffer(w * 3), NNNE = buffer(w * 3 - stride), NNNEE = buffer(w * 3 - stride * 2);
      NNNNW = buffer(w * 4 + stride), NNNN = buffer(w * 4), NNNNE = buffer(w * 4 - stride);
      NNNNN = buffer(w * 5);
      NNNNNN = buffer(w * 6);
      WWp1 = buffer(stride * 2 + 1), Wp1 = buffer(stride + 1), p1 = buffer(1), NWp1 = buffer(w + stride + 1), Np1 = buffer(w + 1), NEp1 = buffer(w - stride + 1), NNp1 = buffer(w * 2 + 1);
      WWp2 = buffer(stride * 2 + 2), Wp2 = buffer(stride + 2), p2 = buffer(2), NWp2 = buffer(w + stride + 2), Np2 = buffer(w + 2), NEp2 = buffer(w - stride + 2), NNp2 = buffer(w * 2 + 2);

      int j = -1;
      mapContexts[++j] = clamp4(N + p1 - Np1, W, NW, N, NE);
      mapContexts[++j] = clamp4(N + p2 - Np2, W, NW, N, NE);
      mapContexts[++j] = (W + clamp4(NE * 3 - NNE * 3 + NNNE, W, N, NE, NEE)) / 2;
      mapContexts[++j] = clamp4((W + clip(NE * 2 - NNE)) / 2, W, NW, N, NE);
      mapContexts[++j] = (W + NEE) / 2;
      mapContexts[++j] = clip((WWW - 4 * WW + 6 * W + clip(NE * 4 - NNE * 6 + NNNE * 4 - NNNNE)) / 4);
      mapContexts[++j] = clip((-WWWW + 5 * WWW - 10 * WW + 10 * W + clamp4(NE * 4 - NNE * 6 + NNNE * 4 - NNNNE, N, NE, NEE, NEEE)) / 5);
      mapContexts[++j] = clip((-4 * WW + 15 * W + 10 * clip(NE * 3 - NNE * 3 + NNNE) - clip(NEEE * 3 - NNEEE * 3 + buffer(w * 3 - 3 * stride))) / 20);
      mapContexts[++j] = clip((-3 * WW + 8 * W + clamp4(NEE * 3 - NNEE * 3 + NNNEE, NE, NEE, NEEE, NEEEE)) / 6);
      mapContexts[++j] = clip((W + clip(NE * 2 - NNE)) / 2 + p1 - (Wp1 + clip(NEp1 * 2 - buffer(w * 2 - stride + 1))) / 2);
      mapContexts[++j] = clip((W + clip(NE * 2 - NNE)) / 2 + p2 - (Wp2 + clip(NEp2 * 2 - buffer(w * 2 - stride + 2))) / 2);
      mapContexts[++j] = clip((-3 * WW + 8 * W + clip(NEE * 2 - NNEE)) / 6 + p1 -(-3 * WWp1 + 8 * Wp1 + clip(buffer(w - stride * 2 + 1) * 2 - buffer(w * 2 - stride * 2 + 1))) / 6);
      mapContexts[++j] = clip((-3 * WW + 8 * W + clip(NEE * 2 - NNEE)) / 6 + p2 -(-3 * WWp2 + 8 * Wp2 + clip(buffer(w - stride * 2 + 2) * 2 - buffer(w * 2 - stride * 2 + 2))) / 6);
      mapContexts[++j] = clip((W + NEE) / 2 + p1 - (Wp1 + buffer(w - stride * 2 + 1)) / 2);
      mapContexts[++j] = clip((W + NEE) / 2 + p2 - (Wp2 + buffer(w - stride * 2 + 2)) / 2);
      mapContexts[++j] = clip((WW + clip(NEE * 2 - NNEE)) / 2 + p1 - (WWp1 + clip(buffer(w - stride * 2 + 1) * 2 - buffer(w * 2 - stride * 2 + 1))) / 2);
      mapContexts[++j] = clip((WW + clip(NEE * 2 - NNEE)) / 2 + p2 - (WWp2 + clip(buffer(w - stride * 2 + 2) * 2 - buffer(w * 2 - stride * 2 + 2))) / 2);
      mapContexts[++j] = clip(WW + NEE - N + p1 - clip(WWp1 + buffer(w - stride * 2 + 1) - Np1));
      mapContexts[++j] = clip(WW + NEE - N + p2 - clip(WWp2 + buffer(w - stride * 2 + 2) - Np2));
      mapContexts[++j] = clip(W + N - NW);
      mapContexts[++j] = clip(W + N - NW + p1 - clip(Wp1 + Np1 - NWp1));
      mapContexts[++j] = clip(W + N - NW + p2 - clip(Wp2 + Np2 - NWp2));
      mapContexts[++j] = clip(W + NE - N);
      mapContexts[++j] = clip(N + NW - NNW);
      mapContexts[++j] = clip(N + NW - NNW + p1 - clip(Np1 + NWp1 - buffer(w * 2 + stride + 1)));
      mapContexts[++j] = clip(N + NW - NNW + p2 - clip(Np2 + NWp2 - buffer(w * 2 + stride + 2)));
      mapContexts[++j] = clip(N + NE - NNE);
      mapContexts[++j] = clip(N + NE - NNE + p1 - clip(Np1 + NEp1 - buffer(w * 2 - stride + 1)));
      mapContexts[++j] = clip(N + NE - NNE + p2 - clip(Np2 + NEp2 - buffer(w * 2 - stride + 2)));
      mapContexts[++j] = clip(N + NN - NNN);
      mapContexts[++j] = clip(N + NN - NNN + p1 - clip(Np1 + NNp1 - buffer(w * 3 + 1)));
      mapContexts[++j] = clip(N + NN - NNN + p2 - clip(Np2 + NNp2 - buffer(w * 3 + 2)));
      mapContexts[++j] = clip(W + WW - WWW);
      mapContexts[++j] = clip(W + WW - WWW + p1 - clip(Wp1 + WWp1 - buffer(stride * 3 + 1)));
      mapContexts[++j] = clip(W + WW - WWW + p2 - clip(Wp2 + WWp2 - buffer(stride * 3 + 2)));
      mapContexts[++j] = clip(W + NEE - NE);
      mapContexts[++j] = clip(W + NEE - NE + p1 - clip(Wp1 + buffer(w - stride * 2 + 1) - NEp1));
      mapContexts[++j] = clip(W + NEE - NE + p2 - clip(Wp2 + buffer(w - stride * 2 + 2) - NEp2));
      mapContexts[++j] = clip(NN + p1 - NNp1);
      mapContexts[++j] = clip(NN + p2 - NNp2);
      mapContexts[++j] = clip(NN + W - NNW);
      mapContexts[++j] = clip(NN + W - NNW + p1 - clip(NNp1 + Wp1 - buffer(w * 2 + stride + 1)));
      mapContexts[++j] = clip(NN + W - NNW + p2 - clip(NNp2 + Wp2 - buffer(w * 2 + stride + 2)));
      mapContexts[++j] = clip(NN + NW - NNNW);
      mapContexts[++j] = clip(NN + NW - NNNW + p1 - clip(NNp1 + NWp1 - buffer(w * 3 + stride + 1)));
      mapContexts[++j] = clip(NN + NW - NNNW + p2 - clip(NNp2 + NWp2 - buffer(w * 3 + stride + 2)));
      mapContexts[++j] = clip(NN + NE - NNNE);
      mapContexts[++j] = clip(NN + NE - NNNE + p1 - clip(NNp1 + NEp1 - buffer(w * 3 - stride + 1)));
      mapContexts[++j] = clip(NN + NE - NNNE + p2 - clip(NNp2 + NEp2 - buffer(w * 3 - stride + 2)));
      mapContexts[++j] = clip(NN + NNNN - NNNNNN);
      mapContexts[++j] = clip(NN + NNNN - NNNNNN + p1 - clip(NNp1 + buffer(w * 4 + 1) - buffer(w * 6 + 1)));
      mapContexts[++j] = clip(NN + NNNN - NNNNNN + p2 - clip(NNp2 + buffer(w * 4 + 2) - buffer(w * 6 + 2)));
      mapContexts[++j] = clip(WW + p1 - WWp1);
      mapContexts[++j] = clip(WW + p2 - WWp2);
      mapContexts[++j] = clip(WW + WWWW - WWWWWW);
      mapContexts[++j] = clip(WW + WWWW - WWWWWW + p1 - clip(WWp1 + buffer(stride * 4 + 1) - buffer(stride * 6 + 1)));
      mapContexts[++j] = clip(WW + WWWW - WWWWWW + p2 - clip(WWp2 + buffer(stride * 4 + 2) - buffer(stride * 6 + 2)));
      mapContexts[++j] = clip(N * 2 - NN + p1 - clip(Np1 * 2 - NNp1));
      mapContexts[++j] = clip(N * 2 - NN + p2 - clip(Np2 * 2 - NNp2));
      mapContexts[++j] = clip(W * 2 - WW + p1 - clip(Wp1 * 2 - WWp1));
      mapContexts[++j] = clip(W * 2 - WW + p2 - clip(Wp2 * 2 - WWp2));
      mapContexts[++j] = clip(N * 3 - NN * 3 + NNN);
      mapContexts[++j] = clamp4(N * 3 - NN * 3 + NNN, W, NW, N, NE);
      mapContexts[++j] = clamp4(W * 3 - WW * 3 + WWW, W, NW, N, NE);
      mapContexts[++j] = clamp4(N * 2 - NN, W, NW, N, NE);
      mapContexts[++j] = clip((NNNNN - 6 * NNNN + 15 * NNN - 20 * NN + 15 * N +
                               clamp4(W * 4 - NWW * 6 + NNWWW * 4 - buffer(w * 3 + 4 * stride), W, NW, N, NN)) / 6);
      mapContexts[++j] = clip((buffer(w * 3 - 3 * stride) - 4 * NNEE + 6 * NE + clip(W * 4 - NW * 6 + NNW * 4 - NNNW)) / 4);
      mapContexts[++j] = clip(((N + 3 * NW) / 4) * 3 - ((NNW + NNWW) / 2) * 3 + (NNNWW * 3 + buffer(w * 3 + 3 * stride)) / 4);
      mapContexts[++j] = clip((W * 2 + NW) - (WW + 2 * NWW) + NWWW);
      mapContexts[++j] = (clip(W * 2 - NW) + clip(W * 2 - NWW) + N + NE) / 4;
      mapContexts[++j] = NNNNNN;
      mapContexts[++j] = (NEEEE + buffer(w - 6 * stride)) / 2;
      mapContexts[++j] = (WWWWWW + WWWW) / 2;
      mapContexts[++j] = ((W + N) * 3 - NW * 2) / 4;
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
      scMapContexts[++j] = W + NEE - NE + p1 - Wp1 - buffer(w - stride * 2 + 1) + NEp1;
      scMapContexts[++j] = W + NEE - NE + p2 - Wp2 - buffer(w - stride * 2 + 2) + NEp2;
      scMapContexts[++j] = N + NN - NNN;
      scMapContexts[++j] = N + NN - NNN + p1 - Np1 - NNp1 + buffer(w * 3 + 1);
      scMapContexts[++j] = N + NN - NNN + p2 - Np2 - NNp2 + buffer(w * 3 + 2);
      scMapContexts[++j] = N + NE - NNE;
      scMapContexts[++j] = N + NE - NNE + p1 - Np1 - NEp1 + buffer(w * 2 - stride + 1);
      scMapContexts[++j] = N + NE - NNE + p2 - Np2 - NEp2 + buffer(w * 2 - stride + 2);
      scMapContexts[++j] = N + NW - NNW;
      scMapContexts[++j] = N + NW - NNW + p1 - Np1 - NWp1 + buffer(w * 2 + stride + 1);
      scMapContexts[++j] = N + NW - NNW + p2 - Np2 - NWp2 + buffer(w * 2 + stride + 2);
      scMapContexts[++j] = NE + NW - NN;
      scMapContexts[++j] = NE + NW - NN + p1 - NEp1 - NWp1 + NNp1;
      scMapContexts[++j] = NE + NW - NN + p2 - NEp2 - NWp2 + NNp2;
      scMapContexts[++j] = NW + W - NWW;
      scMapContexts[++j] = NW + W - NWW + p1 - NWp1 - Wp1 + buffer(w + stride * 2 + 1);
      scMapContexts[++j] = NW + W - NWW + p2 - NWp2 - Wp2 + buffer(w + stride * 2 + 2);
      scMapContexts[++j] = W * 2 - WW;
      scMapContexts[++j] = W * 2 - WW + p1 - Wp1 * 2 + WWp1;
      scMapContexts[++j] = W * 2 - WW + p2 - Wp2 * 2 + WWp2;
      scMapContexts[++j] = N * 2 - NN;
      scMapContexts[++j] = N * 2 - NN + p1 - Np1 * 2 + NNp1;
      scMapContexts[++j] = N * 2 - NN + p2 - Np2 * 2 + NNp2;
      scMapContexts[++j] = NW * 2 - NNWW;
      scMapContexts[++j] = NW * 2 - NNWW + p1 - NWp1 * 2 + buffer(w * 2 + stride * 2 + 1);
      scMapContexts[++j] = NW * 2 - NNWW + p2 - NWp2 * 2 + buffer(w * 2 + stride * 2 + 2);
      scMapContexts[++j] = NE * 2 - NNEE;
      scMapContexts[++j] = NE * 2 - NNEE + p1 - NEp1 * 2 + buffer(w * 2 - stride * 2 + 1);
      scMapContexts[++j] = NE * 2 - NNEE + p2 - NEp2 * 2 + buffer(w * 2 - stride * 2 + 2);
      scMapContexts[++j] = N * 3 - NN * 3 + NNN + p1 - Np1 * 3 + NNp1 * 3 - buffer(w * 3 + 1);
      scMapContexts[++j] = N * 3 - NN * 3 + NNN + p2 - Np2 * 3 + NNp2 * 3 - buffer(w * 3 + 2);
      scMapContexts[++j] = N * 3 - NN * 3 + NNN;
      scMapContexts[++j] = (W + NE * 2 - NNE) / 2;
      scMapContexts[++j] = (W + NE * 3 - NNE * 3 + NNNE) / 2;
      scMapContexts[++j] = (W + NE * 2 - NNE) / 2 + p1 - (Wp1 + NEp1 * 2 - buffer(w * 2 - stride + 1)) / 2;
      scMapContexts[++j] = (W + NE * 2 - NNE) / 2 + p2 - (Wp2 + NEp2 * 2 - buffer(w * 2 - stride + 2)) / 2;
      scMapContexts[++j] = NNE + NE - NNNE;
      scMapContexts[++j] = NNE + W - NN;
      scMapContexts[++j] = NNW + W - NNWW;
      scMapContexts[++j] = 0;
      assert(++j == nSSM);
      j = 0;
      for( int k = (color > 0) ? color - 1 : stride - 1; j < nOLS; j++ ) {
        pOLS[j] = clip(int(floor(ols[j][color].predict(olsCtxs[j]))));
        ols[j][k].update(p1);
      }

      if( isPNG == 0u ) {
        int mean = W + NW + N + NE;
        const int var = (W * W + NW * NW + N * N + NE * NE - mean * mean / 4) >> 2U;
        mean >>= 2U;
        const int logVar = ilog->log(var);

        uint64_t i = 0;
        cm.set(hash(++i, (N + 1) >> 1U, logMeanDiffQt(N, clip(NN * 2 - NNN))));
        cm.set(hash(++i, (W + 1) >> 1U, logMeanDiffQt(W, clip(WW * 2 - WWW))));
        cm.set(hash(++i, clamp4(W + N - NW, W, NW, N, NE), logMeanDiffQt(clip(N + NE - NNE), clip(N + NW - NNW))));
        cm.set(hash(++i, (NNN + N + 4) / 8, clip(N * 3 - NN * 3 + NNN) >> 1));
        cm.set(hash(++i, (WWW + W + 4) / 8, clip(W * 3 - WW * 3 + WWW) >> 1));
        cm.set(hash(++i, color, (W + clip(NE * 3 - NNE * 3 + NNNE)) / 4, logMeanDiffQt(N, (NW + NE) / 2)));
        cm.set(hash(++i, color, clip((-WWWW + 5 * WWW - 10 * WW + 10 * W + clamp4(NE * 4 - NNE * 6 + NNNE * 4 - NNNNE, N, NE, NEE, NEEE)) / 5) / 4));
        cm.set(hash(++i, clip(NEE + N - NNEE), logMeanDiffQt(W, clip(NW + NE - NNE))));
        cm.set(hash(++i, clip(NN + W - NNW), logMeanDiffQt(W, clip(NNW + WW - NNWW))));
        cm.set(hash(++i, color, p1));
        cm.set(hash(++i, color, p2));
        cm.set(hash(++i, color, clip(W + N - NW) / 2, clip(W + p1 - Wp1) / 2));
        cm.set(hash(++i, clip(N * 2 - NN) / 2, logMeanDiffQt(N, clip(NN * 2 - NNN))));
        cm.set(hash(++i, clip(W * 2 - WW) / 2, logMeanDiffQt(W, clip(WW * 2 - WWW))));
        cm.set(hash(++i, clamp4(N * 3 - NN * 3 + NNN, W, NW, N, NE) / 2));
        cm.set(hash(++i, clamp4(W * 3 - WW * 3 + WWW, W, N, NE, NEE) / 2));
        cm.set(hash(++i, color, logMeanDiffQt(W, Wp1), clamp4((p1 * W) / (Wp1 < 1 ? 1 : Wp1), W, N, NE, NEE))); //using max(1,Wp1) results in division by zero in VC2015
        cm.set(hash(++i, color, clamp4(N + p2 - Np2, W, NW, N, NE)));
        cm.set(hash(++i, color, clip(W + N - NW), column[0]));
        cm.set(hash(++i, color, clip(N * 2 - NN), logMeanDiffQt(W, clip(NW * 2 - NNW))));
        cm.set(hash(++i, color, clip(W * 2 - WW), logMeanDiffQt(N, clip(NW * 2 - NWW))));
        cm.set(hash(++i, (W + NEE) / 2, logMeanDiffQt(W, (WW + NE) / 2)));
        cm.set(hash(++i, (clamp4(clip(W * 2 - WW) + clip(N * 2 - NN) - clip(NW * 2 - NNWW), W, NW, N, NE))));
        cm.set(hash(++i, color, W, p2));
        cm.set(hash(++i, N, NN & 0x1FU, NNN & 0x1FU));
        cm.set(hash(++i, W, WW & 0x1FU, WWW & 0x1FU));
        cm.set(hash(++i, color, N, column[0]));
        cm.set(hash(++i, color, clip(W + NEE - NE), logMeanDiffQt(W, clip(WW + NE - N))));
        cm.set(hash(++i, NN, NNNN & 0x1FU, NNNNNN & 0x1FU, column[1]));
        cm.set(hash(++i, WW, WWWW & 0x1FU, WWWWWW & 0x1FU, column[1]));
        cm.set(hash(++i, NNN, NNNNNN & 0x1FU, buffer(w * 9) & 0x1FU, column[1]));
        cm.set(hash(++i, color, column[1]));

        cm.set(hash(++i, color, W, logMeanDiffQt(W, WW)));
        cm.set(hash(++i, color, W, p1));
        cm.set(hash(++i, color, W / 4, logMeanDiffQt(W, p1), logMeanDiffQt(W, p2)));
        cm.set(hash(++i, color, N, logMeanDiffQt(N, NN)));
        cm.set(hash(++i, color, N, p1));
        cm.set(hash(++i, color, N / 4, logMeanDiffQt(N, p1), logMeanDiffQt(N, p2)));
        cm.set(hash(++i, color, (W + N) >> 3U, p1 >> 4U, p2 >> 4U));
        cm.set(hash(++i, color, p1 / 2, p2 / 2));
        cm.set(hash(++i, color, W, p1 - Wp1));
        cm.set(hash(++i, color, W + p1 - Wp1));
        cm.set(hash(++i, color, N, p1 - Np1));
        cm.set(hash(++i, color, N + p1 - Np1));
        cm.set(hash(++i, color, mean, logVar >> 4));

        ctx[0] = (min(color, 
          stride - 1) << 9U) | 
          (static_cast<int>(abs(W - N) > 3) << 8U) | 
          (static_cast<int>(W > N) << 7U) |
          (static_cast<int>(W > NW) << 6U) | 
          (static_cast<int>(abs(N - NW) > 3) << 5U) | 
          (static_cast<int>(N > NW) << 4U) |
          (static_cast<int>(abs(N - NE) > 3) << 3U) | 
          (static_cast<int>(N > NE) << 2U) | 
          (static_cast<int>(W > WW) << 1U) |
          static_cast<int>(N > NN);
        ctx[1] = ((logMeanDiffQt(p1, clip(Np1 + NEp1 - buffer(w * 2 - stride + 1))) >> 1U) << 5U) |
                 ((logMeanDiffQt(clip(N + NE - NNE), clip(N + NW - NNW)) >> 1U) << 2U) | 
                 min(color, stride - 1);
      } else {
        int residuals[5] = {(static_cast<int8_t>(buf(stride + static_cast<int>(x <= stride)))) + 128,
                            (static_cast<int8_t>(buf(1 + static_cast<int>(x < 2)))) + 128,
                            (static_cast<int8_t>(buf(stride + 1 + static_cast<int>(x <= stride)))) + 128,
                            (static_cast<int8_t>(buf(2 + static_cast<int>(x < 3)))) + 128,
                            (static_cast<int8_t>(buf(stride + 2 + static_cast<int>(x <= stride)))) + 128};
        R1 = (residuals[1] * residuals[0]) / max(1, residuals[2]);
        R2 = (residuals[3] * residuals[0]) / max(1, residuals[4]);

        uint64_t i = (filterOn ? filter + 1 : 0) * 1024;
        cm.set(0);
        cm.set(hash(++i, color, clip(W + N - NW) - px, clip(W + p1 - Wp1) - px, R1));
        cm.set(hash(++i, color, clip(W + N - NW) - px, logMeanDiffQt(p1, clip(Wp1 + Np1 - NWp1))));
        cm.set(hash(++i, color, clip(W + N - NW) - px, logMeanDiffQt(p2, clip(Wp2 + Np2 - NWp2)), R2 / 4));
        cm.set(hash(++i, color, clip(W + N - NW) - px, clip(N + NE - NNE) - clip(N + NW - NNW)));
        cm.set(hash(++i, color, clip(W + N - NW + p1 - (Wp1 + Np1 - NWp1)), px, R1));
        cm.set(hash(++i, color, clamp4(W + N - NW, W, NW, N, NE) - px, column[0]));
        cm.set(hash(++i, clamp4(W + N - NW, W, NW, N, NE) / 8, px));
        cm.set(hash(++i, color, N - px, clip(N + p1 - Np1) - px));
        cm.set(hash(++i, color, clip(W + p1 - Wp1) - px, R1));
        cm.set(hash(++i, color, clip(N + p1 - Np1) - px));
        cm.set(hash(++i, color, clip(N + p1 - Np1) - px, clip(N + p2 - Np2) - px));
        cm.set(hash(++i, color, clip(W + p1 - Wp1) - px, clip(W + p2 - Wp2) - px));
        cm.set(hash(++i, color, clip(NW + p1 - NWp1) - px));
        cm.set(hash(++i, color, clip(NW + p1 - NWp1) - px, column[0]));
        cm.set(hash(++i, color, clip(NE + p1 - NEp1) - px, column[0]));
        cm.set(hash(++i, color, clip(NE + N - NNE) - px, clip(NE + p1 - NEp1) - px));
        cm.set(hash(++i, clip(N + NE - NNE) - px, column[0]));
        cm.set(hash(++i, color, clip(NW + N - NNW) - px, clip(NW + p1 - NWp1) - px));
        cm.set(hash(++i, clip(N + NW - NNW) - px, column[0]));
        cm.set(hash(++i, clip(NN + W - NNW) - px, logMeanDiffQt(N, clip(NNN + NW - NNNW))));
        cm.set(hash(++i, clip(W + NEE - NE) - px, logMeanDiffQt(W, clip(WW + NE - N))));
        cm.set(hash(++i, color, clip(N + NN - NNN + buffer(1 + static_cast<int>(color == 0)) - clip(buffer(w + 1 + static_cast<int>(color == 0)) + buffer(w * 2 + 1 + static_cast<int>(color == 0)) - buffer(w * 3 + 1 + static_cast<int>(color == 0)))) - px));
        cm.set(hash(++i, clip(N + NN - NNN) - px, clip(5 * N - 10 * NN + 10 * NNN - 5 * NNNN + NNNNN) - px));
        cm.set(hash(++i, color, clip(N * 2 - NN) - px, logMeanDiffQt(N, clip(NN * 2 - NNN))));
        cm.set(hash(++i, color, clip(W * 2 - WW) - px, logMeanDiffQt(W, clip(WW * 2 - WWW))));
        cm.set(hash(++i, clip(N * 3 - NN * 3 + NNN) - px));
        cm.set(hash(++i, color, clip(N * 3 - NN * 3 + NNN) - px, logMeanDiffQt(W, clip(NW * 2 - NNW))));
        cm.set(hash(++i, clip(W * 3 - WW * 3 + WWW) - px));
        cm.set(hash(++i, color, clip(W * 3 - WW * 3 + WWW) - px, logMeanDiffQt(N, clip(NW * 2 - NWW))));
        cm.set(hash(++i, clip((35 * N - 35 * NNN + 21 * NNNNN - 5 * buffer(w * 7)) / 16) - px));
        cm.set(hash(++i, color, (W + clip(NE * 3 - NNE * 3 + NNNE)) / 2 - px, R2));
        cm.set(hash(++i, color, (W + clamp4(NE * 3 - NNE * 3 + NNNE, W, N, NE, NEE)) / 2 - px, logMeanDiffQt(N, (NW + NE) / 2)));
        cm.set(hash(++i, color, (W + NEE) / 2 - px, R1 / 2));
        cm.set(hash(++i, color, clamp4(clip(W * 2 - WW) + clip(N * 2 - NN) - clip(NW * 2 - NNWW), W, NW, N, NE) - px));
        cm.set(hash(++i, color, buf(stride + static_cast<int>(x <= stride)), buf(1 + static_cast<int>(x < 2)), buf(2 + static_cast<int>(x < 3))));
        cm.set(hash(++i, color, buf(1 + static_cast<int>(x < 2)), px));
        cm.set(hash(++i, buf(w + 1), buf((w + 1) * 2), buf((w + 1) * 3), px));
        ctx[0] = (min(color, stride - 1) << 9U) | 
          (static_cast<int>(abs(W - N) > 3) << 8U) | 
          (static_cast<int>(W > N) << 7U) |
          (static_cast<int>(W > NW) << 6U) |
          (static_cast<int>(abs(N - NW) > 3) << 5U) | 
          (static_cast<int>(N > NW) << 4U) |
          (static_cast<int>(N > NE) << 3U) | 
          min(5, filterOn ? filter + 1 : 0);
        ctx[1] = ((logMeanDiffQt(p1, clip(Np1 + NEp1 - buffer(w * 2 - stride + 1))) >> 1) << 5U) |
                 ((logMeanDiffQt(clip(N + NE - NNE), clip(N + NW - NNW)) >> 1) << 2U) | min(color, stride - 1);
      }

      int i = -1;
      map[++i].setDirect((W & 0xC0U) | ((N & 0xC0U) >> 2U) | ((WW & 0xC0U) >> 4U) | (NN >> 6U));
      map[++i].setDirect((N & 0xC0U) | ((NN & 0xC0U) >> 2U) | ((NE & 0xC0U) >> 4U) | (NEE >> 6U));
      map[++i].setDirect(buf(1 + static_cast<int>((isPNG != 0u) && x < 2)));
      map[++i].setDirect(min(color, stride - 1));
      map[++i].setDirect(0);
      shared->State.Image.plane = min(color, stride - 1);
      shared->State.Image.pixels.W = W;
      shared->State.Image.pixels.N = N;
      shared->State.Image.pixels.NN = NN;
      shared->State.Image.pixels.WW = WW;
      shared->State.Image.pixels.Wp1 = Wp1;
      shared->State.Image.pixels.Np1 = Np1;
      shared->State.Image.ctx = ctx[0] >> 3U;
    }
  }
  if( x > 0 || (isPNG == 0u)) {
    INJECT_SHARED_c0
    uint8_t b = (c0 << (8 - bpos));
    int i = 4;

    map[++i].setDirect(((static_cast<uint8_t>(clip(W + N - NW) - px - b)) * 8 + bpos) |
                       (logMeanDiffQt(clip(N + NE - NNE), clip(N + NW - NNW)) << 11));
    map[++i].setDirect(((static_cast<uint8_t>(clip(N * 2 - NN) - px - b)) * 8 + bpos) | 
                       (logMeanDiffQt(W, clip(NW * 2 - NNW)) << 11));
    map[++i].setDirect(((static_cast<uint8_t>(clip(W * 2 - WW) - px - b)) * 8 + bpos) |
                       (logMeanDiffQt(N, clip(NW * 2 - NWW)) << 11));
    map[++i].setDirect(((static_cast<uint8_t>(clip(W + N - NW) - px - b)) * 8 + bpos) |
                       (logMeanDiffQt(p1, clip(Wp1 + Np1 - NWp1)) << 11));
    map[++i].setDirect(((static_cast<uint8_t>(clip(W + N - NW) - px - b)) * 8 + bpos) |
                       (logMeanDiffQt(p2, clip(Wp2 + Np2 - NWp2)) << 11));
    map[++i].set(hash(W - px - b, N - px - b) * 8 + bpos);
    map[++i].set(hash(W - px - b, WW - px - b) * 8 + bpos);
    map[++i].set(hash(N - px - b, NN - px - b) * 8 + bpos);
    map[++i].set(hash(clip(N + NE - NNE) - px - b, clip(N + NW - NNW) - px - b) * 8 + bpos);
    map[++i].setDirect((min(color, stride - 1) << 11U) | ((static_cast<uint8_t>(clip(N + p1 - Np1) - px - b)) * 8 + bpos));
    map[++i].setDirect((min(color, stride - 1) << 11U) | ((static_cast<uint8_t>(clip(N + p2 - Np2) - px - b)) * 8 + bpos));
    map[++i].setDirect((min(color, stride - 1) << 11U) | ((static_cast<uint8_t>(clip(W + p1 - Wp1) - px - b)) * 8 + bpos));
    map[++i].setDirect((min(color, stride - 1) << 11U) | ((static_cast<uint8_t>(clip(W + p2 - Wp2) - px - b)) * 8 + bpos));
    ++i;
    assert(i == nSM0);

    for( int j = 0; j < nSM1; i++, j++ ) {
      map[i].setDirect((mapContexts[j] - px - b) * 8 + bpos);
    }

    for( int j = 0; i < nSM; i++, j++ ) {
      map[i].setDirect((pOLS[j] - px - b) * 8 + bpos);
    }

    for( int i = 0; i < nSSM; i++ ) {
      SCMap[i].set((scMapContexts[i] - px - b) * 8 + bpos);
    }
  }
}

void Image24BitModel::init() {
  stride = 3 + alpha;
  w = info & 0xFFFFFFU;
  padding = w % stride;
  x = color = line = px = 0;
  filterOn = false;
  columns[0] = max(1, w / max(1, ilog2(w) * 3));
  columns[1] = max(1, columns[0] / max(1, ilog2(columns[0])));
  if( lastPos > 0 && lastWasPNG != isPNG ) {
    for( int i = 0; i < nSM; i++ ) {
      map[i].reset(0);
    }
  }
  lastWasPNG = isPNG;
  buffer.fill(0x7F);
}

void Image24BitModel::setParam(int info0, uint32_t alpha0, uint32_t isPNG0) {
  info = info0;
  alpha = alpha0;
  isPNG = isPNG0;
}

void Image24BitModel::mix(Mixer &m) {
  INJECT_SHARED_bpos
  if( bpos == 0 ) {
    INJECT_SHARED_pos
    if((color < 0) || (pos - lastPos != 1)) {
      init();
    } else {
      x++;
      if( x >= w + static_cast<int>(isPNG)) {
        x = 0;
        line++;
      }
    }
    lastPos = pos;
  }

  update();

  // predict next bit
  if( x > 0 || (isPNG == 0u)) {
    cm.mix(m);

    for( int i = 0; i < nSM; i++ ) {
      map[i].mix(m);
    }

    for( int i = 0; i < nSSM; i++ ) {
      SCMap[i].mix(m);
    }

    if( ++col >= stride * 8 ) {
      col = 0;
    }
    m.set(5 + (((line & 0x7u) << 5u) | col), 5 + 256, 5);
    m.set(min(63, column[0]) + ((ctx[0] >> 3U) & 0xC0U), 256);
    m.set(min(127, column[1]) + ((ctx[0] >> 2U) & 0x180U), 512);
    m.set((ctx[0] & 0x7FCU) | (bpos >> 1), 2048);
    INJECT_SHARED_c0
    m.set(col + (isPNG != 0u ? (ctx[0] & 7) + 1 : static_cast<int>(c0 == ((0x100U | ((N + W) / 2)) >> (8 - bpos)))) * 32, 8 * 32);
    m.set(((isPNG != 0u ? p1 : 0) >> 4U) * stride + (x % stride) + min(5, filterOn ? filter + 1 : 0) * 64, 6 * 64);
    m.set(c0 + 256 * static_cast<int>((isPNG != 0u) && abs(R1 - 128) > 8), 256 * 2);
    m.set((ctx[1] << 2U) | (bpos >> 1U), 1024);
    m.set(finalize64(hash(logMeanDiffQt(W, WW, 5), logMeanDiffQt(N, NN, 5), logMeanDiffQt(W, N, 5), ilog2(W), color), 13), 8192);
    m.set(finalize64(hash(ctx[0], column[0] / 8), 13), 8192);
    m.set(finalize64(hash(logQt(N, 5), logMeanDiffQt(N, NN, 3), c0), 13), 8192);
    m.set(finalize64(hash(logQt(W, 5), logMeanDiffQt(W, WW, 3), c0), 13), 8192);
    m.set(min(255, (x + line) / 32), 256);
  } else {
    m.add(-2048 + ((filter >> (7 - bpos)) & 1U) * 4096);
    m.set(min(4, filter), MIXERCONTEXTS);
  }
}
