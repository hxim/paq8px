#include "Image8BitModel.hpp"

Image8BitModel::Image8BitModel(Shared* const sh, const uint64_t size) : 
  shared(sh), 
  cm(sh, size, nCM, 64, CM_USE_RUN_STATS),
  map { /* StationaryMap: BitsOfContext, InputBits, Scale, Limit  */
    /*nSM0: 0- 1*/ {sh, 0,8,64,1023}, {sh,15,1,64,1023},
    /*nSM1: 0- 4*/ {sh,11,1,64,1023}, {sh,11,1,64,1023}, {sh,11,1,64,1023}, {sh,11,1,64,1023}, {sh,11,1,64,1023},
    /*nSM1: 5- 9*/ {sh,11,1,64,1023}, {sh,11,1,64,1023}, {sh,11,1,64,1023}, {sh,11,1,64,1023}, {sh,11,1,64,1023},
    /*nSM1:10-14*/ {sh,11,1,64,1023}, {sh,11,1,64,1023}, {sh,11,1,64,1023}, {sh,11,1,64,1023}, {sh,11,1,64,1023},
    /*nSM1:15-19*/ {sh,11,1,64,1023}, {sh,11,1,64,1023}, {sh,11,1,64,1023}, {sh,11,1,64,1023}, {sh,11,1,64,1023},
    /*nSM1:20-24*/ {sh,11,1,64,1023}, {sh,11,1,64,1023}, {sh,11,1,64,1023}, {sh,11,1,64,1023}, {sh,11,1,64,1023},
    /*nSM1:25-29*/ {sh,11,1,64,1023}, {sh,11,1,64,1023}, {sh,11,1,64,1023}, {sh,11,1,64,1023}, {sh,11,1,64,1023},
    /*nSM1:30-34*/ {sh,11,1,64,1023}, {sh,11,1,64,1023}, {sh,11,1,64,1023}, {sh,11,1,64,1023}, {sh,11,1,64,1023},
    /*nSM1:35-39*/ {sh,11,1,64,1023}, {sh,11,1,64,1023}, {sh,11,1,64,1023}, {sh,11,1,64,1023}, {sh,11,1,64,1023},
    /*nSM1:40-44*/ {sh,11,1,64,1023}, {sh,11,1,64,1023}, {sh,11,1,64,1023}, {sh,11,1,64,1023}, {sh,11,1,64,1023},
    /*nSM1:45-49*/ {sh,11,1,64,1023}, {sh,11,1,64,1023}, {sh,11,1,64,1023}, {sh,11,1,64,1023}, {sh,11,1,64,1023},
    /*nSM1:50-54*/ {sh,11,1,64,1023}, {sh,11,1,64,1023}, {sh,11,1,64,1023}, {sh,11,1,64,1023}, {sh,11,1,64,1023},
    /*nOLS:   0- 4*/ {sh,11,1,64,1023}, {sh,11,1,64,1023}, {sh,11,1,64,1023}, {sh,11,1,64,1023}, {sh,11,1,64,1023}
  },
  pltMap {   /* SmallStationaryContextMap: BitsOfContext, InputBits, Rate, Scale */
    {sh,11,1,7,64}, {sh,11,1,7,64}, {sh,11,1,7,64}, {sh,11,1,7,64}
  },
  sceneMap { /* IndirectMap: BitsOfContext, InputBits, Scale, Limit */
    {sh,8,8,64,255}, {sh,8,8,64,255}, {sh,22,1,64,255}, {sh,11,1,64,255}, {sh,11,1,64,255}
  },
  iCtx {     /* IndirectContext<U8>: BitsPerContext, InputBits */
    {16,8}, {16,8}, {16,8}, {16,8}
  }
  {}

void Image8BitModel::setParam(int info0, uint32_t gray0, uint32_t isPNG0) {
  w = info0;
  gray = gray0;
  isPNG = isPNG0;
}

void Image8BitModel::mix(Mixer &m) {
  // Select nearby pixels as context
  INJECT_SHARED_bpos
  if( bpos == 0 ) {
    INJECT_SHARED_buf
    INJECT_SHARED_pos
    if( pos != lastPos + 1 ) {
      x = line = px = jump = 0;
      filterOn = false;
      columns[0] = max(1, w / max(1, ilog2(w) * 2));
      columns[1] = max(1, columns[0] / max(1, ilog2(columns[0])));
      if( gray != 0u ) {
        if((lastPos != 0u) && lastWasPNG != isPNG ) {
          for( int i = 0; i < nSM; i++ ) {
            map[i].reset(0);
          }
        }
        lastWasPNG = isPNG;
      }
      buffer.fill(0x7F);
      prevFramePos = framePos;
      framePos = pos;
      prevFrameWidth = frameWidth;
      frameWidth = w;
    } else {
      x++;
      if( x >= w + static_cast<int>(isPNG)) {
        x = 0;
        line++;
      }
    }
    lastPos = pos;

    INJECT_SHARED_c1
    if( isPNG ) {
      if( x == 1 ) {
        filter = c1;
      } else {
        switch( filter ) {
          case 1: {
            buffer.add(static_cast<uint8_t>(c1 + buffer(1) * static_cast<int>(x > 2 || (x == 0))));
            filterOn = x > 1;
            px = buffer(1);
            break;
          }
          case 2: {
            buffer.add(static_cast<uint8_t>(c1 + buffer(w) * static_cast<int>((filterOn = (line > 0)))));
            px = buffer(w);
            break;
          }
          case 3: {
            buffer.add(static_cast<uint8_t>(c1 + (buffer(w) * static_cast<int>(line > 0) + buffer(1) * static_cast<int>(x > 2 || (x == 0))) / 2));
            filterOn = (x > 1 || line > 0);
            px = (buffer(1) * static_cast<int>(x > 1) + buffer(w) * static_cast<int>(line > 0)) / 2;
            break;
          }
          case 4: {
            buffer.add(static_cast<uint8_t>(c1 + paeth(buffer(1) * static_cast<int>(x > 2 || (x == 0)), buffer(w) * static_cast<int>(line > 0), buffer(w + 1) * static_cast<int>(line > 0 && (x > 2 || (x == 0))))));
            filterOn = (x > 1 || line > 0);
            px = paeth(buffer(1) * static_cast<int>(x > 1), buffer(w) * static_cast<int>(line > 0),
                       buffer(w + 1) * static_cast<int>(x > 1 && line > 0));
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
      }
    } else { // non-png
      buffer.add(c1);
      if( x == 0 ) {
        memset(&jumps[0], 0, sizeof(short) * jumps.size());
        if( line > 0 && w > 8 ) {
          uint8_t bMask = 0xFF - ((1 << gray) - 1);
          uint32_t pMask = bMask * 0x01010101u;
          uint32_t left = 0;
          uint32_t right = 0;
          int l = min(w, static_cast<int>(jumps.size()));
          int end = l - 4;
          do {
            left = ((buffer(l - x) << 24U) | (buffer(l - x - 1) << 16U) | (buffer(l - x - 2) << 8U) | buffer(l - x - 3)) & pMask;
            int i = end;
            while( i >= x + 4 ) {
              right = ((buffer(l - i - 3) << 24U) | (buffer(l - i - 2) << 16U) | (buffer(l - i - 1) << 8U) | buffer(l - i)) & pMask;
              if( left == right ) {
                int j = (i + 3 - x - 1) / 2;
                int k = 0;
                for( ; k <= j; k++ ) {
                  if( k < 4 || (buffer(l - x - k) & bMask) == (buffer(l - i - 3 + k) & bMask)) {
                    jumps[x + k] = -(x + (l - i - 3) + 2 * k);
                    jumps[i + 3 - k] = i + 3 - x - 2 * k;
                  } else {
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
            if( x > end ) {
              break;
            }
          } while( x + 4 < l );
          x = 0;
        }
      }
    }

    if((x != 0) || (!isPNG)) {
      column[0] = (x - isPNG) / columns[0];
      column[1] = (x - isPNG) / columns[1];
      WWWWW = buffer(5), WWWW = buffer(4), WWW = buffer(3), WW = buffer(2), W = buffer(1);
      NWWWW = buffer(w + 4), NWWW = buffer(w + 3), NWW = buffer(w + 2), NW = buffer(w + 1), N = buffer(w), NE = buffer(w - 1), NEE = buffer(
              w - 2), NEEE = buffer(w - 3), NEEEE = buffer(w - 4);
      NNWWW = buffer(w * 2 + 3), NNWW = buffer(w * 2 + 2), NNW = buffer(w * 2 + 1), NN = buffer(w * 2), NNE = buffer(
              w * 2 - 1), NNEE = buffer(w * 2 - 2), NNEEE = buffer(w * 2 - 3);
      NNNWW = buffer(w * 3 + 2), NNNW = buffer(w * 3 + 1), NNN = buffer(w * 3), NNNE = buffer(w * 3 - 1), NNNEE = buffer(w * 3 - 2);
      NNNNW = buffer(w * 4 + 1), NNNN = buffer(w * 4), NNNNE = buffer(w * 4 - 1);
      NNNNN = buffer(w * 5);
      NNNNNN = buffer(w * 6);
      if( prevFramePos > 0 && prevFrameWidth == w ) {
        int offset = prevFramePos + line * w + x;
        prvFrmPx = buf[offset];
        if( gray != 0u ) {
          sceneOls.update(W);
          sceneOls.add(W);
          sceneOls.add(NW);
          sceneOls.add(N);
          sceneOls.add(NE);
          for( int i = -1; i < 2; i++ ) {
            for( int j = -1; j < 2; j++ ) {
              sceneOls.add(buf[offset + i * w + j]);
            }
          }
          prvFrmPrediction = clip(int(floor(sceneOls.predict())));
        } else {
          prvFrmPrediction = W;
        }
      } else {
        prvFrmPx = prvFrmPrediction = W;
      }
      sceneMap[0].setDirect(prvFrmPx);
      sceneMap[1].setDirect(prvFrmPrediction);

      int j = 0;
      jump = jumps[min(x, static_cast<int>(jumps.size()) - 1)];
      uint64_t i = (filterOn ? (filter + 1) * 64 : 0) + (gray * 1024);
      cm.set(hash(++i, (jump != 0) ? (0x100 | buffer(abs(jump))) * (1 - 2 * static_cast<int>(jump < 0)) : N, line & 3U));
      if( !gray ) {
        for( j = 0; j < nPltMaps; j++ ) {
          iCtx[j] += W;
        }
        iCtx[0] = W | (NE << 8U);
        iCtx[1] = W | (N << 8U);
        iCtx[2] = W | (WW << 8U);
        iCtx[3] = N | (NN << 8U);
        cm.set(hash(++i, W, px));
        cm.set(hash(++i, W, px, column[0]));
        cm.set(hash(++i, N, px));
        cm.set(hash(++i, N, px, column[0]));
        cm.set(hash(++i, NW, px));
        cm.set(hash(++i, NW, px, column[0]));
        cm.set(hash(++i, NE, px));
        cm.set(hash(++i, NE, px, column[0]));
        cm.set(hash(++i, NWW, px));
        cm.set(hash(++i, NEE, px));
        cm.set(hash(++i, WW, px));
        cm.set(hash(++i, NN, px));
        cm.set(hash(++i, W, N, px));
        cm.set(hash(++i, W, NW, px));
        cm.set(hash(++i, W, NE, px));
        cm.set(hash(++i, W, NEE, px));
        cm.set(hash(++i, W, NWW, px));
        cm.set(hash(++i, N, NW, px));
        cm.set(hash(++i, N, NE, px));
        cm.set(hash(++i, NW, NE, px));
        cm.set(hash(++i, W, WW, px));
        cm.set(hash(++i, N, NN, px));
        cm.set(hash(++i, NW, NNWW, px));
        cm.set(hash(++i, NE, NNEE, px));
        cm.set(hash(++i, NW, NWW, px));
        cm.set(hash(++i, NW, NNW, px));
        cm.set(hash(++i, NE, NEE, px));
        cm.set(hash(++i, NE, NNE, px));
        cm.set(hash(++i, N, NNW, px));
        cm.set(hash(++i, N, NNE, px));
        cm.set(hash(++i, N, NNN, px));
        cm.set(hash(++i, W, WWW, px));
        cm.set(hash(++i, WW, NEE, px));
        cm.set(hash(++i, WW, NN, px));
        cm.set(hash(++i, W, buffer(w - 3), px));
        cm.set(hash(++i, W, buffer(w - 4), px));
        cm.set(hash(++i, W, N, NW, px));
        cm.set(hash(++i, N, NN, NNN, px));
        cm.set(hash(++i, W, NE, NEE, px));
        cm.set(hash(++i, W, NW, N, NE, px));
        cm.set(hash(++i, N, NE, NN, NNE, px));
        cm.set(hash(++i, N, NW, NNW, NN, px));
        cm.set(hash(++i, W, WW, NWW, NW, px));
        cm.set(hash(++i, W, NW << 8U | N, WW << 8U | NWW, px));
        cm.set(hash(++i, px, column[0]));
        cm.set(hash(++i, px));
        cm.set(hash(++i, N, px, column[1]));
        cm.set(hash(++i, W, px, column[1]));
        for( int j = 0; j < nPltMaps; j++ ) {
          cm.set(hash(++i, iCtx[j](), px));
        }
        
        ctx = min(0x1F, (x - isPNG) / min(0x20, columns[0]));
        res = W;
      } else { // gray
        mapContexts[j++] = clamp4(W + N - NW, W, NW, N, NE);
        mapContexts[j++] = clip(W + N - NW);
        mapContexts[j++] = clamp4(W + NE - N, W, NW, N, NE);
        mapContexts[j++] = clip(W + NE - N);
        mapContexts[j++] = clamp4(N + NW - NNW, W, NW, N, NE);
        mapContexts[j++] = clip(N + NW - NNW);
        mapContexts[j++] = clamp4(N + NE - NNE, W, N, NE, NEE);
        mapContexts[j++] = clip(N + NE - NNE);
        mapContexts[j++] = (W + NEE) / 2;
        mapContexts[j++] = clip(N * 3 - NN * 3 + NNN);
        mapContexts[j++] = clip(W * 3 - WW * 3 + WWW);
        mapContexts[j++] = (W + clip(NE * 3 - NNE * 3 + buffer(w * 3 - 1))) / 2;
        mapContexts[j++] = (W + clip(NEE * 3 - buffer(w * 2 - 3) * 3 + buffer(w * 3 - 4))) / 2;
        mapContexts[j++] = clip(NN + buffer(w * 4) - buffer(w * 6));
        mapContexts[j++] = clip(WW + buffer(4) - buffer(6));
        mapContexts[j++] = clip((buffer(w * 5) - 6 * buffer(w * 4) + 15 * NNN - 20 * NN + 15 * N + clamp4(W * 2 - NWW, W, NW, N, NN)) / 6);
        mapContexts[j++] = clip((-3 * WW + 8 * W + clamp4(NEE * 3 - NNEE * 3 + buffer(w * 3 - 2), NE, NEE, buffer(w - 3), buffer(w - 4))) / 6);
        mapContexts[j++] = clip(NN + NW - buffer(w * 3 + 1));
        mapContexts[j++] = clip(NN + NE - buffer(w * 3 - 1));
        mapContexts[j++] = clip((W * 2 + NW) - (WW + 2 * NWW) + buffer(w + 3));
        mapContexts[j++] = clip(((NW + NWW) / 2) * 3 - buffer(w * 2 + 3) * 3 + (buffer(w * 3 + 4) + buffer(w * 3 + 5)) / 2);
        mapContexts[j++] = clip(NEE + NE - buffer(w * 2 - 3));
        mapContexts[j++] = clip(NWW + WW - buffer(w + 4));
        mapContexts[j++] = clip(((W + NW) * 3 - NWW * 6 + buffer(w + 3) + buffer(w * 2 + 3)) / 2);
        mapContexts[j++] = clip((NE * 2 + NNE) - (NNEE + buffer(w * 3 - 2) * 2) + buffer(w * 4 - 3));
        mapContexts[j++] = buffer(w * 6);
        mapContexts[j++] = (buffer(w - 4) + buffer(w - 6)) / 2;
        mapContexts[j++] = (buffer(4) + buffer(6)) / 2;
        mapContexts[j++] = (W + N + buffer(w - 5) + buffer(w - 7)) / 4;
        mapContexts[j++] = clip(buffer(w - 3) + W - NEE);
        mapContexts[j++] = clip(4 * NNN - 3 * buffer(w * 4));
        mapContexts[j++] = clip(N + NN - NNN);
        mapContexts[j++] = clip(W + WW - WWW);
        mapContexts[j++] = clip(W + NEE - NE);
        mapContexts[j++] = clip(WW + NEE - N);
        mapContexts[j++] = (clip(W * 2 - NW) + clip(W * 2 - NWW) + N + NE) / 4;
        mapContexts[j++] = clamp4(N * 2 - NN, W, N, NE, NEE);
        mapContexts[j++] = (N + NNN) / 2;
        mapContexts[j++] = clip(NN + W - NNW);
        mapContexts[j++] = clip(NWW + N - NNWW);
        mapContexts[j++] = clip((4 * WWW - 15 * WW + 20 * W + clip(NEE * 2 - NNEE)) / 10);
        mapContexts[j++] = clip((buffer(w * 3 - 3) - 4 * NNEE + 6 * NE + clip(W * 3 - NW * 3 + NNW)) / 4);
        mapContexts[j++] = clip((N * 2 + NE) - (NN + 2 * NNE) + buffer(w * 3 - 1));
        mapContexts[j++] = clip((NW * 2 + NNW) - (NNWW + buffer(w * 3 + 2) * 2) + buffer(w * 4 + 3));
        mapContexts[j++] = clip(NNWW + W - buffer(w * 2 + 3));
        mapContexts[j++] = clip((-buffer(w * 4) + 5 * NNN - 10 * NN + 10 * N + clip(W * 4 - NWW * 6 + buffer(w * 2 + 3) * 4 - buffer(w * 3 + 4))) / 5);
        mapContexts[j++] = clip(NEE + clip(buffer(w - 3) * 2 - buffer(w * 2 - 4)) - buffer(w - 4));
        mapContexts[j++] = clip(NW + W - NWW);
        mapContexts[j++] = clip((N * 2 + NW) - (NN + 2 * NNW) + buffer(w * 3 + 1));
        mapContexts[j++] = clip(NN + clip(NEE * 2 - buffer(w * 2 - 3)) - NNE);
        mapContexts[j++] = clip((-buffer(4) + 5 * WWW - 10 * WW + 10 * W + clip(NE * 2 - NNE)) / 5);
        mapContexts[j++] = clip((-buffer(5) + 4 * buffer(4) - 5 * WWW + 5 * W + clip(NE * 2 - NNE)) / 4);
        mapContexts[j++] = clip((WWW - 4 * WW + 6 * W + clip(NE * 3 - NNE * 3 + buffer(w * 3 - 1))) / 4);
        mapContexts[j++] = clip((-NNEE + 3 * NE + clip(W * 4 - NW * 6 + NNW * 4 - buffer(w * 3 + 1))) / 3);
        mapContexts[j++] = ((W + N) * 3 - NW * 2) / 4;
        for( j = 0; j < nOLS; j++ ) {
          ols[j].update(W);
          pOLS[j] = clip(int(floor(ols[j].predict(olsCtxs[j]))));
        }

        cm.set(0);
        cm.set(hash(++i, N, px));
        cm.set(hash(++i, N - px));
        cm.set(hash(++i, W, px));
        cm.set(hash(++i, NW, px));
        cm.set(hash(++i, NE, px));
        cm.set(hash(++i, N, NN, px));
        cm.set(hash(++i, W, WW, px));
        cm.set(hash(++i, NE, NNEE, px));
        cm.set(hash(++i, NW, NNWW, px));
        cm.set(hash(++i, W, NEE, px));
        cm.set(hash(++i, (clamp4(W + N - NW, W, NW, N, NE) - px) / 2, logMeanDiffQt(clip(N + NE - NNE), clip(N + NW - NNW))));
        cm.set(hash(++i, (W - px) / 4, (NE - px) / 4, column[0]));
        cm.set(hash(++i, (clip(W * 2 - WW) - px) / 4, (clip(N * 2 - NN) - px) / 4));
        cm.set(hash(++i, (clamp4(N + NE - NNE, W, N, NE, NEE) - px) / 4, column[0]));
        cm.set(hash(++i, (clamp4(N + NW - NNW, W, NW, N, NE) - px) / 4, column[0]));
        cm.set(hash(++i, (W + NEE) / 4, px, column[0]));
        cm.set(hash(++i, clip(W + N - NW) - px, column[0]));
        cm.set(hash(++i, clamp4(N * 3 - NN * 3 + NNN, W, N, NN, NE), px, logMeanDiffQt(W, clip(NW * 2 - NNW))));
        cm.set(hash(++i, clamp4(W * 3 - WW * 3 + WWW, W, N, NE, NEE), px, logMeanDiffQt(N, clip(NW * 2 - NWW))));
        cm.set(hash(++i, (W + clamp4(NE * 3 - NNE * 3 + NNNE, W, N, NE, NEE)) / 2, px, logMeanDiffQt(N, (NW + NE) / 2)));
        cm.set(hash(++i, (N + NNN) / 8, clip(N * 3 - NN * 3 + NNN) / 4, px));
        cm.set(hash(++i, (W + WWW) / 8, clip(W * 3 - WW * 3 + WWW) / 4, px));
        cm.set(hash(++i, clip((-buffer(4) + 5 * WWW - 10 * WW + 10 * W + clamp4(NE * 4 - NNE * 6 + buffer(w * 3 - 1) * 4 - buffer(w * 4 - 1), N, NE, buffer(w - 2), buffer(w - 3))) / 5) - px));
        cm.set(hash(++i, clip(N * 2 - NN) - px, logMeanDiffQt(N, clip(NN * 2 - NNN))));
        cm.set(hash(++i, clip(W * 2 - WW) - px, logMeanDiffQt(NE, clip(N * 2 - NW))));

        if( isPNG ) {
          ctx = (static_cast<int>(abs(W - N) > 8) << 10U) | (static_cast<int>(W > N) << 9U) | (static_cast<int>(abs(N - NW) > 8) << 8U) |
                (static_cast<int>(N > NW) << 7U) | (static_cast<int>(abs(N - NE) > 8) << 6U) | (static_cast<int>(N > NE) << 5U) |
                (static_cast<int>(W > WW) << 4U) | (static_cast<int>(N > NN) << 3U) | min(5, filterOn ? filter + 1 : 0);
        } else {
          ctx = min(0x1F, x / max(1, w / min(32, columns[0]))) |
                (((static_cast<int>(abs(W - N) * 16 > W + N) << 1U) | static_cast<int>(abs(N - NW) > 8)) << 5U) | ((W + N) & 0x180U);
        }

        res = clamp4(W + N - NW, W, NW, N, NE) - px;
      }

      shared->State.Image.pixels.W = W;
      shared->State.Image.pixels.N = N;
      shared->State.Image.pixels.NN = NN;
      shared->State.Image.pixels.WW = WW;
      shared->State.Image.ctx = ctx >> gray;
    }
  }
  INJECT_SHARED_c0
  uint8_t b = (c0 << (8 - bpos));
  if((x != 0) || !isPNG) {
    if( gray ) {
      int i = 0;
      map[i++].setDirect(0);
      map[i++].setDirect(((static_cast<uint8_t>(clip(W + N - NW) - px - b)) * 8 + bpos) |
                         (logMeanDiffQt(clip(N + NE - NNE), clip(N + NW - NNW)) << 11U));

      for( int j = 0; j < nSM1; i++, j++ ) {
        map[i].setDirect((mapContexts[j] - px - b) * 8 + bpos);
      }

      for( int j = 0; i < nSM; i++, j++ ) {
        map[i].setDirect((pOLS[j] - px - b) * 8 + bpos);
      }
    }
    sceneMap[2].setDirect(finalize64(hash(x, line), 19) * 8 + bpos);
    sceneMap[3].setDirect((prvFrmPx - b) * 8 + bpos);
    sceneMap[4].setDirect((prvFrmPrediction - b) * 8 + bpos);
  }

  // predict next bit
  if((x != 0) || !isPNG) {
    cm.mix(m);
    if( gray ) {
      for( int i = 0; i < nSM; i++ ) {
        map[i].mix(m);
      }
    } else {
      for( int i = 0; i < nPltMaps; i++ ) {
        pltMap[i].set((bpos << 8U) | iCtx[i]());
        pltMap[i].mix(m);
      }
    }
    for( int i = 0; i < 5; i++ ) {
      const int scale = (prevFramePos > 0 && prevFrameWidth == w) ? 64 : 0;
      sceneMap[i].setScale(scale);
      sceneMap[i].mix(m);
    }

    col = (col + 1) & 7U;
    m.set(5 + ctx, 2048 + 5);
    m.set(col * 2 + static_cast<int>(isPNG && c0 == ((0x100U | res) >> (8 - bpos))) +
          min(5, filterOn ? filter + 1 : 0) * 16, 6 * 16);
    m.set(((isPNG ? px : N + W) >> 4U) + min(5, filterOn ? filter + 1 : 0) * 32, 6 * 32);
    m.set(c0, 256);
    m.set((static_cast<int>(abs((W - N)) > 4) << 9U) | (static_cast<int>(abs((N - NE)) > 4) << 8U) |
          (static_cast<int>(abs((W - NW)) > 4) << 7U) | (static_cast<int>(W > N) << 6U) | (static_cast<int>(N > NE) << 5U) |
          (static_cast<int>(W > NW) << 4U) | (static_cast<int>(W > WW) << 3U) | (static_cast<int>(N > NN) << 2U) |
          (static_cast<int>(NW > NNWW) << 1U) | static_cast<int>(NE > NNEE), 1024);
    m.set(min(63, column[0]), 64);
    m.set(min(127, column[1]), 128);
    m.set(min(255, (x + line) / 32), 256);
  } else {
    m.add(-2048 + ((filter >> (7 - bpos)) & 1U) * 4096);
    m.set(min(4, filter), MIXERCONTEXTS);
  }
}
