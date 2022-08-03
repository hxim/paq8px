#include "Image8BitModel.hpp"

Image8BitModel::Image8BitModel(Shared* const sh, const uint64_t size) : 
  shared(sh), 
  cm(sh, size, nCM, 64),
  map { /* StationaryMap : BitsOfContext, InputBits, Scale=64, Rate=16  */
    /*nSM0: 0- 1*/ {sh, 0,8}, {sh,15,1},
    /*nSM1: 0- 4*/ {sh,11,1}, {sh,11,1}, {sh,11,1}, {sh,11,1}, {sh,11,1},
    /*nSM1: 5- 9*/ {sh,11,1}, {sh,11,1}, {sh,11,1}, {sh,11,1}, {sh,11,1},
    /*nSM1:10-14*/ {sh,11,1}, {sh,11,1}, {sh,11,1}, {sh,11,1}, {sh,11,1},
    /*nSM1:15-19*/ {sh,11,1}, {sh,11,1}, {sh,11,1}, {sh,11,1}, {sh,11,1},
    /*nSM1:20-24*/ {sh,11,1}, {sh,11,1}, {sh,11,1}, {sh,11,1}, {sh,11,1},
    /*nSM1:25-29*/ {sh,11,1}, {sh,11,1}, {sh,11,1}, {sh,11,1}, {sh,11,1},
    /*nSM1:30-34*/ {sh,11,1}, {sh,11,1}, {sh,11,1}, {sh,11,1}, {sh,11,1},
    /*nSM1:35-39*/ {sh,11,1}, {sh,11,1}, {sh,11,1}, {sh,11,1}, {sh,11,1},
    /*nSM1:40-44*/ {sh,11,1}, {sh,11,1}, {sh,11,1}, {sh,11,1}, {sh,11,1},
    /*nSM1:45-49*/ {sh,11,1}, {sh,11,1}, {sh,11,1}, {sh,11,1}, {sh,11,1},
    /*nSM1:50-54*/ {sh,11,1}, {sh,11,1}, {sh,11,1}, {sh,11,1}, {sh,11,1},
    /*nOLS: 0- 4*/ {sh,11,1}, {sh,11,1}, {sh,11,1}, {sh,11,1}, {sh,11,1}
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

void Image8BitModel::setParam(int width, uint32_t isGray) {
  this->w = width;
  this->isGray = isGray;
}

void Image8BitModel::mix(Mixer &m) {
  // Select nearby pixels as context
  INJECT_SHARED_bpos
  if( bpos == 0 ) {
    INJECT_SHARED_buf
    INJECT_SHARED_pos
    if( pos != lastPos + 1 ) {
      x = line = jump = 0;
      filterOn = false;
      columns[0] = max(1, w / max(1, ilog2(w) * 2));
      columns[1] = max(1, columns[0] / max(1, ilog2(columns[0])));
      if( isGray ) {
        if(lastPos != 0 && false) { // todo: when shall we reset ?
          for( int i = 0; i < nSM; i++ ) {
            map[i].reset();
          }
        }
      }
      else if(frameWidth != w) {
        for (int i = 0; i < nPltMaps; i++) {
          iCtx[i].reset();
          pltMap[i].reset();
        }
      }
      prevFramePos = framePos;
      framePos = pos;
      prevFrameWidth = frameWidth;
      frameWidth = w;
    } else {
      x++;
      if( x >= w ) {
        x = 0;
        line++;
      }
    }
    lastPos = pos;

    INJECT_SHARED_c1
    if( x == 0 ) {
      memset(&jumps[0], 0, sizeof(short) * jumps.size());
      if( line > 0 && w > 8 ) {
        uint8_t bMask = 0xFF - ((1 << isGray) - 1);
        uint32_t pMask = bMask * 0x01010101;
        uint32_t left = 0;
        uint32_t right = 0;
        int l = min(w, static_cast<int>(jumps.size()));
        int end = l - 4;
        do {
          left = ((buf(l - x) << 24) | (buf(l - x - 1) << 16) | (buf(l - x - 2) << 8) | buf(l - x - 3)) & pMask;
          int i = end;
          while( i >= x + 4 ) {
            right = ((buf(l - i - 3) << 24) | (buf(l - i - 2) << 16) | (buf(l - i - 1) << 8) | buf(l - i)) & pMask;
            if( left == right ) {
              int j = (i + 3 - x - 1) / 2;
              int k = 0;
              for( ; k <= j; k++ ) {
                if( k < 4 || (buf(l - x - k) & bMask) == (buf(l - i - 3 + k) & bMask)) {
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

    column[0] = x / columns[0];
    column[1] = x / columns[1];
    WWWWW = buf(5), WWWW = buf(4), WWW = buf(3), WW = buf(2), W = buf(1);
    NWWWW = buf(w + 4), NWWW = buf(w + 3), NWW = buf(w + 2), NW = buf(w + 1), N = buf(w), NE = buf(w - 1), NEE = buf(
            w - 2), NEEE = buf(w - 3), NEEEE = buf(w - 4);
    NNWWW = buf(w * 2 + 3), NNWW = buf(w * 2 + 2), NNW = buf(w * 2 + 1), NN = buf(w * 2), NNE = buf(
            w * 2 - 1), NNEE = buf(w * 2 - 2), NNEEE = buf(w * 2 - 3);
    NNNWW = buf(w * 3 + 2), NNNW = buf(w * 3 + 1), NNN = buf(w * 3), NNNE = buf(w * 3 - 1), NNNEE = buf(w * 3 - 2);
    NNNNW = buf(w * 4 + 1), NNNN = buf(w * 4), NNNNE = buf(w * 4 - 1);
    NNNNN = buf(w * 5);
    NNNNNN = buf(w * 6);
    if( prevFramePos > 0 && prevFrameWidth == w ) {
      int offset = prevFramePos + line * w + x;
      prvFrmPx = buf[offset];
      if( isGray != 0 ) {
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
    uint64_t i = isGray * 1024;
    const uint8_t R_ = CM_USE_RUN_STATS;
    cm.set(R_, hash(++i, (jump != 0) ? (0x100 | buf(abs(jump))) * (1 - 2 * static_cast<int>(jump < 0)) : N, line & 3));
    if( !isGray ) {
      for( j = 0; j < nPltMaps; j++ ) {
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
      cm.set(R_, hash(++i, W, buf(w - 3)));
      cm.set(R_, hash(++i, W, buf(w - 4)));
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
      for( int j = 0; j < nPltMaps; j++ ) {
        cm.set(R_, hash(++i, iCtx[j]()));
      }
      
      ctx = min(0x1F, x / min(0x20, columns[0]));
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
      mapContexts[j++] = (W + clip(NE * 3 - NNE * 3 + buf(w * 3 - 1))) / 2;
      mapContexts[j++] = (W + clip(NEE * 3 - buf(w * 2 - 3) * 3 + buf(w * 3 - 4))) / 2;
      mapContexts[j++] = clip(NN + buf(w * 4) - buf(w * 6));
      mapContexts[j++] = clip(WW + buf(4) - buf(6));
      mapContexts[j++] = clip((buf(w * 5) - 6 * buf(w * 4) + 15 * NNN - 20 * NN + 15 * N + clamp4(W * 2 - NWW, W, NW, N, NN)) / 6);
      mapContexts[j++] = clip((-3 * WW + 8 * W + clamp4(NEE * 3 - NNEE * 3 + buf(w * 3 - 2), NE, NEE, buf(w - 3), buf(w - 4))) / 6);
      mapContexts[j++] = clip(NN + NW - buf(w * 3 + 1));
      mapContexts[j++] = clip(NN + NE - buf(w * 3 - 1));
      mapContexts[j++] = clip((W * 2 + NW) - (WW + 2 * NWW) + buf(w + 3));
      mapContexts[j++] = clip(((NW + NWW) / 2) * 3 - buf(w * 2 + 3) * 3 + (buf(w * 3 + 4) + buf(w * 3 + 5)) / 2);
      mapContexts[j++] = clip(NEE + NE - buf(w * 2 - 3));
      mapContexts[j++] = clip(NWW + WW - buf(w + 4));
      mapContexts[j++] = clip(((W + NW) * 3 - NWW * 6 + buf(w + 3) + buf(w * 2 + 3)) / 2);
      mapContexts[j++] = clip((NE * 2 + NNE) - (NNEE + buf(w * 3 - 2) * 2) + buf(w * 4 - 3));
      mapContexts[j++] = buf(w * 6);
      mapContexts[j++] = (buf(w - 4) + buf(w - 6)) / 2;
      mapContexts[j++] = (buf(4) + buf(6)) / 2;
      mapContexts[j++] = (W + N + buf(w - 5) + buf(w - 7)) / 4;
      mapContexts[j++] = clip(buf(w - 3) + W - NEE);
      mapContexts[j++] = clip(4 * NNN - 3 * buf(w * 4));
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
      mapContexts[j++] = clip((buf(w * 3 - 3) - 4 * NNEE + 6 * NE + clip(W * 3 - NW * 3 + NNW)) / 4);
      mapContexts[j++] = clip((N * 2 + NE) - (NN + 2 * NNE) + buf(w * 3 - 1));
      mapContexts[j++] = clip((NW * 2 + NNW) - (NNWW + buf(w * 3 + 2) * 2) + buf(w * 4 + 3));
      mapContexts[j++] = clip(NNWW + W - buf(w * 2 + 3));
      mapContexts[j++] = clip((-buf(w * 4) + 5 * NNN - 10 * NN + 10 * N + clip(W * 4 - NWW * 6 + buf(w * 2 + 3) * 4 - buf(w * 3 + 4))) / 5);
      mapContexts[j++] = clip(NEE + clip(buf(w - 3) * 2 - buf(w * 2 - 4)) - buf(w - 4));
      mapContexts[j++] = clip(NW + W - NWW);
      mapContexts[j++] = clip((N * 2 + NW) - (NN + 2 * NNW) + buf(w * 3 + 1));
      mapContexts[j++] = clip(NN + clip(NEE * 2 - buf(w * 2 - 3)) - NNE);
      mapContexts[j++] = clip((-buf(4) + 5 * WWW - 10 * WW + 10 * W + clip(NE * 2 - NNE)) / 5);
      mapContexts[j++] = clip((-buf(5) + 4 * buf(4) - 5 * WWW + 5 * W + clip(NE * 2 - NNE)) / 4);
      mapContexts[j++] = clip((WWW - 4 * WW + 6 * W + clip(NE * 3 - NNE * 3 + buf(w * 3 - 1))) / 4);
      mapContexts[j++] = clip((-NNEE + 3 * NE + clip(W * 4 - NW * 6 + NNW * 4 - buf(w * 3 + 1))) / 3);
      mapContexts[j++] = ((W + N) * 3 - NW * 2) / 4;
      for( j = 0; j < nOLS; j++ ) {
        ols[j].update(W);
        pOLS[j] = clip(int(floor(ols[j].predict(olsCtxs[j]))));
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
      cm.set(R_, hash(++i, clip((-buf(4) + 5 * WWW - 10 * WW + 10 * W + clamp4(NE * 4 - NNE * 6 + buf(w * 3 - 1) * 4 - buf(w * 4 - 1), N, NE, buf(w - 2), buf(w - 3))) / 5)));
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
  if( isGray ) {
    int i = 0;
    map[i++].set(0);
    map[i++].set(((static_cast<uint8_t>(clip(W + N - NW) - b)) * 8 + bpos) |
                       (DiffQt(clip(N + NE - NNE), clip(N + NW - NNW)) << 11));

    for( int j = 0; j < nSM1; i++, j++ ) {
      map[i].set((mapContexts[j] - b) * 8 + bpos);
    }

    for( int j = 0; i < nSM; i++, j++ ) {
      map[i].set((pOLS[j] - b) * 8 + bpos);
    }
  }
  sceneMap[2].setDirect(finalize64(hash(x, line), 19) * 8 + bpos);
  sceneMap[3].setDirect((prvFrmPx - b) * 8 + bpos);
  sceneMap[4].setDirect((prvFrmPrediction - b) * 8 + bpos);

  // predict next bit
  cm.mix(m);
  if( isGray ) {
    for( int i = 0; i < nSM; i++ ) {
      map[i].mix(m);
    }
  } else {
    for( int i = 0; i < nPltMaps; i++ ) {
      pltMap[i].set((bpos << 8) | iCtx[i]());
      pltMap[i].mix(m);
    }
  }
  for( int i = 0; i < nIM; i++ ) {
    const int scale = (prevFramePos > 0 && prevFrameWidth == w) ? 64 : 0;
    sceneMap[i].setScale(scale);
    sceneMap[i].mix(m);
  }

  col = (col + 1) & 7;
  m.set(ctx, 512);
  m.set(col << 1 | static_cast<int>(c0 == ((0x100 | res) >> (8 - bpos))), 16);
  m.set((N + W) >> 4, 32);
  m.set(c0-1, 255);
  m.set((static_cast<int>(abs(W - N) > 4) << 9) | (static_cast<int>(abs(N - NE) > 4) << 8) |
        (static_cast<int>(abs(W - NW) > 4) << 7) | (static_cast<int>(W > N) << 6) | (static_cast<int>(N > NE) << 5) |
        (static_cast<int>(W > NW) << 4) | (static_cast<int>(W > WW) << 3) | (static_cast<int>(N > NN) << 2) |
        (static_cast<int>(NW > NNWW) << 1) | static_cast<int>(NE > NNEE), 1024);
  m.set(min(63, column[0]), 64);
  m.set(min(127, column[1]), 128);
  m.set(min(255, (x + line) / 32), 256);
}
