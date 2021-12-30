#pragma once

#include "../ContextMap2.hpp"
#include "../OLS.hpp"
#include "../SmallStationaryContextMap.hpp"
#include "../LargeStationaryMap.hpp"
#include "../StationaryMap.hpp"
#include "ImageModelsCommon.hpp"
#include <cmath>

/**
 * Model for filtered (PNG) or unfiltered 24/32-bit image data
 */
class Image24BitModel {
private:
    static constexpr int nSM0 = 13;
    static constexpr int nSM1 = 76;
    static constexpr int nOLS = 6;
    static constexpr int nSM = nSM0 + nSM1 + nOLS;
    static constexpr int nLSM = 4;
    static constexpr int nSSM = 58;
    static constexpr int nCM = 45;
    Ilog *ilog = &Ilog::getInstance();

public:
    static constexpr int MIXERINPUTS = 
      nSSM * SmallStationaryContextMap::MIXERINPUTS + 
      nLSM * LargeStationaryMap::MIXERINPUTS +
      nSM * StationaryMap::MIXERINPUTS +
      nCM * (ContextMap2::MIXERINPUTS); //593
    static constexpr int MIXERCONTEXTS = (16 * 8) +(5 + 256) + 256 + 512 + 2048 + (8 * 32) + (6 * 64) + (256 * 2) + 1024 + 8192 + 8192 + 8192 + 8192 + 256; //38405
    static constexpr int MIXERCONTEXTSETS = 15;

    Shared * const shared;
    ContextMap2 cm;
    SmallStationaryContextMap SCMap[nSSM];
    LargeStationaryMap mapL;
    StationaryMap map[nSM];
    RingBuffer<uint8_t> buffer {0x100000}; // internal rotating buffer for (PNG unfiltered) pixel data (1 MB)
    //pixel neighborhood
    uint8_t WWWWWW = 0, WWWWW = 0, WWWW = 0, WWW = 0, WW = 0, W = 0;
    uint8_t NWWWW = 0, NWWW = 0, NWW = 0, NW = 0, N = 0, NE = 0, NEE = 0, NEEE = 0, NEEEE = 0;
    uint8_t NNWWW = 0, NNWW = 0, NNW = 0, NN = 0, NNE = 0, NNEE = 0, NNEEE = 0;
    uint8_t NNNWW = 0, NNNW = 0, NNN = 0, NNNE = 0, NNNEE = 0;
    uint8_t NNNNW = 0, NNNN = 0, NNNNE = 0;
    uint8_t NNNNN = 0;
    uint8_t NNNNNN = 0;
    uint8_t WWp1 = 0, Wp1 = 0, p1 = 0, NWp1 = 0, Np1 = 0, NEp1 = 0, NNp1 = 0;
    uint8_t WWp2 = 0, Wp2 = 0, p2 = 0, NWp2 = 0, Np2 = 0, NEp2 = 0, NNp2 = 0;
    uint8_t px = 0; // current PNG filter prediction
    int info = 0;
    uint32_t alpha = 0, isPNG = 0;
    int color = -1;
    int stride = 3;
    int col = 0;
    int ctx[2] {}, padding = 0, filter = 0, x = 0, w = 0, line = 0, R1 = 0, R2 = 0;
    uint32_t lastPos = 0, lastWasPNG = 0;
    bool filterOn = false;
    int columns[2] = {1, 1}, column[2] {};
    short mapContexts[nSM1] = { 0 }, scMapContexts[nSSM] = { 0 };
    uint8_t pOLS[nOLS] = {0};
    static constexpr double lambda[nOLS] = {0.98, 0.87, 0.9, 0.8, 0.9, 0.7};
    static constexpr int num[nOLS] = {32, 12, 15, 10, 14, 8};
    OLS<double, uint8_t> ols[nOLS][4] = {{{shared,num[0], 1, lambda[0]}, {shared,num[0], 1, lambda[0]}, {shared,num[0], 1, lambda[0]}, {shared,num[0], 1, lambda[0]}},
                                         {{shared,num[1], 1, lambda[1]}, {shared,num[1], 1, lambda[1]}, {shared,num[1], 1, lambda[1]}, {shared,num[1], 1, lambda[1]}},
                                         {{shared,num[2], 1, lambda[2]}, {shared,num[2], 1, lambda[2]}, {shared,num[2], 1, lambda[2]}, {shared,num[2], 1, lambda[2]}},
                                         {{shared,num[3], 1, lambda[3]}, {shared,num[3], 1, lambda[3]}, {shared,num[3], 1, lambda[3]}, {shared,num[3], 1, lambda[3]}},
                                         {{shared,num[4], 1, lambda[4]}, {shared,num[4], 1, lambda[4]}, {shared,num[4], 1, lambda[4]}, {shared,num[4], 1, lambda[4]}},
                                         {{shared,num[5], 1, lambda[5]}, {shared,num[5], 1, lambda[5]}, {shared,num[5], 1, lambda[5]}, {shared,num[5], 1, lambda[5]}}};
    const uint8_t *olsCtx1[32] = {&WWWWWW, &WWWWW, &WWWW, &WWW, &WW, &W, &NWWWW, &NWWW, &NWW, &NW, &N, &NE, &NEE, &NEEE, &NEEEE, &NNWWW,
                                  &NNWW, &NNW, &NN, &NNE, &NNEE, &NNEEE, &NNNWW, &NNNW, &NNN, &NNNE, &NNNEE, &NNNNW, &NNNN, &NNNNE, &NNNNN,
                                  &NNNNNN};
    const uint8_t *olsCtx2[12] = {&WWW, &WW, &W, &NWW, &NW, &N, &NE, &NEE, &NNW, &NN, &NNE, &NNN};
    const uint8_t *olsCtx3[15] = {&N, &NE, &NEE, &NEEE, &NEEEE, &NN, &NNE, &NNEE, &NNEEE, &NNN, &NNNE, &NNNEE, &NNNN, &NNNNE, &NNNNN};
    const uint8_t *olsCtx4[10] = {&N, &NE, &NEE, &NEEE, &NN, &NNE, &NNEE, &NNN, &NNNE, &NNNN};
    const uint8_t *olsCtx5[14] = {&WWWW, &WWW, &WW, &W, &NWWW, &NWW, &NW, &N, &NNWW, &NNW, &NN, &NNNW, &NNN, &NNNN};
    const uint8_t *olsCtx6[8] = {&WWW, &WW, &W, &NNN, &NN, &N, &p1, &p2};
    const uint8_t **olsCtxs[nOLS] = {&olsCtx1[0], &olsCtx2[0], &olsCtx3[0], &olsCtx4[0], &olsCtx5[0], &olsCtx6[0]};

    Image24BitModel(Shared* const sh, uint64_t size);
    void update();

    /**
     * New image.
     */
    void init();
    void setParam(int width, uint32_t alpha0, uint32_t isPNG0);
    void mix(Mixer &m);
};
