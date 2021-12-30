#pragma once

#include "../Array.hpp"
#include "../ContextMap2.hpp"
#include "../IndirectContext.hpp"
#include "../IndirectMap.hpp"
#include "../OLS.hpp"
#include "../SmallStationaryContextMap.hpp"
#include "../StationaryMap.hpp"
#include "ImageModelsCommon.hpp"
#include <cstdint>

/**
 * Model for 8-bit image data
 */
class Image8BitModel {
private:
    static constexpr int nSM0 = 2;
    static constexpr int nSM1 = 55;
    static constexpr int nOLS = 5;
    static constexpr int nSM = nSM0 + nSM1 + nOLS;
    static constexpr int nPltMaps = 4;
    static constexpr int nCM = 49 + nPltMaps;
    static constexpr int nIM = 5;

public:
    static constexpr int MIXERINPUTS =
      nSM * StationaryMap::MIXERINPUTS + 
      nCM * (ContextMap2::MIXERINPUTS + ContextMap2::MIXERINPUTS_RUN_STATS) + 
      nPltMaps * SmallStationaryContextMap::MIXERINPUTS +
      nIM * IndirectMap::MIXERINPUTS; //469
    static constexpr int MIXERCONTEXTS = (2048 + 5) + 6 * 16 + 6 * 32 + 256 + 1024 + 64 + 128 + 256; /**< 4069 */
    static constexpr int MIXERCONTEXTSETS = 8;

    Shared * const shared;
    ContextMap2 cm;
    StationaryMap map[nSM];
    SmallStationaryContextMap pltMap[nPltMaps];  /**< palette maps, not used for grayscale images */
    IndirectMap sceneMap[nIM];
    IndirectContext<uint8_t> iCtx[nPltMaps]; /**< palette contexts, not used for grayscale images */
    RingBuffer<uint8_t> buffer {0x100000}; /**< internal rotating buffer for (PNG unfiltered) pixel data (1MB) */
    Array<short> jumps {0x8000};
    //pixel neighborhood
    uint8_t WWWWWW = 0, WWWWW = 0, WWWW = 0, WWW = 0, WW = 0, W = 0;
    uint8_t NWWWW = 0, NWWW = 0, NWW = 0, NW = 0, N = 0, NE = 0, NEE = 0, NEEE = 0, NEEEE = 0;
    uint8_t NNWWW = 0, NNWW = 0, NNW = 0, NN = 0, NNE = 0, NNEE = 0, NNEEE = 0;
    uint8_t NNNWW = 0, NNNW = 0, NNN = 0, NNNE = 0, NNNEE = 0;
    uint8_t NNNNW = 0, NNNN = 0, NNNNE = 0;
    uint8_t NNNNN = 0;
    uint8_t NNNNNN = 0;
    uint8_t px = 0; /**< current PNG filter prediction */
    uint8_t res = 0; /**< expected residual */
    uint8_t prvFrmPx = 0; /**< corresponding pixel in previous frame */
    uint8_t prvFrmPrediction = 0; /**< prediction for corresponding pixel in previous frame */
    uint32_t lastPos = 0;
    uint32_t lastWasPNG = 0;
    uint32_t isGray = 0;
    uint32_t isPNG = 0;
    int w = 0;
    int ctx = 0;
    int col = 0;
    int line = 0;
    int x = 0;
    int filter = 0;
    int jump = 0;
    int framePos = 0;
    int prevFramePos = 0;
    int frameWidth = 0;
    int prevFrameWidth = 0;
    bool filterOn = false;
    int columns[2] = {1, 1}, column[2] {};
    uint8_t mapContexts[nSM1] = {0};
    uint8_t pOLS[nOLS] = {0};
    static constexpr double lambda[nOLS] = {0.996, 0.87, 0.93, 0.8, 0.9};
    static constexpr int num[nOLS] = {32, 12, 15, 10, 14};
    OLS<double, uint8_t> ols[nOLS] = {{shared,num[0], 1, lambda[0]},
                                      {shared,num[1], 1, lambda[1]},
                                      {shared,num[2], 1, lambda[2]},
                                      {shared,num[3], 1, lambda[3]},
                                      {shared,num[4], 1, lambda[4]}};
    OLS<double, uint8_t> sceneOls {shared,13, 1, 0.994};
    const uint8_t *olsCtx1[32] = {&WWWWWW, &WWWWW, &WWWW, &WWW, &WW, &W, &NWWWW, &NWWW, &NWW, &NW, &N, &NE, &NEE, &NEEE, &NEEEE, &NNWWW,
                                  &NNWW, &NNW, &NN, &NNE, &NNEE, &NNEEE, &NNNWW, &NNNW, &NNN, &NNNE, &NNNEE, &NNNNW, &NNNN, &NNNNE, &NNNNN,
                                  &NNNNNN};
    const uint8_t *olsCtx2[12] = {&WWW, &WW, &W, &NWW, &NW, &N, &NE, &NEE, &NNW, &NN, &NNE, &NNN};
    const uint8_t *olsCtx3[15] = {&N, &NE, &NEE, &NEEE, &NEEEE, &NN, &NNE, &NNEE, &NNEEE, &NNN, &NNNE, &NNNEE, &NNNN, &NNNNE, &NNNNN};
    const uint8_t *olsCtx4[10] = {&N, &NE, &NEE, &NEEE, &NN, &NNE, &NNEE, &NNN, &NNNE, &NNNN};
    const uint8_t *olsCtx5[14] = {&WWWW, &WWW, &WW, &W, &NWWW, &NWW, &NW, &N, &NNWW, &NNW, &NN, &NNNW, &NNN, &NNNN};
    const uint8_t **olsCtxs[nOLS] = {&olsCtx1[0], &olsCtx2[0], &olsCtx3[0], &olsCtx4[0], &olsCtx5[0]};

    Image8BitModel(Shared* const sh, uint64_t size);
    void setParam(int info0, uint32_t gray0, uint32_t isPNG0);
    void mix(Mixer &m);
};
