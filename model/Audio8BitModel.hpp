#pragma once

#include "AudioModel.hpp"
#include "../Ilog.hpp"
#include "../LMS.hpp"
#include "../OLS.hpp"
#include "../SmallStationaryContextMap.hpp"
#include <cmath>
#include <cstdint>

class Audio8BitModel : AudioModel {
private:
    static constexpr int nOLS = 8;
    static constexpr int nLMS = 3;
    static constexpr int nSSM = nOLS + nLMS + 3;
    static constexpr int nCtx = 3;
    SmallStationaryContextMap sMap1B[nSSM][nCtx];
    OLS<double, int8_t> ols[nOLS][2] {{{shared,128, 24, 0.9975}, {shared,128, 24, 0.9975}},
                                      {{shared,90,  30, 0.9965}, {shared,90,  30, 0.9965}},
                                      {{shared,90,  31, 0.996},  {shared,90,  31, 0.996}},
                                      {{shared,90,  32, 0.995},  {shared,90,  32, 0.995}},
                                      {{shared,90,  33, 0.995},  {shared,90,  33, 0.995}},
                                      {{shared,90,  34, 0.9985}, {shared,90,  34, 0.9985}},
                                      {{shared,28,  4,  0.98},   {shared,28,  4,  0.98}},
                                      {{shared,28,  3,  0.992},  {shared,28,  3,  0.992}}};
    LMS<float, int8_t> lms[nLMS][2] {{{1280, 640, 3e-5f,   2e-5f}, {1280, 640, 3e-5f,   2e-5f}},
                                     {{640,  64,  8e-5f,   1e-5f}, {640,  64,  8e-5f,   1e-5f}},
                                     {{2450, 8,   1.6e-5f, 1e-6f}, {2450, 8,   1.6e-5f, 1e-6f}}};
    int prd[nSSM][2][2] {0};
    int residuals[nSSM][2] {0};
    int ch = 0;
    uint32_t mask = 0;
    uint32_t errLog = 0;
    uint32_t mxCtx = 0;

public:
    static constexpr int MIXERINPUTS = nCtx * nSSM * SmallStationaryContextMap::MIXERINPUTS; // 84
    static constexpr int MIXERCONTEXTS = 4096 + 2048 + 2048 + 256 + 10; // 8458
    static constexpr int MIXERCONTEXTSETS = 5;

    int stereo = 0;

    explicit Audio8BitModel(Shared* const sh);
    void setParam(int info);
    void mix(Mixer &m);
};
