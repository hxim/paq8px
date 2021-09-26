#ifndef PAQ8PX_AUDIO16BITMODEL_HPP
#define PAQ8PX_AUDIO16BITMODEL_HPP

#include "AudioModel.hpp"
#include "../LMS.hpp"
#include "../OLS.hpp"
#include "../SmallStationaryContextMap.hpp"
#include "../utils.hpp"
#include "../BitCount.hpp"
#include <cstdint>

class Audio16BitModel : AudioModel {
private:
    static constexpr int nOLS = 8;
    static constexpr int nLMS = 3;
    static constexpr int nSSM = nOLS + nLMS + 3;
    static constexpr int nCtx = 4;
    SmallStationaryContextMap sMap1B[nSSM][nCtx];
    OLS<double, short> ols[nOLS][2] {{{shared,128, 24, 0.9975}, {shared,128, 24, 0.9975}},
                                     {{shared,90,  30, 0.997},  {shared,90,  30, 0.997}},
                                     {{shared,90,  31, 0.996},  {shared,90,  31, 0.996}},
                                     {{shared,90,  32, 0.995},  {shared,90,  32, 0.995}},
                                     {{shared,90,  33, 0.995},  {shared,90,  33, 0.995}},
                                     {{shared,90,  34, 0.9985}, {shared,90,  34, 0.9985}},
                                     {{shared,28,  4,  0.98},   {shared,28,  4,  0.98}},
                                     {{shared,32,  3,  0.992},  {shared,32,  3,  0.992}}};
    LMS<float, short> lms[nLMS][2] {{{1280, 640, 5e-5f, 5e-5f}, {1280, 640, 5e-5f, 5e-5f}},
                                    {{640,  64,  7e-5f, 1e-5f}, {640,  64,  7e-5f, 1e-5f}},
                                    {{2450, 8,   2e-5f, 2e-6f}, {2450, 8,   2e-5f, 2e-6f}}};
    int prd[nSSM][2][2] {0};
    int residuals[nSSM][2] {0};
    int ch = 0;
    int lsb = 0;
    uint32_t mask = 0;
    uint32_t errLog = 0;
    uint32_t mxCtx = 0;
    short sample = 0;

public:
    static constexpr int MIXERINPUTS = nCtx * nSSM * SmallStationaryContextMap::MIXERINPUTS; // 112
    static constexpr int MIXERCONTEXTS = 8192 + 4096 + 2560 + 256 + 20; // 15124
    static constexpr int MIXERCONTEXTSETS = 5;

    int stereo = 0;

    explicit Audio16BitModel(Shared* const sh);
    void setParam(int info);
    void mix(Mixer &m);
};

#endif //PAQ8PX_AUDIO16BITMODEL_HPP
