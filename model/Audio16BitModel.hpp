#pragma once

#include "AudioModel.hpp"
#include "../LMS.hpp"
#include "../OLS.hpp"
#include "../SmallStationaryContextMap.hpp"
#include "../Utils.hpp"
#include "../BitCount.hpp"
#include <cstdint>

class Audio16BitModel : AudioModel {
private:
  static constexpr int nOLS = 8;
  static constexpr int nLMS = 3;
  static constexpr int nSSM = nOLS + nLMS + 3;
  static constexpr int nCtx = 4;
  SmallStationaryContextMap sMap1B[nSSM][nCtx];

  static constexpr int num[nOLS] = { 128, 90, 90, 90, 90, 90, 28, 32 };
  static constexpr int solveInterval[nOLS] = { 24, 30, 31, 32, 33, 34, 4, 3 };
  static constexpr double lambda[nOLS] = { 0.9975, 0.997,0.996, 0.995, 0.995, 0.9985, 0.98, 0.992 };
  static constexpr double nu = 0.001;
  OLS_double ols[nOLS][2] {
    {{num[0],solveInterval[0],lambda[0],nu}, {num[0],solveInterval[0],lambda[0],nu}},
    {{num[1],solveInterval[1],lambda[1],nu}, {num[1],solveInterval[1],lambda[1],nu}},
    {{num[2],solveInterval[2],lambda[2],nu}, {num[2],solveInterval[2],lambda[2],nu}},
    {{num[3],solveInterval[3],lambda[3],nu}, {num[3],solveInterval[3],lambda[3],nu}},
    {{num[4],solveInterval[4],lambda[4],nu}, {num[4],solveInterval[4],lambda[4],nu}},
    {{num[5],solveInterval[5],lambda[5],nu}, {num[5],solveInterval[5],lambda[5],nu}},
    {{num[6],solveInterval[6],lambda[6],nu}, {num[6],solveInterval[6],lambda[6],nu}},
    {{num[7],solveInterval[7],lambda[7],nu}, {num[7],solveInterval[7],lambda[7],nu}}
  };

  std::unique_ptr<LMS> lms[nLMS][2];
  int prd[nSSM][2][2] {0};
  int residuals[nSSM][2] {0};
  uint32_t ch = 0;
  uint32_t lsb = 0;
  uint32_t mask = 0;
  uint32_t errLog = 0;
  uint32_t mxCtx = 0;
  short sample = 0;

public:
  static constexpr int MIXERINPUTS = nCtx * nSSM * SmallStationaryContextMap::MIXERINPUTS; // 112
  static constexpr int MIXERCONTEXTS = 8192 + 4096 + 2560 + 256 + 20; // 15124
  static constexpr int MIXERCONTEXTSETS = 5;

  uint32_t stereo = 0;

  explicit Audio16BitModel(Shared* const sh);
  void setParam(int info);
  void mix(Mixer &m);
};
