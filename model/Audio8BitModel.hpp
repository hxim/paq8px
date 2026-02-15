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

  static constexpr int num[nOLS] = { 128, 90, 90, 90, 90, 90, 28, 28 };
  static constexpr int solveInterval[nOLS] = { 24, 30, 31, 32, 33, 34, 4, 3 };
  static constexpr float lambda[nOLS] = { 0.9975f, 0.9965f, 0.996f, 0.995f, 0.995f, 0.9985f, 0.98f, 0.992f };
  static constexpr float nu = 0.001f;
  OLS_float ols[nOLS][2] {
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
