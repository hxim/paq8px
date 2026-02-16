#pragma once

#include "AudioModel.hpp"
#include "../LMS.hpp"
#include "../OLS_factory.hpp"
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
  std::unique_ptr<OLS_double> ols[nOLS][2]; // 2: channels

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
