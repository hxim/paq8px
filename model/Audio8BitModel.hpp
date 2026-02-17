#pragma once

#include "AudioModel.hpp"
#include "../Ilog.hpp"
#include "../LMS.hpp"
#include "../OLS_factory.hpp"
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

  std::unique_ptr<LMS> lms[nLMS][2]; // 2: channels

  static constexpr int num[nOLS] = { 128, 90, 90, 90, 90, 90, 28, 28 };
  static constexpr int solveInterval[nOLS] = { 24, 30, 31, 32, 33, 34, 4, 3 };
  static constexpr float lambda[nOLS] = { 0.9975f, 0.9965f, 0.996f, 0.995f, 0.995f, 0.9985f, 0.98f, 0.992f };
  static constexpr float nu = 0.001f;
  std::unique_ptr<OLS_float> ols[nOLS][2];

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
