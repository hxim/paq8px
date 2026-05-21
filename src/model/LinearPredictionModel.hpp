#pragma once

#include "../OLS_factory.hpp"
#include "../ResidualMap.hpp"
#include <cmath>

class LinearPredictionModel {
private:
  static constexpr int nOLS = 3;
  static constexpr int nRM = nOLS + 4;
  const Shared * const shared;
  ResidualMap mapR;
  static constexpr float lambda[nOLS] = { 1.0f - 1.0f / 162.0f, 1.0f - 1.0f / 162.0f, 1.0f - 1.0f / 162.0f }; //~ 0.9938
  static constexpr int num[nOLS] = { 32, 32, 32};
  static constexpr int solveInterval[nOLS] = { 4, 4, 4 };
  static constexpr float nu = 0.001f;
  std::unique_ptr<OLS_float> ols[nOLS];
  short prd[nRM] {0};
  int predErrBuf[nRM]{0};

public:
  static constexpr int MIXERINPUTS = nRM * ResidualMap::MIXERINPUTS; // 14
  static constexpr int MIXERCONTEXTS = 0;
  static constexpr int MIXERCONTEXTSETS = 0;
  LinearPredictionModel(const Shared* const sh);
  void mix(Mixer &m);
};
