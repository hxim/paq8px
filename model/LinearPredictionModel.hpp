#pragma once

#include "../OLS.hpp"
#include "../SmallStationaryContextMap.hpp"
#include <cmath>

class LinearPredictionModel {
private:
  static constexpr int nOLS = 3;
  static constexpr int nSSM = nOLS + 2;
  const Shared * const shared;
  SmallStationaryContextMap sMap[nSSM];
  static constexpr float nu = 0.001f;
  OLS_float ols[nOLS] {
    {32, 4, 0.995f, nu},
    {32, 4, 0.995f, nu},
    {32, 4, 0.995f, nu}
  };
  uint8_t prd[nSSM] {0};

public:
  static constexpr int MIXERINPUTS = nSSM * SmallStationaryContextMap::MIXERINPUTS; // 10
  static constexpr int MIXERCONTEXTS = 0;
  static constexpr int MIXERCONTEXTSETS = 0;
  LinearPredictionModel(const Shared* const sh);
  void mix(Mixer &m);
};
