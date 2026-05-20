#pragma once

#include "../Shared.hpp"
#include "../Mixer.hpp"
#include "../Random.hpp"
#include "../StateMap.hpp"
#include "../LargeStationaryMap.hpp"

/**
 * Model for 1-bit image data
 */
class Image1BitModel {
private:
  static constexpr int N = 17;
  static constexpr int C = (1 << 2) + (1 << 4) + (1 << 4) + (1 << 6) + (1 << 8) + (1 << 8) + (1 << 8) + (1 << 8) + (1 << 8) + (1 << 9) + (1 << 10) + (1 << 12) + (1 << 12) + 5 + 13 + 25 + 41; //11192
  static constexpr int nLSM = 5;

  const Shared* const shared;
  LargeStationaryMap mapL;
  StateMap stateMap;
  Random rnd;
  Array<uint32_t> cxt{ N };  // context indexes
  Array<uint8_t> t{ C };  // state byte per context
  Array<uint32_t> counts{ C }; // 0/1 counts per context in high and low words

  uint32_t w{};
  void add(Mixer& m, uint32_t n0, uint32_t n1);

  uint32_t r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0, r8 = 0; /**< last 8 rows, bit 8 is over (=N) the current pixel */

public:
  static constexpr int MIXERINPUTS = 3 * N + 4 * 2 + nLSM *LargeStationaryMap::MIXERINPUTS; //74
  static constexpr int MIXERCONTEXTS = 16 + 64 + 41; //121
  static constexpr int MIXERCONTEXTSETS = 3;
  Image1BitModel(const Shared* const sh);
  void setParam(int info0);
  void update();
  void mix(Mixer &m);
};
