#pragma once

#include "../Shared.hpp"
#include "../Mixer.hpp"
#include "../Random.hpp"
#include "../RingBuffer.hpp"
#include "../StateMap.hpp"

/**
 * Model for 1-bit image data
 */
class Image1BitModel {
private:
  static constexpr int s = 11;
  const Shared * const shared;
  Random rnd;
  int w = 0;
  uint32_t r0 = 0, r1 = 0, r2 = 0, r3 = 0; /**< last 4 rows, bit 8 is over current pixel */
  Array<uint8_t> t {0x23000}; /**< model: cxt -> state */
  int cxt[s] {}; // contexts
  StateMap sm;

public:
  static constexpr int MIXERINPUTS = s;
  static constexpr int MIXERCONTEXTS = 4;
  static constexpr int MIXERCONTEXTSETS = 1;
  Image1BitModel(const Shared* const sh);
  void setParam(int info0);
  void mix(Mixer &m);
};
