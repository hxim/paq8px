#pragma once

#include "IPredictor.hpp"
#include "Mixer.hpp"
#include "Stretch.hpp"
#include "UpdateBroadcaster.hpp"

/**
 * Map for modelling contexts of (nearly-)stationary data.
 * The context is looked up directly. For each bit modelled, a 16bit prediction is stored.
 * The adaptation rate is controlled by the caller, see @ref SmallStationaryContextMap::mix().
 *
 * Uses (2^(BitsOfContext+1))*((2^InputBits)-1) bytes of memory.
 */
class SmallStationaryContextMap : IPredictor {
public:
  static constexpr int MIXERINPUTS = 2;

private:
  const Shared * const shared;
  Array<uint16_t> data;
  const uint32_t mask;
  const uint32_t stride;
  const uint32_t bTotal;
  uint32_t b {};
  uint32_t context {};
  uint32_t bCount {};
  uint16_t *cp {};
  const int rate;
  int scale;

public:
  /**
    * Construct using (2^(BitsOfContext+1))*((2^InputBits)-1) bytes of memory.
    * @param bitsOfContext How many bits to use for each context. Higher bits are discarded.
    * @param inputBits How many bits [1..8] of input are to be modelled for each context. New contexts must be set at those intervals.
    * @param rate
    * @param scale
    */
  SmallStationaryContextMap(const Shared* const sh, int bitsOfContext, int inputBits, int rate, int scale);
  void set(uint32_t ctx);
  void reset();
  void update() override;
  void mix(Mixer &m);
};
