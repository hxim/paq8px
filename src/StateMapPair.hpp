#pragma once

#include "AdaptiveMap.hpp"

/**
 * A @ref StateMapPair maps a context to two probabilities.
 */
class StateMapPair : AdaptiveMap, IPredictor {
private:
  uint32_t cxt{};

public:
  int limit_slow; //1..1023, optimal:  1023
  int limit_fast; //1..1023, optimal: 4..64

  /**
   * Creates a @ref StateMap with @ref n contexts using 4*2*n bytes memory.
   * @param n number of contexts
   * @param lim_slow
   * @param lim_fast
   */
  StateMapPair(const Shared* const sh, int n, int lim_slow, int lim_fast);
  ~StateMapPair() override = default;

  void reset();
  void update() override;

  void p(uint32_t cx, int& p_slow, int& p_fast);
};
