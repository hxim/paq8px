#include "StateTable.hpp"

auto StateTable::next(uint8_t const state, const int y) -> uint8_t {
  return stateTable[state][y];
}

auto StateTable::next(uint8_t const oldState, const int y, Random &rnd) -> uint8_t {
  const uint8_t newState = stateTable[oldState][y];
  if (newState >= 205 && newState >= oldState + 4) {
    // Apply probabilistic increment for contexts with strong trend of 0s or 1s.
    // Applicable to states 149 and above, but we'll start applying it from states (205-208).
    // That means...
    // For states 205..208, 209..212, ... 249..252 a group of 4 states is not represented by
    // the counts indicated in the state table. An exponential scale is used instead.
    // The highest group (249..252) represents the top of this scale, where we can not increment anymore.
    // Reaching the highest states (249-252) from states (201-204) takes 12 steps.
    // thus the probability of reaching the highest states (249-252) is ~ 1/2048 
    if (rnd(1) != 0) { // random bit: p()=1/2 -> the probability to advance to a higher state is 1/2
      return oldState; // don't advance
    }
  }
  return newState;
}

void StateTable::update(uint8_t *const state, const int y, Random &rnd) {
  *state = next(*state, y, rnd);
}

auto StateTable::group(const uint8_t state) -> uint8_t {
  return stateGroup[state];
}

auto StateTable::prio(const uint8_t state) -> uint8_t {
  return statePrio[state];
}
