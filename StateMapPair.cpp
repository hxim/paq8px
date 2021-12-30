#include "StateMapPair.hpp"

StateMapPair::StateMapPair(const Shared* const sh, const int n, const int lim_slow, int lim_fast) :
  AdaptiveMap(sh, n * 2), limit_slow(lim_slow), limit_fast(lim_fast) {
  reset();
}

void StateMapPair::reset() {
  uint32_t n = t.size();
  for (uint32_t i = 0; i < n; ++i) {
    t[i] = 2048 << 20 | 5; //initial p=0.5, initial count=5
  }
}

void StateMapPair::update() {
  AdaptiveMap::update(&t[cxt * 2 + 0], limit_slow);
  AdaptiveMap::update(&t[cxt * 2 + 1], limit_fast);
}

void StateMapPair::p(uint32_t cx, int& p_slow, int& p_fast) {
  shared->GetUpdateBroadcaster()->subscribe(this);
  cxt = cx;
  p_slow = t[cxt * 2 + 0] >> 20;
  p_fast = t[cxt * 2 + 1] >> 20;
}
