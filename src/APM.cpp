#include "APM.hpp"
#include "Array.hpp"
#include "Squash.hpp"
#include "Stretch.hpp"
#include "UpdateBroadcaster.hpp"

APM::APM(const Shared* const sh, const int n, const int s, const int limit) : AdaptiveMap(sh, n * s), N(n * s), steps(s), cxt(0), limit(limit) {
  assert(s > 4); // number of steps - must be a positive integer bigger than 4
  for( int i = 0; i < N; ++i ) {
    int p = ((i % steps * 2 + 1) * 4096) / (steps * 2) - 2048;
    t[i] = (uint32_t(squash(p)) << 20) + 6; //initial count: 6
  }
}

void APM::update() {
  assert(cxt >= 0 && cxt < N);
  AdaptiveMap::update(&t[cxt], limit);
}

int APM::p(int pr, int cx) {
  shared->GetUpdateBroadcaster()->subscribe(this);
  assert(pr >= 0 && pr < 4096);
  assert(cx >= 0 && cx < N / steps);
  pr = (stretch(pr) + 2048) * (steps - 1);
  int wt = pr & 0x0fff; // interpolation weight (0..4095)
  cx = cx * steps + (pr >> 12);
  assert(cx >= 0 && cx < N - 1);
  cxt = cx + (wt >> 11);
  pr = ((t[cx] >> 13) * (4096 - wt) + (t[cx + 1] >> 13) * wt) >> 19;
  return pr;
}
