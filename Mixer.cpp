#include "Mixer.hpp"

Mixer::Mixer(const Shared* const sh, const int n, const int m, const int s) : shared(sh),
  n(n), m(m), s(s), 
  scaleFactor(0), tx(n), wx(n * m), cxt(s), info(s), rates(s), pr(s) {
  for( uint64_t i = 0; i < s; ++i ) {
    pr[i] = 2048; //initial p=0.5
    rates[i] = MAX_LEARNING_RATE;
    info[i].reset();
  }
  reset();
}

void Mixer::add(const int x) {
  assert(nx < n);
  assert(x == short(x));
  tx[nx++] = static_cast<short>(x);
}

void Mixer::set(const uint32_t cx, const uint32_t range) {
  assert(numContexts < s);
  assert(cx < range);
  assert(base + range <= m);
  cxt[numContexts++] = base + cx;
  base += range;
}

void Mixer::skip(const uint32_t range) {
  assert(numContexts < s);
  assert(base + range <= m);
  cxt[numContexts++] = UINT32_MAX; // flag for skipping
  base += range;
}

void Mixer::reset() {
  nx = 0;
  base = 0;
  numContexts = 0;
}
