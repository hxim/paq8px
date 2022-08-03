#include "APM1.hpp"
#include "Array.hpp"
#include "UpdateBroadcaster.hpp"
#include "Squash.hpp"
#include "Stretch.hpp"

APM1::APM1(const Shared* const sh, const int n, const int r) : shared(sh), index(0), n(n), t(n * 33), rate(r) {
  assert(n > 0 && rate > 0 && rate < 32);
  // maps p, cxt -> p initially
  for( int i = 0; i < n; ++i ) {
    for( int j = 0; j < 33; ++j ) {
      if( i == 0 ) {
        t[i * 33 + j] = squash((j - 16) * 128) * 16;
      } else {
        t[i * 33 + j] = t[j];
      }
    }
  }
}

int APM1::p(int pr, const int cxt) {
  shared->GetUpdateBroadcaster()->subscribe(this);
  assert(pr >= 0 && pr < 4096 && cxt >= 0 && cxt < n);
  pr = stretch(pr);
  const int w = pr & 127; // interpolation weight (33 points)
  index = ((pr + 2048) >> 7) + cxt * 33;
  return (t[index] * (128 - w) + t[index + 1] * w) >> 11;
}

void APM1::update() {
  INJECT_SHARED_y
  const int g = (y << 16) + (y << rate) - y - y;
  t[index] += (g - t[index]) >> rate;
  t[index + 1] += (g - t[index + 1]) >> rate;
}
