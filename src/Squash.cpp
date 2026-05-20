#include "Squash.hpp"
#include <cassert>
#include <cmath>
#include <cstdint>


class squash_table {
public:
  short t[4095];
  int squash(int d /*-2047..2047*/) {
    /**
     * return p = 1/(1 + exp(-d)), d scaled by 8 bits, p scaled by 12 bits
     * @param d
     * @return
     */
    if (d < -2047)return 1;
    if (d > 2047)return 4095;
    float p = 1.0f / (1.0f + exp(-d / 256.0f));
    p *= 4096.0;
    uint32_t pi = (uint32_t)round(p);
    if (pi > 4095)pi = 4095;
    if (pi < 1)pi = 1;
    return pi;
  }

  squash_table() {
    for (int i = -2047; i <= 2047; i++) {
      t[i + 2047] = squash(i);
    }
  }

} sqt;

int squash(int d /*-2047..2047*/) {
  if (d < -2047)return 1;
  if (d > 2047)return 4095;
  return sqt.t[d + 2047];
}

