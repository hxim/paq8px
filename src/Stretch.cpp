#include "Stretch.hpp"
#include <cassert>
#include <cmath>
#include <cstdint>

class stretch_table {
public:
  short t[4096];
  int stretch(int p) {
    /**
   * Inverse of @ref squash. d = ln(p/(1-p)), d scaled by 8 bits, p by 12 bits.
   * d has range -2047 to 2047 representing -8 to 8. p has range 0 to 4095.
   */
    assert(p >= 0 && p <= 4095);
    if (p == 0)p = 1;
    float f = p / 4096.0f;
    float d = log(f / (1.0f - f)) * 256.0f;
    int32_t di = (int32_t)round(d);
    if (di > 2047)di = 2047;
    if (di < -2047)di = -2047;
    return di;
  }

  stretch_table() {
    for (int i = 0; i <= 4095; i++) {
      t[i] = stretch(i);
    }
  }

} str;


int stretch(int p) {
  return str.t[p];
}

