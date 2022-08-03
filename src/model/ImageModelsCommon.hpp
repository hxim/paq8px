#ifndef PAQ8PX_IMAGEMODELSCOMMON_HPP
#define PAQ8PX_IMAGEMODELSCOMMON_HPP

static inline uint8_t paeth(uint8_t const W, uint8_t const N, uint8_t const NW) {
  int p = W + N - NW;
  int pW = abs(p - static_cast<int>(W));
  int pN = abs(p - static_cast<int>(N));
  int pNW = abs(p - static_cast<int>(NW));
  if (pW <= pN && pW <= pNW) {
    return W;
  }
  if (pN <= pNW) {
    return N;
  }
  return NW;
}

#endif