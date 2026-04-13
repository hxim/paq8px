#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>

// PNG Paeth predictor from the World Wide Web Consortium / PNG Development Group spec.
static inline uint8_t paeth(uint8_t const W, uint8_t const N, uint8_t const NW) {
  int p = W + N - NW;
  int pW = abs(p - W);
  int pN = abs(p - N);
  int pNW = abs(p - NW);
  if (pW <= pN && pW <= pNW) {
    return W;
  }
  if (pN <= pNW) {
    return N;
  }
  return NW;
}


// CALIC-style Gradient Adjusted Predictor (GAP).
static inline int gap(
  uint8_t const W, uint8_t const N,
  uint8_t const NW, uint8_t const NE,
  uint8_t const WW, uint8_t const NNE,
  uint8_t const NN
) {
  int dH = abs(W - WW) + abs(N - NW) + abs(NE - N);
  int dV = abs(W - NW) + abs(N - NN) + abs(NE - NNE);

  if (dH > dV) {
    return N;
  }
  if (dV > dH) {
    return W;
  }
  return (N + W - NW);
}

//signed difference between two 8-bit pixel values, quantized
//used by 8-bit and 24-bit image models
ALWAYS_INLINE
int DiffQt(const uint8_t a, const uint8_t b) {
  int d = (a > b) ? (a - b) : (b - a);
  if (d <= 2)d = d;
  else if (d <= 5)d = 3; //3..5
  else if (d <= 9)d = 4; //6..9
  else if (d <= 14)d = 5; //10..14
  else if (d <= 23)d = 6; //15..23
  else d = 7; //24..255
  const int sign = a > b ? 8 : 0;
  return (sign | d);
}
