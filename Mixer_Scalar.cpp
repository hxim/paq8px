#include "Mixer_Scalar.hpp"

static constexpr int SIMD_WIDTH_SCALAR = 4 / sizeof(short); // processes 2 shorts at once

Mixer_Scalar::Mixer_Scalar(const Shared* const sh, const int n, const int m, const int s, const int promoted)
  : Mixer(sh, n, m, s, SIMD_WIDTH_SCALAR) {
  initSecondLayer(promoted);
}

void Mixer_Scalar::initSecondLayer(const int promoted) {
  if (s > 1) {
    mp = new Mixer_Scalar(shared, s + promoted, 1, 1, 0);
  }
}

int Mixer_Scalar::dotProduct(const short* const w, const size_t n) {
  int sum = 0;
  for (size_t i = 0; i < n; i += 2) {
    sum += (tx[i] * w[i] + tx[i + 1] * w[i + 1]) >> 8;
  }
  return sum;
}

void Mixer_Scalar::train(short* const w, const size_t n, const int e) {
  for (size_t i = 0; i < n; i++) {
    int wt = w[i] + ((((tx[i] * e * 2) >> 16) + 1) >> 1);
    if (wt < -32768) {
      wt = -32768;
    }
    else if (wt > 32767) {
      wt = 32767;
    }
    *reinterpret_cast<short*>(&w[i]) = wt;
  }
}
