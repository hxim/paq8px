#include "Mixer_Scalar.hpp"

static constexpr int SIMD_WIDTH_SCALAR = 4 / sizeof(short); // processes 2 shorts at once

Mixer_Scalar::Mixer_Scalar(const Shared* const sh, const int n, const int m, const int s, const int promoted)
  : Mixer(sh, n, m, s, promoted, SIMD_WIDTH_SCALAR) {
}

int Mixer_Scalar::dotProduct(const short* const t, const short* const w, const size_t n) {
  int sum = 0;
  for (size_t i = 0; i < n; i += 2) {
    sum += (t[i] * w[i] + t[i + 1] * w[i + 1]) >> 8;
  }
  return sum;
}

int Mixer_Scalar::dotProduct2(const short* const t, const short* const w0, const short* const w1, const size_t n, int& sum1) {
  int s0 = 0;
  int s1 = 0;
  for (size_t i = 0; i < n; i += 2) {
    const int t0 = t[i];
    const int t1 = t[i + 1];
    s0 += (t0 * w0[i] + t1 * w0[i + 1]) >> 8;
    s1 += (t0 * w1[i] + t1 * w1[i + 1]) >> 8;
  }
  sum1 = s1;
  return s0;
}

ALWAYS_INLINE
static int addsSaturate16(const int a, const int b) { // _mm_adds_epi16
  const int sum = a + b;
  if (sum < -32768) { return -32768; }
  if (sum > 32767) { return 32767; }
  return sum;
}

void Mixer_Scalar::train(const short* const t, short* const w, const size_t n, const int e) {
  for (size_t i = 0; i < n; i++) {
    const int t2 = addsSaturate16(t[i], t[i]);        // _mm_adds_epi16(t, t)
    const int prod = (t2 * e) >> 16;                  // _mm_mulhi_epi16
    const int rounded = addsSaturate16(prod, 1) >> 1; // _mm_adds_epi16(.,1), _mm_srai_epi16(.,1)
    w[i] = static_cast<short>(addsSaturate16(rounded, w[i]));
  }
}
