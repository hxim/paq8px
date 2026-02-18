#include "Mixer_SSE2.hpp"

#ifdef X64_SIMD_AVAILABLE

static constexpr int SIMD_WIDTH_SSE2 = 16 / sizeof(short); // 8 shorts per 128-bit lane

Mixer_SSE2::Mixer_SSE2(const Shared* const sh, const int n, const int m, const int s, const int promoted)
  : Mixer(sh, n, m, s, SIMD_WIDTH_SSE2) {
  if (s > 1) {
    mp = new Mixer_SSE2(shared, s + promoted, 1, 1, 0);
  }
}

#if (defined(__GNUC__) || defined(__clang__))
__attribute__((target("sse2")))
#endif
int Mixer_SSE2::dotProduct(const short* const w, const size_t n) {
  __m128i sum = _mm_setzero_si128();

  for (size_t i = 0; i < n; i += 8) {
    __m128i tmp = _mm_madd_epi16(*(__m128i*) & tx[i], *(__m128i*) & w[i]);
    tmp = _mm_srai_epi32(tmp, 8);
    sum = _mm_add_epi32(sum, tmp);
  }

  sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
  sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));
  return _mm_cvtsi128_si32(sum);
}

#if (defined(__GNUC__) || defined(__clang__))
__attribute__((target("sse2")))
#endif
int Mixer_SSE2::dotProduct2(const short* const w0, const short* const w1, const size_t n, int& sum1) {
  __m128i s0 = _mm_setzero_si128();
  __m128i s1 = _mm_setzero_si128();
  for (size_t i = 0; i < n; i += 8) {
    const __m128i t = *(__m128i*) & tx[i];
    __m128i tmp0 = _mm_madd_epi16(t, *(__m128i*) & w0[i]);
    __m128i tmp1 = _mm_madd_epi16(t, *(__m128i*) & w1[i]);
    s0 = _mm_add_epi32(s0, _mm_srai_epi32(tmp0, 8));
    s1 = _mm_add_epi32(s1, _mm_srai_epi32(tmp1, 8));
  }
  s0 = _mm_add_epi32(s0, _mm_srli_si128(s0, 8));
  s0 = _mm_add_epi32(s0, _mm_srli_si128(s0, 4));
  s1 = _mm_add_epi32(s1, _mm_srli_si128(s1, 8));
  s1 = _mm_add_epi32(s1, _mm_srli_si128(s1, 4));
  sum1 = _mm_cvtsi128_si32(s1);
  return _mm_cvtsi128_si32(s0);
}

#if (defined(__GNUC__) || defined(__clang__))
__attribute__((target("sse2")))
#endif
void Mixer_SSE2::train(short* const w, const size_t n, const int e) {
  const __m128i one = _mm_set1_epi16(1);
  const __m128i err = _mm_set1_epi16(short(e));

  for (size_t i = 0; i < n; i += 8) {
    __m128i tmp = _mm_adds_epi16(*(__m128i*) & tx[i], *(__m128i*) & tx[i]);
    tmp = _mm_mulhi_epi16(tmp, err);
    tmp = _mm_adds_epi16(tmp, one);
    tmp = _mm_srai_epi16(tmp, 1);
    tmp = _mm_adds_epi16(tmp, *reinterpret_cast<__m128i*>(&w[i]));
    *reinterpret_cast<__m128i*>(&w[i]) = tmp;
  }
}

#endif // X64_SIMD_AVAILABLE
