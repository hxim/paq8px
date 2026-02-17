#include "Mixer_AVX512.hpp"

#ifdef X64_SIMD_AVAILABLE

static constexpr int SIMD_WIDTH_AVX512 = 64 / sizeof(short); // 32 shorts per 512-bit lane

Mixer_AVX512::Mixer_AVX512(const Shared* const sh, const int n, const int m, const int s, const int promoted)
  : Mixer(sh, n, m, s, SIMD_WIDTH_AVX512) {
  initSecondLayer(promoted);
}

void Mixer_AVX512::initSecondLayer(const int promoted) {
  if (s > 1) {
    mp = new Mixer_AVX512(shared, s + promoted, 1, 1, 0);
  }
}

#if (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx512bw")))
#endif
int Mixer_AVX512::dotProduct(const short* const w, const size_t n) {
  __m512i sum = _mm512_setzero_si512();

  for (size_t i = 0; i < n; i += 32) {
    __m512i tmp = _mm512_madd_epi16(*(__m512i*)&tx[i], *(__m512i*)&w[i]);
    tmp = _mm512_srai_epi32(tmp, 8);
    sum = _mm512_add_epi32(sum, tmp);
  }

  __m256i lo = _mm512_extracti64x4_epi64(sum, 0);
  __m256i hi = _mm512_extracti64x4_epi64(sum, 1);

  __m256i newSum1 = _mm256_add_epi32(lo, hi);
  __m128i newSum2 = _mm_add_epi32(_mm256_extractf128_si256(newSum1, 0), _mm256_extractf128_si256(newSum1, 1));
  newSum2 = _mm_add_epi32(newSum2, _mm_srli_si128(newSum2, 8));
  newSum2 = _mm_add_epi32(newSum2, _mm_srli_si128(newSum2, 4));
  return _mm_cvtsi128_si32(newSum2);
}

#if (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx512bw")))
#endif
void Mixer_AVX512::train(short* const w, const size_t n, const int e) {
  const __m512i one = _mm512_set1_epi16(1);
  const __m512i err = _mm512_set1_epi16(short(e));

  for (size_t i = 0; i < n; i += 32) {
    __m512i tmp = _mm512_adds_epi16(*(__m512i*)&tx[i], *(__m512i*)&tx[i]);
    tmp = _mm512_mulhi_epi16(tmp, err);
    tmp = _mm512_adds_epi16(tmp, one);
    tmp = _mm512_srai_epi16(tmp, 1);
    tmp = _mm512_adds_epi16(tmp, *reinterpret_cast<__m512i*>(&w[i]));
    *reinterpret_cast<__m512i*>(&w[i]) = tmp;
  }
}

#endif // X64_SIMD_AVAILABLE
