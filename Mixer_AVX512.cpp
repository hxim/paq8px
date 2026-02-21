#include "Mixer_AVX512.hpp"

#ifdef X64_SIMD_AVAILABLE

static constexpr int SIMD_WIDTH_AVX512 = 64 / sizeof(short); // 32 shorts per 512-bit lane

Mixer_AVX512::Mixer_AVX512(const Shared* const sh, const int n, const int m, const int s, const int promoted)
  : Mixer(sh, n, m, s, SIMD_WIDTH_AVX512) {
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
int Mixer_AVX512::dotProduct2(const short* const w0, const short* const w1, const size_t n, int& sum1) {
  __m512i s0 = _mm512_setzero_si512();
  __m512i s1 = _mm512_setzero_si512();

  for (size_t i = 0; i < n; i += 32) {
    const __m512i t = *(__m512i*)&tx[i];
    __m512i tmp0 = _mm512_madd_epi16(t, *(__m512i*)&w0[i]);
    __m512i tmp1 = _mm512_madd_epi16(t, *(__m512i*)&w1[i]);
    s0 = _mm512_add_epi32(s0, _mm512_srai_epi32(tmp0, 8));
    s1 = _mm512_add_epi32(s1, _mm512_srai_epi32(tmp1, 8));
  }

  __m256i lo0 = _mm512_extracti64x4_epi64(s0, 0);
  __m256i hi0 = _mm512_extracti64x4_epi64(s0, 1);
  __m256i sum0 = _mm256_add_epi32(lo0, hi0);
  __m128i r0 = _mm_add_epi32(_mm256_extractf128_si256(sum0, 0), _mm256_extractf128_si256(sum0, 1));
  r0 = _mm_add_epi32(r0, _mm_srli_si128(r0, 8));
  r0 = _mm_add_epi32(r0, _mm_srli_si128(r0, 4));

  __m256i lo1 = _mm512_extracti64x4_epi64(s1, 0);
  __m256i hi1 = _mm512_extracti64x4_epi64(s1, 1);
  __m256i sum1v = _mm256_add_epi32(lo1, hi1);
  __m128i r1 = _mm_add_epi32(_mm256_extractf128_si256(sum1v, 0), _mm256_extractf128_si256(sum1v, 1));
  r1 = _mm_add_epi32(r1, _mm_srli_si128(r1, 8));
  r1 = _mm_add_epi32(r1, _mm_srli_si128(r1, 4));

  sum1 = _mm_cvtsi128_si32(r1);
  return _mm_cvtsi128_si32(r0);
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
