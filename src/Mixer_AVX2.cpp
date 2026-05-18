#include "Mixer_AVX2.hpp"

#ifdef X64_SIMD_AVAILABLE

static constexpr int SIMD_WIDTH_AVX2 = 32 / sizeof(short); // 16 shorts per 256-bit lane

Mixer_AVX2::Mixer_AVX2(const Shared* const sh, const int n, const int m, const int s, const int promoted)
  : Mixer(sh, n, m, s, SIMD_WIDTH_AVX2) {
  if (s > 1) {
    mp = new Mixer_AVX2(shared, s + promoted, 1, 1, 0);
  }
}

#if (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
#endif
int Mixer_AVX2::dotProduct(const short* const w, const size_t n) {
  __m256i sum = _mm256_setzero_si256();

  for (size_t i = 0; i < n; i += 16) {
    __m256i tmp = _mm256_madd_epi16(*(__m256i*) & tx[i], *(__m256i*) & w[i]);
    tmp = _mm256_srai_epi32(tmp, 8);
    sum = _mm256_add_epi32(sum, tmp);
  }

  __m128i lo = _mm256_extractf128_si256(sum, 0);
  __m128i hi = _mm256_extractf128_si256(sum, 1);

  __m128i newSum = _mm_hadd_epi32(lo, hi);
  newSum = _mm_add_epi32(newSum, _mm_srli_si128(newSum, 8));
  newSum = _mm_add_epi32(newSum, _mm_srli_si128(newSum, 4));
  return _mm_cvtsi128_si32(newSum);
}

#if (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
#endif
int Mixer_AVX2::dotProduct2(const short* const w0, const short* const w1, const size_t n, int& sum1) {
  __m256i s0 = _mm256_setzero_si256();
  __m256i s1 = _mm256_setzero_si256();

  for (size_t i = 0; i < n; i += 16) {
    const __m256i t = *(__m256i*) & tx[i];
    __m256i tmp0 = _mm256_madd_epi16(t, *(__m256i*) & w0[i]);
    __m256i tmp1 = _mm256_madd_epi16(t, *(__m256i*) & w1[i]);
    s0 = _mm256_add_epi32(s0, _mm256_srai_epi32(tmp0, 8));
    s1 = _mm256_add_epi32(s1, _mm256_srai_epi32(tmp1, 8));
  }

  __m128i lo0 = _mm256_extractf128_si256(s0, 0);
  __m128i hi0 = _mm256_extractf128_si256(s0, 1);
  __m128i r0 = _mm_hadd_epi32(lo0, hi0);
  r0 = _mm_add_epi32(r0, _mm_srli_si128(r0, 8));
  r0 = _mm_add_epi32(r0, _mm_srli_si128(r0, 4));

  __m128i lo1 = _mm256_extractf128_si256(s1, 0);
  __m128i hi1 = _mm256_extractf128_si256(s1, 1);
  __m128i r1 = _mm_hadd_epi32(lo1, hi1);
  r1 = _mm_add_epi32(r1, _mm_srli_si128(r1, 8));
  r1 = _mm_add_epi32(r1, _mm_srli_si128(r1, 4));

  sum1 = _mm_cvtsi128_si32(r1);
  return _mm_cvtsi128_si32(r0);
}

#if (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2")))
#endif
void Mixer_AVX2::train(short* const w, const size_t n, const int e) {
  const __m256i one = _mm256_set1_epi16(1);
  const __m256i err = _mm256_set1_epi16(short(e));

  for (size_t i = 0; i < n; i += 16) {
    __m256i tmp = _mm256_adds_epi16(*(__m256i*) & tx[i], *(__m256i*) & tx[i]);
    tmp = _mm256_mulhi_epi16(tmp, err);
    tmp = _mm256_adds_epi16(tmp, one);
    tmp = _mm256_srai_epi16(tmp, 1);
    tmp = _mm256_adds_epi16(tmp, *reinterpret_cast<__m256i*>(&w[i]));
    *reinterpret_cast<__m256i*>(&w[i]) = tmp;
  }
}

#endif // X64_SIMD_AVAILABLE
