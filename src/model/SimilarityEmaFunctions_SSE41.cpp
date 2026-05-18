#include "SimilarityEmaFunctions_SSE41.hpp"

#ifdef X64_SIMD_AVAILABLE

#include <smmintrin.h> // SSE4.1

// Process 8 uint16 EMA values per iteration using SSE4.1.
//
// Fixed-point layout: 8.8 (8 integer bits, 8 fractional bits)
//   current      = diff << 8             → max 128 << 8 = 32768       (fits in uint16)
//   blend        = (r * one_minus_alpha + current * alpha) >> 6
//   intermediate = r * one_minus_alpha   → max 65535 * 64 = 4,194,240 (needs 32-bit lanes)
//   result       = blend >> 6            → max 65535                  (fits back in uint16)
//
// Strategy: widen 8 x uint16 → two groups of 4 x uint32, compute blend in 32-bit lanes,
// pack results back to uint16 with _mm_packus_epi32 (requires SSE4.1).

#if (defined(__GNUC__) || defined(__clang__))
__attribute__((target("sse4.1")))
#endif
void SimilarityEmaFunctions_SSE41::update_and_find(
  uint16_t* ema_buf1,
  uint16_t* ema_buf2,
  const uint8_t* history_buffer_ptr,
  const size_t count,
  const uint8_t c1,
  const uint32_t ema_alpha1,
  const uint32_t ema_alpha2,
  uint32_t* match_index1,
  uint32_t* match_index2,
  uint32_t* match_score1,
  uint32_t* match_score2
) {

  match_index1[0] = 0;
  match_index1[1] = 0;
  match_score1[0] = UINT16_MAX;
  match_score1[1] = UINT16_MAX;

  match_index2[0] = 0;
  match_index2[1] = 0;
  match_score2[0] = UINT16_MAX;
  match_score2[1] = UINT16_MAX;

  const uint32_t ema_one_minus_alpha1 = 64 - ema_alpha1;
  const uint32_t ema_one_minus_alpha2 = 64 - ema_alpha2;

  // rabs is computed as abs_epi8(sub_epi8(pred, c1)), then zero-extended to 32-bit.
  // This mirrors: abs(int8_t(x1 - x2));
  const __m128i vc1_bytes = _mm_set1_epi8((int8_t)c1);

  const __m128i va1 = _mm_set1_epi32(ema_alpha1);
  const __m128i vb1 = _mm_set1_epi32(ema_one_minus_alpha1);
  const __m128i va2 = _mm_set1_epi32(ema_alpha2);
  const __m128i vb2 = _mm_set1_epi32(ema_one_minus_alpha2);

  // SSE has no unsigned 16-bit compare; simulate u16 < by XORing both sides with 0x8000
  // to shift the unsigned range into the signed range before _mm_cmplt_epi16.
  const __m128i v_sign_flip = _mm_set1_epi16((int16_t)0x8000);

  // Current second-best thresholds for fast reject.
  __m128i vbest2_r1 = _mm_set1_epi16((int16_t)match_score1[1]);
  __m128i vbest2_r2 = _mm_set1_epi16((int16_t)match_score2[1]);

  assert(count % 8 == 0);
  for (size_t i = 0; i < count; i += 8) {
    // --- Load predictors and compute rabs ---
    // sub_epi8 wraps into int8 range naturally; abs_epi8 gives absolute value (0..128).
    // _mm_abs_epi8 on 0x80 (-128) yields 128 unsigned — correct for our 0..128 range.
    __m128i pred_bytes = _mm_loadl_epi64((const __m128i*)(history_buffer_ptr + i));
    __m128i abs_bytes = _mm_abs_epi8(_mm_sub_epi8(pred_bytes, vc1_bytes));

    __m128i abs_lo = _mm_cvtepu8_epi32(abs_bytes);
    __m128i abs_hi = _mm_cvtepu8_epi32(_mm_srli_si128(abs_bytes, 4));

    __m128i cur_lo = _mm_slli_epi32(abs_lo, 8); // 8.8 fixed-point, 0..32768
    __m128i cur_hi = _mm_slli_epi32(abs_hi, 8);

    // --- Load + Update EMA1 ---
    __m128i ema16_1 = _mm_load_si128((const __m128i*)(ema_buf1 + i));
    __m128i ema_lo1 = _mm_cvtepu16_epi32(ema16_1);
    __m128i ema_hi1 = _mm_cvtepu16_epi32(_mm_srli_si128(ema16_1, 8));

    __m128i tmp_lo1 = _mm_add_epi32(_mm_mullo_epi32(ema_lo1, vb1),
      _mm_mullo_epi32(cur_lo, va1));
    __m128i tmp_hi1 = _mm_add_epi32(_mm_mullo_epi32(ema_hi1, vb1),
      _mm_mullo_epi32(cur_hi, va1));

    __m128i res_lo1 = _mm_srli_epi32(tmp_lo1, 6);
    __m128i res_hi1 = _mm_srli_epi32(tmp_hi1, 6);
    __m128i new_ema1 = _mm_packus_epi32(res_lo1, res_hi1);

    // --- Load + Update EMA2 ---
    __m128i ema16_2 = _mm_load_si128((const __m128i*)(ema_buf2 + i));
    __m128i ema_lo2 = _mm_cvtepu16_epi32(ema16_2);
    __m128i ema_hi2 = _mm_cvtepu16_epi32(_mm_srli_si128(ema16_2, 8));

    __m128i tmp_lo2 = _mm_add_epi32(_mm_mullo_epi32(ema_lo2, vb2),
      _mm_mullo_epi32(cur_lo, va2));
    __m128i tmp_hi2 = _mm_add_epi32(_mm_mullo_epi32(ema_hi2, vb2),
      _mm_mullo_epi32(cur_hi, va2));

    __m128i res_lo2 = _mm_srli_epi32(tmp_lo2, 6);
    __m128i res_hi2 = _mm_srli_epi32(tmp_hi2, 6);
    __m128i new_ema2 = _mm_packus_epi32(res_lo2, res_hi2);

    // --- Store updated EMA values ---
    _mm_store_si128((__m128i*)(ema_buf1 + i), new_ema1);
    _mm_store_si128((__m128i*)(ema_buf2 + i), new_ema2);

    // --- Fast reject: skip scalar top-2 update if no lane beat the current second-best ---
    __m128i cmp1 = _mm_cmpgt_epi16(_mm_sub_epi16(vbest2_r1, v_sign_flip),
      _mm_sub_epi16(new_ema1, v_sign_flip));

    __m128i cmp2 = _mm_cmpgt_epi16(_mm_sub_epi16(vbest2_r2, v_sign_flip),
      _mm_sub_epi16(new_ema2, v_sign_flip));

    if (_mm_testz_si128(cmp1, cmp1) && _mm_testz_si128(cmp2, cmp2))
      continue;

    // Fast reject: we reach here only if at least one lane beat the second-best
    // which is roughly 30/8192 of the times

    // --- Scalar top-2 update ---
    // Iterates longest-to-shortest; on equal scores, shorter periods win ties.
    for (int k = 0; k < 8; k++) {
      uint32_t idx = (uint32_t)(i + k);
      uint32_t r1 = ema_buf1[idx];
      uint32_t r2 = ema_buf2[idx];

      if (r1 < match_score1[1]) {
        if (r1 < match_score1[0]) {
          match_index1[1] = match_index1[0];
          match_score1[1] = match_score1[0];
          match_index1[0] = idx;
          match_score1[0] = r1;
        }
        else {
          match_index1[1] = idx;
          match_score1[1] = r1;
        }
      }

      if (r2 < match_score2[1]) {
        if (r2 < match_score2[0]) {
          match_index2[1] = match_index2[0];
          match_score2[1] = match_score2[0];
          match_index2[0] = idx;
          match_score2[0] = r2;
        }
        else {
          match_index2[1] = idx;
          match_score2[1] = r2;
        }
      }
    }

    // Refresh thresholds
    vbest2_r1 = _mm_set1_epi16((int16_t)match_score1[1]);
    vbest2_r2 = _mm_set1_epi16((int16_t)match_score2[1]);
  }
}

#endif
