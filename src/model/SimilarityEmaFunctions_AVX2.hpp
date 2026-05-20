#pragma once

#include "../Utils.hpp"
#ifdef X64_SIMD_AVAILABLE

#include <cstdint>
#include <cstddef>

/**
 * AVX2 EMA update implementation.
 * See the scalar implementation for reference.
 */

class SimilarityEmaFunctions_AVX2
{
public:
  // AVX2 EMA update + top-2 min-index search (16 values per iteration).
  // Preconditions:
  //   - ema_buf1, ema_buf2: 32-byte aligned, length >= count
  //   - history_buffer_ptr[0..count): contiguous block, no ring-buffer wrap
  //   - count: multiple of 16
  static void update_and_find(
    uint16_t* ema_buf1,           // [in/out] 8.8 fixed-point EMA buffer, slow model
    uint16_t* ema_buf2,           // [in/out] 8.8 fixed-point EMA buffer, fast model
    const uint8_t* history_buffer_ptr,    // contiguous portion of the ring buffer with history data, length = count
    const size_t count,           // number of distances; multiple of 16
    const uint8_t c1,             // incoming byte to be used for calculating the distances (basis of the update)
    const uint32_t ema_alpha1,    // EMA alpha, slow model (fixed-point, denominator 64)
    const uint32_t ema_alpha2,    // EMA alpha, fast model (fixed-point, denominator 64)
    uint32_t* match_index1,       // [out] top-2 ema_buf1 indices (best, second-best)
    uint32_t* match_index2,       // [out] top-2 ema_buf2 indices (best, second-best)
    uint32_t* match_score1,       // [out] top-2 ema_buf1 scores  (best, second-best)
    uint32_t* match_score2);      // [out] top-2 ema_buf2 scores  (best, second-best)

};

#endif
