#pragma once

#include <cstdint>
#include <cstddef>

/**
 * Scalar (non-SIMD) EMA update implementations.
 *
 * Provides two update paths:
 *   update_with_wrapping  - warmup and ring-buffer wrap cases; processes a single EMA buffer
 *                           with index masking for non-contiguous access.
 *   update_and_find       - steady-state path; processes two EMA buffers (slow and fast) over a contiguous
 *                           block (no ring-buffer wrap).
 *
 * Fixed-point layout: 8.8 (uint16_t), denominator 256.
 * Denominator for the EMA weights: 64 (alpha and one_minus_alpha sum to 64).
 */

class SimilarityEmaFunctions_Scalar
{
public:
  // EMA update with ring-buffer index masking.
  // Used during warmup (count < MAX_MATCH_DISTANCE) and on the rare ring-buffer wrap.
  // Processes a single EMA buffer; for the steady-state two-buffer path use update_and_find.
  static void update_with_wrapping(
    uint16_t* ema_buf,          // [in/out] 8.8 fixed-point EMA values; length >= ema_start + count
    const uint8_t* history_buffer_ptr,  // ring buffer base pointer to history data
    const size_t buf_base,      // logical start index into the ring buffer
    const size_t buffer_mask,   // ring buffer size minus 1 (for index masking)
    const size_t count,         // number of distances to process
    const size_t ema_start,     // starting index into ema_buf[] (= MAX_MATCH_DISTANCE - count)
    const uint8_t c1,           // incoming byte to be used for calculating the distances (basis of the update)
    const uint32_t ema_alpha,   // EMA alpha (fixed-point, denominator 64)
    uint32_t* match_index,      // [out] top-2 ema_buf indices (best, second-best)
    uint32_t* match_score);     // [out] top-2 EMA scores      (best, second-best)

  // EMA update for two buffers over a contiguous block (no ring-buffer wrap).
  // Steady-state path; satisfies the SimilarityEmaUpdateFunction signature.
  static void update_and_find(
    uint16_t* ema_buf1,           // [in/out] 8.8 fixed-point EMA buffer, slow model; length = count
    uint16_t* ema_buf2,           // [in/out] 8.8 fixed-point EMA buffer, fast model; length = count
    const uint8_t* history_buffer_ptr,    // contiguous portion of the ring buffer with history data, length = count
    const size_t count,           // number of distances to process
    const uint8_t c1,             // incoming byte to be used for calculating the distances (basis of the update)
    const uint32_t ema_alpha1,    // EMA alpha for slow model (fixed-point, denominator 64)
    const uint32_t ema_alpha2,    // EMA alpha for fast model (fixed-point, denominator 64)
    uint32_t* match_index1,       // [out] top-2 ema_buf1 indices (best, second-best)
    uint32_t* match_index2,       // [out] top-2 ema_buf2 indices (best, second-best)
    uint32_t* match_score1,       // [out] top-2 ema_buf1 scores  (best, second-best)
    uint32_t* match_score2);      // [out] top-2 ema_buf2 scores  (best, second-best)
};
