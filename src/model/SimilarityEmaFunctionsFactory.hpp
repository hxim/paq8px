#pragma once

#include <cstdint>
#include <cstddef>
#include "../Shared.hpp"
#include "SimilarityEmaFunctions_Scalar.hpp"
#ifdef X64_SIMD_AVAILABLE
#include "SimilarityEmaFunctions_SSE41.hpp"
#include "SimilarityEmaFunctions_AVX2.hpp"
#endif

// Function pointer type for the steady-state EMA update path (contiguous block, no ring-buffer wrap).
// Updates ema_buf1 and ema_buf2 in-place (8.8 fixed-point uint16_t values),
// and writes the top-2 lowest-score indices and scores into match_index/match_score.
using SimilarityEmaUpdateFunction = void(*)(
  uint16_t* ema_buf1,       // 8.8 fixed-point EMA buffer for slow model; length = count
  uint16_t* ema_buf2,       // 8.8 fixed-point EMA buffer for fast model; length = count
  const uint8_t* history_buffer_ptr, // contiguous history block, length = count (no wrap)
  const size_t count,       // number of distances/periods to process; multiple of 16 (AVX2 stride)
  const uint8_t c1,         // incoming byte to be used for calculating the distances (basis of the update)
  const uint32_t ema_alpha1,// EMA alpha for slow model (fixed-point, denominator 64)
  const uint32_t ema_alpha2,// EMA alpha for fast model (fixed-point, denominator 64)
  uint32_t* match_index1,   // [out] top-2 ema_buf1 indices (best, second-best)
  uint32_t* match_index2,   // [out] top-2 ema_buf2 indices (best, second-best)
  uint32_t* match_score1,   // [out] top-2 ema_buf1 scores  (best, second-best)
  uint32_t* match_score2);  // [out] top-2 ema_buf2 scores  (best, second-best)

/**
 * Factory for selecting the best available EMA update function at runtime.
 */

class SimilarityEmaFunctionsFactory
{
public:
  static SimilarityEmaUpdateFunction getEmaUpdateFunction(const Shared* shared);
};
