#pragma once

#include <memory>
#include "SimilarityEmaFunctionsFactory.hpp"
#include "SimilarityModel.hpp"

/**
 * Owns a pair of SimilarityModel instances (slow and fast)
 * and dispatches EMA buffer updates through the best available SIMD path
 * (AVX2, SSE4.1, or scalar) selected at construction time.
 *
 * The slow model (alpha=1) tracks stable long-range structure;
 * the fast model (alpha=7) adapts quickly to changes.
 * Both share the same MAX_MATCH_DISTANCE, derived from the compression level.
 */

class SimilarityModelPair
{
  Shared* const shared;
  std::unique_ptr<SimilarityModel> similarityModel_slow; // alpha=1: slow adaptation, stable predictions
  std::unique_ptr<SimilarityModel> similarityModel_fast; // alpha=7: fast adaptation, responsive to change
  SimilarityEmaUpdateFunction emaUpdateFunction;         // best available SIMD path, selected at construction

  size_t MAX_MATCH_DISTANCE;                             // shared search window size in bytes for both models (typical: 8192)

  // EMA smoothing factors: weight on new sample: alpha/64, weight on history: (64-alpha)/64
  uint32_t EMA_ALPHA_SLOW;                               // for the slow model (typical: 1), fixed-point with denominator of 64
  uint32_t EMA_ALPHA_FAST;                               // for the fast model (typical: 5-9), fixed-point with denominator of 64
  size_t warmup = 0;                                     // bytes processed so far; caps EMA loop range

  // Maximum match/record search distance (in bytes) indexed by compression level (1–12).
  // values must be multiples of 16 (AVX2 stride)
  static constexpr uint32_t MAX_MATCH_DISTANCE_TABLE[12] = {
   // max match distance        // level
     768,   1024,  1536,  2048, // 1,  2,  3,  4
    3072,   4096,  6144,  8192, // 5,  6,  7,  8
   12288,  16384, 24576, 32768  // 9, 10, 11, 12
  };

public:
  SimilarityModelPair(Shared* const sh, const uint64_t mem);
  void update();
  void mix(Mixer& m);
};
