#pragma once

#include "../ResidualMap.hpp"
#include "../ContextMap2.hpp"
#include "SimilarityEmaFunctionsFactory.hpp"

/**
 * Combined similarity-based match model and record model.
 * Matches and record lengths are detected based on byte similarity (EMA of |x1-x2|) rather than equality.
 *
 * Match model:
 *   - best for content with undetected audio, image, or other 1D numeric streams.
 *
 * Record model:
 *   - for content with shorter-period record structures for which byte equality based detection struggles to find the record length
 *     such as undetected 8/24/32-bit raw image data or other 2D numeric streams.
 *
 * Maintains two EMA buffers (slow + fast alpha) via SimilarityModelPair.
 * Best and second-best match periods and record periods are tracked by index into ema_buf[].
 */

class SimilarityModel
{
private:
  static constexpr int nRM1 = 16;  
  static constexpr int nRM2 = 2;
  static constexpr int nCM = 8;

  Shared* const shared;
  ResidualMap mapR1;
  ResidualMap mapR2;
  ContextMap2 cm;
 
  uint32_t record_len = 1;    // best candidate period length (in bytes)
  uint32_t record_score = 0;  // sum of EMA scores at 1x..4x record_len (lower = better)
  uint16_t mctx1 = 0;         // mixer context
  uint16_t mctx2 = 0;         // mixer context

public:
  static constexpr int MIXERINPUTS =
    (nRM1 + nRM2) * ResidualMap::MIXERINPUTS +
    nCM * (ContextMap2::MIXERINPUTS + ContextMap2::MIXERINPUTS_RUN_STATS + ContextMap2::MIXERINPUTS_BYTE_HISTORY); // 92

  static constexpr int MIXERCONTEXTS = (3 * 32) + (3 * 2); // 102
  static constexpr int MIXERCONTEXTSETS = 2;

  SimilarityModel(Shared* const sh, const uint64_t size, size_t max_match_distance, size_t max_record_length);
  void reset();

  void update(uint32_t warmup);
  void mix(Mixer& m);

  // exposed public members for SimilarityModelPair

  const size_t MAX_MATCH_DISTANCE; // typically 8192
  const size_t MAX_RECORD_LENGTH;  // typically 1024 or 512

  Array<uint16_t, 32> ema_buf;        // 8.8 fixed-point EMA of |x-pred| for each distance/period; size = MAX_MATCH_DISTANCE
  uint32_t match_index[2] = { 0, 0 }; // ema_buf[] indices of the best and second-best match period
  uint32_t match_score[2] = { 0, 0 }; // EMA scores (lower = better) for match_index[0] and [1]
};
