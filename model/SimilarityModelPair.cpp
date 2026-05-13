#include "SimilarityModelPair.hpp"

SimilarityModelPair::SimilarityModelPair(Shared* const sh, const uint64_t mem): shared(sh) {

  MAX_MATCH_DISTANCE = MAX_MATCH_DISTANCE_TABLE[shared->level - 1]; // common for slow and fast models

  EMA_ALPHA_SLOW = 1; // alpha = 1/64 (1-alpha = 63/64)
  EMA_ALPHA_FAST = 7; // alpha = 7/64 (1-alpha = 57/64)

  // Record length is capped relative to MAX_MATCH_DISTANCE so that
  // the 4x-period lookback used in update() always stays within the buffer.
  const size_t max_record_length_slow = min(1024llu, MAX_MATCH_DISTANCE / 4);
  // Fast model uses a shorter record window
  const size_t max_record_length_fast = min(512llu, max_record_length_slow / 2);

  similarityModel_slow = std::make_unique<SimilarityModel>(sh, mem / 16, MAX_MATCH_DISTANCE, max_record_length_slow);
  similarityModel_fast = std::make_unique<SimilarityModel>(sh, mem / 32, MAX_MATCH_DISTANCE, max_record_length_fast);

  // Runtime SIMD dispatch: AVX2 > SSE4.1 > scalar
  emaUpdateFunction = SimilarityEmaFunctionsFactory::getEmaUpdateFunction(shared);
}

void SimilarityModelPair::update() {

  // from slow model
  const auto ema_buf1 = &similarityModel_slow->ema_buf[0];
  const auto match_index1 = similarityModel_slow->match_index;
  const auto match_score1 = similarityModel_slow->match_score;

  // from fast model
  const auto ema_buf2 = &similarityModel_fast->ema_buf[0];
  const auto match_index2 = similarityModel_fast->match_index;
  const auto match_score2 = similarityModel_fast->match_score;

  uint8_t* history_buffer_ptr;
  size_t buffer_mask;
  size_t head_pos;
  shared->buf.leakInternals(history_buffer_ptr, buffer_mask, head_pos);
  head_pos--; // point to last known byte (c1)
  INJECT_SHARED_c1
  INJECT_SHARED_buf
  const size_t byte_count = min(warmup, MAX_MATCH_DISTANCE); // number of valid history bytes available
  warmup++; // total bytes processed; capped to MAX_MATCH_DISTANCE

  // EMA update loop
  const size_t buf_base = head_pos - byte_count;           // logical start of the history window
  const size_t ema_base = MAX_MATCH_DISTANCE - byte_count; // corresponding start offset in ema_buf[] during warmup
  
  // EMA update: two paths depending on warmup state and ring-buffer wrap
  //   1. scalar: initial warmup (byte_count < MAX_MATCH_DISTANCE) OR very rare ring-buffer wrap
  //   2. vectorized: steady-state with no ring-buffer wrap
  const size_t buf_start_physical = buf_base & buffer_mask; // physical start index in ring buffer
  if (byte_count == MAX_MATCH_DISTANCE && (buf_start_physical + MAX_MATCH_DISTANCE) <= (buffer_mask + 1)) {
    // Path 2: steady-state, no wrap — contiguous block, vectorized path
    emaUpdateFunction(
      ema_buf1,
      ema_buf2,
      history_buffer_ptr + buf_start_physical,
      MAX_MATCH_DISTANCE,
      c1,
      EMA_ALPHA_SLOW,
      EMA_ALPHA_FAST,
      match_index1,
      match_index2,
      match_score1,
      match_score2);
  }
  else {
    // Path 1: warmup or rare ring-buffer wrap — scalar with masking
    SimilarityEmaFunctions_Scalar::update_with_wrapping(
      ema_buf1,
      history_buffer_ptr,
      buf_base,
      buffer_mask,
      byte_count,
      ema_base,
      c1,
      EMA_ALPHA_SLOW,
      match_index1,
      match_score1);
    SimilarityEmaFunctions_Scalar::update_with_wrapping(
      ema_buf2,
      history_buffer_ptr,
      buf_base,
      buffer_mask,
      byte_count,
      ema_base,
      c1,
      EMA_ALPHA_FAST,
      match_index2,
      match_score2);
  }

}

void SimilarityModelPair::mix(Mixer& m) {
  INJECT_SHARED_bpos
  if (bpos == 0) {
    update();
    similarityModel_slow->update(warmup);
    similarityModel_fast->update(warmup);
  }
  similarityModel_slow->mix(m);
  similarityModel_fast->mix(m);
}
