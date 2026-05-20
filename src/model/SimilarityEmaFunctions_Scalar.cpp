#include "SimilarityEmaFunctions_Scalar.hpp"
#include "../Utils.hpp"

// Scalar EMA update with ring-buffer index masking (warmup and rare wrap case).
// Fixed-point note: diff (0..128) is shifted left 8 to form the 8.8 fixed-point current sample (0..32768).
// Overflow check: max intermediate = 32768 * 64 = 2'097'152 < 2^21, fits in uint32.
void SimilarityEmaFunctions_Scalar::update_with_wrapping(
  uint16_t* ema_buf, // 8.8 fixed-point: value = stored / 256.0
  const uint8_t* history_buffer_ptr,
  const size_t buf_base,
  const size_t buffer_mask,
  const size_t count,
  const size_t ema_start,
  const uint8_t c1,
  const uint32_t ema_alpha,
  uint32_t* match_index,
  uint32_t* match_score) {

  match_index[0] = 0; // index of the period with the lowest EMA score
  match_index[1] = 0; // index of the period with the second-lowest EMA score
  match_score[0] = UINT16_MAX;
  match_score[1] = UINT16_MAX;

  const uint32_t ema_one_minus_alpha = 64 - ema_alpha; // (1 - alpha) in fixed-point, denominator 64

  for (size_t i = 0; i < count; i++) {

    uint8_t pred = history_buffer_ptr[(buf_base + i) & buffer_mask];
    uint32_t diff = rabs(pred, c1); // 0..128
    uint32_t current = diff << 8;               // 8.8 fixed-point, 0..32768
    size_t ema_idx = ema_start + i;
    uint32_t r = ema_buf[ema_idx];
    r = ((r * ema_one_minus_alpha) + (current * ema_alpha)) >> 6; // divide by 64 to renormalize
    ema_buf[ema_idx] = static_cast<uint16_t>(r);

    // Note: iterating longest-to-shortest means on equal scores, longer periods win
    //       this way we'll start at the beginning of a sequence, not at the end
    if (r < match_score[1]) {
      if (r < match_score[0]) {
        match_index[1] = match_index[0];
        match_index[0] = ema_idx;
        match_score[1] = match_score[0];
        match_score[0] = r;
      }
      else {
        match_index[1] = ema_idx;
        match_score[1] = r;
      }
    }
  }
}


// Scalar EMA update for two buffers over a contiguous block (no ring-buffer masking).
// Processes slow and fast EMA buffers in a single pass for cache efficiency.
// See update_with_wrapping for fixed-point layout and overflow notes.
void SimilarityEmaFunctions_Scalar::update_and_find(
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
)
{

  match_index1[0] = 0; // index of the period with the lowest EMA score
  match_index1[1] = 0; // index of the period with the second-lowest EMA score
  match_score1[0] = UINT16_MAX;
  match_score1[1] = UINT16_MAX;

  match_index2[0] = 0; // index of the period with the lowest EMA score
  match_index2[1] = 0; // index of the period with the second-lowest EMA score
  match_score2[0] = UINT16_MAX;
  match_score2[1] = UINT16_MAX;

  const uint32_t ema_one_minus_alpha1 = 64 - ema_alpha1; // (1 - alpha1) in fixed-point, denominator 64
  const uint32_t ema_one_minus_alpha2 = 64 - ema_alpha2; // (1 - alpha2) in fixed-point, denominator 64

  for (size_t i = 0; i < count; i++) {
    uint8_t pred = history_buffer_ptr[i];
    uint32_t diff = rabs(pred, c1); // 0..128
    uint32_t current = diff << 8;               // 8.8 fixed-point, 0..32768

    uint32_t r1 = ema_buf1[i];
    r1 = ((r1 * ema_one_minus_alpha1) + (current * ema_alpha1)) >> 6;
    ema_buf1[i] = static_cast<uint16_t>(r1);

    uint32_t r2 = ema_buf2[i];
    r2 = ((r2 * ema_one_minus_alpha2) + (current * ema_alpha2)) >> 6;
    ema_buf2[i] = static_cast<uint16_t>(r2);

    // Note: iterating longest-to-shortest means on equal scores, longer periods win
    //       this way we'll start at the beginning of a sequence, not at the end

    if (r1 < match_score1[1]) {
      if (r1 < match_score1[0]) {
        match_index1[1] = match_index1[0];
        match_score1[1] = match_score1[0];
        match_index1[0] = i;
        match_score1[0] = r1;
      }
      else {
        match_index1[1] = i;
        match_score1[1] = r1;
      }
    }

    if (r2 < match_score2[1]) {
      if (r2 < match_score2[0]) {
        match_index2[1] = match_index2[0];
        match_score2[1] = match_score2[0];
        match_index2[0] = i;
        match_score2[0] = r2;
      }
      else {
        match_index2[1] = i;
        match_score2[1] = r2;
      }
    }
  }
}

