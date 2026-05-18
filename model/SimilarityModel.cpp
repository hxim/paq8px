#include "SimilarityModel.hpp"
#include "../Hash.hpp"

SimilarityModel::SimilarityModel(Shared* const sh, const uint64_t size, size_t max_match_distance, size_t max_record_length) :
  shared(sh),
  /* ResidualMap: numContexts, histogramsPerContext, scale=64 */
  mapR1{ sh, nRM1, 32, 64 },
  mapR2{ sh, nRM2, 64 * 16, 64 },
  /* ContextMap2: size, numContexts, scale */
  cm(sh, size, nCM, 64),
  MAX_MATCH_DISTANCE(max_match_distance),
  MAX_RECORD_LENGTH(max_record_length),
  ema_buf(MAX_MATCH_DISTANCE)
{
  assert(MAX_MATCH_DISTANCE % 16 == 0);   // AVX2 loop stride
  assert(MAX_MATCH_DISTANCE >= MAX_RECORD_LENGTH * 4);
  assert(sh->buf.size() > MAX_MATCH_DISTANCE);

  for (int i = 0; i < MAX_MATCH_DISTANCE; i++) {
    ema_buf[i] = 64 << 8; // 64 = expected rabs value for random x1,x2 values (in fixed point)
  }
}

void SimilarityModel::update(uint32_t warmup) {

  // Find best record period
  const size_t count1 = min(warmup, MAX_RECORD_LENGTH);
  record_score = UINT32_MAX;

  // if two periods have identical ema values, the one with the shorter period wins
  //   why: - shorter periods are usually the 'correct' ones when one detected period length is a multiple of the other
  //        - shorter periods are closer to the current byte position which is usually a somewhat stronger context
  for (size_t i = 0; i < count1; i++) {
    const size_t rec_len = count1 - i; // candidate period length, decreasing as i increases
    uint32_t r = (
      ema_buf[MAX_MATCH_DISTANCE - rec_len * 1] +
      ema_buf[MAX_MATCH_DISTANCE - rec_len * 2] +
      ema_buf[MAX_MATCH_DISTANCE - rec_len * 3] +
      ema_buf[MAX_MATCH_DISTANCE - rec_len * 4]
      );
    if (r <= record_score) {
      record_len = rec_len;
      record_score = r;
    }
  }
}

void SimilarityModel::mix(Mixer& m) {
  INJECT_SHARED_bpos
  if (bpos == 0) {
    // predicted byte according to the best/second-best match period
    // period = MAX_MATCH_DISTANCE - idx, buf distance = period
    INJECT_SHARED_buf
    uint8_t expected_byte0 = buf(MAX_MATCH_DISTANCE - match_index[0]); // the expected byte based on the best match
    uint8_t expected_byte1 = buf(MAX_MATCH_DISTANCE - match_index[1]); // the expected byte based on the second best match

    // match model

    uint32_t histogram_id0 = min(match_score[0] >> 8, 31);
    uint32_t histogram_id1 = min(match_score[1] >> 8, 31);
    mapR1.set(expected_byte0, histogram_id0);
    mapR1.set(expected_byte1, histogram_id1);

    uint32_t r = min(rabs(expected_byte0, expected_byte1) >> 1, 63); // 0..63
    uint32_t histogram_id3 = r << 4 | min(match_score[0] >> 9, 15);
    mapR2.set(expected_byte0, histogram_id3);

    const uint8_t W = buf(1);
    const uint8_t WW = buf(2);
    const uint8_t WWWW = buf(4);
    short horizontal_extrapolation0;
    short horizontal_extrapolation1;
    short horizontal_extrapolation2;
    short horizontal_extrapolation3;
    {
      uint32_t dist = MAX_MATCH_DISTANCE - match_index[0]; // period = distance
      const uint8_t N0 = buf(dist);
      const uint8_t NW0 = buf(dist + 1);
      const uint8_t NWW0 = buf(dist + 2);
      const uint8_t NWWWW0 = buf(dist + 4);
      //W+N-NW (horizontal extrapolation)
      mapR1.set(horizontal_extrapolation0 = W + N0 - NW0, min(rabs(W, WW + NW0 - NWW0), 31));
      //WW+NW-NWW (horizontal extrapolation)
      mapR1.set(horizontal_extrapolation1 = WW + N0 - NWW0, min(rabs(WW, WWWW + NWW0 - NWWWW0), 31));
    }

    {
      uint32_t dist = MAX_MATCH_DISTANCE - match_index[1]; // period = distance
      const uint8_t N0 = buf(dist);
      const uint8_t NW0 = buf(dist + 1);
      const uint8_t NWW0 = buf(dist + 2);
      const uint8_t NWWWW0 = buf(dist + 4);
      //W+N-NW (horizontal extrapolation)
      mapR1.set(horizontal_extrapolation2 = W + N0 - NW0, min(rabs(W, WW + NW0 - NWW0), 31));
      //WW+NW-NWW (horizontal extrapolation)
      mapR1.set(horizontal_extrapolation3 = WW + N0 - NWW0, min(rabs(WW, WWWW + NWW0 - NWWWW0), 31));
    }

    // record model

    const uint8_t N = buf(1 * record_len);
    const uint8_t NN = buf(2 * record_len);
    const uint8_t NNN = buf(3 * record_len);
    const uint8_t NNNN = buf(4 * record_len);

    // predict based on the row with the lowest score
    auto best_N = N;
    auto best_NW = buf(record_len + 1);
    auto best_NWW = buf(record_len + 2);
    auto best_NWWWW = buf(record_len + 4);
    auto score_N = ema_buf[MAX_MATCH_DISTANCE - record_len * 1];
    auto score_NN = ema_buf[MAX_MATCH_DISTANCE - record_len * 2];
    auto score_NNN = ema_buf[MAX_MATCH_DISTANCE - record_len * 3];
    auto score_NNNN = ema_buf[MAX_MATCH_DISTANCE - record_len * 4];
    auto best_score = score_N;
    if (score_NN < best_score) {
      best_N = NN;
      best_NW = buf(record_len * 2 + 1);
      best_NWW = buf(record_len * 2 + 2);
      best_NWWWW = buf(record_len * 2 + 4);
      best_score = score_NN;
    }
    if (score_NNN < best_score) {
      best_N = NNN;
      best_NW = buf(record_len * 3 + 1);
      best_NWW = buf(record_len * 3 + 2);
      best_NWWWW = buf(record_len * 3 + 4);
      best_score = score_NNN;
    }
    if (score_NNNN < best_score) {
      best_N = NNNN;
      best_NW = buf(record_len * 4 + 1);
      best_NWW = buf(record_len * 4 + 2);
      best_NWWWW = buf(record_len * 4 + 4);
      best_score = score_NNNN;
    }
    best_score >>= 8;
    score_N >>= 8;
    score_NN >>= 8;
    score_NNN >>= 8;
    score_NNNN >>= 8;

    // measure of consistency of the record-predictions
    uint32_t column_score = (rabs(best_N, N) + rabs(best_N, NN) + rabs(best_N, NNN) + rabs(best_N, NNNN));
    // prediction based on the lowest score among the rows
    mapR2.set(best_N, min(best_score >> 1, 31) << 5 | min(column_score >> 2, 31)); // weak

    // W+N-NW (horizontal extrapolation)
    mapR1.set(W + best_N - best_NW, min(rabs(W, WW + best_NW - best_NWW), 31));
    // WW+NW-NWW (horizontal extrapolation)
    mapR1.set(WW + best_N - best_NWW, min(rabs(WW, WWWW + best_NWW - best_NWWWW), 31));

    // predict based on individual row score
    mapR1.set(N, min(score_N >> 1, 31)); // +1 to map only 0 to 0
    mapR1.set(NN, min(score_NN >> 1, 31));
    mapR1.set(NNN, min(score_NNN >> 1, 31));
    mapR1.set(NNNN, min(score_NNNN >> 1, 31));

    // vertical extrapolations (o2, o3, o4)

    const uint8_t NW = buf(record_len + 1);
    const uint8_t NWW = buf(record_len + 2);
    const uint8_t NNW = buf(2 * record_len + 1);
    short o2_N = 2 * N - NN;
    int o2_N_err_with_W = 2 * NW - NNW;
    mapR1.set(o2_N, min(rabs(W, o2_N_err_with_W), 31));

    const uint8_t NNNW = buf(3 * record_len + 1);
    const uint8_t NNNNW = buf(4 * record_len + 1);
    const uint8_t NNNNN = buf(5 * record_len);

    short o3_N = 3 * N - 3 * NN + NNN;  // N + (N - NN) + ((N - NN) - (NN - NNN))
    int o3_N_err_with_N = 3 * NN - 3 * NNN + NNNN;
    int o3_N_err_with_W = 3 * NW - 3 * NNW + NNNW;
    mapR1.set(o3_N, min(rabs(N, o3_N_err_with_N) + rabs(W, o3_N_err_with_W), 63) >> 1);

    short o4_N = 4 * N - 6 * NN + 4 * NNN - NNNN;
    int o4_N_err_with_N = 4 * NN - 6 * NNN + 4 * NNNN - NNNNN;
    int o4_N_err_with_W = 4 * NW - 6 * NNW + 4 * NNNW - NNNNW;
    mapR1.set(o4_N, min(rabs(N, o4_N_err_with_N) + rabs(W, o4_N_err_with_W), 63) >> 1);

    // average

    uint32_t lowest = min(min(min(N, NN), NNN), NNNN);
    uint32_t highest = max(max(max(N, NN), NNN), NNNN);
    uint32_t span = highest - lowest;
    mapR1.set((N + NN + NNN + NNNN + 2) >> 2, min(span >> 1, 31)); // weak

    // cm contexts

    uint64_t i = 0;
    const uint8_t RH = CM_USE_RUN_STATS | CM_USE_BYTE_HISTORY;
    cm.set(RH, hash(N, NN, NNN, NNNN));
    i++;

    // highly adaptive contexts
    cm.set(RH, hash(i++, expected_byte0)); //strong
    cm.set(RH, hash(i++, expected_byte1)); //strong
    cm.set(RH, hash(i++, horizontal_extrapolation0 + 256)); //strong
    cm.set(RH, hash(i++, horizontal_extrapolation1 + 256)); //mid
    cm.set(RH, hash(i++, horizontal_extrapolation2 + 256)); //mid
    cm.set(RH, hash(i++, o2_N + 256)); //mid (rarely useful)
    cm.set(RH, hash(i++, best_N)); //mid

    assert(i == nCM);

    // mixer contexts

    uint32_t column_ctx = column_score <= 1 ? 0 : column_score < (record_score >> 10) * 3 ? 1 : 2; // 0..2
    uint32_t match_ctx = match_score[0] == 0 ? 0 : 1 + min((match_score[0] >> 8) >> 2, 30); // 0..31

    mctx1 =
      column_ctx << 5 |
      match_ctx;

    mctx2 =
      column_ctx << 1 |
      (record_score == 0);
  }

  // for every bit

  mapR1.mix(m);
  mapR2.mix(m);
  cm.mix(m);

  m.set(mctx1, 3 * 32);
  m.set(mctx2, 3 * 2);
}
