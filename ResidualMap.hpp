#pragma once

#include "IPredictor.hpp"
#include "Mixer.hpp"
#include "Stretch.hpp"
#include "UpdateBroadcaster.hpp"
/**
 * Residual histogram model: tracks the distribution of (partial_actual_value − predicted_value) residuals.
 * The partial_actual_value is the already known bits of the actual byte.
 * Calculating the predicted_value is the responsibility of the caller.
 * Determining the context is also the responsibility of the caller.
 * The distribution is modelling the probability of the next bit in the partial byte being a "1".
 * The context is direct (not hashed). For each byte context modelled, the exact next bit counts are stored.
 *
 * For each context we maintain a running prefix-sum histogram (cumulative counts) over 256 residual bins.
 * Residuals of the full byte would be centered around 0 usually following a Laplacian distribution.
 * A perfect prediction of the full byte gives residual 0.
 * The partial residual uses only the bits known so far, so it converges toward the full residual as more bits are decoded.
 * 
 * The histogram is circular (mod 256) to handle underflows of the difference (partial_actual_value − predicted_value).
 * As a speed optimization, the center is moved from #0 to #192. Since residuals cluster near 0 (mapped to bin #192),
 * the suffix increment loop [bin, BINS) is shorter on average when the center is near the high end of the array.
 * It also reduces how often a query window wraps around, keeping lookups cache-friendly.
 *
 * Read  (mix):    O(numContexts); O(1) per context — four array lookups and two subtractions per window.
 * Write (update): O(BINS) — increment all suffix entries by 1. Sequential, cache-friendly.
 * Halve (aging):  O(BINS) — halve all entries when the total count would not fit in 16 bits.
 *
 * Layout: sums[base+k] = count[0]+...+count[k], k=0..BINS-1.
 *         sums[base+BINS-1] = total.
 */
class ResidualMap : IPredictor
{
public:
  static constexpr int MIXERINPUTS = 2;
  static constexpr int BINS = 256;

private:
  const Shared* const shared;
  int scale;

  const uint32_t numContexts;
  const uint32_t histogramsPerContext;
  uint32_t currentContextIndex;

  Array<short>    predictions;  // pixel value prediction per active context
  Array<uint32_t> bases;        // base offset into sums[] per active context; UINT32_MAX = skipped
  Array<uint16_t> sums;         // prefix-sum histograms: BINS entries per context per distroset

public:
  ResidualMap(const Shared* const sh, const int numContexts, const int histogramsPerContext, const int scale = 64);

  void setscale(const int scale);

  // set: 'prediction' is the predicted byte value; 'distro' selects the histogram set for this context.
  void set(const short prediction, const uint32_t histogram_id);
  void skip();

  // either set() or skip() must be called exactly once per context slot (0..numContexts-1) before mix().
  void mix(Mixer& m);

  // update(): Called once per byte (at bitPosition==0) via UpdateBroadcaster.
  // Walks all active contexts and increments the suffix of their histogram
  // at the bin corresponding to (actual_byte − prediction), then ages if needed.
  void update() override;


  void print();
};
