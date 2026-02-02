#pragma once

/**
 * Least Mean Squares predictor with RMSProp-style adaptive learning rates
 *
 * Implements an adaptive filter for audio prediction using:
 * - Two components with separate learning rates and update patterns
 * - RMSProp gradient normalization for stable learning
 * - Dual circular buffers for different history types
 *
 * Buffer structure:
 * - Component 's': Same-channel recent history (updated every sample)
 * - Component 'd': Other-channel history (updated less frequently)
 *
 * Use case: Stereo audio where 's' tracks same channel and 'd'
 * tracks the other channel for cross-channel correlation
 */
class LMS
{
private:
  float* weights;
  float* eg;                // RMSProp gradient accumulator
  float* buffer;            // Sample history buffer [s same-channel | d other-channel]
  float sameChannelRate;    // Learning rate for same-channel weights
  float otherChannelRate;   // Learning rate for other-channel weights
  float rho;                // RMSProp decay rate
  float eps;                // Numerical stability constant
  float prediction;         // Last prediction value
  int s;                    // Same-channel buffer size (updated every sample)
  int d;                    // Other-channel buffer size

public:
  /**
   * Construct an LMS predictor
   * @param s Same-channel buffer size (updated every sample)
   * @param d Other-channel buffer size
   * @param sameChannelRate Learning rate for same-channel weights
   * @param otherChannelRate Learning rate for other-channel weights
   */
  LMS(int s, int d, float sameChannelRate, float otherChannelRate);

  /**
   * Destructor - frees allocated memory
   */
  ~LMS();

  // Prevent copying (we manage raw pointers)
  LMS(const LMS&) = delete;
  LMS& operator=(const LMS&) = delete;

  /**
   * Generate prediction for the next sample
   * Updates the other-channel buffer (d component)
   * @param sample Input sample (from the other channel)
   * @return Predicted value
   */
  float predict(int sample);

  /**
   * Update weights based on prediction error
   * Updates the same-channel buffer (s component)
   * @param sample Input sample (from this channel)
   */
  void update(int sample);

  /**
   * Reset all weights, gradients, and buffer to zero
   */
  void reset();
};
