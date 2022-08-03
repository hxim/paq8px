#pragma once

/**
 * Common interface for all probability predictors (probability map and mixer classes).
 * It declares the update() event handler.
 * A probability predictor predicts the probability that the next bit is 1.

 * After each bit becomes known, all probability predictors participated
 * in the prediction must update their internal state to improve future
 * predictions (adapting). In order to achieve this, a probability
 * predictor must
 *   1) inherit from the IPredictor interface,
 *   2) implement the update() interface method,
 *   3) subscribe for the broadcast in p() or mix(),
 *   4) receive the update event from UpdateBroadcaster and update its internal state in update().
 */
class IPredictor {
public:
  virtual void update() = 0;
  virtual ~IPredictor() = default;
};
