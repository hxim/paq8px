#pragma once

#include <cstdint>
#include <cassert>
#include "Shared.hpp"

/**
 * APM1 maps a probability and a context into a new probability that bit y will next be 1.
 * After each guess it updates its state to improve future guesses.
 * Probabilities are scaled 12 bits (0-4095).
 */
class APM1 : IPredictor {
private:
  const Shared * const shared;
  int index; /**< last p, context */
  const int n; /**< number of contexts */
  Array<uint16_t> t; /**< [n][33]:  p, context -> p */
  const int rate;

public:
  /**
    * Creates an instance with @ref n contexts and learning rate @ref r.
    * @ref r determines the learning rate. Smaller = faster, must be in the range (0, 32).
    * Uses 66*n bytes memory.
    * @param n the number of contexts
    * @param r the learning rate
    */
  APM1(const Shared* const sh, int n, int r);
  /**
    * Returns adjusted probability in context @ref cx (0 to n-1).
    * @param pr initial (pre-adjusted) probability
    * @param cxt the context
    * @return adjusted probability
    */
  int p(int pr, int cxt);
  void update() override;
};
