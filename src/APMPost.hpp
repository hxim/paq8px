#pragma once

#include <cstdint>
#include <cassert>
#include "Shared.hpp"

/**
 * APMPost maps a 12-bit probability (0-4095) to an n-bit probability (as defined by the Arithmetic Encoder)
 * After each guess it updates its state to improve future guesses.
 */
class APMPost : IPredictor {
private:
  const Shared * const shared;
  uint32_t index; /**< last p, context */
  const uint32_t n; /**< number of contexts */
  Array<uint64_t> t;

public:
  /**
    * Creates an instance with @ref n contexts.
    * @param n the number of contexts
    */
  APMPost(const Shared* sh, uint32_t n);
  /**
    * Returns adjusted probability in context @ref cx (0 to n-1).
    * @param pr initial (pre-adjusted) probability
    * @param cxt the context
    * @return adjusted probability
    */
  uint32_t p(uint32_t pr, uint32_t cxt);
  void update() override;

  void print() {
    for (int i = 0; i < 4096; i++) {
      printf("%d\t%d\t%d\n",i,(int)(t[i]>>32), (int)(t[i]&0xffffffff));
    }
  }
};
