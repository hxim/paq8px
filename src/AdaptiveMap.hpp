#pragma once

#include <cstdint>
#include "Array.hpp"
#include "DivisionTable.hpp"
#include "Shared.hpp"

/**
 * This is the base class for StateMap and APM.
 * Purpose: common members are here
 */
class AdaptiveMap {
protected:
  const Shared * const shared;
  Array<uint32_t> t; /**< cxt -> prediction in high 22 bits, count in low 10 bits */
  int *dt; /**< Pointer to division table */
  AdaptiveMap(const Shared* const sh, int n);
  void update(uint32_t *p, int limit);
};
