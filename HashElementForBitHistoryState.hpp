#pragma once

#include <cstdint>
#include "StateTable.hpp"

struct HashElementForBitHistoryState {
  uint8_t bitState;

  // priority for hash replacement strategy
  inline uint8_t prio() {
    return StateTable::prio(bitState);
  }
};

static_assert (sizeof(HashElementForBitHistoryState) == 1);
