#ifndef PAQ8PX_HASHELEMENTFORBITHISTORYSTATE_HPP
#define PAQ8PX_HASHELEMENTFORBITHISTORYSTATE_HPP

#include <cstdint>
#include "StateTable.hpp"

struct HashElementForBitHistoryState { // sizeof(HashElemetForContextMap) = 1
  uint8_t bitState;

  // priority for hash replacement strategy
  inline uint8_t prio() {
    return StateTable::prio(bitState);
  }
};

#endif //PAQ8PX_HASHELEMENTFORBITHISTORYSTATE_HPP
