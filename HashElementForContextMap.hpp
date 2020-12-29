#ifndef PAQ8PX_HASHELEMENTFORCONTEXTMAP_HPP
#define PAQ8PX_HASHELEMENTFORCONTEXTMAP_HPP

#include <cstdint>
#include "StateTable.hpp"

struct HashElementForContextMap { // sizeof(HashElemetForContextMap) = 7
  union {
    struct {
      uint8_t bit;
      uint8_t bit0;
      uint8_t bit1;
      uint8_t count;
      uint8_t value;
      uint8_t unused1;
      uint8_t unused2;
    } byteStats;
    struct {
      uint8_t bit;
      uint8_t bit0;
      uint8_t bit1;
      uint8_t bit00;
      uint8_t bit01;
      uint8_t bit10;
      uint8_t bit11;
    } bitStats;
  };

  // priority for hash replacement strategy
  inline uint8_t prio() {
    return StateTable::prio(byteStats.bit);
  }
};

#endif //PAQ8PX_HASHELEMENTFORCONTEXTMAP_HPP
