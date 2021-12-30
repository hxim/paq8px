#pragma once

#include <cstdint>
#include "StateTable.hpp"

struct HashElementForContextMap { // sizeof(HashElemetForContextMap) = 7
  uint8_t bitState;  // state of bit0 (1st slot) / state of bit2 (2nd slot)  /  state of bit5 (in slot 2)
  uint8_t bitState0; // state of bit1 when bit0 was 0 (1st slot)  /  state of bit3 when bit2 was 0 (2nd slot)  /  state of bit6 when bit5 was 0 (3rd slot)
  uint8_t bitState1; // state of bit1 when bit0 was 1 (1st slot)  /  state of bit3 when bit2 was 1 (2nd slot)  /  state of bit6 when bit5 was 1 (3rd slot)
  union {
    struct { // information in 1st slot
      uint8_t runcount; // run count of byte1
      uint8_t byte1; // the last seen byte
      uint8_t byte2; 
      uint8_t byte3;
    } byteStats;
    struct { // information in 2nd and 3rd slots
      uint8_t bitState00; // state of bit4 when bit2+bit3 was 00 (2nd slot)  /  state of bit7 when bit5+bit6 was 00 (3rd slot)
      uint8_t bitState01; // state of bit4 when bit2+bit3 was 01 (2nd slot)  /  state of bit7 when bit5+bit6 was 01 (3rd slot)
      uint8_t bitState10; // state of bit4 when bit2+bit3 was 10 (2nd slot)  /  state of bit7 when bit5+bit6 was 10 (3rd slot)
      uint8_t bitState11; // state of bit4 when bit2+bit3 was 11 (2nd slot)  /  state of bit7 when bit5+bit6 was 11 (3rd slot)
    } bitStates;
  };

  // priority for hash replacement strategy
  inline uint8_t prio() {
    return StateTable::prio(bitState);
  }
};
