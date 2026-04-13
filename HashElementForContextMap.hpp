#pragma once

#include <cstdint>
#include "StateTable.hpp"

struct HashElementForContextMap {
  union {

    uint8_t states[7];

    uint8_t priority;  //the first byte in any slot being the entry state serves as a priority for hash replacement strategy as well

    struct { // structure when in slot 0
      uint8_t bitState;  // state of bit0; it is also called the 'bytestate' as this is the first bit of a byte, thus it is always updated for every processed byte 
      uint8_t bitState0; // state of bit1 when bit0 was 0
      uint8_t bitState1; // state of bit1 when bit0 was 1
      uint8_t runcount;  // run count of byte1
      uint8_t byte1;     // the last seen byte
      uint8_t byte2;
      uint8_t byte3;
    } slot0;

    struct { // structure when in slot 1 or slot 2
      uint8_t bitState;   // state of bit2 (slot 1)  /  state of bit5 (slot 2)
      uint8_t bitState0;  // state of bit3 when bit2 was 0 (slot 1)  /  state of bit6 when bit5 was 0 (slot 2)
      uint8_t bitState1;  // state of bit3 when bit2 was 1 (slot 1)  /  state of bit6 when bit5 was 1 (slot 2)
      uint8_t bitState00; // state of bit4 when bit2+bit3 was 00 (slot 1)  /  state of bit7 when bit5+bit6 was 00 (slot 2)
      uint8_t bitState01; // state of bit4 when bit2+bit3 was 01 (slot 1)  /  state of bit7 when bit5+bit6 was 01 (slot 2)
      uint8_t bitState10; // state of bit4 when bit2+bit3 was 10 (slot 1)  /  state of bit7 when bit5+bit6 was 10 (slot 2)
      uint8_t bitState11; // state of bit4 when bit2+bit3 was 11 (slot 1)  /  state of bit7 when bit5+bit6 was 11 (slot 2)
    } slot1and2;
  
  };

  // priority for hash replacement strategy
  inline uint8_t prio() {
    return StateTable::prio(priority);
  }
};

static_assert (sizeof(HashElementForContextMap) == 7);

