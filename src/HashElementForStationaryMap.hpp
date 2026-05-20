#pragma once

#include <cstdint>

struct HashElementForStationaryMap {

  uint32_t value;

  // priority for hash replacement strategy
  uint32_t prio() {
    return (value >> 16) + (value & 0xffff); // to be tuned
  }

};

static_assert(sizeof(HashElementForStationaryMap) == 4);
