#pragma once

#include <cstdint>

struct HashElementForStationaryMap {

  uint32_t value;

  // priority for hash replacement strategy
  uint32_t prio() const {
    return value & 1023; // count
  }

};

static_assert(sizeof(HashElementForStationaryMap) == 4);
