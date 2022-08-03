#pragma once

#include <cstdint>

static int VLICost(uint64_t n) {
  int cost = 1;
  while (n > 0x7F) {
    n >>= 7;
    cost++;
  }
  return cost;
}
