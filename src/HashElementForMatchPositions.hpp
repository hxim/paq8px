#pragma once

#include <cstdint>

struct HashElementForMatchPositions {
  static constexpr size_t N = 3;
  uint32_t matchPositions[N];
  void Add(uint32_t pos) {
    if (N > 1) {
      memmove(&matchPositions[1], &matchPositions[0], (N - 1) * sizeof(matchPositions[0]));
    }
    matchPositions[0] = pos;
  }
};

static_assert(sizeof(HashElementForMatchPositions) == 3 * 4);
