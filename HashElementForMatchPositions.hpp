#ifndef PAQ8PX_HASHELEMENTFORMATCHPOSITIONS_HPP
#define PAQ8PX_HASHELEMENTFORMATCHPOSITIONS_HPP

#include <cstdint>

struct HashElementForMatchPositions { // sizeof(HashElementForMatchPositions) = 3*4 = 12
  static constexpr size_t N = 3;
  uint32_t matchPositions[N];
  void Add(uint32_t pos) {
    if (N > 1) {
      memmove(&matchPositions[1], &matchPositions[0], (N - 1) * sizeof(matchPositions[0]));
    }
    matchPositions[0] = pos;
  }
};

#endif //PAQ8PX_HASHELEMENTFORMATCHPOSITIONS_HPP
