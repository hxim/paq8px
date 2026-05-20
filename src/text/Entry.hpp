#pragma once

#include <cstdint>

#pragma pack(push, 1)

struct Entry {
  short prefix;
  uint8_t suffix;
  bool termination;
  uint32_t embedding;
};
#pragma pack(pop)
