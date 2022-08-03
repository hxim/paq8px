#pragma once

#include "Sentence.hpp"
#include <cstdint>

class Paragraph {
public:
  uint32_t sentenceCount;
  uint32_t typeCount[Sentence::Types::Count];
  uint32_t typeMask;
};
