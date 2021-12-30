#pragma once

#include "Word.hpp"
#include <cstdint>

class Segment {
public:
    Word firstWord; /**< useful following questions */
    uint32_t wordCount {};
    uint32_t numCount {};
};
