#pragma once

#include "Array.hpp"
#include <cstdint>

/**
 * Move To Front List
 */
class MTFList {
private:
    int root, Index;
    Array<int, 16> previous;
    Array<int, 16> next;
public:
    explicit MTFList(uint16_t n);
    auto getFirst() -> int;
    auto getNext() -> int;
    void moveToFront(int i);
};
