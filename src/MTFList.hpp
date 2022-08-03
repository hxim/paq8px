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
  int getFirst();
  int getNext();
  void moveToFront(int i);
};
