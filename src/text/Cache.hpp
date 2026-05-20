#pragma once

#include "../Utils.hpp"
#include "../Array.hpp"

template<class T, const uint32_t Size>
class Cache {
  static_assert(Size > 1 && isPowerOf2(Size), "cache size must be a power of 2 bigger than 1");

private:
  Array<T> data;
  uint32_t Index;

public:
  static const uint32_t size = Size;

  explicit Cache() : data(Size) { Index = 0; }

  T& operator()(uint32_t i) {
    return data[(Index - i) & (Size - 1)];
  }

  void operator++(int) {
    Index++;
  }

  void operator--(int) {
    Index--;
  }

  T& next() {
    return Index++, *((T *) memset(&data[Index & (Size - 1)], 0, sizeof(T)));
  }
};
