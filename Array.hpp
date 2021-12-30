#pragma once

#include <cstdint>
#include "Utils.hpp"
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include "ProgramChecker.hpp"

#ifdef NDEBUG
#define CHECK_INDEX(index, upper_bound)((void) 0)
#else

static void CHECK_INDEX(uint64_t index, uint64_t upperBound) {
  if( index >= upperBound ) {
    fprintf(stderr, "%" PRIu64 " out of upper bound %" PRIu64 "\n", index, upperBound);
    BACKTRACE()
    quit();
  }
}

#endif

/**
 * Array<T, Align> a(n); allocates memory for n elements of T.
 * The base address is aligned if the "alignment" parameter is given.
 * Constructors for T are not called, the allocated memory is initialized to 0s.
 * It's the caller's responsibility to populate the array with elements.
 * Parameters are checked and indexing is bounds checked if assertions are on.
 * Use of copy and assignment constructors are not supported.
 * @tparam T
 * @tparam Align
 */
template<class T, const int Align = 16>
class Array {
private:
    uint64_t usedSize {};
    uint64_t reservedSize {};
    char *ptr {}; /**< Address of allocated memory (may not be aligned) */
    T *data;   /**< Aligned base address of the elements, (ptr <= T) */
    ProgramChecker *programChecker = ProgramChecker::getInstance();
    void create(uint64_t requestedSize);

    [[nodiscard]] inline uint64_t padding() const { return Align - 1; }

    [[nodiscard]] inline uint64_t allocatedBytes() const {
      return (reservedSize == 0) ? 0 : reservedSize * sizeof(T) + padding();
    }

    /**
    * Assignment operator is private so that it cannot be called
    */
    auto operator=(Array const& /*unused*/) -> Array& { return *this; }

public:
    explicit Array(uint64_t requestedSize) { create(requestedSize); }

    ~Array();

    T &operator[](uint64_t i) {
      CHECK_INDEX(i, usedSize);
      return data[i];
    }

    const T &operator[](uint64_t i) const {
      CHECK_INDEX(i, usedSize);
      return data[i];
    }

    /**
     * @return the number of T elements currently in the array.
     */
    [[nodiscard]] uint64_t size() const { return usedSize; }

    /**
     * Grows or shrinks the array.
     * @param newSize the new size of the array
     */
    void resize(uint64_t newSize);

    /**
     * Removes the last element by reducing the size by one (but does not free memory).
     */
    void popBack() {
      assert(usedSize > 0);
      --usedSize;
    }

    /**
     * appends x to the end of the array and reserves space for more elements if needed.
     * @param x the element to append
     */
    void pushBack(const T &x);
    /**
     * Prevent copying
     * Remark: GCC complains if this member is private, so it is public
     */
    Array(const Array &) { assert(false); }

};

template<class T, const int Align>
void Array<T, Align>::create(uint64_t requestedSize) {
#ifdef VERBOSE
  printf("Created Array of size %" PRIu64 "\n", requestedSize);
#endif
  assert(isPowerOf2(Align));
  usedSize = reservedSize = requestedSize;
  if( requestedSize == 0 ) {
    data = nullptr;
    ptr = nullptr;
    return;
  }
  const uint64_t bytesToAllocate = allocatedBytes();
  ptr = (char *) calloc(bytesToAllocate, 1);
  if( ptr == nullptr ) {
    quit("Out of memory.");
  }
  uint64_t pad = padding();
  data = (T *) (((uintptr_t) ptr + pad) & ~(uintptr_t) pad);
  assert(ptr <= (char *) data && (char *) data <= ptr + Align);
  assert(((uintptr_t) data & (Align - 1)) == 0); //aligned as expected?
  programChecker->alloc(bytesToAllocate);
}

template<class T, const int Align>
void Array<T, Align>::resize(uint64_t newSize) {
  if( newSize <= reservedSize ) {
    usedSize = newSize;
    return;
  }
  char *oldPtr = ptr;
  T *oldData = data;
  const uint64_t oldSize = usedSize;
  programChecker->free(allocatedBytes());
  create(newSize);
  if( oldSize > 0 ) {
    assert(oldPtr != nullptr && oldData != nullptr);
    memcpy(data, oldData, sizeof(T) * oldSize);
  }
  if( oldPtr != nullptr ) {
    free(oldPtr);
  }
}

template<class T, const int Align>
void Array<T, Align>::pushBack(const T &x) {
  if( usedSize == reservedSize ) {
    const uint64_t oldSize = usedSize;
    const uint64_t newSize = usedSize * 2 + 16;
    resize(newSize);
    usedSize = oldSize;
  }
  data[usedSize++] = x;
}

template<class T, const int Align>
Array<T, Align>::~Array() {
  programChecker->free(allocatedBytes());
  free(ptr);
  usedSize = reservedSize = 0;
  data = nullptr;
  ptr = nullptr;
}
