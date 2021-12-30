#pragma once

#include "Array.hpp"
#include "Hash.hpp"
#include "Utils.hpp"
#include <cstdint>

/**
 * A BH maps a 32 bit hash to an array of b bytes (2-byte checksum and b-2 values)
 * The bucket selector index must be a multiplicative hash.
 *
 * If a collision is detected, up to "searchLimit" nearby locations in the same
 * cache line are tested and the first matching checksum or
 * empty element is returned.
 *
 * Collision detection and resolution policy:
 * 2 byte checksum with LRU replacement
 * If neither a match nor an empty element is found the lowest priority
 * element is replaced by the new element (except last 2 MRU elements).

 * @tparam B
 */
template<uint64_t B>
class BH {
    uint32_t searchLimit = 8;
    Array<uint8_t> t; /**< elements */
    const uint32_t mask; /**< size-1 */
    const int hashBits;

public:
    /**
     * Creates @ref n element table with @ref b bytes each. @ref n must be a power of 2.
     * The first two bytes of each element is reserved for a checksum to detect collisions.
     * The remaining b-2 bytes are values, prioritized by the first value. This byte is 0 to mark an unused element.
     * @param n number of elements in the table
     */
    explicit BH(uint64_t n) : t(n * B), mask(uint32_t(n - 1)), hashBits(ilog2(mask + 1)) {
      assert(B >= 2 && n > 0 && isPowerOf2(n));
    }

    /**
     * Returns a pointer to the ctx'th element, such that
     * t[ctx][-2]..t[ctx][-1] is a checksum of @ref ctx, t[ctx][0] is the priority, and
     * t[ctx][1..B-3] are other values (0-255).
     * @tparam B number of bytes in each slot including the checksum and priority
     * @param ctx bucket selector
     * @return pointer to the ctx'th element
     */
    auto operator[](uint64_t ctx) -> uint8_t *;
};

template<uint64_t B>
inline auto BH<B>::operator[](const uint64_t ctx) -> uint8_t * {
  const uint16_t chk = checksum16(ctx, hashBits);
  const uint32_t i = finalize64(ctx, hashBits) * searchLimit & mask;
  uint8_t *p = nullptr;
  uint16_t *cp = nullptr;
  uint32_t j = 0;
  for( j = 0; j < searchLimit; ++j ) {
    p = &t[(i + j) * B];
    cp = reinterpret_cast<uint16_t *>(p);
    if( p[2] == 0 ) { // prio/state byte is zero -> empty slot
      *cp = chk; //occupy
      break;
    }
    if( *cp == chk ) {
      break; // found
    }
  }
  // move to front what we've found
  if( j == 0 ) {
    return p + 2; // we are already at the front -> nothing to do
  }
  static uint8_t tmp[B]; // temporary array for element to move to front
  if( j == searchLimit ) { // element was not found
    --j; // candidate to overwrite is the last one
    memset(tmp, 0, B); // zero all
    memcpy(tmp, &chk, 2); // 2 checksum bytes
    /*for (int k = j - 1; k > 2; k--) { // try to find a lower priority element, but protect the top 2
      if (t[(i + j) * B + 2] > t[(i + k) * B + 2])
        j = k;
    }*/
    if (searchLimit > 2 && t[(i + j) * B + 2] > t[(i + j - 1) * B + 2]) {
      --j;
    }
  } else { // element was found, or empty slot occupied
    memcpy(tmp, cp, B);
  }
  memmove(&t[(i + 1) * B], &t[i * B], j * B);
  memcpy(&t[i * B], tmp, B);
  return &t[i * B + 2];
}
