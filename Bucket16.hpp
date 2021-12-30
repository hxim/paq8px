#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>
#include "Random.hpp"
#include "HashElementForContextMap.hpp"
#include "HashElementForStationaryMap.hpp"
#include "HashElementForBitHistoryState.hpp"

/**
 * Hash bucket to be used in a hash table
 * A hash bucket consists of a list of HashElements
 * Each hash element consists of a 16-bit checksum for collision detection
 * and an arbitrary-sized value:
 *   For LargeStationaryMap: two 16-bit counts               ( sizeof(HashElement) = 2+2 = 4 bytes )
 *   For LargeIndirectContext: byte history                  ( sizeof(HashElement) = 2+4 = 6 bytes )
 *   For ContextMap and ContextMap2: bit and byte statistics ( sizeof(HashElement) = 2+7 = 9 bytes )
 *
 */

#pragma pack(push,1)
template<typename T>
struct HashElement {
  uint16_t checksum;
  T value;
};
#pragma pack(pop)

template<typename T, int ElementsInBucket>
class Bucket16 {
private:
  HashElement<T> elements[ElementsInBucket];
public:

  void reset() {
    for (size_t i = 0; i < ElementsInBucket; i++)
      elements[i] = {};
  }

  void stat(uint64_t& used, uint64_t& empty) {
    for (size_t i = 0; i < ElementsInBucket; i++)
      if (elements[i].checksum == 0)
        empty++;
      else
        used++;
  }

  T* find(uint16_t checksum, Random* rnd) {

    checksum += checksum == 0; //don't allow 0 checksums (0 checksums are used for empty slots)

    if (elements[0].checksum == checksum) //there is a high chance that we'll find it in the first slot, so go for it
      return &elements[0].value;

    for (size_t i = 1; i < ElementsInBucket; ++i) {
      if (elements[i].checksum == checksum) { // found matching checksum
        T value = elements[i].value;
        //shift elements down
        memmove(&elements[1], &elements[0], i * sizeof(HashElement<T>));
        //move element to front (re-create)
        elements[0].checksum = checksum;
        elements[0].value = value;
        return &elements[0].value;
      }
      if (elements[i].checksum == 0) { // found empty slot
        //shift elements down (free the first slot for the new element)
        memmove(&elements[1], &elements[0], i * sizeof(HashElement<T>)); // i==0 is OK
        goto create_element;
      }
    }

    //no match and no empty slot -> overwrite an existing element with the new (empty) one

    //Replacement strategy:
    // - In case the hash elements represent an indirect context (we have content but no priority): overwrite the last used element
    // - In case the hash elements are counts/statistics: overwrite one of the the last used elements having low priority.
    //
    //   The priority of an element is established by the counts it represents: large counts usually mean that the element is worthy to keep.
    //   When an element is accessed recently it is probably worthy to keep as it represents recent statistics,
    //   when an element is accessed long ago it is probably obolete and may be overwritten.
    //   In order to keep track of the order the elements were accessed we move the most recently accessed element to the front: thus
    //   recently acessed elements are usually somewhere in the front of the bucket, rarely accessed element slowly move to the back. The
    //   last elements in the bucket are cantidates for eviction.
    //
    //   Overwriting the lowest priority element (with no regard of its position) is in favor of homogenous (semi-stationary) files.
    //   Overwriting the last element (with no regard of its priority) is in favor of mixed-content files.
    //   In order to favor both, we are using a probabilistic replacement startegy where most recently accessed and/or higher priority
    //   elements have higher chance to stay in the bucket = rarely accessed and/or low priority elements have higher chance to be 
    //   evicted (overwritten).
    //
    //   The the 2 most recently accessed elements are always protected from overwriting.
    {
      size_t minElementIdx = ElementsInBucket - 1;
      if constexpr (std::is_same<T, HashElementForContextMap>::value ||
                    std::is_same<T, HashElementForStationaryMap>::value || 
                    std::is_same<T, HashElementForBitHistoryState>::value)
      {
        uint32_t RND = rnd->operator()(32);
        if ((RND & 63) >= 1) {
          RND >>= 6;
          uint32_t minPrio = elements[ElementsInBucket - 1].value.prio();
          uint32_t thisPrio = elements[ElementsInBucket - 2].value.prio();
          if (thisPrio < minPrio) {
            minPrio = thisPrio;
            minElementIdx = ElementsInBucket - 2;
          }
          if ((RND & 63) >= 4) {
            RND >>= 6;
            thisPrio = elements[ElementsInBucket - 3].value.prio();
            if (thisPrio < minPrio) {
              minPrio = thisPrio;
              minElementIdx = ElementsInBucket - 3;
            }
            if ((RND & 63) >= 8) {
              RND >>= 6;
              thisPrio = elements[ElementsInBucket - 4].value.prio();
              if (thisPrio < minPrio) {
                minPrio = thisPrio;
                minElementIdx = ElementsInBucket - 4;
              }
              if ((RND & 63) >= 16) {
                //RND >>= 6; //not necessary
                thisPrio = elements[ElementsInBucket - 5].value.prio();
                if (thisPrio < minPrio) {
                  minPrio = thisPrio;
                  minElementIdx = ElementsInBucket - 5;
                }
              }
            }
          }
        }
      }

      //shift elements down (make room for the new element in the first slot)
      //at the same time owerwrite the element at position "minElementIdx"
      memmove(&elements[1], &elements[0], minElementIdx * sizeof(HashElement<T>));
    }

  create_element:
    elements[0].checksum = checksum;
    elements[0].value = {};
    return &elements[0].value;
  }
};
