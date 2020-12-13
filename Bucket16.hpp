#ifndef PAQ8PX_BUCKET16_HPP
#define PAQ8PX_BUCKET16_HPP

#include <cstdint>
#include <cstring>

/**
 * Hash bucket for 16-bit values
 *
 */

constexpr int ElementsInBucket = 7;

struct HashElement16 { // 6 bytes
  uint16_t checksum;
  uint16_t n0;
  uint16_t n1;
};

class Bucket16 { // sizeof(Bucket16) = 7*6 = 42 bytes
private:
  HashElement16 elements[ElementsInBucket];
public:

  void reset() {
    memset(&elements[0], 0, sizeof(elements));
  }

  uint16_t* find(uint16_t checksum) {
    checksum += checksum == 0; //don't allow 0 checksums
    for (int i = 0; i < ElementsInBucket; ++i) {
      if (elements[i].checksum == checksum) { // found matching checksum
        if (i != 0) {
          uint16_t n0 = elements[i].n0;
          uint16_t n1 = elements[i].n1;
          //shift elements
          memmove(&elements[1], &elements[0], i * sizeof(HashElement16));
          //move element to front
          elements[0].checksum = checksum;
          elements[0].n0 = n0;
          elements[0].n1 = n1;
        }
        return &elements[0].n0;
      }
      if (elements[i].checksum == 0) { // found empty slot
        if (i != 0) {
          //shift elements
          memmove(&elements[1], &elements[0], i * sizeof(HashElement16));
        }
        //create new element
        elements[0].checksum = checksum;
        elements[0].n0 = 0;
        elements[0].n1 = 0;
        return &elements[0].n0;
      }
    }
    //no match and no empty slot -> overwrite the last accessed element with an empty one
    memmove(&elements[1], &elements[0], (ElementsInBucket - 1) * sizeof(HashElement16));
    //create new element
    elements[0].checksum = checksum;
    elements[0].n0 = 0;
    elements[0].n1 = 0;
    return &elements[0].n0;
  }
};

#endif //PAQ8PX_BUCKET16_HPP
