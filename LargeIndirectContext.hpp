#pragma once

#include "Array.hpp"
#include "Bucket16.hpp"
#include "Hash.hpp"
#include <cassert>
#include <cstdint>

template<typename T>
class LargeIndirectContext {
private:
    Array<Bucket16<T, 7>> data;
    const uint32_t hashBits, inputBits;

public:
  LargeIndirectContext(const int hashBits, const int inputBits) :
      data(UINT64_C(1) << hashBits),
      hashBits(hashBits),
      inputBits(inputBits)
    {
      assert(hashBits > 0 && hashBits <= 24);
      assert(inputBits > 0 && inputBits <= 8);
    }

    void reset() {
      for (uint64_t i = 0; i < data.size(); ++i) {
        data[i].reset();
      }
    }

    void set(const uint64_t contextHash, const uint8_t c) {
      assert(c < (1 << inputBits));
      uint32_t* ptr = data[finalize64(contextHash, hashBits)].find(checksum16(contextHash, hashBits), nullptr);
      *ptr = (*ptr) << inputBits | c;
    };

    uint32_t get(const uint64_t contextHash) {
      return *data[finalize64(contextHash, hashBits)].find(checksum16(contextHash, hashBits), nullptr);
    };

};
