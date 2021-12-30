#pragma once

#include "Array.hpp"
#include <cassert>
#include <cstdint>

template<typename T>
class IndirectContext {
private:
    Array<T> data;
    T *ctx;
    const uint32_t ctxMask, inputMask, inputBits, contextBits;

public:
    IndirectContext(const int bitsPerContext, const int inputBits, const int contextBits = sizeof(T)*8) :
      data(UINT64_C(1) << bitsPerContext), ctx(&data[0]),
      ctxMask((UINT32_C(1) << bitsPerContext) - 1), 
      inputMask((UINT32_C(1) << inputBits) - 1), 
      inputBits(inputBits),
      contextBits(contextBits) {
      assert(bitsPerContext > 0 && bitsPerContext <= 20);
      assert(inputBits > 0 && inputBits <= 8);
      assert(contextBits <= sizeof(T)*8);
      if (contextBits < sizeof(T) * 8) // need for leading bit -> include it
        reset(); 
    }

    void reset() {
      for (uint64_t i = 0; i < data.size(); ++i) {
        data[i] = contextBits < sizeof(T) * 8 ? 1 : 0; // 1: leading bit to indicate the number of used bits
      }
    }

    void operator+=(const uint32_t i) {
      assert(i <= inputMask);
      // note: when the context is fully mature, we need to keep the leading bit in front of the contextbits as the MSB
      T leadingBit = (*ctx) & (1 << contextBits); 
      (*ctx) <<= inputBits;
      (*ctx) |= i | leadingBit;
      (*ctx) &= (1 << (contextBits + 1)) - 1;
    };

    void operator=(const uint32_t i) {
      ctx = &data[i & ctxMask];
    }

    auto operator()() -> T & {
      return *ctx;
    };
};
