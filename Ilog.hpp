#pragma once

#include "Array.hpp"

/**
 * ilog(x) = round(log2(x) * 16), 0 <= x < 64K
 */
class Ilog {
public:
    static Ilog& getInstance();
    [[nodiscard]] auto log(uint16_t x) const -> int;
private:
    /**
     * Compute lookup table by numerical integration of 1/x
     */
    Ilog() {
      uint32_t x = 14155776;
      for( uint32_t i = 2; i < 65536; ++i ) {
        x += 774541002 / (i * 2 - 1); // numerator is 2^29/ln 2
        t[i] = x >> 24U;
      }
    }

    /**
     * Copy constructor is private so that it cannot be called
     */
    Ilog(Ilog const & /*unused*/) {}

    /**
     * Assignment operator is private so that it cannot be called
     */
    Ilog& operator=(Ilog const & /*unused*/) { return *this; }

    Array<uint8_t> t = Array<uint8_t>(65536);
};

/**
 * llog(x) accepts 32 bits
 * @param x
 * @return
 */
int llog(uint32_t x);
