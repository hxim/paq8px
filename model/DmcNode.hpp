#pragma once

#include <cstdint>

/**
 * c0,c1: adaptive counts of zeroes and ones;
 *        fixed point numbers with 6 integer and 10 fractional bits, i.e. scaling factor = 1024;
 *        thus the values 0 .. 65535 represent real counts of 0.0 .. 63.999
 * nx0, nx1: indexes of next DMC nodes in the state graph
 * state: bit history state - as in a contextmap
 */
struct DMCNode { // 12 bytes
private:
    uint32_t _nx0, _nx1; /**< packed: their higher 28 bits are nx0, nx1; the lower 4+4 bits give the bit history state byte */

public:
    uint16_t c0, c1;
    [[nodiscard]] auto getState() const -> uint8_t;
    void setState(uint8_t state);
    [[nodiscard]] auto getNx0() const -> uint32_t;
    void setNx0(uint32_t nx0);
    [[nodiscard]] auto getNx1() const -> uint32_t;
    void setNx1(uint32_t nx1);
};
