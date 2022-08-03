#include "ArithmeticEncoder.hpp"
#include <cassert>
#include <cstdint>

ArithmeticEncoder::ArithmeticEncoder(File* f) : x1(0), x2(0xffffffff), x(0), pending_bits(0), bits_in_B(0), B(0), archive(f) {};

  // read the archive bit by bit
  int ArithmeticEncoder::bit_read() {
    if (bits_in_B == 0) {
      B = archive->getchar(); // EOF is OK
      bits_in_B = 8;
    }
    bits_in_B--; //7..0
    return (B >> bits_in_B) & 1;
  }

  // write the archive bit by bit
  void ArithmeticEncoder::bit_write(const int bit) {
    B = (B << 1) | bit;
    bits_in_B++;
    if (bits_in_B == 8) {
      archive->putChar(B);
      B = 0;
      bits_in_B = 0;
    }
  }

  void ArithmeticEncoder::bit_write_with_pending(const int bit) {
    bit_write(bit);
    for (; pending_bits > 0; pending_bits--)
      bit_write(bit ^ 1);
  }

  void ArithmeticEncoder::flush() {
    do {
      bit_write_with_pending(x1 >> 31); // pad pending byte from x1
      x1 <<= 1;
    } while (bits_in_B != 0);
  }

void ArithmeticEncoder::prefetch() {
  for (int i = 0; i < 32; ++i)
    x = (x << 1) | bit_read();
}

void ArithmeticEncoder::encodeBit(uint32_t p, const int bit) {
  if (p == 0) p++;
  assert(p > 0 && p < (1u << PRECISION));
  uint32_t xmid = x1 + uint32_t((uint64_t(x2 - x1) * p) >> PRECISION);
  assert(xmid >= x1 && xmid < x2);
  bit !=0 ? (x2 = xmid) : (x1 = xmid + 1);
  while (((x1 ^ x2) >> 31) == 0) {  // pass equal leading bits of range
    bit_write_with_pending(x2 >> 31);
    x1 <<= 1;
    x2 = (x2 << 1) | 1;
  }
  while (x1 >= 0x40000000 && x2 < 0xC0000000) {
    pending_bits++;
    x1 = (x1 << 1) & 0x7FFFFFFF;
    x2 = (x2 << 1) | 0x80000001;
  }
}

int ArithmeticEncoder::decodeBit(uint32_t p) {
  if (p == 0) p++;
  assert(p > 0 && p < (1u<< PRECISION));
  uint32_t xmid = x1 + uint32_t((uint64_t(x2 - x1) * p) >> PRECISION);
  assert(xmid >= x1 && xmid < x2);
  int bit = x <= xmid;
  bit != 0 ? (x2 = xmid) : (x1 = xmid + 1);
  while (((x1 ^ x2) >> 31) == 0) {  // pass equal leading bits of range
    x1 <<= 1;
    x2 = (x2 << 1) | 1;
    x = (x << 1) | bit_read();
  }
  while (x1 >= 0x40000000 && x2 < 0xC0000000) {
    x1 = (x1 << 1) & 0x7FFFFFFF;
    x2 = (x2 << 1) | 0x80000001;
    x = (x << 1) ^ 0x80000000;
    x += bit_read();
  }
  return bit;
}