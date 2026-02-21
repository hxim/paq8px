#include "ArithmeticEncoder.hpp"
#include <cassert>
#include <cstdint>

ArithmeticEncoder::ArithmeticEncoder(File* f)
  : x1(0), x2(UINT32_MAX), x(0), pending_bits(0), bits_in_B(0), B(0), archive(f) {
}

// Read one bit from the archive, buffering a full byte at a time.
int ArithmeticEncoder::bit_read() {
  if (bits_in_B == 0) {
    B = archive->getchar(); // EOF is handled gracefully by the caller
    bits_in_B = 8;
  }
  bits_in_B--; //7..0
  return (B >> bits_in_B) & 1;
}

// Write one bit to the archive, flushing a full byte when the buffer is full.
void ArithmeticEncoder::bit_write(const int bit) {
  B = (B << 1) | bit;
  bits_in_B++;
  if (bits_in_B == 8) {
    archive->putChar(B);
    B = 0;
    bits_in_B = 0;
  }
}

// Write one bit, then emit all pending bits with the opposite value (E3 carry resolution).
void ArithmeticEncoder::bit_write_with_pending(const int bit) {
  bit_write(bit);
  for (; pending_bits > 0; pending_bits--)
    bit_write(bit ^ 1);
}

// Flush the encoder: emit enough bits from x1 to complete the final byte.
void ArithmeticEncoder::flush() {
  do {
    bit_write_with_pending(x1 >> 31);
    x1 <<= 1;
  } while (bits_in_B != 0);
}

// Prime the decode window with the first 32 bits of compressed data.
void ArithmeticEncoder::prefetch() {
  for (int i = 0; i < 32; ++i)
    x = (x << 1) | bit_read();
}

// Encode one bit. p is the probability (scaled by 2^PRECISION) that bit == 1.
void ArithmeticEncoder::encodeBit(uint32_t p, const int bit) {
  if (p == 0) p++;
  assert(p > 0 && p < (1u << PRECISION));

  // Split [x1, x2] at xmid proportional to p; round to nearest for accuracy.
  const uint32_t xmid = x1 + uint32_t((uint64_t(x2 - x1) * p) >> PRECISION);
  assert(xmid >= x1 && xmid < x2);

  bit != 0 ? (x2 = xmid) : (x1 = xmid + 1);

  // Emit leading bits that are identical in x1 and x2.
  while (((x1 ^ x2) >> 31) == 0) {
    bit_write_with_pending(x2 >> 31);
    x1 <<= 1;
    x2 = (x2 << 1) | 1;
  }

  // Rescale when the interval straddles the midpoint; defer the carry bit.
  while (x1 >= 0x40000000 && x2 < 0xC0000000) {
    pending_bits++;
    x1 = (x1 << 1) & 0x7FFFFFFF;
    x2 = (x2 << 1) | 0x80000001;
  }
}

// Decode one bit. p is the probability (scaled by 2^PRECISION) that the bit is 1.
int ArithmeticEncoder::decodeBit(uint32_t p) {
  if (p == 0) p++;
  assert(p > 0 && p < (1u << PRECISION));

  // Mirror the encoder's midpoint split.
  const uint32_t xmid = x1 + uint32_t((uint64_t(x2 - x1) * p) >> PRECISION);
  assert(xmid >= x1 && xmid < x2);

  const int bit = (x <= xmid) ? 1 : 0;
  bit != 0 ? (x2 = xmid) : (x1 = xmid + 1);

  // Consume leading bits and refill x from the bitstream.
  while (((x1 ^ x2) >> 31) == 0) {
    x1 <<= 1;
    x2 = (x2 << 1) | 1;
    x = (x << 1) | bit_read();
  }

  // Undo midpoint rescaling and keep x in sync with the interval.
  while (x1 >= 0x40000000 && x2 < 0xC0000000) {
    x1 = (x1 << 1) & 0x7FFFFFFF;
    x2 = (x2 << 1) | 0x80000001;
    x = (x << 1) ^ 0x80000000;
    x += bit_read();
  }

  return bit;
}
