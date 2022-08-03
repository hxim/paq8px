#pragma once

#include "FileDisk.hpp"

class BitFileDisk : public FileDisk {
private:
  static size_t constexpr BufWidth = sizeof(size_t) * 8;
  size_t buffer, bits_left;
public:
  BitFileDisk(bool const input) :
    FileDisk(),
    buffer(0)
  {
    bits_left = (input) ? 0 : BufWidth;
  }
  void putBits(size_t const code, size_t const bits) {
    if (bits_left >= bits)
      buffer = (buffer << bits) | code, bits_left -= bits;
    else {
      buffer = (buffer << bits_left) | (code >> (bits - bits_left));
      for (int32_t k = static_cast<int32_t>(BufWidth) - 8; k >= 0; putChar((buffer >> k) & 0xff), k -= 8);
      buffer = code & ((UINT64_C(1) << (bits - bits_left)) - 1), bits_left = BufWidth - (bits - bits_left);
    }
  }
  void flush() {
    if (bits_left !=BufWidth) {
      buffer <<= bits_left;
      for (int32_t i = 8; i <= static_cast<int32_t>(BufWidth - bits_left); putChar((buffer >> (BufWidth - i)) & 0xff), i += 8);
    }
  }
  size_t getBits(size_t const bits) {
    if (bits_left < bits) {
      int c;
      do {
        if ((c = getchar()) == EOF) break;
        buffer |= static_cast<size_t>(c & 0xffu) << (BufWidth - bits_left - 8);
      } while (((bits_left += 8) + 8) <= BufWidth);
    }
    size_t const code = buffer >> (BufWidth - bits);
    buffer <<= bits, bits_left -= bits;
    return code;
  }
};
