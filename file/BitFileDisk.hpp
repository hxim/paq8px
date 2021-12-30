#pragma once

#include "FileDisk.hpp"

class BitFileDisk : public FileDisk {
private:
  static std::size_t constexpr BufWidth = sizeof(std::size_t) * 8u;
  std::size_t buffer, bits_left;
public:
  BitFileDisk(bool const input) :
    FileDisk(),
    buffer(0u)
  {
    bits_left = (input) ? 0u : BufWidth;
  }
  void putBits(std::size_t const code, std::size_t const bits) {
    if (bits_left >= bits)
      buffer = (buffer << bits) | code, bits_left -= bits;
    else {
      buffer = (buffer << bits_left) | (code >> (bits - bits_left));
      for (std::int32_t k = static_cast<std::int32_t>(BufWidth) - 8; k >= 0; putChar((buffer >> k) & 0xFFu), k -= 8);
      buffer = code & ((1ull << (bits - bits_left)) - 1ull), bits_left = BufWidth - (bits - bits_left);
    }
  }
  void flush() {
    if (bits_left !=BufWidth) {
      buffer <<= bits_left;
      for (std::int32_t i = 8; i <= static_cast<std::int32_t>(BufWidth - bits_left); putChar((buffer >> (BufWidth - i)) & 0xFFu), i += 8);
    }
  }
  std::size_t getBits(std::size_t const bits) {
    if (bits_left < bits) {
      int c;
      do {
        if ((c = getchar()) == EOF) break;
        buffer |= static_cast<std::size_t>(c & 0xFFu) << (BufWidth - bits_left - 8);
      } while (((bits_left += 8) + 8) <= BufWidth);
    }
    std::size_t const code = buffer >> (BufWidth - bits);
    buffer <<= bits, bits_left -= bits;
    return code;
  }
};
