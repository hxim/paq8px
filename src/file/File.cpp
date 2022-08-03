#include "File.hpp"

void File::append(const char *s) {
  for( int i = 0; s[i] != 0; i++ ) {
    putChar(static_cast<uint8_t>(s[i]));
  }
}

uint32_t  File::get32() { return (getchar() << 24) | (getchar() << 16) | (getchar() << 8) | (getchar()); }

void File::put32(uint32_t x) {
  putChar((x >> 24) & 255);
  putChar((x >> 16) & 255);
  putChar((x >> 8) & 255);
  putChar(x & 255);
}

uint64_t File::getVLI() {
  uint64_t i = 0;
  int k = 0;
  uint8_t b = 0;
  do {
    b = getchar();
    i |= static_cast<uint64_t>(b & 0x7F) << k;
    k += 7;
  } while((b >> 7) > 0 );
  return i;
}

void File::putVLI(uint64_t i) {
  while( i > 0x7F ) {
    putChar(0x80 | (i & 0x7F));
    i >>= 7;
  }
  putChar(uint8_t(i));
}
