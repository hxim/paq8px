#include "Ilog.hpp"

int Ilog::log(uint16_t x) const { return static_cast<int>(t[x]); }

Ilog& Ilog::getInstance() {
  static Ilog instance;
  return instance;
}

Ilog *ilog = &Ilog::getInstance();

int llog(uint32_t x) {
  if( x >= 0x1000000 ) {
    return 256 + ilog->log(uint16_t(x >> 16));
  }
  if( x >= 0x10000 ) {
    return 128 + ilog->log(uint16_t(x >> 8));
  }

  return ilog->log(uint16_t(x));
}
