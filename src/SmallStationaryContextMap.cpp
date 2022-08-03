#include "SmallStationaryContextMap.hpp"

SmallStationaryContextMap::SmallStationaryContextMap(const Shared* const sh, const int bitsOfContext, const int inputBits, const int rate, const int scale) : 
  shared(sh),
  data((UINT64_C(1) << bitsOfContext) * ((UINT64_C(1) << inputBits) - 1)),
  mask((1U << bitsOfContext) - 1), 
  stride((1U << inputBits) - 1),
  bTotal(inputBits), rate(rate), scale(scale) {
  assert(inputBits > 0 && inputBits <= 8);
  reset();
  set(0);
}

void SmallStationaryContextMap::set(uint32_t ctx) {
  context = (ctx & mask) * stride;
  bCount = b = 0;
}

void SmallStationaryContextMap::reset() {
  for( uint32_t i = 0; i < data.size(); ++i ) {
    data[i] = 0x7FFF;
  }
  cp = &data[0];
}

void SmallStationaryContextMap::update() {
  INJECT_SHARED_y
  *cp += ((y << 16) - (*cp) + (1 << (rate - 1))) >> rate;
  b += static_cast<uint32_t>((y != 0) && b > 0);
}

void SmallStationaryContextMap::mix(Mixer &m) {
  shared->GetUpdateBroadcaster()->subscribe(this);
  cp = &data[context + b];
  const int prediction = (*cp) >> 4;
  m.add((stretch(prediction) * scale) >> 8);
  m.add(((prediction - 2048) * scale) >> 9);
  bCount++;
  b += b + 1;
  assert(bCount <= bTotal);
}
