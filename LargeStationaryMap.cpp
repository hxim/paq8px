#include "LargeStationaryMap.hpp"

LargeStationaryMap::LargeStationaryMap(const Shared* const sh, const int hashMaskBits, const int scale, const int rate) :
  shared(sh),
  data((1ULL << hashMaskBits)),
  maskBits(hashMaskBits),
  scale(scale),
  rate(rate) {
#ifdef VERBOSE
  printf("Created LargeStationaryMap with hashMaskBits = %d, %d, scale = %d, rate = %d\n", hashMaskBits, scale, rate);
#endif
  assert(hashMaskBits > 0);
  assert(hashMaskBits <= 24); // 24 is just a reasonable limit for memory use 
  assert(9 <= rate && rate <= 16); // 9 is just a reasonable lower bound, 16 is a hard bound
  reset();
  set(0);
}

void LargeStationaryMap::set(uint64_t ctx) {
  context = ctx;
}

void LargeStationaryMap::setscale(int scale) {
  this->scale = scale;
}

void LargeStationaryMap::reset() {
  for( uint32_t i = 0; i < data.size(); ++i ) {
    data[i].reset();
  }
}

void LargeStationaryMap::update() {
  INJECT_SHARED_y

  uint32_t counts, n0, n1;
  n0 = cp[0];
  n1 = cp[1];

  n0 += 1 - y;
  n1 += y;
  int shift = (n0 | n1) >> rate; //near-stationary: rate=16; adaptive: smaller rate
  n0 >>= shift;
  n1 >>= shift;

  cp[0] = n0;
  cp[1] = n1;

  context = hash(context, y);
}

void LargeStationaryMap::mix(Mixer &m) {
  shared->GetUpdateBroadcaster()->subscribe(this);
  uint32_t n0, n1, sum;
  int p1, st, bitIsUncertain, p0;
  
  uint32_t hashkey = finalize64(context, maskBits);
  uint16_t checksum = checksum16(context, maskBits);
  cp = data[hashkey].find(checksum);
  n0 = cp[0];
  n1 = cp[1];

  sum = n0 + n1;
  p1 = ((n1 * 2 + 1) << 12) / (sum * 2 + 2);
  st = (stretch(p1) * scale) >> 8;
  m.add(st);
  m.add(((p1 - 2048) * scale) >> 9);
  bitIsUncertain = int(sum <= 1 || (n0 != 0 && n1 != 0));
  m.add((bitIsUncertain - 1) & st); // when both counts are nonzero add(0) otherwise add(st)
  //p0 = 4095 - p1;
  //m.add((((p1 & (-!n0)) - (p0 & (-!n1))) * scale) >> 10);
}
