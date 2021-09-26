#include "StationaryMap.hpp"

StationaryMap::StationaryMap(const Shared* const sh, const int bitsOfContext, const int inputBits, const int scale, const int rate) : 
  shared(sh),
  data((UINT64_C(1) << bitsOfContext) * ((UINT64_C(1) << inputBits) - 1)),
  mask((1 << bitsOfContext) - 1), 
  stride((1 << inputBits) - 1), bTotal(inputBits), scale(scale), rate(rate),
  context(0), bCount(0), b(0), cp(nullptr)
{
  assert(inputBits > 0 && inputBits <= 8);
  assert(bitsOfContext + inputBits <= 24);
  assert(9 <= rate && rate <= 16); // 9 is just a reasonable lower bound, 16 is a hard bound
}

void StationaryMap::set(uint32_t ctx) {
  context = (ctx & mask) * stride;
  bCount = b = 0;
}

void StationaryMap::setScale(int scale) {
  this->scale = scale;
}

void StationaryMap::setRate(int rate) {
  this->rate = rate;
}

void StationaryMap::reset() {
  for( uint32_t i = 0; i < data.size(); ++i ) {
    data[i] = 0;
  }
}

void StationaryMap::update() {
  INJECT_SHARED_y

  uint32_t counts, n0, n1;
  counts = *cp;
  n0 = counts >> 16;
  n1 = counts & 0xffff;

  n0 += 1 - y;
  n1 += y;
  int shift = (n0 | n1) >> rate; // shift: 0 or 1
  n0 >>= shift;
  n1 >>= shift;

  *cp = n0 << 16 | n1;

  b += static_cast<uint32_t>((y != 0) && b > 0);
}

void StationaryMap::mix(Mixer &m) {
  shared->GetUpdateBroadcaster()->subscribe(this);
  uint32_t counts, n0, n1, sum;
  int p1, st, bitIsUncertain, p0;

  cp = &data[context + b];
  counts = *cp;
  n0 = counts >> 16;
  n1 = counts & 0xffff;

  sum = n0 + n1;
  p1 = ((n1 * 2 + 1) << 12) / (sum * 2 + 2);
  st = (stretch(p1) * scale) >> 8;
  m.add(st);
  m.add(((p1 - 2048) * scale) >> 9);
  bitIsUncertain = int(sum <= 1 || (n0 != 0 && n1 != 0));
  m.add((bitIsUncertain - 1) & st); // when both counts are nonzero add(0) otherwise add(st)
  //p0 = 4095 - p1;
  //m.add((((p1 & (-!n0)) - (p0 & (-!n1))) * scale) >> 10);

  bCount++;
  b += b + 1;
  assert(bCount <= bTotal);
}
