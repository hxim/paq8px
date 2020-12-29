#include "IndirectMap.hpp"
#include "Stretch.hpp"

IndirectMap::IndirectMap(const Shared* const sh, const int bitsOfContext, const int inputBits, const int scale, const int limit) : 
  shared(sh), data((UINT64_C(1) << bitsOfContext) * ((UINT64_C(1) << inputBits) - 1)),
  sm {sh, 1, 256, 1023, StateMap::BitHistory}, mask((1U << bitsOfContext) - 1),
  maskBits(bitsOfContext), stride((1U << inputBits) - 1), bTotal(inputBits), scale(scale) {
#ifdef VERBOSE
  printf("Created IndirectMap with bitsOfContext = %d, inputBits = %d, scale = %d, limit = %d\n", bitsOfContext, inputBits, scale, limit);
#endif
  assert(inputBits > 0 && inputBits <= 8);
  assert(bitsOfContext + inputBits <= 24);
  cp = nullptr;
  setDirect(0);
  sm.setLimit(limit);
}

void IndirectMap::setDirect(const uint32_t ctx) {
  context = (ctx & mask) * stride;
  bCount = b = 0;
}

void IndirectMap::set(const uint64_t ctx) {
  context = (finalize64(ctx, maskBits)) * stride;
  bCount = b = 0;
}

void IndirectMap::update() {
  INJECT_SHARED_y
  StateTable::update(cp, y, rnd);
  b += static_cast<uint32_t>((y != 0) && b > 0);
}

void IndirectMap::setScale(const int Scale) { this->scale = Scale; }

void IndirectMap::mix(Mixer &m) {
  shared->GetUpdateBroadcaster()->subscribe(this);
  cp = &data[context + b];
  const uint8_t state = *cp;
  const int p1 = sm.p1(state);
  m.add((stretch(p1) * scale) >> 8);
  m.add(((p1 - 2048) * scale) >> 9);
  bCount++;
  b += b + 1;
  assert(bCount <= bTotal);
}
