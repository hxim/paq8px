#include "LargeStationaryMap.hpp"

LargeStationaryMap::LargeStationaryMap(const Shared* const sh, const int contexts, const int hashBits, const int scale) :
  shared(sh),
  rnd(),
  data((UINT64_C(1) << hashBits)),
  hashBits(hashBits),
  scale(scale),
  numContexts(contexts),
  currentContextIndex(0),
  contextPointers(contexts) {
  assert(hashBits > 0);
  assert(hashBits <= 24); // 24 is just a reasonable limit for memory use 
  reset();
  dt = DivisionTable::getDT(); 
}

void LargeStationaryMap::set(const uint64_t contextHash) {
  assert(currentContextIndex < numContexts);
  uint32_t hashkey = finalize64(contextHash, hashBits);
  uint16_t checksum = checksum16(contextHash, hashBits);
  uint32_t* cp = &data[hashkey].find(checksum, &rnd)->value;
  // Note: the hashtable's MTF logic may physically move elements within a bucket.
  // If two contexts share the same bucket, the second find() shifts the first
  // element down, invalidating the pointer stored for it. Both pointers then
  // refer to the same element: one context's stats are never updated, the other's
  // are updated twice, and mix() reads the wrong stats for the first context.
  // This is rare and behaves like an undetected hash collision
  // -> acceptable, nothing to do.
  contextPointers[currentContextIndex] = cp;
  currentContextIndex++;
}

void LargeStationaryMap::setscale(const int scale) {
  this->scale = scale;
}

void LargeStationaryMap::reset() {
  for (uint32_t i = 0; i < data.size(); i++) {
    data[i].reset();
  }
}

void LargeStationaryMap::update() {
  assert(currentContextIndex <= numContexts);
  while (currentContextIndex > 0) {
    currentContextIndex--;
    uint32_t* const cp = contextPointers[currentContextIndex];
    if (cp == nullptr) {
      continue; // skipped context
    }
    update(cp);
  }
}

void LargeStationaryMap::update(uint32_t* cp) {

  // XOR cells with (2048<<22 = 0x80000000) on read and on write back:
  // The hashtable zero-initialises new slots unconditionally, with no hook to
  // customise it. A raw zero would decode as p1=0 but we need p1 = 0.5
  // XORing with (2048<<22) on read maps zero storage to p1=0.5,
  // and the same XOR in update() maps the probability back before storing.
  const uint32_t cell = *cp ^ 0x80000000;
  const int sum = cell & 1023;      //sum=n0+n1
  const int64_t y = shared->State.y;
  const int64_t target_22bit = y << 22;
  const int64_t p1_22bit = cell >> 10;  //prediction
  const int delta_22bit = int(((target_22bit - p1_22bit) * int64_t(dt[sum])) >> 30); // see derivation above
  (*cp) = 0x80000000 ^ (uint32_t(p1_22bit + delta_22bit) << 10) | (sum < 1023 ? sum + 1 : sum);

}

void LargeStationaryMap::mix(Mixer& m) {
  shared->GetUpdateBroadcaster()->subscribe(this);
  assert(currentContextIndex == numContexts);
  for (size_t i = 0; i < currentContextIndex; i++) {
    const uint32_t* const cp = contextPointers[i];
    const uint32_t cell = *cp ^ 0x80000000;

    int p1 = cell >> 20; // top 12 bits of the probability
    const int st = (stretch(p1) * scale) >> 8;
    m.add(st);
    m.add(((p1 - 2048) * scale) >> 9);

    // recover n1 for yet another mixer input
    const uint32_t p1_22bit = cell >> 10;
    const uint32_t sum = cell & 1023;
    const uint32_t n1 = ((sum + 1) * p1_22bit) >> 22; // 0..1023 // `sum+1` instead of `sum` is to counteract truncation bias in p1
    const int bitIsUncertain = int(sum <= 1 || (n1 < sum - 1 && n1 != 0)); // `n1 < sum-1` instead of `n1 != sum` to counteract truncation bias in p1
    m.add((bitIsUncertain - 1) & st); // when both counts are nonzero add(0) otherwise add(st)
  }
}

void LargeStationaryMap::subscribe() {
  shared->GetUpdateBroadcaster()->subscribe(this);
}

void LargeStationaryMap::skip() {
  assert(currentContextIndex < numContexts);
  contextPointers[currentContextIndex] = nullptr; // mark for skipping
  currentContextIndex++;
}
