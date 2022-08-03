#include "LargeStationaryMap.hpp"

LargeStationaryMap::LargeStationaryMap(const Shared* const sh, const int contexts, const int hashBits, const int scale, const int rate) :
  shared(sh),
  rnd(),
  data((UINT64_C(1) << hashBits)),
  hashBits(hashBits),
  scale(scale),
  rate(rate),
  numContexts(contexts),
  currentContextIndex(0),
  contextHashes(contexts) {
  assert(hashBits > 0);
  assert(hashBits <= 24); // 24 is just a reasonable limit for memory use 
  assert(9 <= rate && rate <= 16); // 9 is just a reasonable lower bound, 16 is a hard bound
  reset();
}

void LargeStationaryMap::set(const uint64_t contextHash) {
  assert(currentContextIndex < numContexts);
  contextHashes[currentContextIndex] = contextHash;
  currentContextIndex++;
}

void LargeStationaryMap::setscale(const int scale) {
  this->scale = scale;
}

void LargeStationaryMap::reset() {
  for( uint32_t i = 0; i < data.size(); ++i ) {
    data[i].reset();
  }
}

void LargeStationaryMap::update() {
  assert(currentContextIndex <= numContexts);
  while (currentContextIndex > 0) {
    currentContextIndex--;
    const uint64_t contextHash = contextHashes[currentContextIndex];
    if (contextHash == 0) {
      continue; // skipped context
    }
    uint32_t hashkey = finalize64(contextHash, hashBits);
    uint16_t checksum = checksum16(contextHash, hashBits);
    uint32_t *cp = &data[hashkey].find(checksum, &rnd)->value;
    update(cp);
  }
}

void LargeStationaryMap::update(uint32_t *cp) {
  INJECT_SHARED_y

  uint32_t n0, n1, value;
  value = *cp;
  n0 = value >> 16;
  n1 = value & 0xffff;

  n0 += 1 - y;
  n1 += y;
  int shift = (n0 | n1) >> rate; // shift: 0 or 1
  n0 >>= shift;
  n1 >>= shift;

  *cp = n0 << 16 | n1;
}

void LargeStationaryMap::mix(Mixer &m) {
  shared->GetUpdateBroadcaster()->subscribe(this);
  uint32_t n0, n1, value, sum;
  int p1, st, bitIsUncertain;
  assert(currentContextIndex == numContexts);
  for (size_t i = 0; i < currentContextIndex; i++) {
    uint64_t contextHash = contextHashes[i];
    uint32_t hashkey = finalize64(contextHash, hashBits);
    uint16_t checksum = checksum16(contextHash, hashBits);
    value = data[hashkey].find(checksum, &rnd)->value;
    n0 = value >> 16;
    n1 = value & 0xffff;

    sum = n0 + n1;
    if (sum == 0) {
      m.add(0);
      m.add(0);
      m.add(0);
    }
    else {
      p1 = ((n1 * 2 + 1) << 12) / (sum * 2 + 2);
      st = (stretch(p1) * scale) >> 8;
      m.add(st);
      m.add(((p1 - 2048) * scale) >> 9);
      bitIsUncertain = int(sum <= 1 || (n0 != 0 && n1 != 0));
      m.add((bitIsUncertain - 1) & st); // when both counts are nonzero add(0) otherwise add(st)
    }
  }
}

void LargeStationaryMap::subscribe() {
  shared->GetUpdateBroadcaster()->subscribe(this);
}

void LargeStationaryMap::skip() {
  assert(currentContextIndex < numContexts);
  contextHashes[currentContextIndex] = 0; // mark for skipping
  currentContextIndex++;
}