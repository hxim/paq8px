#include "StationaryMapForJpegModel.hpp"

StationaryMapForJpegModel::StationaryMapForJpegModel(const Shared* const sh, const uint32_t numContexts, const int bitsOfMemory,
  const int inputBits, const int scale) :
  shared(sh),
  data(UINT64_C(1) << (bitsOfMemory - 8)), // 256 bytes per group
  activeContexts(numContexts),
  numContexts(numContexts),
  groupBits(bitsOfMemory - 8),
  scale(scale),
  bCount(0), b(0), subscribed(false) {
  assert(inputBits == static_cast<int>(InputBits)); // the memory layout depends on it
  static_cast<void>(inputBits); // only used by the assert
  assert(numContexts > 0);
  assert(groupBits >= 1); // at least two groups
  assert(groupBits + ChecksumBits <= 64); // the group index and the checksum must both fit in the 64 bit hash
}

void StationaryMapForJpegModel::set(const uint32_t i, const uint64_t ctxHash) {
  assert(i < numContexts);
  bCount = b = 0;
  ActiveContext& ctx = activeContexts[i];
  const uint64_t h = hash(i, ctxHash);
  // The group index is taken from the top groupBits bits of the hash and the
  // checksum from the 24 bits below them, so the two don't overlap.
  ctx.group = &data[finalize64(h, groupBits)];
  const uint32_t c = checksum24(h, groupBits);
  ctx.chk = c + static_cast<uint32_t>((c & 0xffff) == 0); // a low half of 0 is reserved for empty elements
  prefetch(ctx.group); // the lookup only reads this first cache line
}

void StationaryMapForJpegModel::claim(Group* const g, const uint32_t e, const uint32_t chk) {
  // Start from empty counters, so the new context does not inherit statistics
  // from the context that used this element before.
  g->root[e] = 0;
  memset(g->tail[e], 0, sizeof(g->tail[e]));
  g->chkLo[e] = static_cast<uint16_t>(chk);
  g->chkHi[e] = static_cast<uint8_t>(chk >> 16);
}

uint32_t StationaryMapForJpegModel::victimRank(const Group* const g, const uint64_t ranking) {
  // In 1 out of 64 replacements, ignore the use counts and simply take the least
  // recently used element. Otherwise an old element with a high count could stay
  // at the back of the group forever.
  if (rnd(6) == 0) {
    return ElementsPerGroup - 1;
  }
  uint32_t victim = ElementsPerGroup - 1;
  uint32_t victimUses = useCount(g->root[elementAt(ranking, ElementsPerGroup - 1)]);
  // On a tie, replace the less recently used one (only a strictly smaller count wins).
  for (uint32_t r = ElementsPerGroup - 2; r >= Protected; --r) {
    const uint32_t uses = useCount(g->root[elementAt(ranking, r)]);
    if (uses < victimUses) {
      victimUses = uses;
      victim = r;
    }
  }
  return victim;
}

void StationaryMapForJpegModel::lookup(ActiveContext& ctx) {
  Group* const g = ctx.group;
  const uint16_t chkLo = static_cast<uint16_t>(ctx.chk);
  const uint8_t chkHi = static_cast<uint8_t>(ctx.chk >> 16);
  const uint64_t ranking = g->order ^ Identity;

  uint32_t r = ElementsPerGroup; // stays ElementsPerGroup if the group is full and the context is not in it
  bool isNew = true; // true unless the context was found

  for (uint32_t rank = 0; rank < ElementsPerGroup; ++rank) {
    const uint32_t e = elementAt(ranking, rank);
    const uint16_t lo = g->chkLo[e];
    if (lo == chkLo && g->chkHi[e] == chkHi) { // found the context
      r = rank;
      isNew = false;
      break;
    }
    if (lo == 0) { // empty element; all later ranks are empty too
      r = rank;
      break;
    }
  }

  if (r == ElementsPerGroup) { // group is full and the context is not in it: replace an element
    r = victimRank(g, ranking);
  }

  const uint32_t e = elementAt(ranking, r);
  if (r != 0) { // is not at the front yet?
    g->order = moveToFront(ranking, r) ^ Identity;
  }
  if (isNew) {
    claim(g, e, ctx.chk);
  }

  ctx.counter = &g->root[e]; // the first bit of the chunk uses the root counter
  ctx.tail = &g->tail[e][0];
  // The tail is first needed on the second bit of the chunk, so fetch it now.
  // A tail is 24 bytes, so two of the eight tails cross a cache line boundary:
  // prefetch both its first and its last counter.
  prefetch(ctx.tail);
  prefetch(ctx.tail + (Counters - 2));
}

void StationaryMapForJpegModel::setScale(const int scale) {
  this->scale = scale;
}

void StationaryMapForJpegModel::reset() {
  // An all-zero group is valid: empty, with its elements ranked in index order
  // (see Identity for why).
  memset(&data[0], 0, data.size() * sizeof(Group));
  bCount = b = 0;
}

int StationaryMapForJpegModel::mix(Mixer& m, const uint32_t i) {
  if (!subscribed) { // subscribe once per bit, not once per context, so update() runs once per bit
    shared->GetUpdateBroadcaster()->subscribe(this);
    subscribed = true;
  }
  assert(i < numContexts);
  assert(b < Counters);

  ActiveContext& ctx = activeContexts[i];
  if (b == 0) { // first bit of the chunk: the group prefetched by set() should be in cache by now
    lookup(ctx); // sets ctx.counter and ctx.tail
  }
  else {
    ctx.counter = ctx.tail + (b - 1); // counters[b]; the tail starts at counters[1]
  }

  const uint32_t counts = *ctx.counter;
  const uint32_t n0 = counts >> 16;
  const uint32_t n1 = counts & 0xffff;
  const uint32_t sum = n0 + n1;

  const int p1 = static_cast<int>(((n1 * 2 + 1) << 12) / (sum * 2 + 2));
  const int st = (stretch(p1) * scale) >> 8;
  m.add(st);
  m.add(((p1 - 2048) * scale) >> 9);
  const int bitIsUncertain = static_cast<int>(sum <= 1 || (n0 != 0 && n1 != 0));
  // Adds st only if the bit has always had the same value (seen at least twice); otherwise adds 0.
  m.add((bitIsUncertain - 1) & st);

  return st;
}

void StationaryMapForJpegModel::update() {
  INJECT_SHARED_y

  // y == 0: add 1 to n0 (high half); y == 1: add 1 to n1 (low half).
  const uint32_t inc = 0x00010000u >> (y << 4);

  for (uint32_t i = 0; i < numContexts; ++i) {
    uint32_t* const counter = activeContexts[i].counter; // the counter that mix() just used
    uint32_t c = *counter + inc; // both counts are at most 0x7fff before this, so the increment cannot overflow into the other half
    if (c & 0x80008000u) { // rare: at most once per 16k updates of this counter
      // When either count reaches 0x8000, halve both counts. This keeps each
      // count within its 16 bits and lets the statistics keep adapting.
      // Packed version of:
      //   n0 = c >> 16;
      //   n1 = c & 0xffff;
      //   n0 >>= 1;
      //   n1 >>= 1;
      //   c = n0 << 16 | n1;
      // Shifting right moves the lowest bit of n0 into bit 15 (the top of n1);
      // the mask clears it. Bit 31 is always 0 after a right shift.
      c = (c >> 1) & 0x7fff7fffu;
    }
    *counter = c;
  }

  b = 2 * b + 1 + y;
  bCount++;
  assert(bCount <= InputBits);
  subscribed = false;
}
