#include "ResidualMap.hpp"

ResidualMap::ResidualMap(const Shared* const sh, const int numContexts, const int histogramsPerContext, const int scale) :
  shared(sh),
  scale(scale),
  numContexts(numContexts),
  histogramsPerContext(histogramsPerContext),
  currentContextIndex(0),
  predictions(numContexts),
  bases(numContexts),
  sums(BINS * numContexts * histogramsPerContext) {
}

void ResidualMap::setscale(const int scale) {
  this->scale = scale;
}

void ResidualMap::set(const short prediction, const uint32_t histogram_id) {
  assert(histogram_id < histogramsPerContext);
  assert(currentContextIndex < numContexts);
  predictions[currentContextIndex] = prediction;
  bases[currentContextIndex] = (currentContextIndex * histogramsPerContext + histogram_id) * BINS;
  currentContextIndex++;
}

void ResidualMap::skip() {
  assert(currentContextIndex < numContexts);
  bases[currentContextIndex] = UINT32_MAX;
  currentContextIndex++;
}

void ResidualMap::mix(Mixer& m) {
  INJECT_SHARED_bpos
    INJECT_SHARED_c0
    if (bpos == 7)
      shared->GetUpdateBroadcaster()->subscribe(this);
  assert(currentContextIndex == numContexts);

  for (size_t i = 0; i < currentContextIndex; i++) {
    const uint32_t base = bases[i];
    if (base == UINT32_MAX) { // skipped context
      m.add(0);
      m.add(0);
      continue;
    }

    const uint16_t* s = &sums[base];
    const uint8_t  c1 = c0 << (8 - bpos);
    const int   range = 1 << (7 - bpos);  // 128 down to 1

    // n0: bins [offset, offset+range) mod 256
    // n1: bins [mid,    mid+range)    mod 256
    // s[k] = count[0]+...+count[k], k=0..BINS-1, s[BINS-1]=total.
    // Circular range [a, a+len): no wrap → s[a+len-1]-s[a-1]; wrap → s[BINS-1]-s[a-1]+s[a+len-BINS-1].
    // boundary cases:
    //   a==0 → s[a-1] treated as 0.
    //   offset+range == BINS is the exact-fit case: s[BINS-1] is the last valid entry, no wrap needed.
    const int bin0_offset = (192 + c1 - predictions[i]) & 255;
    const int bin1_offset = (bin0_offset + range) & 255;

    // n0: observation count in the "next bit = 0" residual window (lower half)
    // n1: observation count in the "next bit = 1" residual window (upper half)
    uint64_t n0, n1;
    // n0
    if (bin0_offset + range <= BINS)
      n0 = (bin0_offset + range == 0 ? 0 : s[bin0_offset + range - 1]) - (bin0_offset == 0 ? 0 : s[bin0_offset - 1]);
    else
      n0 = s[BINS - 1] - (bin0_offset == 0 ? 0 : s[bin0_offset - 1]) + s[bin0_offset + range - BINS - 1];

    // n1
    if (bin1_offset + range <= BINS)
      n1 = (bin1_offset + range == 0 ? 0 : s[bin1_offset + range - 1]) - (bin1_offset == 0 ? 0 : s[bin1_offset - 1]);
    else
      n1 = s[BINS - 1] - (bin1_offset == 0 ? 0 : s[bin1_offset - 1]) + s[bin1_offset + range - BINS - 1];

    n0 = n0 * 2 + 1;
    n1 = n1 * 2 + 1;
    const uint64_t sum = n0 + n1;
    const int p1 = static_cast<int>((n1 << 12) / sum);
    m.add((stretch(p1) * scale) >> 8);
    m.add(((p1 - 2048) * scale) >> 9);
  }
}

void ResidualMap::update() {
  assert(shared->State.bitPosition == 0);
  assert(currentContextIndex == numContexts);

  while (currentContextIndex > 0) {
    currentContextIndex--;
    const uint32_t base = bases[currentContextIndex];
    if (base == UINT32_MAX)
      continue;

    INJECT_SHARED_c1
    const int bin = (192 + c1 - predictions[currentContextIndex]) & 255; // moving the center to bin #192 gives better cache locality on read and on updates

    // Increment all prefix entries after bin: sums[bin .. BINS-1] += 1.
    uint16_t* s = &sums[base];
    for (int k = bin; k < BINS; k++)
      s[k]++;

    // Aging: when the total reaches UINT16_MAX halve all prefix entries.
    // A bin's count is s[k]-s[k-1] (s[-1] treated as 0); total is s[BINS-1].
    if (s[BINS - 1] == UINT16_MAX) {
      for (int k = 0; k < BINS; k++)
        s[k] >>= 1;
    }
  }
}

void ResidualMap::print() {
  for (size_t i = 0; i < numContexts; i++) {
    for (size_t j = 0; j < histogramsPerContext; j++) {
      const uint16_t* s = &sums[(i * histogramsPerContext + j) * BINS];
      printf("%zu\t%zu", i, j);
      for (int k = 0; k < BINS; k++)
        printf("\t%d", k == 0 ? s[0] : s[k] - s[k - 1]);
      printf("\n");
    }
  }
}
