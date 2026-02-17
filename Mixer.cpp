#include "Mixer.hpp"

#include "BitCount.hpp"
#include "Squash.hpp"

ALWAYS_INLINE
static int scaleDotProduct(const int dp, const int scaleFactor) {
  return (dp * scaleFactor) >> 16;
}

ALWAYS_INLINE
static int clipDotProduct(int dp) {
  if (dp < -2047) {
    dp = -2047;
  }
  else if (dp > 2047) {
    dp = 2047;
  }
  return dp;
}

ALWAYS_INLINE
static void addDotProductToNextMixer(Mixer* const mp, const int dp) {
  mp->add(dp);
}

ALWAYS_INLINE
static int processDotProduct(Mixer* const mp, int dp, const int scaleFactor) {
  dp = scaleDotProduct(dp, scaleFactor);
  dp = clipDotProduct(dp);
  addDotProductToNextMixer(mp, dp);
  return squash(dp);
}

[[gnu::cold]] [[gnu::noinline]]
static int updateLearningRateAdaptive(ErrorInfo& info, int rate, const int err) {
  const uint32_t logErr = min(0xF, ilog2(abs(err)));
  info.sum -= square(info.data[1] >> 28);
  info.data[1] <<= 4;
  info.data[1] |= info.data[0] >> 28;
  info.data[0] <<= 4;
  info.data[0] |= logErr;
  info.sum += square(logErr);
  info.collected += info.collected < 4096;
  info.mask <<= 1;
  info.mask |= (logErr <= ((info.data[0] >> 4) & 0xF));
  const uint32_t count = bitCount(info.mask);
  if (info.collected >= 64 && (info.sum > 1500 + uint32_t(rate >> 10) || count < 9 || (info.mask & 0xFF) == 0)) {
    rate = 7 * 65536;
    info.reset();
  }
  else if (info.collected == 4096 && info.sum >= 56 && info.sum <= 144 && count > 28 - uint32_t(rate >> 16) &&
    ((info.mask & 0xFF) == 0xFF)) {
    rate = max(rate - 65536, 2 * 65536);
    info.reset();
  }
  return rate;
}

ALWAYS_INLINE
static int updateLearningRate(const bool isAdaptiveLearningRate, ErrorInfo& info, int rate, const int err, const int lowerLimitOfLearningRate) {
  if (isAdaptiveLearningRate) {
    rate = updateLearningRateAdaptive(info, rate, err);
  }
  //linear learning rate decay
  if (rate > lowerLimitOfLearningRate) {
    rate--;
  }
  return rate;
}

Mixer::Mixer(const Shared* const sh, const int n, const int m, const int s, const int simdWidth) :
  shared(sh),
  n((n + (simdWidth - 1)) & -(simdWidth)),
  m(m), s(s),
  lowerLimitOfLearningRate(s == 1 ? MIN_LEARNING_RATE_S1 : MIN_LEARNING_RATE_SN),
  isAdaptiveLearningRate(sh->GetOptionAdaptiveLearningRate()),
  scaleFactor(0),
  tx((n + (simdWidth - 1)) & -(simdWidth)),
  wx(((n + (simdWidth - 1)) & -(simdWidth))* m),
  cxt(s), info(s), rates(s), pr(s),
  mp(nullptr),
  simdWidth(simdWidth) {
  assert((this->n & (simdWidth - 1)) == 0);
  assert(this->m > 0);
  assert(this->s > 0);
  for (size_t i = 0; i < s; ++i) {
    pr[i] = 2048; //initial p=0.5
    rates[i] = MAX_LEARNING_RATE;
  }
  const short initialWeight = s == 1 ? 8192 : 128;
  for (size_t i = 0; i < this->n * m; ++i) {
    wx[i] = initialWeight;
  }
  reset();
}

Mixer::~Mixer() {
  delete mp;
}

void Mixer::setScaleFactor(const int sf0, const int sf1) {
  scaleFactor = sf0;
  if (mp != nullptr) {
    mp->setScaleFactor(sf1, 0);
  }
}

void Mixer::promote(const int x) {
  if (mp != nullptr) {
    mp->add(x);
  }
}

void Mixer::update() {
  INJECT_SHARED_y
    const int target = y << 12;
  if (nx > 0) {
    for (size_t i = 0; i < numContexts; ++i) {
      const int err = target - pr[i];
      const int rate = rates[i] = updateLearningRate(isAdaptiveLearningRate, info[i], rates[i], err, lowerLimitOfLearningRate);
      train(&wx[cxt[i] * n], nx, (err * rate) >> 16);
    }
  }
  reset();
}

int Mixer::p() {
  shared->GetUpdateBroadcaster()->subscribe(this);
  assert(scaleFactor > 0);
  //pad input to a multiple of simdWidth
  while (nx & (simdWidth - 1)) {
    tx[nx++] = 0;
  }
  if (mp != nullptr) { // first mixer layer: feed results to second layer
    for (size_t i = 0; i < numContexts; ++i) {
      const int dp = dotProduct(&wx[cxt[i] * n], nx);
      pr[i] = processDotProduct(mp, dp, scaleFactor);
    }
    mp->set(0, 1);
    return mp->p();
  }
  else { // second (last) mixer layer: return prediction directly
    const int dp = scaleDotProduct(dotProduct(&wx[cxt[0] * n], nx), scaleFactor);
    return pr[0] = squash(dp);
  }
}

void Mixer::add(const int x) {
  assert(nx < n);
  assert(x == short(x));
  tx[nx++] = static_cast<short>(x);
}

void Mixer::set(const uint32_t cx, const uint32_t range) {
  assert(numContexts < s);
  assert(cx < range);
  assert(base + range <= m);
  cxt[numContexts++] = base + cx;
  base += range;
}

void Mixer::reset() {
  nx = 0;
  base = 0;
  numContexts = 0;
}
