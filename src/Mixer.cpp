#include "Mixer.hpp"

#include "BitCount.hpp"
#include "Squash.hpp"

#include <climits>
#include <initializer_list>

// Maximum stretched probability accepted from a model.
static constexpr int STRETCH_LIMIT = 2047;

// Maximum stretched value stored as a layer output. squash() saturates above
// 2047, but later layers can still use the full value through their linear
// dot product.
static constexpr int CLAMP_LIMIT = 4095;

// A skip input is the difference of two layer outputs, so its range is twice
// the normal layer-output range.
static constexpr int SKIP_LIMIT = 2 * CLAMP_LIMIT;
static_assert(SKIP_LIMIT <= 16383, "skip difference would saturate train()");

// Largest error between the target (0 or 4096) and a 12-bit prediction.
static constexpr int MAX_ERROR = 4095;

ALWAYS_INLINE
static int scaleDotProduct(const int dp, const int scaleFactor) {
  // Use 64 bits for the multiplication, then scale and clamp the result.
  const int64_t scaled = (int64_t(dp) * scaleFactor) >> 16;
  if (scaled < -CLAMP_LIMIT) { return -CLAMP_LIMIT; }
  if (scaled > CLAMP_LIMIT) { return CLAMP_LIMIT; }
  return int(scaled);
}

// Rounds @p v up to the next multiple of @p simdWidth.
static size_t padToWidth(const int v, const int simdWidth) {
  return size_t((v + (simdWidth - 1)) & -(simdWidth));
}

Mixer::Mixer(const Shared* const sh, const int n, const int m, const int s, const int promoted, const int simdWidth) :
  shared(sh),
  numInputs(size_t(n)),
  numPromotedInputs(size_t(promoted)),
  numPromoted(0),
  numAdded(0),
  bitContexts(uint32_t(m), uint32_t(s)),
  singleContext(1, 1),
  inputLayer(bitContexts, padToWidth(n, simdWidth),
    /*maxInput*/ STRETCH_LIMIT, /*errorLimit*/ 56, /*initialWeight*/ 128,
    /*lowerLimit*/ learningRate(DEFAULT_END_RATE_INPUT)),
  middleLayer(bitContexts, padToWidth(s + promoted, simdWidth),
    /*maxInput*/ CLAMP_LIMIT, /*errorLimit*/ 56, /*initialWeight*/ 4096,
    /*lowerLimit*/ learningRate(DEFAULT_END_RATE_MIDDLE)),
  // The combining layer has s middle outputs, s skip inputs, and promoted inputs.
  combiningLayer(singleContext, padToWidth(2 * s + promoted, simdWidth),
    /*maxInput*/ SKIP_LIMIT, /*errorLimit*/ 6, /*initialWeight*/ 8192,
    /*lowerLimit*/ learningRate(DEFAULT_END_RATE_COMBINING)) {
  assert(n > 0);
  assert(m > 0);
  assert(s > 0);
  assert(promoted >= 0);

  assert(isWellFormed(inputLayer, simdWidth));
  assert(isWellFormed(middleLayer, simdWidth));
  assert(isWellFormed(combiningLayer, simdWidth));

  // Set the input layout of the middle and combining layers.
  middleLayer.routed = { 0, size_t(s) };
  middleLayer.promoted = { size_t(s), numPromotedInputs };
  combiningLayer.routed = { 0, size_t(s) };
  combiningLayer.skip = { size_t(s), size_t(s) };
  combiningLayer.promoted = { size_t(2 * s), numPromotedInputs };
  assert(middleLayer.promoted.end() <= middleLayer.n);
  assert(combiningLayer.promoted.end() <= combiningLayer.n);

  // Skip weights are deliberately initialized to zero (see the class comment).
  setWeightBlock(combiningLayer, combiningLayer.skip, 0);

  reset();
}

Mixer::~Mixer() = default;

bool Mixer::isWellFormed(const Layer& layer, const int simdWidth) {
  // Inputs must be SIMD-aligned, and the layer must be small enough that its
  // dot products stay within the integer range supported by the SIMD kernels.
  return (layer.n & size_t(simdWidth - 1)) == 0
    && int64_t(layer.n) * layer.maxInput * 128 <= INT_MAX;
}

void Mixer::setWeightBlock(Layer& layer, const Block& block, const short value) {
  assert(block.end() <= layer.n);
  for (uint32_t r = 0; r < layer.contexts.m; ++r) {
    short* const w = &layer.wx[size_t(r) * layer.n];
    for (size_t i = 0; i < block.count; ++i) {
      w[block.offset + i] = value;
    }
  }
}

void Mixer::forward(Layer& layer) {
  assert(layer.scaleFactor > 0);

  const short* const t = &layer.tx[0];
  const ContextSet& ctx = layer.contexts;
  const size_t count = ctx.count;
  const size_t end = count & ~size_t(1);
  size_t i = 0;
  for (; i < end; i += 2) {
    int dp1 = 0;
    int dp0 = dotProduct2(t,
      &layer.wx[ctx.cxt[i + 0] * layer.n],
      &layer.wx[ctx.cxt[i + 1] * layer.n], layer.n, dp1);
    dp0 = scaleDotProduct(dp0, layer.scaleFactor);
    dp1 = scaleDotProduct(dp1, layer.scaleFactor);
    layer.out[i + 0] = static_cast<short>(dp0);
    layer.out[i + 1] = static_cast<short>(dp1);
    layer.pr[i + 0] = static_cast<short>(squash(dp0));
    layer.pr[i + 1] = static_cast<short>(squash(dp1));
  }
  if (i < count) {
    int dp = dotProduct(t, &layer.wx[ctx.cxt[i] * layer.n], layer.n);
    dp = scaleDotProduct(dp, layer.scaleFactor);
    layer.out[i] = static_cast<short>(dp);
    layer.pr[i] = static_cast<short>(squash(dp));
  }
}

void Mixer::routeOutputs(const Layer& from, Layer& to, const Block& block) {
  assert(block.count == from.contexts.count);
  assert(block.end() <= to.n);
  short* const dst = &to.tx[block.offset];
  for (size_t i = 0; i < block.count; ++i) {
    dst[i] = from.out[i];
  }
}

void Mixer::routeSkipConnection() {
  const Block& block = combiningLayer.skip;

  // Input output i and middle output i use the same mixer context
  assert(&inputLayer.contexts == &middleLayer.contexts);
  assert(block.count == inputLayer.contexts.count);

  short* const dst = &combiningLayer.tx[block.offset];
  for (size_t i = 0; i < block.count; ++i) {
    // The difference measures how the middle layer changed that signal.
    const int diff = int(inputLayer.out[i]) - int(middleLayer.out[i]);
    dst[i] = static_cast<short>(diff);
  }
}

void Mixer::setScaleFactor(const int sf0, const int sf1, const int sf2) {
  inputLayer.scaleFactor = sf0;
  middleLayer.scaleFactor = sf1;
  combiningLayer.scaleFactor = sf2;
}

void Mixer::setLowerLimitOfLearningRate(const int lr0, const int lr1, const int lr2) {
  assert(lr0 >= MIN_LEARNING_RATE_UNITS && lr0 <= MAX_LEARNING_RATE_UNITS);
  assert(lr1 >= MIN_LEARNING_RATE_UNITS && lr1 <= MAX_LEARNING_RATE_UNITS);
  assert(lr2 >= MIN_LEARNING_RATE_UNITS && lr2 <= MAX_LEARNING_RATE_UNITS);
  inputLayer.lowerLimitOfLearningRate = learningRate(lr0);
  middleLayer.lowerLimitOfLearningRate = learningRate(lr1);
  combiningLayer.lowerLimitOfLearningRate = learningRate(lr2);
}

void Mixer::promote(const int x) {
  assert(x >= -STRETCH_LIMIT && x <= STRETCH_LIMIT);
  assert(numPromoted < numPromotedInputs);

  const short v = static_cast<short>(x);
  middleLayer.tx[middleLayer.promoted.offset + numPromoted] = v;
  combiningLayer.tx[combiningLayer.promoted.offset + numPromoted] = v;
  ++numPromoted;
}

void Mixer::update() {
  // Keep the fixed-point intermediate values within the integer and short
  // ranges used by train().
  static_assert(int64_t(MAX_ERROR) * (MAX_LEARNING_RATE >> 2) <= INT_MAX,
    "error * rate overflows int");
  static_assert((int64_t(MAX_ERROR) * (MAX_LEARNING_RATE >> 2)) >> 16 <= SHRT_MAX,
    "scaled error does not fit a short");

  INJECT_SHARED_y
    const int target = y << 12;

  // Each layer trains its weight vectors selected for the last prediction.
  // The input and middle layers use the same context selections; the combining
  // layer uses its single weight vector.
  for (Layer* const layerPtr : { &inputLayer, & middleLayer, & combiningLayer }) {
    Layer& layer = *layerPtr;
    const ContextSet& ctx = layer.contexts;
    const int rate = layer.rate >> 2;

    for (size_t i = 0; i < ctx.count; ++i) {
      const int err = target - layer.pr[i];
      if (err < -layer.errorLimit || err > layer.errorLimit) {
        train(&layer.tx[0], &layer.wx[ctx.cxt[i] * layer.n], layer.n, (err * rate) >> 16);
      }
    }

    // Decay the learning rate one step per bit until it reaches its floor.
    if (layer.rate > layer.lowerLimitOfLearningRate) {
      layer.rate--;
    }
  }

  reset();
}

int Mixer::p() {
  shared->GetUpdateBroadcaster()->subscribe(this);

  // All per-bit context selections and promoted inputs must be present before
  // computing the final prediction.
  //assert(numAdded == numInputs);
  assert(bitContexts.count == bitContexts.s);
  assert(numPromoted == numPromotedInputs);

  // Input layer -> middle layer -> combining layer, with the input layer's
  // outputs also reaching the combining layer through the skip connection.
  // Promoted inputs were written directly into both later layers.
  forward(inputLayer);
  routeOutputs(inputLayer, middleLayer, middleLayer.routed);

  forward(middleLayer);
  routeOutputs(middleLayer, combiningLayer, combiningLayer.routed);
  routeSkipConnection();

  // The combining layer has one weight vector, selected by its single context.
  singleContext.select(0, 1);
  forward(combiningLayer);

  return combiningLayer.pr[0];
}

void Mixer::add(const int x) {
  assert(x >= -STRETCH_LIMIT && x <= STRETCH_LIMIT);
  assert(numAdded < numInputs);
  inputLayer.tx[numAdded++] = static_cast<short>(x);
}

void Mixer::set(const uint32_t cx, const uint32_t range) {
  // Select one weight vector from the next range belonging to this context.
  bitContexts.select(cx, range);
}

void Mixer::reset() {
  // Start the next bit with fresh input, promotion, and context-selection state.
  numAdded = 0;
  numPromoted = 0;
  bitContexts.reset();
  singleContext.reset();
}
