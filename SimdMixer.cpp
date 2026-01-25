#include "SimdMixer.hpp"

#include "BitCount.hpp"
#include "Squash.hpp"

//the compiler needs to see these implementations for inlining + optimization
#include "MixerFunctions_SIMD_None.hpp"
#include "MixerFunctions_SIMD_SSE2.hpp"
#include "MixerFunctions_SIMD_AVX2.hpp"
#include "MixerFunctions_SIMD_AVX512.hpp"
#include "MixerFunctions_SIMD_Neon.hpp"

ALWAYS_INLINE
static int scaleDotProduct(int dp, int scaleFactor) {
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
static void addDotProductToNextMixer(Mixer* mp, int dp) {
  mp->add(dp);
}

ALWAYS_INLINE
static int processDotProduct(Mixer* mp, int dp, int scaleFactor) {
  dp = scaleDotProduct(dp, scaleFactor);
  dp = clipDotProduct(dp);
  addDotProductToNextMixer(mp, dp);
  const int pr = squash(dp);
  return pr;
}

[[gnu::cold]] [[gnu::noinline]]
static int updatelearning_rate_Adaptive(ErrorInfo& info, int rate, const int err) {
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
  if (isAdaptiveLearningRate)
    rate = updatelearning_rate_Adaptive(info, rate, err);
  //linear learning rate decay
  if (rate > lowerLimitOfLearningRate)
    rate--;
  return rate;
}

/**
  * Define SIMD padding requirements.
  */
template<SIMDType simd>
constexpr int SIMDMixer<simd>::simdWidth() const {
  if (simd == SIMDType::SIMD_AVX512) {
    return 64 / sizeof(short); // 512 bit (64 byte) data size
  }
  else if (simd == SIMDType::SIMD_AVX2) {
    return 32 / sizeof(short); // 256 bit (32 byte) data size
  }
  else if (simd == SIMDType::SIMD_SSE2 || simd == SIMDType::SIMD_NEON) {
    return 16 / sizeof(short); // 128 bit (16 byte) data size
  }
  else if (simd == SIMDType::SIMD_NONE) {
    return 4 / sizeof(short); // Processes 2 shorts at once -> width is 4 bytes
  }
  else {
    static_assert("Unknown SIMD parameter");
  }
}

template<SIMDType simd>
SIMDMixer<simd>::SIMDMixer(const Shared* const sh, const int n, const int m, const int s, const int promoted) :
  Mixer(sh, ((n + (simdWidth() - 1)) & -(simdWidth())), m, s) {
  assert((this->n & (simdWidth() - 1)) == 0);
  assert(this->m > 0);
  assert(this->s > 0);
  mp = (s > 1) ? new SIMDMixer<simd>(sh, s + promoted, 1, 1, 0) : nullptr;
}

template<SIMDType simd>
SIMDMixer<simd>::~SIMDMixer() {
  delete mp;
}

template<SIMDType simd>
void SIMDMixer<simd>::setScaleFactor(const int sf0, const int sf1) {
  scaleFactor = sf0;
  if (mp) {
    mp->setScaleFactor(sf1, 0);
  }
}

template<SIMDType simd>
void SIMDMixer<simd>::promote(int x) {
  if (mp != nullptr)
    mp->add(x);
}

/**
  * Adjust weights to minimize coding cost of last prediction.
  * Trains the network where the expected output is the last bit (in the shared variable y).
  */
template<SIMDType simd>
void SIMDMixer<simd>::update() {
  INJECT_SHARED_y
  const int target = y << 12;
  if (nx > 0) {
    for (uint64_t i = 0; i < numContexts; ++i) {
      int err = target - pr[i];
      const int rate = rates[i] = updateLearningRate(isAdaptiveLearningRate, info[i], rates[i], err, lowerLimitOfLearningRate);
      if (simd == SIMDType::SIMD_NONE) {
        trainSimdNone(&tx[0], &wx[cxt[i] * n], nx, (err * rate) >> 16);
      }
#ifdef X64_SIMD_AVAILABLE
      else if (simd == SIMDType::SIMD_SSE2) {
        trainSimdSse2(&tx[0], &wx[cxt[i] * n], nx, (err * rate) >> 16);
      }
      else if (simd == SIMDType::SIMD_AVX2) {
        trainSimdAvx2(&tx[0], &wx[cxt[i] * n], nx, (err * rate) >> 16);
      }
      else if (simd == SIMDType::SIMD_AVX512) {
        trainSimdAvx512(&tx[0], &wx[cxt[i] * n], nx, (err * rate) >> 16);
      }
#endif
#ifdef ARM_NEON_AVAILABLE
      else if (simd == SIMDType::SIMD_NEON) {
        trainSimdNeon(&tx[0], &wx[cxt[i] * n], nx, (err * rate) >> 16);
      }
#endif
      else {
        static_assert("Unknown SIMD parameter");
      }
    }
  }
  reset();
}

/**
  * Predict next bit
  * @return prediction
  */
template<SIMDType simd>
int SIMDMixer<simd>::p() {
  shared->GetUpdateBroadcaster()->subscribe(this);
  assert(scaleFactor > 0);
  //if(mp)printf("nx: %d, numContexts: %d, base: %d\n",nx, numContexts, base); //for debugging: how many inputs do we have?
  while (nx & (simdWidth() - 1)) {
    tx[nx++] = 0; // pad
  }
  if (mp != nullptr) { // first mixer layer
    for (uint64_t i = 0; i < numContexts; ++i) {
      int dp = 0;
      if (simd == SIMDType::SIMD_NONE) {
        dp = dotProductSimdNone(&tx[0], &wx[cxt[i] * n], nx);
      }
#ifdef X64_SIMD_AVAILABLE
      else if (simd == SIMDType::SIMD_SSE2) {
        dp = dotProductSimdSse2(&tx[0], &wx[cxt[i] * n], nx);
      }
      else if (simd == SIMDType::SIMD_AVX2) {
        dp = dotProductSimdAvx2(&tx[0], &wx[cxt[i] * n], nx);
      }
      else if (simd == SIMDType::SIMD_AVX512) {
        dp = dotProductSimdAvx512(&tx[0], &wx[cxt[i] * n], nx);
      }
#endif
#ifdef ARM_NEON_AVAILABLE
      else if (simd == SIMDType::SIMD_NEON) {
        dp = dotProductSimdNeon(&tx[0], &wx[cxt[i] * n], nx);
      }
#endif
      else {
        static_assert("Unknown SIMD parameter");
      }
      pr[i] = processDotProduct(mp, dp, scaleFactor);
    }
    mp->set(0, 1);
    return mp->p();
  }
  else {  // secont (last) mixer layer
    int dp;
    if (simd == SIMDType::SIMD_NONE) {
      dp = dotProductSimdNone(&tx[0], &wx[cxt[0] * n], nx);
    }
#ifdef X64_SIMD_AVAILABLE
    else if (simd == SIMDType::SIMD_SSE2) {
      dp = dotProductSimdSse2(&tx[0], &wx[cxt[0] * n], nx);
    }
    else if (simd == SIMDType::SIMD_AVX2) {
      dp = dotProductSimdAvx2(&tx[0], &wx[cxt[0] * n], nx);
    }
    else if (simd == SIMDType::SIMD_AVX512) {
      dp = dotProductSimdAvx512(&tx[0], &wx[cxt[0] * n], nx);
    }
#endif
#ifdef ARM_NEON_AVAILABLE
    else if (simd == SIMDType::SIMD_NEON) {
      dp = dotProductSimdNeon(&tx[0], &wx[cxt[0] * n], nx);
    }
#endif
    else {
      static_assert("Unknown SIMD parameter");
    }
    dp = scaleDotProduct(dp, scaleFactor);
    return pr[0] = squash(dp);
  }
}

template class SIMDMixer<SIMDType::SIMD_NONE>;
#ifdef X64_SIMD_AVAILABLE
template class SIMDMixer<SIMDType::SIMD_SSE2>;
template class SIMDMixer<SIMDType::SIMD_AVX2>;
template class SIMDMixer<SIMDType::SIMD_AVX512>;
#endif
#ifdef ARM_NEON_AVAILABLE
template class SIMDMixer<SIMDType::SIMD_NEON>;
#endif