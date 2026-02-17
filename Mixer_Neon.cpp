#include "Mixer_Neon.hpp"

#ifdef ARM_NEON_AVAILABLE

static constexpr int SIMD_WIDTH_NEON = 16 / sizeof(short); // 8 shorts per 128-bit lane

// NEON helpers emulating the x86 intrinsics used in the algorithm
#if (defined(__GNUC__) || defined(__clang__))
static inline int32x4_t neon_mulhi_epi16(int32x4_t a, int32x4_t b) {
  int32x4_t rl = vmull_s16(vget_low_s16(vreinterpretq_s16_s32(a)), vget_low_s16(vreinterpretq_s16_s32(b)));
  int32x4_t rh = vmull_s16(vget_high_s16(vreinterpretq_s16_s32(a)), vget_high_s16(vreinterpretq_s16_s32(b)));
  uint16x8x2_t r = vuzpq_u16(vreinterpretq_u16_s32(rl), vreinterpretq_u16_s32(rh));
  return vreinterpretq_s32_u16(r.val[1]);
}

static inline int32x4_t neon_madd_epi16(int32x4_t a, int32x4_t b) {
  int32x4_t pl = vmull_s16(vget_low_s16(vreinterpretq_s16_s32(a)), vget_low_s16(vreinterpretq_s16_s32(b)));
  int32x4_t ph = vmull_s16(vget_high_s16(vreinterpretq_s16_s32(a)), vget_high_s16(vreinterpretq_s16_s32(b)));
  int32x2_t rl = vpadd_s32(vget_low_s32(pl), vget_high_s32(pl));
  int32x2_t rh = vpadd_s32(vget_low_s32(ph), vget_high_s32(ph));
  return vcombine_s32(rl, rh);
}
#endif

Mixer_Neon::Mixer_Neon(const Shared* const sh, const int n, const int m, const int s, const int promoted)
  : Mixer(sh, n, m, s, SIMD_WIDTH_NEON) {
  initSecondLayer(promoted);
}

void Mixer_Neon::initSecondLayer(const int promoted) {
  if (s > 1) {
    mp = new Mixer_Neon(shared, s + promoted, 1, 1, 0);
  }
}

int Mixer_Neon::dotProduct(const short* const w, const size_t n) {
  int32x4_t sum = vdupq_n_s32(0);

  for (size_t i = 0; i < n; i += 8) {
    int32x4_t tmp = neon_madd_epi16(*(int32x4_t*)&tx[i], *(int32x4_t*)&w[i]);
    tmp = vshrq_n_s32(tmp, 8);
    sum = vaddq_s32(sum, tmp);
  }

  sum = vaddq_s32(sum, vreinterpretq_s32_s8(vextq_s8(vreinterpretq_s8_s32(sum), vdupq_n_s8(0), 8)));
  sum = vaddq_s32(sum, vreinterpretq_s32_s8(vextq_s8(vreinterpretq_s8_s32(sum), vdupq_n_s8(0), 4)));
  return vgetq_lane_s32(sum, 0);
}

void Mixer_Neon::train(short* const w, const size_t n, const int e) {
  const int32x4_t one = vreinterpretq_s32_s16(vdupq_n_s16(1));
  const int32x4_t err = vreinterpretq_s32_s16(vdupq_n_s16(short(e)));

  for (size_t i = 0; i < n; i += 8) {
    int32x4_t tmp = vreinterpretq_s32_s16(vqaddq_s16(vreinterpretq_s16_s32(*(int32x4_t*)&tx[i]), vreinterpretq_s16_s32(*(int32x4_t*)&tx[i])));
    tmp = neon_mulhi_epi16(tmp, err);
    tmp = vreinterpretq_s32_s16(vqaddq_s16(vreinterpretq_s16_s32(tmp), vreinterpretq_s16_s32(one)));
    tmp = vreinterpretq_s32_s16(vshrq_n_s16(vreinterpretq_s16_s32(tmp), 1));
    tmp = vreinterpretq_s32_s16(vqaddq_s16(vreinterpretq_s16_s32(tmp), vreinterpretq_s16_s32(*reinterpret_cast<int32x4_t*>(&w[i]))));
    *reinterpret_cast<int32x4_t*>(&w[i]) = tmp;
  }
}

#endif // ARM_NEON_AVAILABLE
