#include "APMPost.hpp"
#include "ArithmeticEncoder.hpp"
#include "UpdateBroadcaster.hpp"

APMPost::APMPost(const Shared* const sh, const uint32_t n) : shared(sh), index(0), n(n), t(n* UINT64_C(4096)) {
  assert(n > 0);
  for(size_t i = 0; i < n; ++i ) {
    for(uint64_t j = 0; j < 4096; ++j ) {
      t[i * 4096 + j] = ((4096 - j) >> 1) << 32 | (j >> 1);
    }
  }
}


uint32_t APMPost::p(const uint32_t pr, const uint32_t cxt) {
  shared->GetUpdateBroadcaster()->subscribe(this);
  assert(pr < 4096 && cxt < n);
  index = cxt * 4096 + pr;
  uint64_t n0 = t[index] >> 32;
  uint64_t n1 = t[index] &0xffffffff;
  n0 = n0 * 2 + 1;
  n1 = n1 * 2 + 1;
  constexpr int PRECISION = ArithmeticEncoder::PRECISION;
  return static_cast<uint32_t>((n1 << PRECISION) / (n0+n1));
}

void APMPost::update() {
  INJECT_SHARED_y
  uint64_t n0, n1, value;
  value = t[index];
  n0 = value >> 32;
  n1 = value & 0xffffffff;

  n0 += 1 - y;
  n1 += y;
  int shift = (n0 | n1) >> 32; // shift: 0 or 1
  n0 >>= shift;
  n1 >>= shift;

  t[index] = n0 << 32 | n1;
}
