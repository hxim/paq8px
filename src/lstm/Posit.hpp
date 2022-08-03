#ifndef PAQ8PX_POSIT_HPP
#define PAQ8PX_POSIT_HPP

#include "../Clz.hpp"

template<std::int32_t n, std::int32_t es>
class Posit {
private:
  static std::int32_t constexpr pow2_es = 1 << es, max_exp = pow2_es * (n - 2);
  static std::uint32_t constexpr left_bits(std::uint32_t const value, std::uint32_t const bits) {
    return value & ((~0ul) << (32u - bits));
  }
public:
  static std::uint32_t Encode(float const in) {
    union {
      float f;
      std::uint32_t u;
    } num;
    num.f = in;
    std::int32_t exp = std::max<std::int32_t>(-max_exp, std::min<std::int32_t>(max_exp, static_cast<std::int32_t>((num.u >> 23u) & 0xFFu) - 127));
    num.u = (num.u & 0x807FFFFFu) | (static_cast<std::uint32_t>(exp + 127) << 23u);
    std::int32_t reg = (exp / pow2_es) - (exp % pow2_es < 0);
    std::int32_t rs = std::max<std::int32_t>(-reg + 1, reg + 2);
    exp -= pow2_es * reg;
    std::uint32_t res = num.u << 9u; // fraction
    res = left_bits(exp << (32 - es), es) | (res >> es); // exponent
    res = ((reg < 0) ? 1u << (31 + reg) : left_bits(~0ul, reg + 1)) | (res >> rs); // regime
    res >>= 1u;
    res = ((num.u >> 31u) > 0u) ? left_bits(-static_cast<std::int32_t>(left_bits(res, n)), n) : left_bits(res, n);
    res >>= 32u - n;
    return res;
  }
  static float Decode(std::uint32_t const v) {
    union {
      float f;
      std::uint32_t u;
    } num;
    num.u = v << (32u - n);
    std::uint32_t sign = num.u >> 31u;
    if (sign > 0u)
      num.u = left_bits(-static_cast<std::int32_t>(left_bits(num.u, n)), n);
    std::int32_t lz = clz(num.u << 1u), lo = clz(((~num.u) << 1u));
    std::int32_t rs = std::min<std::int32_t>(n - 1, std::max<std::int32_t>(lz, lo) + 1);
    std::int32_t reg = (lz == 0) ? lo - 1 - ((~num.u) == 0u) : -lz;
    std::int32_t exp = (num.u << (rs + 1)) >> (32 - es);
    exp += pow2_es * reg;
    exp += 127;
    num.u = num.u << (1 + rs + es);
    num.u = (static_cast<std::uint32_t>(exp) << 24u) | (num.u >> 8u);
    num.u = (sign << 31u) | (num.u >> 1u);
    if ((num.u << 1u) == 0u)
      num.u++;
    return num.f;
  }
};

#endif //PAQ8PX_POSIT_HPP
