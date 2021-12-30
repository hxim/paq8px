#pragma once

#include "../Shared.hpp"
#include "../Mixer.hpp"
#include "../APM.hpp"
#include "../IndirectContext.hpp"
#include <valarray>

template <std::size_t Bits = 8>
class LstmModel {
public:
  static constexpr int MIXERINPUTS = 5;
  static constexpr int MIXERCONTEXTS = 8 * 256 + 8 * 100;
  static constexpr int MIXERCONTEXTSETS = 2;
  static constexpr std::size_t Size = 1u << Bits;
protected:  
  const Shared* const shared;
  std::valarray<float> probs;
  APM apm1, apm2, apm3;
  IndirectContext<std::uint16_t> iCtx;
  std::size_t top, mid, bot;
  std::uint8_t expected;
public:
  LstmModel(
    const Shared* const sh,
    std::size_t const num_cells,
    std::size_t const num_layers,
    std::size_t const horizon,
    float const learning_rate,
    float const gradient_clip) :
    shared(sh),
    probs(1.f / Size, Size),
    apm1{ sh, 0x10000u, 24, 255 }, apm2{ sh, 0x800u, 24, 255 }, apm3{ sh, 1024, 24, 255 },
    iCtx{ 11, 1, 9 },
    top(Size - 1), mid(0), bot(0),
    expected(0)
  {}
  virtual ~LstmModel() = default;
  virtual void mix(Mixer& m) = 0;
};
