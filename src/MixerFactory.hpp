#pragma once

#include "Mixer.hpp"
#include "Shared.hpp"
#include "Mixer_Scalar.hpp"
#include "Mixer_SSE2.hpp"
#include "Mixer_AVX2.hpp"
#include "Mixer_AVX512.hpp"
#include "Mixer_Neon.hpp"

class MixerFactory
{
private:
  const Shared* const shared;
public:
  MixerFactory(const Shared* const sh);
  Mixer* createMixer(int n, int m, int s, int promoted) const;
};
