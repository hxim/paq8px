#pragma once

#include "Mixer.hpp"
#include "SystemDefines.hpp"

#ifdef X64_SIMD_AVAILABLE

class Mixer_AVX2 : public Mixer
{
public:
  Mixer_AVX2(const Shared* sh, int n, int m, int s, int promoted);
protected:
  int  dotProduct(const short* w, const size_t n) override;
  void train(short* w, const size_t n, const int e) override;
  void initSecondLayer(int promoted) override;
};

#endif // X64_SIMD_AVAILABLE
