#pragma once

#include "Mixer.hpp"
#include "SystemDefines.hpp"

#ifdef X64_SIMD_AVAILABLE

class Mixer_SSE2 : public Mixer
{
public:
  Mixer_SSE2(const Shared* sh, int n, int m, int s, int promoted);
protected:
  int  dotProduct(const short* w, const size_t n) override;
  int  dotProduct2(const short* w0, const short* w1, const size_t n, int& sum1) override;
  void train(short* w, const size_t n, const int e) override;
};

#endif // X64_SIMD_AVAILABLE
