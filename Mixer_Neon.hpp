#pragma once

#include "Mixer.hpp"
#include "SystemDefines.hpp"

#ifdef ARM_NEON_AVAILABLE

class Mixer_Neon : public Mixer
{
public:
  Mixer_Neon(const Shared* sh, int n, int m, int s, int promoted);
protected:
  int  dotProduct(const short* w, const size_t n) override;
  int  dotProduct2(const short* w0, const short* w1, const size_t n, int& sum1) override;
  void train(short* w, const size_t n, const int e) override;
};

#endif // ARM_NEON_AVAILABLE
