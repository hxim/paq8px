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
  void train(short* w, const size_t n, const int e) override;
  void initSecondLayer(int promoted) override;
};

#endif // ARM_NEON_AVAILABLE
