#pragma once

#include "Mixer.hpp"

class Mixer_Scalar : public Mixer
{
public:
  Mixer_Scalar(const Shared* sh, int n, int m, int s, int promoted);
protected:
  int  dotProduct(const short* w, const size_t n) override;
  void train(short* w, const size_t n, int e) override;
  void initSecondLayer(int promoted) override;
};
