#pragma once

#include "Mixer.hpp"

class Mixer_Scalar : public Mixer
{
public:
  Mixer_Scalar(const Shared* sh, int n, int m, int s, int promoted);
protected:
  int  dotProduct(const short* t, const short* w, size_t n) override;
  int  dotProduct2(const short* t, const short* w0, const short* w1, size_t n, int& sum1) override;
  void train(const short* t, short* w, size_t n, int e) override;
};
