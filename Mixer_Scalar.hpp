#pragma once

#include "Mixer.hpp"

class Mixer_Scalar : public Mixer
{
public:
  Mixer_Scalar(const Shared* sh, int n, int m, int s, int promoted);
protected:
  int  dotProduct(const short* w, const size_t n) override;
  int  dotProduct2(const short* w0, const short* w1, const size_t n, int& sum1) override;
  void train(short* w, const size_t n, const int e) override;
};
