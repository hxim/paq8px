#pragma once

#include "UpdateBroadcaster.hpp"
#include "Mixer.hpp"

/**
 * For training @ref NormalModel, @ref WordModel and @ref ExeModel.
 * Always returns p=0.5 and discards all inputs on update.
 */
class DummyMixer : public Mixer
{
public:
  DummyMixer(const Shared* sh, int n, int m, int s, int promoted);
  void update() override;
  int p() override;
protected:
  int  dotProduct(const short* /*t*/, const short* /*w*/, const size_t /*n*/) override { return 0; }
  int  dotProduct2(const short* /*t*/, const short* /*w0*/, const short* /*w1*/, const size_t /*n*/, int& sum1) override { sum1 = 0; return 0; }
  void train(const short* /*t*/, short* /*w*/, const size_t /*n*/, const int /*e*/) override {}
};
