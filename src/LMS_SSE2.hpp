#pragma once

#include "LMS.hpp"

#ifdef X64_SIMD_AVAILABLE

class LMS_SSE2 : public LMS
{
public:
  LMS_SSE2(const int s, const int d, const float sameChannelRate, const float otherChannelRate) :
    LMS(s, d, sameChannelRate, otherChannelRate) {
  }

  virtual float predict(int sample) override;

  virtual void update(int sample) override;
};

#endif
