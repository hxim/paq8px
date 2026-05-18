#pragma once

#include "LMS.hpp"

class LMS_Scalar : public LMS
{
public:
  LMS_Scalar(const int s, const int d, const float sameChannelRate, const float otherChannelRate) :
    LMS(s, d, sameChannelRate, otherChannelRate) {
  }

  virtual float predict(int sample) override;

  virtual void update(int sample) override;
};
