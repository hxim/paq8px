#pragma once

#include "../MixerFactory.hpp"

class ContextModelBlock {

private:
  Shared* const shared;
  Mixer* m;

public:
  ContextModelBlock(Shared* const sh, const MixerFactory* const mf);
  int p();

  ~ContextModelBlock();

};
