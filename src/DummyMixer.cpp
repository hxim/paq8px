#include "DummyMixer.hpp"

DummyMixer::DummyMixer(const Shared* const sh, const int n, const int m, const int s, const int promoted)
  : Mixer(sh, n, m, s, promoted, /*simdWidth=*/1) {
}

void DummyMixer::update() { reset(); }

int DummyMixer::p() {
  shared->GetUpdateBroadcaster()->subscribe(this);
  return 2048;
}
