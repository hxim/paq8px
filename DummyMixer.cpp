#include "DummyMixer.hpp"

DummyMixer::DummyMixer(const Shared* const sh, const int n, const int m, const int s)
  : Mixer(sh, n, m, s, /*simdWidth=*/1) {
}

void DummyMixer::update() { reset(); }

int DummyMixer::p() {
  shared->GetUpdateBroadcaster()->subscribe(this);
  return 2048;
}

void DummyMixer::initSecondLayer(int /*promoted*/) {
  // DummyMixer never creates a second layer
}
