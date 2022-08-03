#include "DmcForest.hpp"

DmcForest::DmcForest(const Shared* const sh, const uint64_t size) : shared(sh), dmcModels(MODELS) {
  for( int i = MODELS - 1; i >= 0; i-- ) {
    dmcModels[i] = new DmcModel(sh, size / dmcMem[i], dmcParams[i]);
  }
}

DmcForest::~DmcForest() {
  for( int i = MODELS - 1; i >= 0; i-- ) {
    delete dmcModels[i];
  }
}

void DmcForest::mix(Mixer &m) {
  int i = MODELS;

  // the slow models predict individually
  m.add(dmcModels[--i]->stw() >> 5);
  m.add(dmcModels[--i]->stw() >> 5);

  // the fast models are combined for stability
  while( i > 0 ) {
    const int st1 = dmcModels[--i]->stw();
    const int st2 = dmcModels[--i]->stw();
    m.add((st1 + st2) >> 6);
  }

  // reset models when their structure can't adapt anymore
  // the two slow models are never reset
  INJECT_SHARED_bpos
  if( bpos == 0 ) {
    for( int j = MODELS - 3; j >= 0; j-- ) {
      if( dmcModels[j]->isFull()) {
        dmcModels[j]->resetStateGraph(dmcParams[j]);
      }
    }
  }
}
