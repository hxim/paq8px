#include "ContextModelBlock.hpp"
#include "../MixerFactory.hpp"
#include "BlockModel.hpp"

ContextModelBlock::ContextModelBlock(Shared* const sh, const MixerFactory* const mf) : shared(sh) {
  m = mf->createMixer(
    1 +  //bias
    BlockModel::MIXERINPUTS
    ,
    BlockModel::MIXERCONTEXTS
    ,
    BlockModel::MIXERCONTEXTSETS
    ,
    0
  );
  m->setScaleFactor(10240, 3072);
}

int ContextModelBlock::p() {

  m->add(256); //network bias

  static BlockModel blockModel{ shared, 32768 };
  blockModel.mix(*m);

  return m->p();
}

ContextModelBlock::~ContextModelBlock() {
  delete m;
}
