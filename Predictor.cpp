#include "Predictor.hpp"
#include "ArithmeticEncoder.hpp"

Predictor::Predictor(Shared* const sh) : shared(sh), sse(sh), pr(1<<(ArithmeticEncoder::PRECISION-1)) {
  shared->reset();
  models = new Models(sh);
  contextModel = new ContextModel(sh, models);
  //initiate pre-training
  if((shared->options & OPTION_TRAINTXT) != 0U ) {
    trainText("english.dic", 3);
    trainText("english.exp", 1);
  }
  if((shared->options & OPTION_TRAINEXE) != 0U ) {
    trainExe();
  }
}

Predictor::~Predictor() {
  delete models;
  delete contextModel;
}

auto Predictor::p() -> int { 
  // predict
  pr = contextModel->p();
  // SSE Stage
  pr = sse.p(pr);
  return pr;
}

void Predictor::update(uint8_t y) {
  // update global context: y, pos, bitPosition, c0, c4, c8, buf
  shared->update(y);
  shared->State.misses = shared->State.misses << 1 | ((pr >> (ArithmeticEncoder::PRECISION-1)) != y);
}

void Predictor::trainText(const char *const dictionary, int iterations) {
  NormalModel &normalModel = models->normalModel();
  WordModel &wordModel = models->wordModel();
  DummyMixer mDummy(shared, 
    NormalModel::MIXERINPUTS + WordModel::MIXERINPUTS, 
    NormalModel::MIXERCONTEXTS_PRE + WordModel::MIXERCONTEXTS,
    NormalModel::MIXERCONTEXTSETS_PRE + WordModel::MIXERCONTEXTSETS);
  shared->State.blockType = TEXT;
  INJECT_SHARED_pos
  INJECT_SHARED_blockPos
  assert(pos == 0 && blockPos == 0);
  FileDisk f;
  printf("Pre-training models with text...");
  OpenFromMyFolder::anotherFile(&f, dictionary);
  int c = 0;
  int trainingByteCount = 0;
  while( iterations-- > 0 ) {
    f.setpos(0);
    c = SPACE;
    trainingByteCount = 0;
    do {
      trainingByteCount++;
      uint8_t c1 = c == NEW_LINE ? SPACE : c;
      if( c != CARRIAGE_RETURN ) {
        for( int bpos = 0; bpos < 8; bpos++ ) {
          normalModel.mix(mDummy); //update (train) model
#ifndef DISABLE_TEXTMODEL
          wordModel.mix(mDummy); //update (train) model
#endif
          mDummy.p();
          int y = (c1 >> (7 - bpos)) & 1U;
          shared->update(y);
        }
      }
      // emulate a space before and after each word/expression
      // reset models in between
      if( c == NEW_LINE ) {
        normalModel.reset();
#ifndef DISABLE_TEXTMODEL
        wordModel.reset();
#endif
        for( int bpos = 0; bpos < 8; bpos++ ) {
          normalModel.mix(mDummy); //update (train) model
#ifndef DISABLE_TEXTMODEL
          wordModel.mix(mDummy); //update (train) model
#endif
          mDummy.p();
          int y = (c1 >> (7 - bpos)) & 1U;
          shared->update(y);
        }
      }
    } while((c = f.getchar()) != EOF);
  }
  normalModel.reset();
#ifndef DISABLE_TEXTMODEL
  wordModel.reset();
#endif
  shared->reset();
  printf(" done [%s, %d bytes]\n", dictionary, trainingByteCount);
  f.close();
}

void Predictor::trainExe() {
  ExeModel &exeModel = models->exeModel();
  DummyMixer dummyM(shared, ExeModel::MIXERINPUTS, ExeModel::MIXERCONTEXTS, ExeModel::MIXERCONTEXTSETS);
  INJECT_SHARED_pos
  INJECT_SHARED_blockPos
  assert(pos == 0 && blockPos == 0);
  FileDisk f;
  printf("Pre-training x86/x64 model...");
  OpenFromMyFolder::myself(&f);
  int c = 0;
  int trainingByteCount = 0;
  do {
    trainingByteCount++;
    for( uint32_t bpos = 0; bpos < 8; bpos++ ) {
      exeModel.mix(dummyM); //update (train) model
      dummyM.p();
      int y = (c >> (7 - bpos)) & 1U;
      shared->update(y);
    }
  } while((c = f.getchar()) != EOF);
  printf(" done [%d bytes]\n", trainingByteCount);
  f.close();
  shared->reset();
}