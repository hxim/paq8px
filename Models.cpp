#include "Models.hpp"
#include "CharacterNames.hpp"
#include "DummyMixer.hpp"
#include "file/OpenFromMyFolder.hpp"
#include "file/FileDisk.hpp"

/*
 relationship between compression level, shared->mem and NormalModel memory use as an example

 level   shared->mem    NormalModel memory use (shared->mem*32)
 -----   -----------    ----------------------
   1      0.125 MB              4 MB
   2      0.25	MB              8 MB
   3      0.5   MB             16 MB
   4      1.0   MB             32 MB
   5      2.0   MB             64 MB
   6      4.0   MB            128 MB
   7      8.0   MB            256 MB
   8     16.0   MB            512 MB
   9     32.0   MB           1024 MB
  10     64.0   MB           2048 MB
  11    128.0   MB           4096 MB
  12    256.0   MB           8192 MB

*/


Models::Models(Shared* const sh, MixerFactory* mf) : shared(sh), mixerFactory(mf) {}

void Models::trainModelsWhenNeeded() {
  //initiate pre-training
  if (shared->GetOptionTrainTxt()) {
    trainText("english.dic", 3);
    trainText("english.exp", 1);
  }
  if (shared->GetOptionTrainExe()) {
    trainExe();
  }
}

void Models::trainText(const char* const dictionary, int iterations) {
  NormalModel& normalModel = this->normalModel();
  WordModel& wordModel = this->wordModel();
  DummyMixer mDummy(shared,
    NormalModel::MIXERINPUTS + WordModel::MIXERINPUTS_TEXT,
    NormalModel::MIXERCONTEXTS_PRE + WordModel::MIXERCONTEXTS,
    NormalModel::MIXERCONTEXTSETS_PRE + WordModel::MIXERCONTEXTSETS);
  shared->State.blockType = BlockType::TEXT;
  INJECT_SHARED_pos
    INJECT_SHARED_blockPos
    assert(pos == 0 && blockPos == 0);
  FileDisk f;
  printf("Pre-training models with text...");
  OpenFromMyFolder::anotherFile(&f, dictionary);
  int c = 0;
  int trainingByteCount = 0;
  while (iterations-- > 0) {
    f.setpos(0);
    c = SPACE;
    trainingByteCount = 0;
    do {
      trainingByteCount++;
      uint8_t c1 = c == NEW_LINE ? SPACE : c;
      if (c != CARRIAGE_RETURN) {
        for (int bpos = 0; bpos < 8; bpos++) {
          normalModel.mix(mDummy); //update (train) model
#ifndef DISABLE_TEXTMODEL
          wordModel.mix(mDummy); //update (train) model
#endif
          mDummy.p();
          int y = (c1 >> (7 - bpos)) & 1U;
          shared->update(y, false);
        }
      }
      // emulate a space before and after each word/expression
      // reset models in between
      if (c == NEW_LINE) {
        normalModel.reset();
#ifndef DISABLE_TEXTMODEL
        wordModel.reset();
#endif
        for (int bpos = 0; bpos < 8; bpos++) {
          normalModel.mix(mDummy); //update (train) model
#ifndef DISABLE_TEXTMODEL
          wordModel.mix(mDummy); //update (train) model
#endif
          mDummy.p();
          int y = (c1 >> (7 - bpos)) & 1U;
          shared->update(y, false);
        }
      }
    } while ((c = f.getchar()) != EOF);
  }
  normalModel.reset();
#ifndef DISABLE_TEXTMODEL
  wordModel.reset();
#endif
  shared->reset();
  printf(" done [%s, %d bytes]\n", dictionary, trainingByteCount);
  f.close();
}

void Models::trainExe() {
  ExeModel& exeModel = this->exeModel();
  DummyMixer mDummy(shared, ExeModel::MIXERINPUTS, ExeModel::MIXERCONTEXTS, ExeModel::MIXERCONTEXTSETS);
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
    for (uint32_t bpos = 0; bpos < 8; bpos++) {
      exeModel.mix(mDummy); //update (train) model
      mDummy.p();
      int y = (c >> (7 - bpos)) & 1U;
      shared->update(y, false);
    }
  } while ((c = f.getchar()) != EOF);
  printf(" done [%d bytes]\n", trainingByteCount);
  f.close();
  shared->reset();
}

auto Models::normalModel() -> NormalModel & {
  static NormalModel instance {shared, shared->mem * 32};
  return instance;
}

auto Models::dmcForest() -> DmcForest & {
  static DmcForest instance {shared, shared->mem};  /**< Not the actual memory use - see in the model */
  return instance;
}

auto Models::charGroupModel() -> CharGroupModel & {
  static CharGroupModel instance {shared, shared->mem / 2};
  return instance;
}

auto Models::recordModel() -> RecordModel & {
  static RecordModel instance {shared, shared->mem * 2};
  return instance;
}

auto Models::sparseBitModel() -> SparseBitModel& {
  static SparseBitModel instance{ shared, shared->mem / 4 };
  return instance;
}

auto Models::sparseModel() -> SparseModel & {
  static SparseModel instance {shared, shared->mem * 4};
  return instance;
}

auto Models::matchModel() -> MatchModel & {
  static MatchModel instance {shared, shared->mem / 4 /*hashtablesize*/, shared->mem /*mapmemorysize*/ }; /**< Not the actual memory use - see in the model */
  return instance;
}

auto Models::sparseMatchModel() -> SparseMatchModel & {
  static SparseMatchModel instance {shared, shared->mem};
  return instance;
}

auto Models::indirectModel() -> IndirectModel & {
  static IndirectModel instance {shared, shared->mem * 2};
  return instance;
}

auto Models::chartModel() -> ChartModel& {
  static ChartModel instance{ shared, shared->mem * 4 };
  return instance;
}

#ifndef DISABLE_TEXTMODEL

auto Models::textModel() -> TextModel & {
  static TextModel instance {shared, shared->mem * 16};
  return instance;
}

auto Models::wordModel() -> WordModel & {
  static WordModel instance {shared, shared->mem * 16};
  return instance;
}

#else

auto wordModel() -> WordModel & {
      static WordModel instance {};
      return instance;
    }

#endif //DISABLE_TEXTMODEL

auto Models::nestModel() -> NestModel & {
  static NestModel instance {shared, shared->mem};
  return instance;
}

auto Models::xmlModel() -> XMLModel & {
  static XMLModel instance {shared, shared->mem / 4};
  return instance;
}

auto Models::exeModel() -> ExeModel & {
  static ExeModel instance {shared, shared->mem * 4};
  return instance;
}

auto Models::linearPredictionModel() -> LinearPredictionModel & {
  static LinearPredictionModel instance {shared};
  return instance;
}

auto Models::jpegModel() -> JpegModel & {
  static JpegModel instance {shared, mixerFactory, shared->mem}; /**< Not the actual memory use - see in the model */
  return instance;
}

auto Models::image24BitModel() -> Image24BitModel & {
  static Image24BitModel instance {shared, shared->mem * 4};
  return instance;
}

auto Models::image8BitModel() -> Image8BitModel & {
  static Image8BitModel instance {shared, shared->mem * 4};
  return instance;
}

auto Models::image4BitModel() -> Image4BitModel & {
  static Image4BitModel instance {shared, shared->mem / 2};
  return instance;
}

auto Models::image1BitModel() -> Image1BitModel & {
  static Image1BitModel instance {shared};
  return instance;
}

#ifndef DISABLE_AUDIOMODEL

auto Models::audio8BitModel() -> Audio8BitModel & {
  static Audio8BitModel instance {shared};
  return instance;
}

auto Models::audio16BitModel() -> Audio16BitModel & {
  static Audio16BitModel instance {shared};
  return instance;
}

#endif //DISABLE_AUDIOMODEL

auto Models::lstmModel() -> LstmModel<> & {
  static LstmModel<>* instance = LstmFactory<>::CreateLSTM(shared, 200, 2, 100, 0.06f, 16.f);
  return *instance;
}

auto Models::decAlphaModel() -> DECAlphaModel & {
  static DECAlphaModel instance{ shared };
  return instance;
}
