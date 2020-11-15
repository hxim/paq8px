#include "Models.hpp"

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


Models::Models(Shared* const sh) : shared(sh) {}

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
  static SparseModel instance {shared, shared->mem * 2};
  return instance;
}

auto Models::matchModel() -> MatchModel & {
  static MatchModel instance {shared, shared->mem * 4 /*buffermemorysize*/, shared->mem / 32 /*mapmemorysize*/ };
  return instance;
}

auto Models::sparseMatchModel() -> SparseMatchModel & {
  static SparseMatchModel instance {shared, shared->mem};
  return instance;
}

auto Models::indirectModel() -> IndirectModel & {
  static IndirectModel instance {shared, shared->mem};
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
  static JpegModel instance {shared, shared->mem}; /**< Not the actual memory use - see in the model */
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
