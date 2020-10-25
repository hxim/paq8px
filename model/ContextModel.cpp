#include "ContextModel.hpp"

ContextModel::ContextModel(Shared* const sh, Models &models) : shared(sh), models(models) {
  auto mf = new MixerFactory();
  m = mf->createMixer(
    // this is the maximum case: how many mixer inputs, mixer contexts and mixer context sets are needed (max)
    sh,
    1 +  //bias
    MatchModel::MIXERINPUTS + NormalModel::MIXERINPUTS + SparseMatchModel::MIXERINPUTS +
    SparseModel::MIXERINPUTS + RecordModel::MIXERINPUTS + CharGroupModel::MIXERINPUTS +
    TextModel::MIXERINPUTS + WordModel::MIXERINPUTS + IndirectModel::MIXERINPUTS +
    DmcForest::MIXERINPUTS + NestModel::MIXERINPUTS + XMLModel::MIXERINPUTS +
    LinearPredictionModel::MIXERINPUTS + ExeModel::MIXERINPUTS + LstmModel<>::MIXERINPUTS +
    DECAlphaModel::MIXERINPUTS
    ,
    MatchModel::MIXERCONTEXTS + NormalModel::MIXERCONTEXTS + SparseMatchModel::MIXERCONTEXTS +
    SparseModel::MIXERCONTEXTS + RecordModel::MIXERCONTEXTS + CharGroupModel::MIXERCONTEXTS +
    TextModel::MIXERCONTEXTS + WordModel::MIXERCONTEXTS + IndirectModel::MIXERCONTEXTS +
    DmcForest::MIXERCONTEXTS + NestModel::MIXERCONTEXTS + XMLModel::MIXERCONTEXTS +
    LinearPredictionModel::MIXERCONTEXTS + ExeModel::MIXERCONTEXTS + LstmModel<>::MIXERCONTEXTS +
    DECAlphaModel::MIXERCONTEXTS
    ,
    MatchModel::MIXERCONTEXTSETS + NormalModel::MIXERCONTEXTSETS + SparseMatchModel::MIXERCONTEXTSETS +
    SparseModel::MIXERCONTEXTSETS + RecordModel::MIXERCONTEXTSETS + CharGroupModel::MIXERCONTEXTSETS +
    TextModel::MIXERCONTEXTSETS + WordModel::MIXERCONTEXTSETS + IndirectModel::MIXERCONTEXTSETS +
    DmcForest::MIXERCONTEXTSETS + NestModel::MIXERCONTEXTSETS + XMLModel::MIXERCONTEXTSETS +
    LinearPredictionModel::MIXERCONTEXTSETS + ExeModel::MIXERCONTEXTSETS + +LstmModel<>::MIXERCONTEXTSETS +
    DECAlphaModel::MIXERCONTEXTSETS
  );
}

auto ContextModel::p() -> int {
  uint32_t &blpos = shared->State.blockPos;
  // Parse block type and block size
  INJECT_SHARED_bpos
  if( bpos == 0 ) {
    --blockSize;
    blpos++;
    INJECT_SHARED_c1
    if( blockSize == -1 ) {
      nextBlockType = static_cast<BlockType>(c1); //got blockType but don't switch (we don't have all the info yet)
      bytesRead = 0;
      readSize = true;
    } else if( blockSize < 0 ) {
      if( readSize ) {
        bytesRead |= int(c1 & 0x7FU) << ((-blockSize - 2) * 7);
        if((c1 >> 7U) == 0 ) {
          readSize = false;
          if( !hasInfo(nextBlockType)) {
            blockSize = bytesRead;
            if( hasRecursion(nextBlockType)) {
              blockSize = 0;
            }
            blpos = 0;
          } else {
            blockSize = -1;
          }
        }
      } else if( blockSize == -5 ) {
        INJECT_SHARED_c4
        blockSize = bytesRead;
        blockInfo = c4;
        blpos = 0;
      }
    }

    if(blpos == 0 ) {
      blockType = nextBlockType; //got all the info - switch to next blockType
      shared->State.blockType = blockType;
    }
    if( blockSize == 0 ) {
      blockType = DEFAULT;
      shared->State.blockType = blockType;
    }
  }

  m->add(256); //network bias

  MatchModel &matchModel = models.matchModel();
  matchModel.mix(*m);
  NormalModel &normalModel = models.normalModel();
  normalModel.mix(*m);
  if ((shared->options & OPTION_LSTM) != 0u) {
    LstmModel<>& lstmModel = models.lstmModel();
    lstmModel.mix(*m);
  }

  // Test for special block types
  switch( blockType ) {
    case IMAGE1: {
      Image1BitModel &image1BitModel = models.image1BitModel();
      image1BitModel.setParam(blockInfo);
      image1BitModel.mix(*m);
      break;
    }
    case IMAGE4: {
      Image4BitModel &image4BitModel = models.image4BitModel();
      image4BitModel.setParam(blockInfo);
      image4BitModel.mix(*m);
      m->setScaleFactor(2048, 256);
      return m->p();
    }
    case IMAGE8: {
      Image8BitModel &image8BitModel = models.image8BitModel();
      image8BitModel.setParam(blockInfo, 0, 0);
      image8BitModel.mix(*m);
      m->setScaleFactor(1600, 110); // 1500-1800, 100-128
      return m->p();
    }
    case IMAGE8GRAY: {
      Image8BitModel &image8BitModel = models.image8BitModel();
      image8BitModel.setParam(blockInfo, 1, 0);
      image8BitModel.mix(*m);
      m->setScaleFactor(1300, 100); // 1100-1400, 90-110
      return m->p();
    }
    case IMAGE24: {
      Image24BitModel &image24BitModel = models.image24BitModel();
      image24BitModel.setParam(blockInfo, 0, 0);
      image24BitModel.mix(*m);
      m->setScaleFactor(1024, 100); // 800-1300, 90-110
      return m->p();
    }
    case IMAGE32: {
      Image24BitModel &image24BitModel = models.image24BitModel();
      image24BitModel.setParam(blockInfo, 1, 0);
      image24BitModel.mix(*m);
      m->setScaleFactor(2048, 128);
      return m->p();
    }
    case PNG8: {
      Image8BitModel &image8BitModel = models.image8BitModel();
      image8BitModel.setParam(blockInfo, 0, 1);
      image8BitModel.mix(*m);
      m->setScaleFactor(1600, 110);
      return m->p();
    }
    case PNG8GRAY: {
      Image8BitModel &image8BitModel = models.image8BitModel();
      image8BitModel.setParam(blockInfo, 1, 1);
      image8BitModel.mix(*m);
      m->setScaleFactor(1300, 110);
      return m->p();
    }
    case PNG24: {
      Image24BitModel &image24BitModel = models.image24BitModel();
      image24BitModel.setParam(blockInfo, 0, 1);
      image24BitModel.mix(*m);
      m->setScaleFactor(1024, 100);
      return m->p();
    }
    case PNG32: {
      Image24BitModel &image24BitModel = models.image24BitModel();
      image24BitModel.setParam(blockInfo, 1, 1);
      image24BitModel.mix(*m);
      m->setScaleFactor(2048, 128);
      return m->p();
    }
#ifndef DISABLE_AUDIOMODEL
    case AUDIO:
    case AUDIO_LE: {
      if((blockInfo & 2U) == 0 ) {
        Audio8BitModel &audio8BitModel = models.audio8BitModel();
        audio8BitModel.setParam(blockInfo);
        audio8BitModel.mix(*m);
        m->setScaleFactor(850, 140); //800-900, 140
      }
      else {
        Audio16BitModel& audio16BitModel = models.audio16BitModel();
        audio16BitModel.setParam(blockInfo);
        audio16BitModel.mix(*m);
        m->setScaleFactor(1024, 128);
      }
      RecordModel& recordModel = models.recordModel();
      recordModel.mix(*m);
      return m->p();
    }
#endif //DISABLE_AUDIOMODEL
    case JPEG: {
      JpegModel &jpegModel = models.jpegModel();
      if( jpegModel.mix(*m) != 0 ) {
        m->setScaleFactor(1024, 256); //850 for larger files, 1400 for smaller files - very sensitive
        return m->p();
      }
      break;
    }
    default:
      break;
  }

  normalModel.mixPost(*m);

  if( blockType != IMAGE1 ) {
    SparseMatchModel &sparseMatchModel = models.sparseMatchModel();
    sparseMatchModel.mix(*m);
    SparseModel &sparseModel = models.sparseModel();
    sparseModel.mix(*m);
    RecordModel &recordModel = models.recordModel();
    recordModel.mix(*m);
    CharGroupModel &charGroupModel = models.charGroupModel();
    charGroupModel.mix(*m);
#ifndef DISABLE_TEXTMODEL
    TextModel &textModel = models.textModel();
    textModel.mix(*m);
    WordModel &wordModel = models.wordModel();
    wordModel.mix(*m);
#endif //DISABLE_TEXTMODEL
    IndirectModel &indirectModel = models.indirectModel();
    indirectModel.mix(*m);
    DmcForest &dmcForest = models.dmcForest();
    dmcForest.mix(*m);
    NestModel &nestModel = models.nestModel();
    nestModel.mix(*m);
    XMLModel &xmlModel = models.xmlModel();
    xmlModel.mix(*m);
    if( blockType != TEXT && blockType != TEXT_EOL ) {
      LinearPredictionModel &linearPredictionModel = models.linearPredictionModel();
      linearPredictionModel.mix(*m);
      if (blockType == DEC_ALPHA) {
        DECAlphaModel& decAlphaModel = models.decAlphaModel();
        decAlphaModel.mix(*m);
      }
      else {
        ExeModel& exeModel = models.exeModel();
        exeModel.mix(*m);
      }
    }
  }

  m->setScaleFactor(940, 80); // 900-1024, 70-110
  return m->p();
}

ContextModel::~ContextModel() {
  delete m;
}
