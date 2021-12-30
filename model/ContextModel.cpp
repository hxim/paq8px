#include "ContextModel.hpp"
#include "ContextModelGeneric.cpp"
#include "ContextModelText.cpp"
#include "ContextModelImage1.cpp"
#include "ContextModelImage4.cpp"
#include "ContextModelImage8.cpp"
#include "ContextModelImage24.cpp"
#include "ContextModelJpeg.cpp"
#include "ContextModelDec.cpp"
#include "ContextModelAudio8.cpp"
#include "ContextModelAudio16.cpp"

ContextModel::ContextModel(Shared* const sh, Models* const models, const MixerFactory* const mf) :
  shared(sh), 
  models(models), 
  mixerFactory(mf)
{}

int ContextModel::p() {
  INJECT_SHARED_bpos
  if( bpos == 0 ) {
    uint32_t& blpos = shared->State.blockPos;
    blpos++;
    if (blpos == 0) {
      
      BlockType blockType = shared->State.blockType;
      int blockInfo = shared->State.blockInfo;

      if (blockType == BlockType::MRB) {
        const uint8_t packingMethod = (blockInfo >> 24) & 3; //0..3
        const uint16_t colorBits = (blockInfo >> 26); //1,4,8
        const int width = (blockInfo >> 12) & 0xfff;
        int widthInBytes;
        if (colorBits == 8) { 
          widthInBytes = ((width + 3) / 4) * 4; 
          blockType = BlockType::IMAGE8;
        }
        else if (colorBits == 4) {
          widthInBytes = ((width + 3) / 4) * 2;
          blockType = BlockType::IMAGE4;
        }
        else if (colorBits == 1) {
          widthInBytes = ((width + 31) / 32) * 4; 
          blockType = BlockType::IMAGE1;
        }
        else
          quit("Unexpected colorBits for MRB");
        blockInfo = widthInBytes;
      }
      else if (blockType == BlockType::DBF) {
        RecordModel& recordModel = models->recordModel();
        uint32_t fixedRecordLenght = blockInfo;
        recordModel.setParam(fixedRecordLenght);
      }
      else if (blockType == BlockType::DEC_ALPHA) {
        RecordModel& recordModel = models->recordModel();
        uint32_t fixedRecordLenght = 16;
        recordModel.setParam(fixedRecordLenght);
      }
      else {
        RecordModel& recordModel = models->recordModel();
        recordModel.setParam(0); //enable automatic record length detection
      }

      bool isText = isTEXT(blockType);
      TextModel& textModel = models->textModel();
      textModel.setParam(isText ? 74 : 64);
      WordModel& wordModel = models->wordModel();
      wordModel.setParam(isText ? 74 : 64);

      switch (blockType) {

        case BlockType::IMAGE1: {
          static ContextModelImage1 contextModelImage1{ shared, models, mixerFactory};
          int width = blockInfo;
          contextModelImage1.setParam(width);
          selectedContextModel = &contextModelImage1;
          break;
        }

        case BlockType::IMAGE4: {
          static ContextModelImage4 contextModelImage4{ shared, models, mixerFactory };
          int width = blockInfo;
          contextModelImage4.setParam(width);
          selectedContextModel = &contextModelImage4;
          break;
        }

        case BlockType::IMAGE8:
        case BlockType::PNG8:
        case BlockType::IMAGE8GRAY:
        case BlockType::PNG8GRAY: {
          static ContextModelImage8 contextModelImage8{ shared, models, mixerFactory };
          int isGray = blockType == BlockType::IMAGE8GRAY || blockType == BlockType::PNG8GRAY;
          int isPNG = blockType == BlockType::PNG8 || blockType == BlockType::PNG8GRAY;
          int width = blockInfo & 0xffffff;
          contextModelImage8.setParam(width, isGray, isPNG);
          selectedContextModel = &contextModelImage8;
          break;
        }

        case BlockType::IMAGE24:
        case BlockType::PNG24:
        case BlockType::IMAGE32:
        case BlockType::PNG32: {
          static ContextModelImage24 contextModelImage24{ shared, models, mixerFactory };
          int isAlpha = blockType == BlockType::IMAGE32 || blockType == BlockType::PNG32;
          int isPNG = blockType == BlockType::PNG24 || blockType == BlockType::PNG32;
          int width = blockInfo & 0xffffff;
          contextModelImage24.setParam(width, isAlpha, isPNG);
          selectedContextModel = &contextModelImage24;
          break;
        }

  #ifndef DISABLE_AUDIOMODEL
        case BlockType::AUDIO:
        case BlockType::AUDIO_LE: {
          if ((blockInfo & 2) == 0) {
            static ContextModelAudio8 contextModelAudio8{ shared, models, mixerFactory };
            contextModelAudio8.setParam(blockInfo);
            selectedContextModel = &contextModelAudio8;
            break;
          }
          else {
            static ContextModelAudio16 contextModelAudio16{ shared, models, mixerFactory };
            contextModelAudio16.setParam(blockInfo);
            selectedContextModel = &contextModelAudio16;
            break;
          }
        }
  #endif //DISABLE_AUDIOMODEL

        case BlockType::JPEG: {
          static ContextModelJpeg contextModelJpeg{ shared, models, mixerFactory };
          selectedContextModel = &contextModelJpeg;
          break;
        }

        case BlockType::DEC_ALPHA: {
          static ContextModelDec contextModelDec{ shared, models, mixerFactory };
          selectedContextModel = &contextModelDec;
          break;
        }

        case BlockType::TEXT:
        case BlockType::TEXT_EOL:
        case BlockType::DBF: {
          static ContextModelText contextModelText{ shared, models, mixerFactory };
          selectedContextModel = &contextModelText;
          break;
        }

        default: {
          static ContextModelGeneric contextModelGeneric{ shared, models, mixerFactory };
          selectedContextModel = &contextModelGeneric;
          break;
        }
      }
    }
  }

  return selectedContextModel->p();

}
