#include "ContextModel.hpp"
#include "ContextModelText.cpp"
#include "ContextModelImage1.cpp"
#include "ContextModelImage4.cpp"
#include "ContextModelImage8.cpp"
#include "ContextModelImage24.cpp"
#include "ContextModelJpeg.cpp"
#include "ContextModelDec.cpp"
#include "ContextModelAudio8.cpp"
#include "ContextModelAudio16.cpp"

ContextModel::ContextModel(Shared* const sh, Models* const models) : shared(sh), models(models) {}

int ContextModel::p() {
  uint32_t &blpos = shared->State.blockPos;
  // Parse block type and block size
  INJECT_SHARED_bpos
  if( bpos == 0 ) {
    --blockSize;
    blpos++;
    INJECT_SHARED_c1
    if( blockSize == -1 ) {
      selectedContextModel = &contextModelGeneric;
      nextBlockType = static_cast<BlockType>(c1); //got blockType but don't switch (we don't have all the info yet)
      bytesRead = 0;
      readSize = true;
    } else if( blockSize < 0 ) {
      selectedContextModel = &contextModelGeneric;
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

    if (blpos == 0) {
      blockType = nextBlockType; //got all the info - switch to next blockType
      shared->State.blockType = blockType;

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
          static ContextModelImage1 contextModelImage1{ shared, models };
          int width = blockInfo;
          contextModelImage1.setParam(width);
          selectedContextModel = &contextModelImage1;
          break;
        }

        case BlockType::IMAGE4: {
          static ContextModelImage4 contextModelImage4{ shared, models };
          int width = blockInfo;
          contextModelImage4.setParam(width);
          selectedContextModel = &contextModelImage4;
          break;
        }

        case BlockType::IMAGE8:
        case BlockType::PNG8:
        case BlockType::IMAGE8GRAY:
        case BlockType::PNG8GRAY: {
          static ContextModelImage8 contextModelImage8{ shared, models };
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
          static ContextModelImage24 contextModelImage24{ shared, models };
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
            static ContextModelAudio8 contextModelAudio8{ shared, models };
            contextModelAudio8.setParam(blockInfo);
            selectedContextModel = &contextModelAudio8;
            break;
          }
          else {
            static ContextModelAudio16 contextModelAudio16{ shared, models };
            contextModelAudio16.setParam(blockInfo);
            selectedContextModel = &contextModelAudio16;
            break;
          }
        }
  #endif //DISABLE_AUDIOMODEL

        case BlockType::JPEG: {
          static ContextModelJpeg contextModelJpeg{ shared, models };
          selectedContextModel = &contextModelJpeg;
          break;
        }

        case BlockType::DEC_ALPHA: {
          static ContextModelDec contextModelDec{ shared, models };
          selectedContextModel = &contextModelDec;
          break;
        }

        case BlockType::TEXT:
        case BlockType::TEXT_EOL:
        case BlockType::DBF: {
          static ContextModelText contextModelText{ shared, models };
          selectedContextModel = &contextModelText;
          break;
        }

        default: {
          selectedContextModel = &contextModelGeneric;
          break;
        }
      }
    }
    if( blockSize == 0 ) {
      blockType = BlockType::DEFAULT;
      shared->State.blockType = blockType;
      selectedContextModel = &contextModelGeneric;
    }
  }

  return selectedContextModel->p();

}
