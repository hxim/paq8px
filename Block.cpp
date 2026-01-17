#include "Block.hpp"

namespace Block {

  void EncodeBlockHeader(Encoder* const encoder, BlockType blockType, uint64_t blockSize, int blockInfo) {
    EncodeBlockType(encoder, blockType);
    EncodeBlockSize(encoder, blockSize);
    if (hasInfo(blockType))
      EncodeInfo(encoder, blockInfo);
    encoder->initContextForBlockModel(blockType, blockInfo);
  }

  uint64_t DecodeBlockHeader(Encoder* const encoder) { //returns blockSize
    BlockType blockType = DecodeBlockType(encoder);
    uint64_t blockSize = DecodeBlockSize(encoder);
    int blockInfo = -1;
    if (hasInfo(blockType))
      blockInfo = DecodeInfo(encoder);

    Shared* shared = encoder->predictorMain->shared;
    shared->State.blockType = blockType;
    shared->State.blockInfo = blockInfo;
    shared->State.blockPos = UINT32_MAX;
    return blockSize;
  }

  void EncodeBlockType(Encoder* const encoder, BlockType blocktype) {
    encoder->setContextForBlockModel(0x10);
    encoder->compressByte(encoder->predictorBlock, uint8_t(blocktype));
    encoder->appendToBlockTypeHistoryForBlockModel(blocktype);
  }

  BlockType DecodeBlockType(Encoder* const encoder) {
    encoder->setContextForBlockModel(0x10);
    BlockType blockType = (BlockType)encoder->decompressByte(encoder->predictorBlock);
    encoder->appendToBlockTypeHistoryForBlockModel(blockType);
    return blockType;
  }

  void EncodeBlockSize(Encoder* const encoder, uint64_t blockSize) {
    encoder->setContextForBlockModel(0x20);
    encoder->compressByte(encoder->predictorBlock, (blockSize >> 24) & 0xFF);
    encoder->setContextForBlockModel(0x21);
    encoder->compressByte(encoder->predictorBlock, (blockSize >> 16) & 0xFF);
    encoder->setContextForBlockModel(0x22);
    encoder->compressByte(encoder->predictorBlock, (blockSize >> 8) & 0xFF);
    encoder->setContextForBlockModel(0x23);
    encoder->compressByte(encoder->predictorBlock, (blockSize) & 0xFF);
  }

  uint64_t DecodeBlockSize(Encoder* const encoder) {
    uint64_t blockSize = 0;
    uint8_t b;
    encoder->setContextForBlockModel(0x20);
    b = encoder->decompressByte(encoder->predictorBlock);
    blockSize = blockSize << 8 | b;
    encoder->setContextForBlockModel(0x21);
    b = encoder->decompressByte(encoder->predictorBlock);
    blockSize = blockSize << 8 | b;
    encoder->setContextForBlockModel(0x22);
    b = encoder->decompressByte(encoder->predictorBlock);
    blockSize = blockSize << 8 | b;
    encoder->setContextForBlockModel(0x23);
    b = encoder->decompressByte(encoder->predictorBlock);
    blockSize = blockSize << 8 | b;
    return blockSize;
  }

  void EncodeInfo(Encoder* const encoder, int info) {
    encoder->setContextForBlockModel(0x30);
    encoder->compressByte(encoder->predictorBlock, (info >> 24) & 0xFF);
    encoder->setContextForBlockModel(0x31);
    encoder->compressByte(encoder->predictorBlock, (info >> 16) & 0xFF);
    encoder->setContextForBlockModel(0x32);
    encoder->compressByte(encoder->predictorBlock, (info >> 8) & 0xFF);
    encoder->setContextForBlockModel(0x33);
    encoder->compressByte(encoder->predictorBlock, (info) & 0xFF);
  }

  int DecodeInfo(Encoder* const encoder) {
    uint32_t info = 0;
    uint8_t b;
    encoder->setContextForBlockModel(0x30);
    b = encoder->decompressByte(encoder->predictorBlock);
    info = info << 8 | b;
    encoder->setContextForBlockModel(0x31);
    b = encoder->decompressByte(encoder->predictorBlock);
    info = info << 8 | b;
    encoder->setContextForBlockModel(0x32);
    b = encoder->decompressByte(encoder->predictorBlock);
    info = info << 8 | b;
    encoder->setContextForBlockModel(0x33);
    b = encoder->decompressByte(encoder->predictorBlock);
    info = info << 8 | b;
    return info;
  }


};
