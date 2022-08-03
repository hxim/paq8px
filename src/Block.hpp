#pragma once

#include "Shared.hpp"
#include "Encoder.hpp"
#include "BlockType.hpp"

namespace Block {

  void EncodeBlockHeader(Encoder* const encoder, BlockType blockType, uint64_t blockSize, int blockInfo);
  uint64_t DecodeBlockHeader(Encoder* const encoder);
  void EncodeBlockSize(Encoder* const encoder, uint64_t blockSize);
  uint64_t DecodeBlockSize(Encoder* const encoder);
  void EncodeBlockType(Encoder* const encoder, BlockType blocktype);
  BlockType DecodeBlockType(Encoder* const encoder);
  void EncodeInfo(Encoder* const encoder, int info);
  int DecodeInfo(Encoder* const encoder);
};
