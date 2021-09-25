#ifndef PAQ8PX_BLOCKTYPE_HPP
#define PAQ8PX_BLOCKTYPE_HPP

#include <type_traits>

enum class BlockType {
  DEFAULT = 0,
  FILECONTAINER,
  JPEG,
  HDR,
  IMAGE1,
  IMAGE4,
  IMAGE8,
  IMAGE8GRAY,
  IMAGE24,
  IMAGE32,
  AUDIO,
  AUDIO_LE,
  EXE,
  CD,
  ZLIB,
  BASE64,
  GIF,
  PNG8,
  PNG8GRAY,
  PNG24,
  PNG32,
  TEXT,
  TEXT_EOL,
  RLE,
  LZW,
  DEC_ALPHA,
  MRB,
  Count
};

inline int operator << (BlockType bt, int shift)
{
  using T = std::underlying_type_t <BlockType>;
  return (static_cast<T>(bt) << shift);
}

bool hasRecursion(BlockType ft);

bool hasInfo(BlockType ft);

bool hasTransform(BlockType ft);

bool isPNG(BlockType ft);

bool isTEXT(BlockType ft);

#endif //PAQ8PX_BLOCKTYPE_HPP