#include <cstdint>
#include "BlockType.hpp"

bool hasRecursion(BlockType ft) {
  return ft == BlockType::CD || ft == BlockType::ZLIB || ft == BlockType::BASE64 || ft == BlockType::GIF || ft == BlockType::RLE || ft == BlockType::LZW;
}

bool hasInfo(BlockType ft) {
  return ft == BlockType::IMAGE1 || ft == BlockType::IMAGE4 || ft == BlockType::IMAGE8 || ft == BlockType::IMAGE8GRAY || ft == BlockType::IMAGE24 || ft == BlockType::IMAGE32 || ft == BlockType::AUDIO ||
    ft == BlockType::AUDIO_LE || ft == BlockType::PNG8 || ft == BlockType::PNG8GRAY || ft == BlockType::PNG24 || ft == BlockType::PNG32 || ft == BlockType::MRB || ft == BlockType::DBF || ft == BlockType::EXE;
}

bool hasTransform(BlockType ft, int info) {
  if (ft == BlockType::MRB) {
    uint8_t packingMethod = (info >> 24) & 3; //0..3
    return packingMethod != 0; //0: uncompressed, 1: rle encoded
  }
  return ft == BlockType::IMAGE24 || ft == BlockType::IMAGE32 || ft == BlockType::AUDIO_LE || ft == BlockType::EXE || ft == BlockType::CD || ft == BlockType::ZLIB || ft == BlockType::BASE64 || ft == BlockType::GIF ||
    ft == BlockType::TEXT_EOL || ft == BlockType::RLE || ft == BlockType::LZW || ft == BlockType::DEC_ALPHA;
}

bool isPNG(BlockType ft) { return ft == BlockType::PNG8 || ft == BlockType::PNG8GRAY || ft == BlockType::PNG24 || ft == BlockType::PNG32; }

bool isTEXT(BlockType ft) { return ft == BlockType::TEXT || ft == BlockType::TEXT_EOL || ft == BlockType::DBF; }
