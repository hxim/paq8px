#include "TextParserStateInfo.hpp"

void TextParserStateInfo::reset(uint64_t startPos) {
  Start = startPos;
  End = startPos - 1;
  invalidCount = 0;
  EOLType = 0;
  UTF8State = 0;
}

uint64_t TextParserStateInfo::realLength() const {
  return End - Start + 1;
}

bool TextParserStateInfo::isLargeText() const {
  return realLength() >= TEXT_FRAGMENT_MIN_SIZE;
}

bool TextParserStateInfo::isSmallText() const {
  return realLength() >= TEXT_SMALL_BLOCK_SIZE;
}
