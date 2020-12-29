#include "Encoder.hpp"


Encoder::Encoder(Shared* const sh, Mode m, File *f) : shared(sh), predictor(sh), ari(f), mode(m), archive(f), alt(nullptr) {
  if( mode == DECOMPRESS ) {
    uint64_t start = size();
    archive->setEnd();
    uint64_t end = size();
    if( end >= (UINT64_C(1) << 31)) {
      quit("Large archives not yet supported.");
    }
    setStatusRange(0.0, static_cast<float>(end));
    archive->setpos(start);
  }
  if( shared->level > 0 && mode == DECOMPRESS ) { // x = first 4 bytes of archive
    ari.prefetch();
  }
}

auto Encoder::getMode() const -> Mode { return mode; }

auto Encoder::size() const -> uint64_t { return archive->curPos(); }

void Encoder::flush() {
  if( mode == COMPRESS && shared->level > 0 ) {
    ari.flush();
  }
}

void Encoder::setFile(File *f) { alt = f; }

void Encoder::compressByte(uint8_t c) {
  assert(mode == COMPRESS);
  if( shared->level == 0 ) {
    archive->putChar(c);
  } else {
    for( int i = 7; i >= 0; --i ) {
      int p = predictor.p();
      int y = (c >> i) & 1;
      ari.encodeBit(p, y);
      predictor.update(y);
    }
  }
}

auto Encoder::decompressByte() -> uint8_t {
  if( mode == COMPRESS ) {
    assert(alt);
    return alt->getchar();
  }
  if( shared->level == 0 ) {
    return archive->getchar();
  } else {
    uint8_t c = 0;
    for( int i = 0; i < 8; ++i ) {
      int p = predictor.p();
      int y = ari.decodeBit(p);
      c = c << 1 | y;
      predictor.update(y);
    }
    return c;
  }
}

void Encoder::encodeBlockSize(uint64_t blockSize) {
  while( blockSize > 0x7FU ) {
    compressByte(0x80 | (blockSize & 0x7FU));
    blockSize >>= 7U;
  }
  compressByte(uint8_t(blockSize));
}

void Encoder::encodeBlockType(BlockType blocktype) {
  compressByte(uint8_t(blocktype));
}

auto Encoder::decodeBlockType() -> BlockType {
  uint8_t b = decompressByte();
  return (BlockType)b;
}

auto Encoder::decodeBlockSize() -> uint64_t {
  uint64_t blockSize = 0;
  uint8_t b = 0;
  int i = 0;
  do {
    b = decompressByte();
    blockSize |= uint64_t((b & 0x7FU) << i);
    i += 7;
  } while((b >> 7U) > 0 );
  return blockSize;
}

void Encoder::encodeInfo(int info) {
  compressByte((info >> 24) & 0xFF);
  compressByte((info >> 16) & 0xFF);
  compressByte((info >> 8) & 0xFF);
  compressByte((info) & 0xFF);
}
auto Encoder::decodeInfo() -> int {
  int info = 0;
  for (int j = 0; j < 4; ++j) {
    info <<= 8;
    info += decompressByte();
  }
  return info;
}

void Encoder::setStatusRange(float perc1, float perc2) {
  p1 = perc1;
  p2 = perc2;
}

void Encoder::printStatus(uint64_t n, uint64_t size) const {
  fprintf(stderr, "%6.2f%%\b\b\b\b\b\b\b", (p1 + (p2 - p1) * n / (size + 1)) * 100);
  fflush(stderr);
}

void Encoder::printStatus() const {
  fprintf(stderr, "%6.2f%%\b\b\b\b\b\b\b", float(size()) / (p2 + 1) * 100);
  fflush(stderr);
}
