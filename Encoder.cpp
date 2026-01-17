#include "Encoder.hpp"

Encoder::Encoder(Predictor* predictorBlock, Predictor* predictorMain, bool doEncoding, Mode m, File *f) :
  doEncoding(doEncoding), 
  ari(f), 
  mode(m), 
  archive(f), 
  alt(nullptr),
  predictorBlock(predictorBlock),
  predictorMain(predictorMain)
{
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
  if( doEncoding && mode == DECOMPRESS ) { // x = first 4 bytes of archive
    ari.prefetch();
  }
}

Mode Encoder::getMode() const { return mode; }

uint64_t Encoder::size() const { return archive->curPos(); }

void Encoder::flush() {
  if(doEncoding && mode == COMPRESS) {
    ari.flush();
  }
}

void Encoder::setFile(File *f) { alt = f; }

void Encoder::compressByte(Predictor *predictor, uint8_t c) {
  assert(mode == COMPRESS);
  if (!doEncoding) {
    archive->putChar(c);
  } else {
    for( int i = 7; i >= 0; --i ) {
      uint32_t p = predictor->p();
      int y = (c >> i) & 1;
      ari.encodeBit(p, y);
      updateModels(predictor, p, y);
    }
    assert(predictor->shared->State.c1 == c);
  }
}

uint8_t Encoder::decompressByte(Predictor *predictor) {
  if( mode == COMPRESS ) {
    assert(alt);
    return alt->getchar();
  }
  if (!doEncoding) {
    return archive->getchar();
  } else {
    for( int i = 0; i < 8; ++i ) {
      int p = predictor->p();
      int y = ari.decodeBit(p);
      updateModels(predictor, p, y);
    }
    return predictor->shared->State.c1;
  }
}

void Encoder::initContextForBlockModel(BlockType blockType, int blockInfo) {
  if (doEncoding) {
    Shared* shared = predictorMain->shared;
    shared->State.blockType = blockType;
    shared->State.blockInfo = blockInfo;
    shared->State.blockPos = UINT32_MAX;
  }
}

void Encoder::setContextForBlockModel(uint8_t context) {
  if (doEncoding)
    predictorBlock->shared->State.blockStateID = context;
}

void Encoder::appendToBlockTypeHistoryForBlockModel(BlockType blockType) {
  if (doEncoding) {
    predictorBlock->shared->State.blockTypeHistory <<= 8;
    predictorBlock->shared->State.blockTypeHistory |= (uint8_t)blockType;
  }
}

void Encoder::updateModels(Predictor* predictor, uint32_t p, int y) {
  bool isMissed = ((p >> (ArithmeticEncoder::PRECISION - 1)) != y);
  predictor->shared->update(y, isMissed);
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
