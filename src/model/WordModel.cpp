#include "WordModel.hpp"

WordModel::WordModel(Shared* const sh, const uint64_t size) : 
  shared(sh),
  cm(sh, size, nCM1 + nCM2_TEXT, 64),
  infoNormal(sh, cm), infoPdf(sh, cm), pdfTextParserState(0)
{}

void WordModel::reset() {
  infoNormal.reset();
  infoPdf.reset();
}

void WordModel::setParam(uint32_t fixedLineLength) {
  infoNormal.fixedLineLength = fixedLineLength;
}

void WordModel::setCmScale(int cmScale) {
  cm.setScale(cmScale);
}

void WordModel::mix(Mixer &m) {
  INJECT_SHARED_bpos
  if( bpos == 0 ) {
    //extract text from pdf
    INJECT_SHARED_c4
    const uint8_t c1 = c4;
    if( c4 == 0x0a42540a /* "\nBT\n" */) {
      pdfTextParserState = 1; // Begin Text
    } else if( c4 == 0x0a45540a /* "\nET\n" */) {
      pdfTextParserState = 0;
    } // end Text
    bool doPdfProcess = true;
    if( pdfTextParserState != 0 ) {
      const uint8_t pC = c4 >> 8;
      if( pC != '\\' ) {
        if( c1 == '[' ) {
          pdfTextParserState |= 2;
        } //array begins
        else if( c1 == ']' ) {
          pdfTextParserState &= (255 - 2);
        } else if( c1 == '(' ) {
          pdfTextParserState |= 4;
          doPdfProcess = false;
        } //signal: start text extraction from the next char
        else if( c1 == ')' ) {
          pdfTextParserState &= (255 - 4);
        } //signal: start pdf gap processing
      }
    }

    INJECT_SHARED_blockType
    const bool isTextBlock = isTEXT(blockType);

    const bool isPdfText = (pdfTextParserState & 4) != 0;
    if( isPdfText ) {
      infoPdf.setParams(isTextBlock);
      //predict the chars after "(", but the "(" must not be processed
      if( doPdfProcess ) {
        //printf("%c",c1); //debug: print the extracted pdf text
        const bool isExtendedChar = false;
        infoPdf.processChar(isExtendedChar);
      }
      infoPdf.predict(pdfTextParserState);
      infoPdf.lineModelSkip();
    } else {
      infoNormal.setParams(isTextBlock);
      const bool isExtendedChar = isTextBlock && c1 >= 128;
      infoNormal.processChar(isExtendedChar);
      infoNormal.predict(pdfTextParserState);
      infoNormal.lineModelPredict();
    }
  }
  cm.mix(m);

  const int order = max(0, cm.order - (nCM1 + nCM2_TEXT - 31)); //0-31
  assert(0 <= order && order <= 31);
  m.set((order >> 1) << 3 | bpos, 16 * 8);
  shared->State.WordModel.order = order;
}
