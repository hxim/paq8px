#include "WordModel.hpp"

#ifndef DISABLE_TEXTMODEL

WordModel::WordModel(Shared* const sh, const uint64_t size) : 
  shared(sh),
  cm(sh, size, nCM1 + nCM2_TEXT, 64),
  infoNormal(sh, cm), infoPdf(sh, cm), pdfTextParserState(0)
{}

void WordModel::reset() {
  infoNormal.reset();
  infoPdf.reset();
}

void WordModel::setParam(int cmScale) {
  cm.setScale(cmScale);
}

void WordModel::mix(Mixer &m) {
  INJECT_SHARED_bpos
  if( bpos == 0 ) {
    //extract text from pdf
    INJECT_SHARED_c4
    const uint8_t c1 = c4;
    if( c4 == 0x0a42540aU /* "\nBT\n" */) {
      pdfTextParserState = 1; // Begin Text
    } else if( c4 == 0x0a45540aU /* "\nET\n" */) {
      pdfTextParserState = 0;
    } // end Text
    bool doPdfProcess = true;
    if( pdfTextParserState != 0 ) {
      const uint8_t pC = c4 >> 8U;
      if( pC != '\\' ) {
        if( c1 == '[' ) {
          pdfTextParserState |= 2U;
        } //array begins
        else if( c1 == ']' ) {
          pdfTextParserState &= (255 - 2);
        } else if( c1 == '(' ) {
          pdfTextParserState |= 4U;
          doPdfProcess = false;
        } //signal: start text extraction from the next char
        else if( c1 == ')' ) {
          pdfTextParserState &= (255 - 4);
        } //signal: start pdf gap processing
      }
    }

    const bool isPdfText = (pdfTextParserState & 4U) != 0;
    if( isPdfText ) {
      const bool isExtendedChar = false;
      //predict the chars after "(", but the "(" must not be processed
      if( doPdfProcess ) {
        //printf("%c",c1); //debug: print the extracted pdf text
        infoPdf.processChar(isExtendedChar);
      }
      infoPdf.predict(pdfTextParserState);
      infoPdf.lineModelSkip();
    } else {
      INJECT_SHARED_blockType
      const bool isTextBlock = isTEXT(blockType);
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

#endif //DISABLE_TEXTMODEL