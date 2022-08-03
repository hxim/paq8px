#include "WordModelInfo.hpp"
#include "../CharacterNames.hpp"

WordModelInfo::WordModelInfo(Shared* const sh, ContextMap2 &contextmap) : shared(sh), cm(contextmap) {
  reset();
}

void WordModelInfo::setParams(bool isTextBlock) {
  this->isTextBlock = isTextBlock;
}

void WordModelInfo::reset() {
  memset(&wordPositions[0], 0, (1 << wPosBits) * sizeof(uint32_t));
  memset(&checksums[0], 0, (1 << wPosBits) * sizeof(uint16_t));
  c4 = 0;
  c = pC = ppC = 0;
  isLetter = isLetterPc = isLetterPpC = false;
  isTextBlock = isNewline = isNewlinePc = false;
  opened = wordLen0 = wordLen1 = exprLen0 = 0;
  line0 = line1 = firstWord = 0;
  word0 = word1 = word2 = word3 = word4 = 0;
  expr0 = expr1 = expr2 = expr3 = expr4 = 0;
  keyword0 = gapToken0 = gapToken1 = 0;
  w = chk = 0;
  firstChar = lineMatch = nl1 = nl2 = 0;
  groups = text0 = 0;
  lastLetter = lastUpper = wordGap = 0;
  mask = expr0Chars = mask2 = f4 = 0;
  lastUpper = maxLastUpper;
  lastLetter = wordGap = maxLastLetter;
  firstChar = lineMatch = -1;
}

void WordModelInfo::shiftWords() {
  word4 = word3;
  word3 = word2;
  word2 = word1;
  word1 = word0;
  wordLen1 = wordLen0;
}

void WordModelInfo::killWords() {
  word4 = word3 = word2 = word1 = 0;
  gapToken1 = 0; //not really necessary - tiny gain
}

void WordModelInfo::processChar(const bool isExtendedChar) {
  //shift history
  ppC = pC;
  pC = c;
  isLetterPpC = isLetterPc;
  isLetterPc = isLetter;
  INJECT_SHARED_c1
  c4 = c4 << 8 | c1;
  c = c1;
  if( c >= 'A' && c <= 'Z' ) {
    c += 'a' - 'A';
    lastUpper = 0;
  }

  isLetter = (c >= 'a' && c <= 'z') || isExtendedChar;
  const bool isNumber =
          (c >= '0' && c <= '9') || (pC >= '0' && pC <= '9' && c == '.' /* decimal point (english) or thousand separator (misc) */);
  isNewlinePc = isNewline;

  if (fixedLineLength == 0) {
    isNewline = isTextBlock ? c == NEW_LINE || c == 0 : c == 0;
  } else {
    INJECT_SHARED_blockPos
    isNewline = blockPos % fixedLineLength == 0;
  }
  lastUpper = min(lastUpper + 1, maxLastUpper);
  lastLetter = min(lastLetter + 1, maxLastLetter);

  mask2 <<= 8;

  if( isLetter || isNumber ) {
    if( wordLen0 == 0 ) { // the beginning of a new word
      if( pC == NEW_LINE && lastLetter == 3 && ppC == '+' ) {
        // correct hyphenation with "+" (book1 from Calgary)
        // (undo some of the recent context changes)
        word0 = word1;
        word1 = word2;
        word2 = word3;
        word3 = word4 = 0;
        wordLen0 = wordLen1;
      } else {
        wordGap = lastLetter;
        gapToken1 = gapToken0;
        if( pC == QUOTE || (pC == APOSTROPHE && !isLetterPpC)) {
          opened = pC;
        }
      }
      gapToken0 = 0;
      mask2 = 0;
    }
    lastLetter = 0;
    word0 = combine64(word0, c);
    w = static_cast<uint16_t>(finalize64(word0, wPosBits));
    chk = checksum16(word0, wPosBits);
    text0 = (text0 << 8 | c) & 0xffffffffff; // last 5 alphanumeric chars (other chars are ignored)
    wordLen0 = min(wordLen0 + 1, maxWordLen);
    //last letter types
    if( isLetter ) {
      if( c == 'e' ) {
        mask2 |= c; // vowel/e // separating 'a' is also ok
      } else if( c == 'a' || c == 'i' || c == 'o' || c == 'u' ) {
        mask2 |= 'a'; // vowel
      } else if( c >= 'b' && c <= 'z' ) {
        if( c == 'y' ) {
          mask2 |= 'y'; // y
        } else if( pC == 't' && c == 'h' ) {
          mask2 = ((mask2 >> 8) & 0x00ffff00) | 't';
        } // consonant/th
        else {
          mask2 |= 'b'; // consonant
        }
      } else {
        mask2 |= 128; // extended_char: c>=128
      }
    } else { // isNumber
      if( c == '.' ) {
        mask2 |= '.'; // decimal point or thousand separator
      } else {
        mask2 |= c == '0' ? '0' : '1'; // number
      }
    }
  } else { //it's not a letter/number
    gapToken0 = combine64(gapToken0, isNewline ? static_cast<uint8_t>(SPACE) : c1);
    if( isNewline && pC == '+' && isLetterPpC ) {
    } //calgary/book1 hyphenation - don't shift again
    else if( c == '?' || pC == '!' || pC == '.' ) {
      killWords(); //end of sentence (probably)
    } else if( c == pC ) {
    } //don't shift when anything repeats
    else if((c == SPACE || isNewline) && (pC == SPACE || isNewlinePc)) {
    } //ignore repeating whitespace
    else {
      shiftWords();
    }
    if( wordLen0 != 0 ) { //beginning of a new non-word token
      INJECT_SHARED_pos
      wordPositions[w] = pos;
      checksums[w] = chk;
      w = 0;
      chk = 0;

      if( c == ':' || c == '=' ) {
        keyword0 = word0; // enwik, world95.txt, html/xml
      }
      if( firstWord == 0 ) {
        firstWord = word0;
      }

      word0 = 0;
      wordLen0 = 0;
      mask2 = 0;
    }

    if( c1 == '.' || c1 == '!' || c1 == '?' ) {
      mask2 |= '!';
    } else if( c1 == ',' || c1 == ';' || c1 == ':' ) {
      mask2 |= ',';
    } else if( c1 == '(' || c1 == '{' || c1 == '[' || c1 == '<' ) {
      mask2 |= '(';
      opened = c1;
    } else if( c1 == ')' || c1 == '}' || c1 == ']' || c1 == '>' ) {
      mask2 |= ')';
      opened = 0;
    } else if( c1 == QUOTE || c1 == APOSTROPHE ) {
      mask2 |= c1;
      opened = 0;
    } else {
      mask2 |= c1; //0, SPACE, NEW_LINE, /\+=%$- etc.
    }
  }

  //const uint8_t characterGroup = shared->State.Text.characterGroup;
  uint8_t g = c1;
  if( g >= 128 ) {
    //utf8 code points (weak context)
    if((g & 0xf8) == 0xf0 ) {
      g = 1;
    } else if((g & 0xf0) == 0xe0 ) {
      g = 2;
    } else if((g & 0xe0) == 0xc0 ) {
      g = 3;
    } else if((g & 0xc0) == 0x80 ) {
      g = 4;
      //the rest of the values
    } else if( g == 0xff ) {
      g = 5;
    } else {
      g = c1 & 0xf0;
    }
  } else if( g >= '0' && g <= '9' ) {
    g = '0';
  } else if( g >= 'a' && g <= 'z' ) {
    g = 'a';
  } else if( g >= 'A' && g <= 'Z' ) {
    g = 'A';
  } else if( g < 32 && !isNewline ) {
    g = 6;
  }
  groups = groups << 8 | g;

  // Expressions (pure words separated by single spaces)
  //
  // Remarks: (1) formatted text files may contain SPACE+NEW_LINE (dickens) or NEW_LINE+SPACE (world95.txt)
  // or just a NEW_LINE instead of a SPACE. (2) quotes and apostrophes are ignored during processing
  if( isLetter ) {
    expr0Chars = expr0Chars << 8 | c; // last 4 consecutive letters
    expr0 = combine64(expr0, c);
    exprLen0 = min(exprLen0 + 1, maxWordLen);
  } else {
    expr0Chars = 0;
    exprLen0 = 0;
    if((c == SPACE || isNewline) && (isLetterPc || pC == APOSTROPHE || pC == QUOTE)) {
      expr4 = expr3;
      expr3 = expr2;
      expr2 = expr1;
      expr1 = expr0;
      expr0 = 0;
    } else if( c == APOSTROPHE || c == QUOTE || (isNewline && pC == SPACE) || (c == SPACE && isNewlinePc)) {
      //ignore
    } else {
      expr4 = expr3 = expr2 = expr1 = expr0 = 0;
    }
  }
}

void WordModelInfo::lineModelPredict() {
  const uint8_t RH = CM_USE_RUN_STATS | CM_USE_BYTE_HISTORY;
  uint64_t i = 1024 * (1 + (isTextBlock << 1 | (fixedLineLength == 0)));
  INJECT_SHARED_pos
  if( isNewline ) { // a new line has just started (or: zero in asciiz or in binary data)
      nl4 = nl3;
      nl3 = nl2;
      nl2 = nl1;
      nl1 = pos;
      firstChar = -1;
      firstWord = 0;
      line1 = line0;
      line0 = 0;
    }
  INJECT_SHARED_c1
  line0 = combine64(line0, c1);
  cm.set(RH, hash(++i, line0));

  uint32_t col = pos - nl1;
  if( col == 1 ) {
    firstChar = groups & 0xff;
  }
  INJECT_SHARED_buf
    const uint8_t cAbove = buf[nl2 + col]; // N
  const uint8_t pCAbove = buf[nl2 + col - 1]; // NW

  const bool isNewLineStart = col == 0 && nl2 > 0;
  const bool isPrevCharMatchAbove = c1 == pCAbove && col != 0 && nl2 != 0;
  const uint32_t aboveCtx = cAbove << 1 | uint32_t(isPrevCharMatchAbove);
  if (isNewLineStart) {
    lineMatch = 0; //first char not yet known = nothing to match
  }
  else if(lineMatch >= 0 && isPrevCharMatchAbove) {
    lineMatch = min(lineMatch + 1, maxLineMatch); //match continues
  }
  else {
    lineMatch = -1; //match does not continue
  }

  // context: matches with the previous line
  if( lineMatch >= 0 ) {
    cm.set(RH, hash(++i, cAbove, lineMatch));
  }
  else {
    cm.skip(RH);
    i++;
  }
  const uint32_t lineLength = nl1 - nl2;
  if( col < lineLength ) {
    cm.set(RH, hash(++i, aboveCtx << 8 | c1));
    cm.set(RH, hash(++i, aboveCtx << 8 | c1, col));
  }
  else {
    cm.skipn(RH, 2);
    i += 2;
  }
  cm.set(RH, hash(++i, lineLength, col, aboveCtx << 8 | (groups & 0xff))); // english_mc
  cm.set(RH, hash(++i, lineLength, col, aboveCtx << 8 | c1)); // english_mc


  //modeling columns in fixed-length lines (such as in silesia/nci)
  const uint8_t cAbove2 = buf[nl3 + col]; // NN
  const uint8_t cAbove3 = buf[nl4 + col]; // NNN
  const int lineLength2 = nl2 - nl3;
  const int lineLength3 = nl3 - nl4;
  const uint32_t gBefore = groups & 0xffff;
  if( cAbove == cAbove2 && lineLength == lineLength2 ) {
    cm.set(RH, hash(++i, gBefore, cAbove));
  }
  else {
    cm.skip(RH);
    i++;
  }
  if( col < lineLength && col < lineLength2 && col < lineLength3 ) {
    cm.set(RH, hash(++i, gBefore, cAbove, cAbove2, cAbove3));
  }
  else {
    cm.skip(RH);
    i++;
  }

  //modeling line content based on previous line content (nci, q.wk3)
  if( lineLength > 1 /* same as line1 != 0 */ ) {
    uint32_t cx = isTextBlock ? groups & 0xff : c1;
    cm.set(RH, hash(++i, line1, col << 8 | cx));
    cm.set(RH, hash(++i, line1, line0));
  }
  else {
    cm.skipn(RH, 2);
    i += 2;
  }

  // modeling line content per column (and NEW_LINE in some extent)
  cm.set(RH, hash(++i, static_cast<uint64_t>(col << 1 | (c1 == SPACE)))); // after space vs after other char in this column // world95.txt
  cm.set(RH, hash(++i, col << 8 | c1));
  cm.set(RH, hash(++i, col, mask & 0x1ff));
  cm.set(RH, hash(++i, col, lineLength)); // the length of the previous line may foretell the content of columns

  cm.set(RH, hash(++i, col << 8 | firstChar, static_cast<uint64_t>(lastUpper < col) << 8 | (groups & 0xff))); // book1 book2 news

  // content of lines, paragraphs
  cm.set(RH, hash(++i, nl1)); //chars occurring in this paragraph (order 0)
  cm.set(RH, hash(++i, nl1, c)); //chars occurring in this paragraph (order 1)
  cm.set(RH, hash(++i, firstChar)); //chars occurring in a paragraph that began with firstChar (order 0)
  cm.set(RH, hash(++i, firstChar, c)); //chars occurring in a paragraph that began with firstChar (order 1)
  cm.set(RH, hash(++i, firstWord)); //chars occurring in a paragraph that began with firstWord (order 0)
  cm.set(RH, hash(++i, firstWord, c)); //chars occurring in a paragraph that began with firstWord (order 1)
  assert(i == 1024 * (1 + (isTextBlock << 1 | (fixedLineLength == 0))) + nCM1);
}

void WordModelInfo::lineModelSkip() {
  const uint8_t RH = CM_USE_RUN_STATS | CM_USE_BYTE_HISTORY;
  cm.skipn(RH, nCM1);
}

void WordModelInfo::predict(const uint8_t pdfTextParserState) {
  INJECT_SHARED_pos
  const uint32_t lastPos = checksums[w] != chk ? 0 : wordPositions[w]; //last occurrence (position) of a whole word or number
  const uint32_t dist = lastPos == 0 ? 0 : min(llog(pos - lastPos + 120) >> 4, 20);
  const bool word0MayEndNow = lastPos != 0;
  const uint8_t mayBeCaps = static_cast<const uint8_t>(uint8_t(c4 >> 8) >= 'A' && uint8_t(c4 >> 8) <= 'Z' && uint8_t(c4) >= 'A' && uint8_t(c4) <= 'Z');
                                                       
  const uint8_t RH = CM_USE_RUN_STATS | CM_USE_BYTE_HISTORY;
  uint64_t i = 0;

  cm.set(RH, hash(++i, text0)); //strong
  
  // expressions in normal text
  if( isTextBlock ) {
    cm.set(RH, hash(++i, expr0, expr1, expr2, expr3, expr4));
    cm.set(RH, hash(++i, expr0, expr1, expr2));
  } else {
    i += 2;
  }

  // sections introduced by keywords (enwik world95.txt html/xml)
  cm.set(RH, hash(++i, gapToken0, keyword0)); // chars occurring in a section introduced by "keyword:" or "keyword=" (order 0 and variable order for the gap)
  cm.set(RH, hash(++i, word0, c, keyword0)); // tokens occurring in a section introduced by "keyword:" or "keyword=" (order 1 and variable order for a word)

  cm.set(RH, hash(++i, word0, dist));
  cm.set(RH, hash(++i, word1, gapToken0, dist));

  cm.set(RH, hash(++i, pos >> 10, word0)); //word/number tokens and characters occurring in this 1K block

  // Simple word morphology (order 1-4)

  const uint32_t wmeMbc = static_cast<uint32_t>(word0MayEndNow) << 1 | mayBeCaps;
  const uint32_t wl = min(wordLen0, 6);
  const uint32_t wlWmeMbc = wl << 2 | wmeMbc;
  cm.set(RH, hash(++i, wlWmeMbc, mask2 /*last 1-4 char types*/));

  if( exprLen0 >= 1 ) {
    const uint32_t wlWmeMbc = min(exprLen0, 1 + 3) << 2 | wmeMbc;
    cm.set(RH, hash(++i, wlWmeMbc, c));
  } else {
    cm.skip(RH);
    i++;
  }

  if( exprLen0 >= 2 ) {
    const uint32_t wlWmeMbc = min(exprLen0, 2 + 3) << 2 | wmeMbc;
    cm.set(RH, hash(++i, wlWmeMbc, expr0Chars & 0xffff));
  } else {
    cm.skip(RH);
    i++;
  }

  if( exprLen0 >= 3 ) {
    const uint32_t wlWmeMbc = min(exprLen0, 3 + 3) << 2 | wmeMbc;
    cm.set(RH, hash(++i, wlWmeMbc, expr0Chars & 0xffffff));
  } else {
    cm.skip(RH);
    i++;
  }

  if( exprLen0 >= 4 ) {
    const uint32_t wlWmeMbc = min(exprLen0, 4 + 3) << 2 | wmeMbc;
    cm.set(RH, hash(++i, wlWmeMbc, expr0Chars));
  } else {
    cm.skip(RH);
    i++;
  }

  cm.set(RH, hash(++i, word0, pdfTextParserState));

  cm.set(RH, hash(++i, word0, gapToken0)); // stronger
  cm.set(RH, hash(++i, c, word0, gapToken1)); // stronger
  cm.set(RH, hash(++i, c, gapToken0, word1)); // stronger //french texts need that "c"
  cm.set(RH, hash(++i, word0, word1)); // stronger
  cm.set(RH, hash(++i, word0, word1, word2)); // weaker
  cm.set(RH, hash(++i, gapToken0, word1, gapToken1, word2)); // stronger
  cm.set(RH, hash(++i, word0, word1, gapToken1, word2)); // weaker
  cm.set(RH, hash(++i, word0, word1, gapToken1)); // weaker

  const uint8_t c1 = c4 & 0xff;
  if( isTextBlock ) {
    cm.set(RH, hash(++i, word0, c1, word2));
    cm.set(RH, hash(++i, word0, c1, word3));
    cm.set(RH, hash(++i, word0, c1, word4));
    cm.set(RH, hash(++i, word0, c1, word1, word4));
    cm.set(RH, hash(++i, word0, c1, word1, word3));
    cm.set(RH, hash(++i, word0, c1, word2, word3));
  } else {
    i += 6;
  }

  const uint8_t g = groups & 0xff;
  cm.set(RH, hash(++i, opened, wlWmeMbc, g, pdfTextParserState));
  cm.set(RH, hash(++i, opened, c, static_cast<uint64_t>(dist != 0), pdfTextParserState));
  cm.set(RH, hash(++i, opened, word0)); // book1, book2, dickens, enwik


  cm.set(RH, hash(++i, groups));
  cm.set(RH, hash(++i, groups, c));
  cm.set(RH, hash(++i, groups, c4 & 0x0000ffff));

  f4 = (f4 << 4) | (c1 == ' ' ? 0 : c1 >> 4);
  cm.set(RH, hash(++i, f4 & 0x0fff));
  cm.set(RH, hash(++i, f4));

  int fl = 0;
  if( c1 != 0 ) {
    if( isalpha(c1) != 0 ) {
      fl = 1;
    } else if( ispunct(c1) != 0 ) {
      fl = 2;
    } else if( isspace(c1) != 0 ) {
      fl = 3;
    } else if(c1 == 0xff ) {
      fl = 4;
    } else if(c1 < 16 ) {
      fl = 5;
    } else if(c1 < 64 ) {
      fl = 6;
    } else {
      fl = 7;
    }
  }
  mask = (mask << 3) | fl;

  cm.set(RH, hash(++i, mask));
  cm.set(RH, hash(++i, mask, c1));
  cm.set(RH, hash(++i, mask, c4 & 0x00ffff00));
  cm.set(RH, hash(++i, mask & 0x1ff, f4 & 0x00fff0)); // for some binary files

  if( isTextBlock ) {
    cm.set(RH, hash(++i, word0, c1, llog(wordGap), mask & 0x01ff,
                (static_cast<uint32_t>(wordLen1 > 3) << 2) |
                (static_cast<uint32_t>(lastUpper < lastLetter + wordLen1) << 1) |
                (static_cast<uint32_t>(lastUpper < wordLen0 + wordLen1 + wordGap)))); //weak
  } else {
    i++;
  }

  assert(int(i) == nCM2_TEXT);
}
