#include "TextModel.hpp"
#include "../CharacterNames.hpp"

TextModel::TextModel(Shared* const sh, const uint64_t size) : 
    shared(sh),
    cm(sh, size, nCM2, 64),
    stemmers(Language::Count - 1), languages(Language::Count - 1), dictionaries(Language::Count - 1), wordPos(0x10000),
    State(Parse::Unknown), pState(State), Lang {{0}, {0}, Language::Unknown, Language::Unknown}, Info {}, parseCtx(0)
{
  stemmers[Language::English - 1] = new EnglishStemmer();
  stemmers[Language::French - 1] = new FrenchStemmer();
  stemmers[Language::German - 1] = new GermanStemmer();
  languages[Language::English - 1] = new English();
  languages[Language::French - 1] = new French();
  languages[Language::German - 1] = new German();
  cWord = &words[Lang.id](0);
  pWord = &words[Lang.id](1);
  cSegment = &segments(0);
  cSentence = &sentences(0);
  cParagraph = &paragraphs(0);
  memset(&bytePos[0], 0, sizeof(bytePos));
}

TextModel::~TextModel() {
  for( int i = 0; i < Language::Count - 1; i++ ) {
    delete stemmers[i];
    delete languages[i];
    delete dictionaries[i];
  }
}

void TextModel::update() {
  Info.lastUpper = min(0xFF, Info.lastUpper + 1), Info.maskUpper <<= 1U;
  Info.lastLetter = min(0x1F, Info.lastLetter + 1);
  Info.lastDigit = min(0xFF, Info.lastDigit + 1);
  Info.lastPunctuation = min(0x3F, Info.lastPunctuation + 1);
  Info.lastNewLine++;
  Info.prevNewLine++;
  Info.lastNest++;
  Info.spaceCount -= (Info.spaces >> 31U);
  Info.spaces <<= 1U;
  Info.masks[0] <<= 2U;
  Info.masks[1] <<= 2U;
  Info.masks[2] <<= 4U;
  Info.masks[3] <<= 3U;
  pState = State;

  INJECT_SHARED_buf
  uint8_t c = buf(1);
  uint8_t lc = tolower(c);
  uint8_t g = (c < 0x80) ? asciiGroup[c] : 31;
  if( g > 4 || g != (Info.masks[4] & 0x1FU)) {
    Info.masks[4] <<= 5U, Info.masks[4] |= g;
  }
  INJECT_SHARED_pos
  bytePos[c] = pos;
  if( c != lc ) {
    c = lc;
    Info.lastUpper = 0, Info.maskUpper |= 1U;
  }
  uint8_t pC = buf(2);
  State = Parse::Unknown;
  parseCtx = hash(State, pWord->Hash[0], c,
                  (ilog2(Info.lastNewLine) + 1) * static_cast<uint32_t>(Info.lastNewLine * 3 > Info.prevNewLine),
                  Info.masks[1] & 0xFCU);

  if((c >= 'a' && c <= 'z') || c == APOSTROPHE || c == '-' || c > 0x7F ) {
    if( Info.wordLength[0] == 0 ) {
      // check for hyphenation with "+" (book1 from Calgary)
      if( pC == NEW_LINE &&
          ((Info.lastLetter == 3 && buf(3) == '+') || (Info.lastLetter == 4 && buf(3) == CARRIAGE_RETURN && buf(4) == '+'))) {
        Info.wordLength[0] = Info.wordLength[1];
        for( int i = Language::Unknown; i < Language::Count; i++ ) {
          words[i]--;
        }
        cWord = pWord, pWord = &words[Lang.pId](1);
        cWord->reset();
        for( uint32_t i = 0; i < Info.wordLength[0]; i++ ) {
          (*cWord) += buf(Info.wordLength[0] - i + Info.lastLetter);
        }
        Info.wordLength[1] = (*pWord).length();
        cSegment->wordCount--;
        cSentence->wordCount--;
      } else {
        Info.wordGap = Info.lastLetter;
        Info.firstLetter = c;
      }
    }
    Info.lastLetter = 0;
    Info.wordLength[0]++;
    Info.masks[0] += (Lang.id != Language::Unknown) ? 1 + static_cast<int>(stemmers[Lang.id - 1]->isVowel(c))
                                                    : 1, Info.masks[1]++, Info.masks[3] += Info.masks[0] & 3U;
    if( c == APOSTROPHE ) {
      Info.masks[2] += 12;
      if( Info.wordLength[0] == 1 ) {
        if( Info.quoteLength == 0 && pC == SPACE ) {
          Info.quoteLength = 1;
        } else if( Info.quoteLength > 0 && Info.lastPunctuation == 1 ) {
          Info.quoteLength = 0;
          State = Parse::AfterQuote;
          parseCtx = hash(State, pC);
        }
      }
    }
    (*cWord) += c;
    cWord->calculateWordHash();
    State = Parse::ReadingWord;
    parseCtx = hash(State, cWord->Hash[0]);
  } else {
    if( cWord->length() > 0 ) {
      if( Lang.id != Language::Unknown ) {
        memcpy(&words[Language::Unknown](0), cWord, sizeof(Word));
      }

      for( int i = Language::Count - 1; i > Language::Unknown; i-- ) {
        Lang.count[i - 1] -= (Lang.mask[i - 1] >> 63U), Lang.mask[i - 1] <<= 1U;
        if( i != Lang.id ) {
          memcpy(&words[i](0), cWord, sizeof(Word));
        }
        if( stemmers[i - 1]->stem(&words[i](0))) {
          Lang.count[i - 1]++, Lang.mask[i - 1] |= 1U;
        }
      }
      Lang.id = Language::Unknown;
      uint32_t best = MIN_RECOGNIZED_WORDS;
      for( int i = Language::Count - 1; i > Language::Unknown; i-- ) {
        if( Lang.count[i - 1] >= best ) {
          best = Lang.count[i - 1] + static_cast<uint32_t>(i == Lang.pId); //bias to prefer the previously detected language
          Lang.id = i;
        }
        words[i]++;
      }
      words[Language::Unknown]++;

      if( Lang.id != Lang.pId ) {
        if constexpr (false) { //turn it on if you'd like to see language-detection info on screen
          if (shared->toScreen) {
            printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b");
          } 
          INJECT_SHARED_blockPos
          switch (Lang.id) {
            case Language::Unknown: {
              printf("[language: Unknown, blPos: %d]\n", blockPos);
              break;
            }
            case Language::English: {
              printf("[language: English, blPos: %d]\n", blockPos);
              break;
            }
            case Language::French: {
              printf("[language: French,  blPos: %d]\n", blockPos);
              break;
            }
            case Language::German: {
              printf("[language: German,  blPos: %d]\n", blockPos);
              break;
            }
          }
        }
        if((shared->GetOptionTrainTxt()) && Lang.id != Language::Unknown && dictionaries[Lang.id - 1] == nullptr ) {
          switch( Lang.id ) {
            case Language::English: {
              dictionaries[Lang.id - 1] = new WordEmbeddingDictionary();
              dictionaries[Lang.id - 1]->loadFromFile("english.emb");
            }
          }
        }
      }

      Lang.pId = Lang.id;
      pWord = &words[Lang.id](1);
      cWord = &words[Lang.id](0);
      cWord->reset();
      if(shared->GetOptionTrainTxt()) {
        if( Lang.id != Language::Unknown && dictionaries[Lang.id - 1] != nullptr ) {
          dictionaries[Lang.id - 1]->getWordEmbedding(pWord);
        }
        wordDistances++;
        for( uint32_t i = Cache<TextModel::WordDistance, 4>::size - 1; i > 0; i-- ) {
          memcpy(&wordDistances(i), &wordDistances(i - 1), sizeof(WordDistance));
        }
        uint32_t minDistance = UINT32_MAX;
        for( uint32_t i = 1; i <= 4; i++ ) {
          uint32_t d = pWord->distanceTo(words[Lang.id](i + 1));
          wordDistances(0).distance[i - 1] = d;
          if( d < minDistance ) {
            minDistance = d;
            wordDistances(0).closest = words[Lang.id](i + 1).embedding;
          }
        }
      }
      wordPos[pWord->Hash[0] & (wordPos.size() - 1)] = pos;
      if( cSegment->wordCount == 0 ) {
        memcpy(&cSegment->firstWord, pWord, sizeof(Word));
      }
      cSegment->wordCount++;
      if( cSentence->wordCount == 0 ) {
        memcpy(&cSentence->firstWord, pWord, sizeof(Word));
      }
      cSentence->wordCount++;
      Info.wordLength[1] = Info.wordLength[0], Info.wordLength[0] = 0;
      Info.quoteLength += static_cast<uint32_t>(Info.quoteLength > 0);
      if( Info.quoteLength > 0x1F ) {
        Info.quoteLength = 0;
      }
      cSentence->verbIndex++, cSentence->nounIndex++, cSentence->capitalIndex++;
      if((pWord->type & Language::Verb) != 0 ) {
        cSentence->verbIndex = 0;
        memcpy(&cSentence->lastVerb, pWord, sizeof(Word));
      }
      if((pWord->type & Language::Noun) != 0 ) {
        cSentence->nounIndex = 0;
        memcpy(&cSentence->lastNoun, pWord, sizeof(Word));
      }
      if( cSentence->wordCount > 1 && Info.lastUpper < Info.wordLength[1] ) {
        cSentence->capitalIndex = 0;
        memcpy(&cSentence->lastCapital, pWord, sizeof(Word));
      }
    }
    bool skip = false;
    switch( c ) {
      case '.': {
        if( Lang.id != Language::Unknown && Info.lastUpper == Info.wordLength[1] && languages[Lang.id - 1]->isAbbreviation(pWord)) {
          State = Parse::WasAbbreviation;
          parseCtx = hash(State, pWord->Hash[0]);
          break;
        }
      }
      case '?':
      case '!': {
        cSentence->type = (c == '.') ? Sentence::Types::Declarative : (c == '?') ? Sentence::Types::Interrogative
                                                                                 : Sentence::Types::Exclamative;
        cSentence->segmentCount++;
        cParagraph->sentenceCount++;
        cParagraph->typeCount[cSentence->type]++;
        cParagraph->typeMask <<= 2U, cParagraph->typeMask |= cSentence->type;
        cSentence = &sentences.next();
        Info.masks[3] += 3;
        skip = true;
      }
      case ',':
      case ';':
      case ':': {
        if( c == ',' ) {
          Info.commas++;
          State = Parse::AfterComma;
          parseCtx = hash(State, ilog2(Info.quoteLength + 1), ilog2(Info.lastNewLine),
                          static_cast<uint64_t>(Info.lastUpper < Info.lastLetter + Info.wordLength[1]));
        } else if( c == ':' ) {
          memcpy(&Info.topicDescriptor, pWord, sizeof(Word));
        }
        if( !skip ) {
          cSentence->segmentCount++;
          Info.masks[3] += 4;
        }
        Info.lastPunctuation = 0, Info.prevPunctuation = c;
        Info.masks[0] += 3, Info.masks[1] += 2, Info.masks[2] += 15;
        cSegment = &segments.next();
        break;
      }
      case NEW_LINE: {
        Info.prevNewLine = Info.lastNewLine, Info.lastNewLine = 0;
        Info.commas = 0;
        if( Info.prevNewLine == 1 || (Info.prevNewLine == 2 && pC == CARRIAGE_RETURN)) {
          cParagraph = &paragraphs.next();
        } else if((Info.lastLetter == 2 && pC == '+') || (Info.lastLetter == 3 && pC == CARRIAGE_RETURN && buf(3) == '+')) {
          parseCtx = hash(Parse::ReadingWord, pWord->Hash[0]);
          State = Parse::PossibleHyphenation;
        }
      }
      case TAB:
      case CARRIAGE_RETURN:
      case SPACE: {
        Info.spaceCount++, Info.spaces |= 1U;
        Info.masks[1] += 3, Info.masks[3] += 5;
        if( c == SPACE && pState == Parse::WasAbbreviation ) {
          State = Parse::AfterAbbreviation;
          parseCtx = hash(State, pWord->Hash[0]);
        }
        break;
      }
      case '(':
        Info.masks[2] += 1;
        Info.masks[3] += 6;
        Info.nestHash += 31;
        Info.lastNest = 0;
        break;
      case '[':
        Info.masks[2] += 2;
        Info.nestHash += 11;
        Info.lastNest = 0;
        break;
      case '{':
        Info.masks[2] += 3;
        Info.nestHash += 17;
        Info.lastNest = 0;
        break;
      case '<':
        Info.masks[2] += 4;
        Info.nestHash += 23;
        Info.lastNest = 0;
        break;
      case 0xAB:
        Info.masks[2] += 5;
        break;
      case ')':
        Info.masks[2] += 6;
        Info.nestHash -= 31;
        Info.lastNest = 0;
        break;
      case ']':
        Info.masks[2] += 7;
        Info.nestHash -= 11;
        Info.lastNest = 0;
        break;
      case '}':
        Info.masks[2] += 8;
        Info.nestHash -= 17;
        Info.lastNest = 0;
        break;
      case '>':
        Info.masks[2] += 9;
        Info.nestHash -= 23;
        Info.lastNest = 0;
        break;
      case 0xBB:
        Info.masks[2] += 10;
        break;
      case QUOTE: {
        Info.masks[2] += 11;
        // start/stop counting
        if( Info.quoteLength == 0 ) {
          Info.quoteLength = 1;
        } else {
          Info.quoteLength = 0;
          State = Parse::AfterQuote;
          parseCtx = hash(State, 0x100U | pC);
        }
        break;
      }
      case '/':
      case '-':
      case '+':
      case '*':
      case '=':
      case '%':
        Info.masks[2] += 13;
        break;
      case '\\':
      case '|':
      case '_':
      case '@':
      case '&':
      case '^':
        Info.masks[2] += 14;
        break;
    }
    if( c >= '0' && c <= '9' ) {
      Info.numbers[0] = Info.numbers[0] * 10 + (c & 0xFU), Info.numLength[0] = min(19, Info.numLength[0] + 1);
      Info.numHashes[0] = combine64(Info.numHashes[0], c);
      Info.expectedDigit = -1;
      if( Info.numLength[0] < Info.numLength[1] && (pState == Parse::ExpectDigit || ((Info.numDiff & 3U) == 0 && Info.numLength[0] <= 1))) {
        uint64_t expectedNum = Info.numbers[1] + (Info.numMask & 3U) - 2;
        uint64_t placeDivisor = 1;
        for( int i = 0; i < Info.numLength[1] - Info.numLength[0]; i++, placeDivisor *= 10 ) { ;
        }
        if( expectedNum / placeDivisor == Info.numbers[0] ) {
          placeDivisor /= 10;
          Info.expectedDigit = (expectedNum / placeDivisor) % 10;
          State = Parse::ExpectDigit;
        }
      } else {
        uint8_t d = buf(Info.numLength[0] + 2);
        if( Info.numLength[0] < 3 && buf(Info.numLength[0] + 1) == ',' && d >= '0' && d <= '9' ) {
          State = Parse::ExpectDigit;
        }
      }
      Info.lastDigit = 0;
      Info.masks[3] += 7;
    } else if( Info.numbers[0] > 0 ) {
      Info.numMask <<= 2U, Info.numMask |= 1U + static_cast<uint32_t>(Info.numbers[0] >= Info.numbers[1]) + static_cast<uint32_t>(Info.numbers[0] > Info.numbers[1]);
      Info.numDiff <<= 2U, Info.numDiff |= min(3, ilog2(abs(static_cast<int>(Info.numbers[0] - Info.numbers[1]))));
      Info.numbers[1] = Info.numbers[0], Info.numbers[0] = 0;
      Info.numHashes[1] = Info.numHashes[0], Info.numHashes[0] = 0;
      Info.numLength[1] = Info.numLength[0], Info.numLength[0] = 0;
      cSegment->numCount++, cSentence->numCount++;
    }
  }
  if( Info.lastNewLine == 1 ) {
    Info.firstChar = (Lang.id != Language::Unknown) ? c : min(c, 96);
  }
  if( Info.lastNest > 512 ) {
    Info.nestHash = 0;
  }
  int leadingBitsSet = 0;
  while(((c >> (7 - leadingBitsSet)) & 1U) != 0 ) {
    leadingBitsSet++;
  }

  if( Info.UTF8Remaining > 0 && leadingBitsSet == 1 ) {
    Info.UTF8Remaining--;
  } else {
    Info.UTF8Remaining = (leadingBitsSet != 1) ? (c != 0xC0 && c != 0xC1 && c < 0xF5) ? (leadingBitsSet - static_cast<int>(leadingBitsSet > 0)) : -1 : 0;
  }
  Info.maskPunctuation = static_cast<int>(bytePos[','] > bytePos['.']) | (static_cast<int>(bytePos[','] > bytePos['!']) << 1U) |
                         (static_cast<int>(bytePos[','] > bytePos['?']) << 2U) | (static_cast<int>(bytePos[','] > bytePos[':']) << 3U) |
                         (static_cast<int>(bytePos[','] > bytePos[';']) << 4U);

  shared->State.Text.firstLetter = Info.firstLetter;
  shared->State.Text.mask = Info.masks[1] & 0xFFU;
}

void TextModel::setContexts() {
  INJECT_SHARED_buf
  const uint8_t c = buf(1);
  const uint8_t lc = tolower(c);
  const uint8_t m2 = Info.masks[2] & 0xFU;
  const uint8_t column = min(0xFF, Info.lastNewLine);
  const uint64_t w = State == Parse::ReadingWord ? cWord->Hash[0] : pWord->Hash[0];

  const uint64_t cWordHash0 = cWord->Hash[0];
  const uint64_t pWordHash0 = pWord->Hash[0];
  const uint64_t pWordHash1 = pWord->Hash[1];

  const uint8_t RH = CM_USE_RUN_STATS | CM_USE_BYTE_HISTORY;

  cm.set(RH, parseCtx);
  uint64_t i = State * 64;
  cm.set(RH, hash(++i, cWordHash0, pWordHash0,
    static_cast<int>(Info.lastUpper < Info.wordLength[0]) |
    (static_cast<int>(Info.lastDigit < Info.wordLength[0] + Info.wordGap) << 1U)
  ));
  cm.set(RH, hash(++i, cWordHash0, words[Lang.pId](2).Hash[0],
    min(10, ilog2(static_cast<uint32_t>(Info.numbers[0]))), static_cast<int>(Info.lastUpper < Info.lastLetter + Info.wordLength[1]) |
    (static_cast<int>(Info.lastLetter > 3) << 1U) | 
    (static_cast<int>(Info.lastLetter > 0 && Info.wordLength[1] < 3) << 2U)
  ));
  cm.set(RH, hash(++i, cWordHash0,
    Info.masks[1] & 0x3FFU, words[Lang.pId](3).Hash[1], static_cast<int>(Info.lastDigit < Info.wordLength[0] + Info.wordGap) | 
    (static_cast<int>(Info.lastUpper < Info.lastLetter + Info.wordLength[1]) << 1U) |
    ((Info.spaces & 0x7F) << 2U)
  ));
  cm.set(RH, hash(++i, cWordHash0, pWordHash1));
  cm.set(RH, hash(++i, cWordHash0, pWordHash1, words[Lang.pId](2).Hash[1]));
  cm.set(RH, hash(++i, w, words[Lang.pId](2).Hash[0], words[Lang.pId](3).Hash[0]));
  cm.set(RH, hash(++i, cWordHash0, c, (cSentence->verbIndex < cSentence->wordCount) ? cSentence->lastVerb.Hash[0] : 0));
  cm.set(RH, hash(++i, pWordHash1, Info.masks[1] & 0xFCU, lc, Info.wordGap));
  cm.set(RH, hash(++i, (Info.lastLetter == 0) ? cWordHash0 : pWordHash0, c, cSegment->firstWord.Hash[1], min(3, ilog2(cSegment->wordCount + 1))));
  cm.set(RH, hash(++i, cWordHash0, c, segments(1).firstWord.Hash[1]));
  cm.set(RH, hash(++i, max(31, lc), Info.masks[1] & 0xFFCU,
    (Info.spaces & 0xFEU) | 
    static_cast<uint32_t>(Info.lastPunctuation < Info.lastLetter), (Info.maskUpper & 0xFFU) | 
    (((0x100U | Info.firstLetter) * static_cast<uint32_t>(Info.wordLength[0] > 1)) << 8U)
  ));
  cm.set(RH, hash(++i, column, min(7, ilog2(Info.lastUpper + 1)), ilog2(Info.lastPunctuation + 1)));
  cm.set(RH, hash(++i,
    (column & 0xF8U) | 
    (Info.masks[1] & 3U) | 
    (static_cast<int>(Info.prevNewLine - Info.lastNewLine > 63) << 2U) |
    (min(3, Info.lastLetter) << 8U) | 
    (Info.firstChar << 10U) | 
    (static_cast<int>(Info.commas > 4) << 18U) |
    (static_cast<int>(m2 >= 1 && m2 <= 5) << 19U) | 
    (static_cast<int>(m2 >= 6 && m2 <= 10) << 20U) |
    (static_cast<int>(m2 == 11 || m2 == 12) << 21U) | 
    (static_cast<int>(Info.lastUpper < column) << 22U) |
    (static_cast<int>(Info.lastDigit < column) << 23U) |
    (static_cast<int>(column < Info.prevNewLine - Info.lastNewLine) << 24U)
  ));
  cm.set(RH, hash(++i, (2 * column) / 3,
    min(13, Info.lastPunctuation) + static_cast<int>(Info.lastPunctuation > 16) + static_cast<int>(Info.lastPunctuation > 32) + Info.maskPunctuation * 16, 
    ilog2(Info.lastUpper + 1), 
    ilog2(Info.prevNewLine - Info.lastNewLine), 
      static_cast<int>((Info.masks[1] & 3U) == 0) | 
      (static_cast<int>(m2 < 6) << 1U) | 
      (static_cast<int>(m2 < 11) << 2U)
  ));
  cm.set(RH, hash(++i, column >> 1U, Info.spaces & 0xFU));
  cm.set(RH, hash(++i, Info.masks[3] & 0x3FU,
    min((max(Info.wordLength[0], 3) - 2) * static_cast<int>(Info.wordLength[0] < 8), 3), 
    Info.firstLetter * static_cast<int>(Info.wordLength[0] < 5),
    w, 
      static_cast<int>(c == buf(2)) | (static_cast<int>(Info.masks[2] > 0) << 1U) |
      (static_cast<int>(Info.lastPunctuation < Info.wordLength[0] + Info.wordGap) << 2U) |
      (static_cast<int>(Info.lastUpper < Info.wordLength[0]) << 3U) |
      (static_cast<int>(Info.lastDigit < Info.wordLength[0] + Info.wordGap) << 4U) |
      (static_cast<int>(Info.lastPunctuation < 2 + Info.wordLength[0] + Info.wordGap + Info.wordLength[1]) << 5U)
  ));
  cm.set(RH, hash(++i, w, c, Info.numHashes[1]));
  INJECT_SHARED_pos
  cm.set(RH, hash(++i, w, c, llog(pos - wordPos[w & (wordPos.size() - 1)]) >> 1));
  cm.set(RH, hash(++i, w, c, Info.topicDescriptor.Hash[0]));
  cm.set(RH, hash(++i, Info.numLength[0], c, Info.topicDescriptor.Hash[0]));
  cm.set(RH, hash(++i, (Info.lastLetter > 0) ? c : 0x100, Info.masks[1] & 0xFFCU, Info.nestHash & 0x7FFU));
  cm.set(RH, hash(++i, w, c, Info.masks[3] & 0x1FFU,
    (static_cast<int>(cSentence->verbIndex == 0 && cSentence->lastVerb.length() > 0) << 6U) |
    (static_cast<int>(Info.wordLength[1] > 3) << 5U) |
    (static_cast<int>(cSegment->wordCount == 0) << 4U) |
    (static_cast<int>(cSentence->segmentCount == 0 && cSentence->wordCount < 2) << 3U) |
    (static_cast<int>(Info.lastPunctuation >= Info.lastLetter + Info.wordLength[1] + Info.wordGap) << 2U) |
    (static_cast<int>(Info.lastUpper < Info.lastLetter + Info.wordLength[1]) << 1U) |
     static_cast<int>(Info.lastUpper < Info.wordLength[0] + Info.wordGap + Info.wordLength[1])
  ));
  cm.set(RH, hash(++i, c, pWordHash1,
    Info.firstLetter * static_cast<int>(Info.wordLength[0] < 6),
      (static_cast<int>(Info.lastPunctuation < Info.wordLength[0] + Info.wordGap) << 1U) |
      static_cast<int>(Info.lastPunctuation >= Info.lastLetter + Info.wordLength[1] + Info.wordGap)
  ));
  cm.set(RH, hash(++i, w, c, words[Lang.pId](1 + static_cast<int>(Info.wordLength[0] == 0)).letters[words[Lang.pId](1 + static_cast<int>(Info.wordLength[0] == 0)).start], Info.firstLetter * static_cast<int>(Info.wordLength[0] < 7)));
  cm.set(RH, hash(++i, column, Info.spaces & 7U, Info.nestHash & 0x7FFU));
  cm.set(RH, hash(++i, cWordHash0,
    static_cast<int>(Info.lastUpper < column) | 
    (static_cast<int>(Info.lastUpper < Info.wordLength[0]) << 1U), min(5, Info.wordLength[0])
  ));
  cm.set(RH, hash(++i, Lang.id, w,
    uint8_t(words[Lang.id](1 + static_cast<int>(State != Parse::ReadingWord)).embedding),
      static_cast<int>(Info.lastUpper < Info.wordLength[0]) |
      (static_cast<int>(cSegment->wordCount == 0) << 1U)
  ));
  assert(i - State * 64 + 1 == nCM2);
}

void TextModel::setParam(int cmScale) {
  cm.setScale(cmScale);
}

void TextModel::mix(Mixer &m) {
  INJECT_SHARED_bpos
  if( bpos == 0 ) {
    update();
    setContexts();
  }
  cm.mix(m);

  const int order = max(0, cm.order - (nCM2 - 15)); //0-15
  assert(0 <= order && order <= 15);
  m.set(order << 3 | bpos, 16 * 8); // 16*8
  shared->State.Text.order = order;

  INJECT_SHARED_c0
  INJECT_SHARED_c1
  const uint8_t characterGroup = shared->State.Text.characterGroup;
  m.set(finalize64(hash((Lang.id != Language::Unknown) ? 1 + static_cast<int>(stemmers[Lang.id - 1]->isVowel(c1)) : 0, Info.masks[1] & 0xFF, c0), 11), 2048);
  m.set(finalize64(hash(ilog2(Info.wordLength[0] + 1), c0, 
    static_cast<int>(Info.lastDigit < Info.wordLength[0] + Info.wordGap) |
    (static_cast<int>(Info.lastUpper < Info.lastLetter + Info.wordLength[1]) << 1U) |
    (static_cast<int>(Info.lastPunctuation < Info.wordLength[0] + Info.wordGap) << 2U) | 
    (static_cast<int>(Info.lastUpper < Info.wordLength[0]) << 3U)), 11), 2048);
  m.set(finalize64(hash(Info.masks[1] & 0x3FFU, characterGroup, 
    static_cast<uint64_t>(Info.lastUpper < Info.wordLength[0]),
    static_cast<uint64_t>(Info.lastUpper < Info.lastLetter + Info.wordLength[1])), 12), 4096);
  m.set(finalize64(hash(Info.spaces & 0x1FFU, characterGroup, 
    static_cast<int>(Info.lastUpper < Info.wordLength[0]) |
    (static_cast<int>(Info.lastUpper < Info.lastLetter + Info.wordLength[1]) << 1U) |
    (static_cast<int>(Info.lastPunctuation < Info.lastLetter) << 2U) |
    (static_cast<int>(Info.lastPunctuation < Info.wordLength[0] + Info.wordGap) << 3U) | 
    (static_cast<int>(Info.lastPunctuation < Info.lastLetter + Info.wordLength[1] + Info.wordGap) << 4U)), 12), 4096);
  m.set(finalize64(hash(Info.firstLetter * static_cast<int>(Info.wordLength[0] < 4), min(6, Info.wordLength[0]), c0), 11), 2048);
  m.set(finalize64(hash((*pWord)[0], (*pWord)(0), min(4, Info.wordLength[0]), static_cast<uint64_t>(Info.lastPunctuation < Info.lastLetter)), 11), 2048);
  m.set(finalize64(hash(min(4, Info.wordLength[0]), characterGroup, 
    static_cast<uint64_t>(Info.lastUpper < Info.wordLength[0]),
    (Info.nestHash > 0) ? Info.nestHash & 0xFFU : 0x100U |
    (Info.firstLetter * static_cast<int>(Info.wordLength[0] > 0 && Info.wordLength[0] < 4))), 12), 4096);
  m.set(finalize64(hash(characterGroup, Info.masks[4] & 0x1FU, (Info.masks[4] >> 5) & 0x1FU), 13), 8192);
  m.set(finalize64(hash(characterGroup, uint8_t(pWord->embedding), Lang.id, State), 11), 2048);
}