#include "EnglishStemmer.hpp"
#include "../CharacterNames.hpp"

bool EnglishStemmer::isConsonant(const char c) {
  return !isVowel(c);
}

bool EnglishStemmer::isShortConsonant(const char c) {
  return !charInArray(c, nonShortConsonants, numNonShortConsonants);
}

bool EnglishStemmer::isDouble(const char c) {
  return charInArray(c, doubles, numDoubles);
}

bool EnglishStemmer::isLiEnding(const char c) {
  return charInArray(c, liEndings, numLiEndings);
}

uint32_t EnglishStemmer::getRegion1(const Word *w) {
  for( int i = 0; i < numExceptionRegion1; i++ ) {
    if( w->startsWith(exceptionsRegion1[i])) {
      return uint32_t(strlen(exceptionsRegion1[i]));
    }
  }
  return getRegion(w, 0);
}

bool EnglishStemmer::endsInShortSyllable(const Word *w) {
  if( w->end == w->start ) {
    return false;
  }
  if( w->end == w->start + 1 ) {
    return isVowel((*w)(1)) && isConsonant((*w)(0));
  }
  return (isConsonant((*w)(2)) && isVowel((*w)(1)) && isConsonant((*w)(0)) && isShortConsonant((*w)(0)));

}

bool EnglishStemmer::isShortWord(const Word *w) {
  return (endsInShortSyllable(w) && getRegion1(w) == w->length());
}

bool EnglishStemmer::hasVowels(const Word *w) {
  for( int i = w->start; i <= w->end; i++ ) {
    if( isVowel(w->letters[i])) {
      return true;
    }
  }
  return false;
}

bool EnglishStemmer::trimApostrophes(Word *w) {
  bool result = false;
  //trim all apostrophes from the beginning
  int cnt = 0;
  while( w->start != w->end && (*w)[0] == APOSTROPHE ) {
    result = true;
    w->start++;
    cnt++;
  }
  //trim the same number of apostrophes from the end (if there are)
  while( w->start != w->end && (*w)(0) == APOSTROPHE ) {
    if( cnt == 0 ) {
      break;
    }
    w->end--;
    cnt--;
  }
  return result;
}

void EnglishStemmer::markYsAsConsonants(Word *w) {
  if((*w)[0] == 'y' ) {
    w->letters[w->start] = 'Y';
  }
  for( int i = w->start + 1; i <= w->end; i++ ) {
    if( isVowel(w->letters[i - 1]) && w->letters[i] == 'y' ) {
      w->letters[i] = 'Y';
    }
  }
}

bool EnglishStemmer::processPrefixes(Word *w) {
  if( w->startsWith("irr") && w->length() > 5 && ((*w)[3] == 'a' || (*w)[3] == 'e')) {
    w->start += 2, w->type |= English::Negation;
  } else if( w->startsWith("over") && w->length() > 5 ) {
    w->start += 4, w->type |= English::PrefixOver;
  } else if( w->startsWith("under") && w->length() > 6 ) {
    w->start += 5, w->type |= English::PrefixUnder;
  } else if( w->startsWith("unn") && w->length() > 5 ) {
    w->start += 2, w->type |= English::Negation;
  } else if( w->startsWith("non") && w->length() > static_cast<uint32_t>(5 + static_cast<int>((*w)[3] == '-'))) {
    w->start += 2 + static_cast<int>((*w)[3] == '-'), w->type |= English::Negation;
  } else {
    return false;
  }
  return true;
}

bool EnglishStemmer::processSuperlatives(Word *w) {
  if( w->endsWith("est") && w->length() > 4 ) {
    uint8_t i = w->end;
    w->end -= 3;
    w->type |= English::AdjectiveSuperlative;

    if((*w)(0) == (*w)(1) && (*w)(0) != 'r' && !(w->length() >= 4 && memcmp("sugg", &w->letters[w->end - 3], 4) == 0)) {
      w->end -= static_cast<int>((((*w)(0) != 'f' && (*w)(0) != 'l' && (*w)(0) != 's') ||
                                  (w->length() > 4 && (*w)(1) == 'l' && ((*w)(2) == 'u' || (*w)(3) == 'u' || (*w)(3) == 'v'))) &&
                                 (!(w->length() == 3 && (*w)(1) == 'd' && (*w)(2) == 'o')));
      if( w->length() == 2 && ((*w)[0] != 'i' || (*w)[1] != 'n')) {
        w->end = i, w->type &= ~English::AdjectiveSuperlative;
      }
    } else {
      switch((*w)(0)) {
        case 'd':
        case 'k':
        case 'm':
        case 'y':
          break;
        case 'g': {
          if( !(w->length() > 3 && ((*w)(1) == 'n' || (*w)(1) == 'r') && memcmp("cong", &w->letters[w->end - 3], 4) != 0)) {
            w->end = i, w->type &= ~English::AdjectiveSuperlative;
          } else {
            w->end += static_cast<int>((*w)(2) == 'a');
          }
          break;
        }
        case 'i': {
          w->letters[w->end] = 'y';
          break;
        }
        case 'l': {
          if( w->end == w->start + 1 || memcmp("mo", &w->letters[w->end - 2], 2) == 0 ) {
            w->end = i, w->type &= ~English::AdjectiveSuperlative;
          } else {
            w->end += static_cast<int>(isConsonant((*w)(1)));
          }
          break;
        }
        case 'n': {
          if( w->length() < 3 || isConsonant((*w)(1)) || isConsonant((*w)(2))) {
            w->end = i, w->type &= ~English::AdjectiveSuperlative;
          }
          break;
        }
        case 'r': {
          if( w->length() > 3 && isVowel((*w)(1)) && isVowel((*w)(2))) {
            w->end += static_cast<int>(((*w)(2) == 'u') && ((*w)(1) == 'a' || (*w)(1) == 'i'));
          } else {
            w->end = i, w->type &= ~English::AdjectiveSuperlative;
          }
          break;
        }
        case 's': {
          w->end++;
          break;
        }
        case 'w': {
          if( !(w->length() > 2 && isVowel((*w)(1)))) {
            w->end = i, w->type &= ~English::AdjectiveSuperlative;
          }
          break;
        }
        case 'h': {
          if( !(w->length() > 2 && isConsonant((*w)(1)))) {
            w->end = i, w->type &= ~English::AdjectiveSuperlative;
          }
          break;
        }
        default: {
          w->end += 3;
          w->type &= ~English::AdjectiveSuperlative;
        }
      }
    }
  }
  return (w->type & English::AdjectiveSuperlative) > 0;
}

bool EnglishStemmer::step0(Word *w) {
  for( int i = 0; i < numSuffixesStep0; i++ ) {
    if( w->endsWith(suffixesStep0[i])) {
      w->end -= uint8_t(strlen(suffixesStep0[i]));
      w->type |= English::Plural;
      return true;
    }
  }
  return false;
}

bool EnglishStemmer::step1A(Word *w) {
  //sses -> replace by ss
  if( w->endsWith("sses")) {
    w->end -= 2;
    w->type |= English::Plural;
    return true;
  }

  //ied+ ies* -> replace by i if preceded by more than one letter, otherwise by ie (so ties -> tie, cries -> cri)
  if( w->endsWith("ied") || w->endsWith("ies")) {
    w->type |= ((*w)(0) == 'd') ? English::PastTense : English::Plural;
    w->end -= w->length() > 4 ? 2 : 1;
    return true;
  }

  //us+ ss -> do nothing
  if( w->endsWith("us") || w->endsWith("ss")) {
    return false;
  }

  //s -> delete if the preceding word part contains a vowel not immediately before the s (so gas and this retain the s, gaps and kiwis lose it)
  if((*w)(0) == 's' && w->length() > 2 ) {
    for( int i = w->start; i <= w->end - 2; i++ ) {
      if( isVowel(w->letters[i])) {
        w->end--;
        w->type |= English::Plural;
        return true;
      }
    }
  }

  if( w->endsWith("n't") && w->length() > 4 ) {
    switch((*w)(3)) {
      case 'a': {
        if((*w)(4) == 'c' ) {
          w->end -= 2; // can't -> can
        } else {
          w->changeSuffix("n't", "ll"); // shan't -> shall
        }
        break;
      }
      case 'i': {
        w->changeSuffix("in't", "m");
        break;
      } // ain't -> am
      case 'o': {
        if((*w)(4) == 'w' ) {
          w->changeSuffix("on't", "ill"); // won't -> will
        } else {
          w->end -= 3; //don't -> do
        }
        break;
      }
      default:
        w->end -= 3; // couldn't -> could, shouldn't -> should, needn't -> need, hasn't -> has
    }
    w->type |= English::Negation;
    return true;
  }

  if( w->endsWith("hood") && w->length() > 7 ) {
    w->end -= 4; //brotherhood -> brother
    return true;
  }
  return false;
}

bool EnglishStemmer::step1B(Word *w, const uint32_t r1) {
  for( int i = 0; i < numSuffixesStep1B; i++ ) {
    if( w->endsWith(suffixesStep1B[i])) {
      switch( i ) {
        case 0:
        case 1: {
          if( suffixInRn(w, r1, suffixesStep1B[i])) {
            w->end -= 1 + i * 2;
          }
          break;
        }
        default: {
          uint8_t j = w->end;
          w->end -= uint8_t(strlen(suffixesStep1B[i]));
          if( hasVowels(w)) {
            if( w->endsWith("at") || w->endsWith("bl") || w->endsWith("iz") || isShortWord(w)) {
              (*w) += 'e';
            } else if( w->length() > 2 ) {
              if((*w)(0) == (*w)(1) && isDouble((*w)(0))) {
                w->end--;
              } else if( i == 2 || i == 3 ) {
                switch((*w)(0)) {
                  case 'c':
                  case 's':
                  case 'v': {
                    w->end += static_cast<int>(!(w->endsWith("ss") || w->endsWith("ias")));
                    break;
                  }
                  case 'd': {
                    static constexpr char nAllowed[4] = {'a', 'e', 'i', 'o'};
                    w->end += static_cast<int>(isVowel((*w)(1)) && (!charInArray((*w)(2), nAllowed, 4)));
                    break;
                  }
                  case 'k': {
                    w->end += static_cast<int>(w->endsWith("uak"));
                    break;
                  }
                  case 'l': {
                    static constexpr char allowed1[10] = {'b', 'c', 'd', 'f', 'g', 'k', 'p', 't', 'y', 'z'};
                    static constexpr char allowed2[4] = {'a', 'i', 'o', 'u'};
                    w->end += static_cast<int>(charInArray((*w)(1), allowed1, 10) ||
                                               (charInArray((*w)(1), allowed2, 4) && isConsonant((*w)(2))));
                    break;
                  }
                }
              } else if( i >= 4 ) {
                switch((*w)(0)) {
                  case 'd': {
                    if( isVowel((*w)(1)) && (*w)(2) != 'a' && (*w)(2) != 'e' && (*w)(2) != 'o' ) {
                      (*w) += 'e';
                    }
                    break;
                  }
                  case 'g': {
                    static constexpr char allowed[7] = {'a', 'd', 'e', 'i', 'l', 'r', 'u'};
                    if( charInArray((*w)(1), allowed, 7) || ((*w)(1) == 'n' &&
                                                             ((*w)(2) == 'e' || ((*w)(2) == 'u' && (*w)(3) != 'b' && (*w)(3) != 'd') ||
                                                              ((*w)(2) == 'a' && ((*w)(3) == 'r' || ((*w)(3) == 'h' && (*w)(4) == 'c'))) ||
                                                              (w->endsWith("ring") && ((*w)(4) == 'c' || (*w)(4) == 'f'))))) {
                      (*w) += 'e';
                    }
                    break;
                  }
                  case 'l': {
                    if( !((*w)(1) == 'l' || (*w)(1) == 'r' || (*w)(1) == 'w' || (isVowel((*w)(1)) && isVowel((*w)(2))))) {
                      (*w) += 'e';
                    }
                    if( w->endsWith("uell") && w->length() > 4 && (*w)(4) != 'q' ) {
                      w->end--;
                    }
                    break;
                  }
                  case 'r': {
                    if((((*w)(1) == 'i' && (*w)(2) != 'a' && (*w)(2) != 'e' && (*w)(2) != 'o') ||
                        ((*w)(1) == 'a' && (!((*w)(2) == 'e' || (*w)(2) == 'o' || ((*w)(2) == 'l' && (*w)(3) == 'l')))) ||
                        ((*w)(1) == 'o' && (!((*w)(2) == 'o' || ((*w)(2) == 't' && (*w)(3) != 's')))) || (*w)(1) == 'c' ||
                        (*w)(1) == 't') && (!w->endsWith("str"))) {
                      (*w) += 'e';
                    }
                    break;
                  }
                  case 't': {
                    if((*w)(1) == 'o' && (*w)(2) != 'g' && (*w)(2) != 'l' && (*w)(2) != 'i' && (*w)(2) != 'o' ) {
                      (*w) += 'e';
                    }
                    break;
                  }
                  case 'u': {
                    if( !(w->length() > 3 && isVowel((*w)(1)) && isVowel((*w)(2)))) {
                      (*w) += 'e';
                    }
                    break;
                  }
                  case 'z': {
                    if( w->endsWith("izz") && w->length() > 3 && ((*w)(3) == 'h' || (*w)(3) == 'u')) {
                      w->end--;
                    } else if((*w)(1) != 't' && (*w)(1) != 'z' ) {
                      (*w) += 'e';
                    }
                    break;
                  }
                  case 'k': {
                    if( w->endsWith("uak")) {
                      (*w) += 'e';
                    }
                    break;
                  }
                  case 'b':
                  case 'c':
                  case 's':
                  case 'v': {
                    if( !(((*w)(0) == 'b' && ((*w)(1) == 'm' || (*w)(1) == 'r')) || w->endsWith("ss") || w->endsWith("ias") ||
                          (*w) == "zinc")) {
                      (*w) += 'e';
                    }
                    break;
                  }
                }
              }
            }
          } else {
            w->end = j;
            return false;
          }
        }
      }
      w->type |= typesStep1B[i];
      return true;
    }
  }
  return false;
}

bool EnglishStemmer::step1C(Word *w) {
  if( w->length() > 2 && tolower((*w)(0)) == 'y' && isConsonant((*w)(1))) {
    w->letters[w->end] = 'i';
    return true;
  }
  return false;
}

bool EnglishStemmer::step2(Word *w, const uint32_t r1) {
  for( int i = 0; i < numSuffixesStep2; i++ ) {
    if( w->endsWith(suffixesStep2[i][0]) && suffixInRn(w, r1, suffixesStep2[i][0])) {
      w->changeSuffix(suffixesStep2[i][0], suffixesStep2[i][1]);
      w->type |= typesStep2[i];
      return true;
    }
  }
  if( w->endsWith("logi") && suffixInRn(w, r1, "ogi")) {
    w->end--;
    return true;
  }
  if( w->endsWith("li")) {
    if( suffixInRn(w, r1, "li") && isLiEnding((*w)(2))) {
      w->end -= 2;
      w->type |= English::AdverbOfManner;
      return true;
    }
    if( w->length() > 3 ) {
      switch((*w)(2)) {
        case 'b': {
          w->letters[w->end] = 'e';
          w->type |= English::AdverbOfManner;
          return true;
        }
        case 'i': {
          if( w->length() > 4 ) {
            w->end -= 2;
            w->type |= English::AdverbOfManner;
            return true;
          }
          break;
        }
        case 'l': {
          if( w->length() > 5 && ((*w)(3) == 'a' || (*w)(3) == 'u')) {
            w->end -= 2;
            w->type |= English::AdverbOfManner;
            return true;
          }
          break;
        }
        case 's': {
          w->end -= 2;
          w->type |= English::AdverbOfManner;
          return true;
        }
        case 'e':
        case 'g':
        case 'm':
        case 'n':
        case 'r':
        case 'w': {
          if( w->length() > static_cast<uint32_t>(4 + static_cast<int>((*w)(2) == 'r'))) {
            w->end -= 2;
            w->type |= English::AdverbOfManner;
            return true;
          }
        }
      }
    }
  }
  return false;
}

bool EnglishStemmer::step3(Word *w, const uint32_t r1, const uint32_t r2) {
  bool res = false;
  for( int i = 0; i < numSuffixesStep3; i++ ) {
    if( w->endsWith(suffixesStep3[i][0]) && suffixInRn(w, r1, suffixesStep3[i][0])) {
      w->changeSuffix(suffixesStep3[i][0], suffixesStep3[i][1]);
      w->type |= typesStep3[i];
      res = true;
      break;
    }
  }
  if( w->endsWith("ative") && suffixInRn(w, r2, "ative")) {
    w->end -= 5;
    w->type |= English::SuffixIVE;
    return true;
  }
  if( w->length() > 5 && w->endsWith("less")) {
    w->end -= 4;
    w->type |= English::AdjectiveWithout;
    return true;
  }
  return res;
}

bool EnglishStemmer::step4(Word *w, const uint32_t r2) {
  bool res = false;
  for( int i = 0; i < numSuffixesStep4; i++ ) {
    if( w->endsWith(suffixesStep4[i]) && suffixInRn(w, r2, suffixesStep4[i])) {
      w->end -= uint8_t(strlen(suffixesStep4[i]) - static_cast<unsigned long>(i >
                                                                              17) /*sion, tion*/); // remove: al   ance   ence   er   ic   able   ible   ant   ement   ment   ent   ism   ate   iti   ous   ive   ize ; ion -> delete if preceded by s or t
      if( !(i == 10 /* ent */ && (*w)(0) == 'm')) { //exception: no typesStep4 should be assigned for 'agreement'
        w->type |= typesStep4[i];
      }
      if( i == 0 && w->endsWith("nti")) {
        w->end--; // ntial -> nt
        res = true; // process ant+ial, ent+ial
        continue;
      }
      return true;
    }
  }
  return res;
}

bool EnglishStemmer::step5(Word *w, const uint32_t r1, const uint32_t r2) {
  if((*w)(0) == 'e' && (*w) != "here" ) {
    if( suffixInRn(w, r2, "e")) {
      w->end--; //e -> delete if in r2, or in r1 and not preceded by a short syllable
    } else if( suffixInRn(w, r1, "e")) {
      w->end--; //e -> delete if in r1 and not preceded by a short syllable
      w->end += static_cast<int>(endsInShortSyllable(w));
    } else {
      return false;
    }
    return true;
  }
  if( w->length() > 1 && (*w)(0) == 'l' && suffixInRn(w, r2, "l") && (*w)(1) == 'l' ) {
    w->end--; //l -> delete if in r2 and preceded by l
    return true;
  }
  return false;
}

bool EnglishStemmer::isVowel(const char c) {
  return charInArray(c, vowels, numVowels);
}

bool EnglishStemmer::stem(Word *w) {
  if( w->length() < 2 ) {
    w->calculateStemHash();
    //w->print(); //for debugging
    return false;
  }
  bool res = trimApostrophes(w);
  res |= processPrefixes(w);
  res |= processSuperlatives(w);
  for( int i = 0; i < numExceptions1; i++ ) {
    if((*w) == exceptions1[i][0] ) {
      if( i < 11 ) {
        size_t len = strlen(exceptions1[i][1]);
        memcpy(&w->letters[w->start], exceptions1[i][1], len);
        w->end = w->start + uint8_t(len - 1);
      }
      w->calculateStemHash();
      w->type |= typesExceptions1[i];
      w->language = Language::English;
      //w->print(); //for debugging
      return (i < 11);
    }
  }

  // start of modified Porter2 Stemmer
  markYsAsConsonants(w);
  uint32_t r1 = getRegion1(w);
  uint32_t r2 = getRegion(w, r1);
  res |= step0(w);
  res |= step1A(w);
  for( int i = 0; i < numExceptions2; i++ ) {
    if((*w) == exceptions2[i] ) {
      w->calculateStemHash();
      w->type |= typesExceptions2[i];
      w->language = Language::English;
      //w->print(); //for debugging
      return res;
    }
  }
  res |= step1B(w, r1);
  res |= step1C(w);
  res |= step2(w, r1);
  res |= step3(w, r1, r2);
  res |= step4(w, r2);
  res |= step5(w, r1, r2);

  for( uint8_t i = w->start; i <= w->end; i++ ) {
    if( w->letters[i] == 'Y' ) {
      w->letters[i] = 'y';
    }
  }
  if((w->type == 0) || w->type == English::Plural ) {
    if( w->matchesAny(maleWords, numMaleWords)) {
      res = true, w->type |= English::Male;
    } else if( w->matchesAny(femaleWords, numFemaleWords)) {
      res = true, w->type |= English::Female;
    }
  }
  if( !res ) {
    res = w->matchesAny(commonWords, numCommonWords);
  }
  w->calculateStemHash();
  if( res ) {
    w->language = Language::English;
  }
  //w->print(); //for debugging
  return res;
}
