#include "FrenchStemmer.hpp"

bool FrenchStemmer::isConsonant(const char c) {
  return !isVowel(c);
}

void FrenchStemmer::convertUtf8(Word *w) {
  for( int i = w->start; i < w->end; i++ ) {
    uint8_t c = w->letters[i + 1] + ((w->letters[i + 1] < 0xA0) ? 0x60 : 0x40);
    if( w->letters[i] == 0xC3 && (isVowel(c) || (w->letters[i + 1] & 0xDF) == 0x87)) {
      w->letters[i] = c;
      if( i + 1 < w->end ) {
        memmove(&w->letters[i + 1], &w->letters[i + 2], w->end - i - 1);
      }
      w->end--;
    }
  }
}

void FrenchStemmer::markVowelsAsConsonants(Word *w) {
  for( int i = w->start; i <= w->end; i++ ) {
    switch( w->letters[i] ) {
      case 'i':
      case 'u': {
        if( i > w->start && i < w->end && (isVowel(w->letters[i - 1]) || (w->letters[i - 1] == 'q' && w->letters[i] == 'u')) &&
            isVowel(w->letters[i + 1])) {
          w->letters[i] = toupper(w->letters[i]);
        }
        break;
      }
      case 'y': {
        if((i > w->start && isVowel(w->letters[i - 1])) || (i < w->end && isVowel(w->letters[i + 1]))) {
          w->letters[i] = toupper(w->letters[i]);
        }
      }
    }
  }
}

uint32_t FrenchStemmer::getRv(Word *w) {
  uint32_t len = w->length();
  uint32_t res = w->start + len;
  if( len >= 3 && ((isVowel(w->letters[w->start]) && isVowel(w->letters[w->start + 1])) || w->startsWith("par") || w->startsWith("col") ||
                   w->startsWith("tap"))) {
    return w->start + 3;
  }
  for( int i = w->start + 1; i <= w->end; i++ ) {
    if( isVowel(w->letters[i])) {
      return i + 1;
    }
  }

  return res;
}

bool FrenchStemmer::step1(Word *w, const uint32_t rv, const uint32_t r1, const uint32_t r2, bool *forceStep2A) {
  int i = 0;
  for( ; i < 11; i++ ) {
    if( w->endsWith(suffixesStep1[i]) && suffixInRn(w, r2, suffixesStep1[i])) {
      w->end -= uint8_t(strlen(suffixesStep1[i]));
      if( i == 3 /*able*/) {
        w->type |= French::Adjective;
      }
      return true;
    }
  }
  for( ; i < 17; i++ ) {
    if( w->endsWith(suffixesStep1[i]) && suffixInRn(w, r2, suffixesStep1[i])) {
      w->end -= uint8_t(strlen(suffixesStep1[i]));
      if( w->endsWith("ic")) {
        w->changeSuffix("c", "qU");
      }
      return true;
    }
  }
  for( ; i < 25; i++ ) {
    if( w->endsWith(suffixesStep1[i]) && suffixInRn(w, r2, suffixesStep1[i])) {
      w->end -= uint8_t(strlen(suffixesStep1[i])) - 1 - static_cast<int>(i < 19) * 2;
      if( i > 22 ) {
        w->end += 2;
        w->letters[w->end] = 't';
      }
      return true;
    }
  }
  for( ; i < 27; i++ ) {
    if( w->endsWith(suffixesStep1[i]) && suffixInRn(w, r1, suffixesStep1[i]) &&
        isConsonant((*w)(static_cast<uint8_t>(strlen(suffixesStep1[i]))))) {
      w->end -= uint8_t(strlen(suffixesStep1[i]));
      return true;
    }
  }
  for( ; i < 29; i++ ) {
    if( w->endsWith(suffixesStep1[i]) && suffixInRn(w, rv, suffixesStep1[i])) {
      w->end -= uint8_t(strlen(suffixesStep1[i]));
      if( w->endsWith("iv") && suffixInRn(w, r2, "iv")) {
        w->end -= 2;
        if( w->endsWith("at") && suffixInRn(w, r2, "at")) {
          w->end -= 2;
        }
      } else if( w->endsWith("eus")) {
        if( suffixInRn(w, r2, "eus")) {
          w->end -= 3;
        } else if( suffixInRn(w, r1, "eus")) {
          w->letters[w->end] = 'x';
        }
      } else if((w->endsWith("abl") && suffixInRn(w, r2, "abl")) || (w->endsWith("iqU") && suffixInRn(w, r2, "iqU"))) {
        w->end -= 3;
      } else if((w->endsWith("i\xE8r") && suffixInRn(w, rv, "i\xE8r")) || (w->endsWith("I\xE8r") && suffixInRn(w, rv, "I\xE8r"))) {
        w->end -= 2;
        w->letters[w->end] = 'i';
      }
      return true;
    }
  }
  for( ; i < 31; i++ ) {
    if( w->endsWith(suffixesStep1[i]) && suffixInRn(w, r2, suffixesStep1[i])) {
      w->end -= uint8_t(strlen(suffixesStep1[i]));
      if( w->endsWith("abil")) {
        if( suffixInRn(w, r2, "abil")) {
          w->end -= 4;
        } else {
          w->end--, w->letters[w->end] = 'l';
        }
      } else if( w->endsWith("ic")) {
        if( suffixInRn(w, r2, "ic")) {
          w->end -= 2;
        } else {
          w->changeSuffix("c", "qU");
        }
      } else if( w->endsWith("iv") && suffixInRn(w, r2, "iv")) {
        w->end -= 2;
      }
      return true;
    }
  }
  for( ; i < 35; i++ ) {
    if( w->endsWith(suffixesStep1[i]) && suffixInRn(w, r2, suffixesStep1[i])) {
      w->end -= uint8_t(strlen(suffixesStep1[i]));
      if( w->endsWith("at") && suffixInRn(w, r2, "at")) {
        w->end -= 2;
        if( w->endsWith("ic")) {
          if( suffixInRn(w, r2, "ic")) {
            w->end -= 2;
          } else {
            w->changeSuffix("c", "qU");
          }
        }
      }
      return true;
    }
  }
  for( ; i < 37; i++ ) {
    if( w->endsWith(suffixesStep1[i])) {
      if( suffixInRn(w, r2, suffixesStep1[i])) {
        w->end -= uint8_t(strlen(suffixesStep1[i]));
        return true;
      }
      if( suffixInRn(w, r1, suffixesStep1[i])) {
        w->changeSuffix(suffixesStep1[i], "eux");
        return true;
      }
    }
  }
  for( ; i < numSuffixesStep1; i++ ) {
    if( w->endsWith(suffixesStep1[i]) && suffixInRn(w, rv + 1, suffixesStep1[i]) &&
        isVowel((*w)(static_cast<uint8_t>(strlen(suffixesStep1[i]))))) {
      w->end -= uint8_t(strlen(suffixesStep1[i]));
      (*forceStep2A) = true;
      return true;
    }
  }
  if( w->endsWith("eaux") || (*w) == "eaux" ) {
    w->end--;
    w->type |= French::Plural;
    return true;
  }
  if( w->endsWith("aux") && suffixInRn(w, r1, "aux")) {
    w->end--, w->letters[w->end] = 'l';
    w->type |= French::Plural;
    return true;
  }
  if( w->endsWith("amment") && suffixInRn(w, rv, "amment")) {
    w->changeSuffix("amment", "ant");
    (*forceStep2A) = true;
    return true;
  } else if( w->endsWith("emment") && suffixInRn(w, rv, "emment")) {
    w->changeSuffix("emment", "ent");
    (*forceStep2A) = true;
    return true;
  }
  return false;
}

bool FrenchStemmer::step2A(Word *w, const uint32_t rv) {
  for( int i = 0; i < numSuffixesStep2A; i++ ) {
    if( w->endsWith(suffixesStep2A[i]) && suffixInRn(w, rv + 1, suffixesStep2A[i]) &&
        isConsonant((*w)(static_cast<uint8_t>(strlen(suffixesStep2A[i]))))) {
      w->end -= uint8_t(strlen(suffixesStep2A[i]));
      if( i == 31 /*ir*/) {
        w->type |= French::Verb;
      }
      return true;
    }
  }
  return false;
}

bool FrenchStemmer::step2B(Word *w, const uint32_t rv, const uint32_t r2) {
  for( int i = 0; i < numSuffixesStep2B; i++ ) {
    if( w->endsWith(suffixesStep2B[i]) && suffixInRn(w, rv, suffixesStep2B[i])) {
      switch( suffixesStep2B[i][0] ) {
        case 'a':
        case '\xE2': {
          w->end -= uint8_t(strlen(suffixesStep2B[i]));
          if( w->endsWith("e") && suffixInRn(w, rv, "e")) {
            w->end--;
          }
          return true;
        }
        default: {
          if( i != 14 || suffixInRn(w, r2, suffixesStep2B[i])) {
            w->end -= uint8_t(strlen(suffixesStep2B[i]));
            return true;
          }
        }
      }
    }
  }
  return false;
}

void FrenchStemmer::step3(Word *w) {
  char *final = reinterpret_cast<char *>(&w->letters[w->end]);
  if((*final) == 'Y' ) {
    (*final) = 'i';
  } else if((*final) == '\xE7' ) {
    (*final) = 'c';
  }
}

bool FrenchStemmer::step4(Word *w, const uint32_t rv, const uint32_t r2) {
  bool res = false;
  if( w->length() >= 2 && w->letters[w->end] == 's' && !charInArray((*w)(1), setStep4, numSetStep4)) {
    w->end--;
    res = true;
  }
  for( int i = 0; i < numSuffixesStep4; i++ ) {
    if( w->endsWith(suffixesStep4[i]) && suffixInRn(w, rv, suffixesStep4[i])) {
      switch( i ) {
        case 2: { //ion
          char prec = (*w)(3);
          if( suffixInRn(w, r2, suffixesStep4[i]) && suffixInRn(w, rv + 1, suffixesStep4[i]) && (prec == 's' || prec == 't')) {
            w->end -= 3;
            return true;
          }
          break;
        }
        case 5: { //e
          w->end--;
          return true;
        }
        case 6: { //\xEB
          if( w->endsWith("gu\xEB")) {
            w->end--;
            return true;
          }
          break;
        }
        default: {
          w->changeSuffix(suffixesStep4[i], "i");
          return true;
        }
      }
    }
  }
  return res;
}

bool  FrenchStemmer::step5(Word *w) {
  for( int i = 0; i < numSuffixesStep5; i++ ) {
    if( w->endsWith(suffixesStep5[i])) {
      w->end--;
      return true;
    }
  }
  return false;
}

bool FrenchStemmer::step6(Word *w) {
  for( int i = w->end; i >= w->start; i-- ) {
    if( isVowel(w->letters[i])) {
      if( i < w->end && (w->letters[i] & 0xFE) == 0xE8 ) {
        w->letters[i] = 'e';
        return true;
      }
      return false;
    }
  }
  return false;
}

bool FrenchStemmer::isVowel(const char c){
  return charInArray(c, vowels, numVowels);
}

bool FrenchStemmer::stem(Word *w) {
  convertUtf8(w);
  if( w->length() < 2 ) {
    w->calculateStemHash();
    return false;
  }
  for( int i = 0; i < numExceptions; i++ ) {
    if((*w) == exceptions[i][0] ) {
      size_t len = strlen(exceptions[i][1]);
      memcpy(&w->letters[w->start], exceptions[i][1], len);
      w->end = w->start + uint8_t(len - 1);
      w->calculateStemHash();
      w->type |= typesExceptions[i];
      w->language = Language::French;
      return true;
    }
  }
  markVowelsAsConsonants(w);
  uint32_t rv = getRv(w);
  uint32_t r1 = getRegion(w, 0);
  uint32_t r2 = getRegion(w, r1);
  bool doNextStep = false;
  bool res = step1(w, rv, r1, r2, &doNextStep);
  doNextStep |= !res;
  if( doNextStep ) {
    doNextStep = !step2A(w, rv);
    res |= !doNextStep;
    if( doNextStep ) {
      res |= step2B(w, rv, r2);
    }
  }
  if( res ) {
    step3(w);
  } else {
    res |= step4(w, rv, r2);
  }
  res |= step5(w);
  res |= step6(w);
  for( int i = w->start; i <= w->end; i++ ) {
    w->letters[i] = tolower(w->letters[i]);
  }
  if( !res ) {
    res = w->matchesAny(commonWords, numCommonWords);
  }
  w->calculateStemHash();
  if( res ) {
    w->language = Language::French;
  }
  return res;
}
