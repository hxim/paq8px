#include "Stemmer.hpp"

uint32_t Stemmer::getRegion(const Word *w, const uint32_t from) {
  bool hasVowel = false;
  for( int i = w->start + from; i <= w->end; i++ ) {
    if( isVowel(w->letters[i])) {
      hasVowel = true;
      continue;
    }
    if( hasVowel ) {
      return i - w->start + 1;
    }
  }
  return w->start + w->length();
}

bool Stemmer::suffixInRn(const Word *w, const uint32_t rn, const char *suffix) {
  return (w->start != w->end && rn <= w->length() - strlen(suffix));
}

bool Stemmer::charInArray(const char c, const char *a, const int len) {
  if( a == nullptr ) {
    return false;
  }
  int i = 0;
  for( ; i < len && c != a[i]; i++ ) { ;
  }
  return i < len;
}
