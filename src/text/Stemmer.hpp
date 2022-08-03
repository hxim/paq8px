#pragma once

#include "Word.hpp"

class Stemmer {
protected:
  uint32_t getRegion(const Word *w, uint32_t from);
  static bool suffixInRn(const Word *w, uint32_t rn, const char *suffix);
  static bool charInArray(char c, const char a[], int len);
public:
  virtual ~Stemmer() = default;
  virtual bool isVowel(char c) = 0;
  virtual bool stem(Word *w) = 0;
};
