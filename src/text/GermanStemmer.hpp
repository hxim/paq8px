#pragma once

#include "German.hpp"
#include "Language.hpp"
#include "Stemmer.hpp"
#include "Word.hpp"
#include <cctype>
#include <cstdint>

/**
 * German suffix stemmer, based on the Porter stemmer.
 */
class GermanStemmer : public Stemmer {
private:
  static constexpr int numVowels = 9;
  static constexpr char vowels[numVowels] = {'a', 'e', 'i', 'o', 'u', 'y', '\xE4', '\xF6', '\xFC'};
  static constexpr int numCommonWords = 10;
  const char *commonWords[numCommonWords] = {"der", "die", "das", "und", "sie", "ich", "mit", "sich", "auf", "nicht"};
  static constexpr int numEndings = 10;
  static constexpr char endings[numEndings] = {'b', 'd', 'f', 'g', 'h', 'k', 'l', 'm', 'n', 't'}; /**< plus 'r' for words ending in 's' */
  static constexpr int numSuffixesStep1 = 6;
  const char *suffixesStep1[numSuffixesStep1] = {"em", "ern", "er", "e", "en", "es"};
  static constexpr int numSuffixesStep2 = 3;
  const char *suffixesStep2[numSuffixesStep2] = {"en", "er", "est"};
  static constexpr int numSuffixesStep3 = 7;
  const char *suffixesStep3[numSuffixesStep3] = {"end", "ung", "ik", "ig", "isch", "lich", "heit"};

  void convertUtf8(Word *w);
  static void replaceSharpS(Word *w);
  void markVowelsAsConsonants(Word *w);
  static inline bool isValidEnding(char c, bool includeR = false);
  bool step1(Word *w, uint32_t r1);
  bool step2(Word *w, uint32_t r1);
  bool step3(Word *w, uint32_t r1, uint32_t r2);

public:
  bool isVowel(char c) final;
  bool  stem(Word *w) override;
};
