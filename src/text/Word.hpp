#pragma once

#include "../Hash.hpp"
#include <cctype>
#include <cmath>
#include <cstdio>

class Word {
private:
  uint64_t calculateHash();
public:
  constexpr static int maxWordSize = 64;
  constexpr static int wordEmbeddingSize = 3;
  uint8_t letters[maxWordSize] {};
  uint8_t start {};
  uint8_t end {};
  uint64_t Hash[2] {};
  uint32_t type {};
  uint32_t language {};
  uint32_t embedding {};
  Word();
  void reset();
  bool operator==(const char *s) const;
  bool operator!=(const char *s) const;
  void operator+=(char c);
  uint32_t operator-(Word w) const;
  uint32_t operator+(Word w) const;
  uint8_t operator[](uint8_t i) const;
  uint8_t operator()(uint8_t i) const;
  uint32_t length() const;
  uint32_t distanceTo(Word w) const;
  void calculateWordHash();

  /**
    * Called by a stemmer after stemming
    */
  void calculateStemHash();
  bool changeSuffix(const char *oldSuffix, const char *newSuffix);
  bool matchesAny(const char **a, int count);
  bool endsWith(const char *suffix) const;
  bool startsWith(const char *prefix) const;
  void print() const;
};
