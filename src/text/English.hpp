#pragma once

#include "Language.hpp"
#include "Word.hpp"

class English : public Language {
private:
  static constexpr int numAbbrev = 6;
  const char *abbreviations[numAbbrev] = {"mr", "mrs", "ms", "dr", "st", "jr"};

public:
  enum Flags {
    Adjective = (1 << 2),
    Plural = (1 << 3),
    Male = (1 << 4),
    Female = (1 << 5),
    Negation = (1 << 6),
    PastTense = (1 << 7) | Verb,
    PresentParticiple = (1 << 8) | Verb,
    AdjectiveSuperlative = (1 << 9) | Adjective,
    AdjectiveWithout = (1 << 10) | Adjective,
    AdjectiveFull = (1 << 11) | Adjective,
    AdverbOfManner = (1 << 12),
    SuffixNESS = (1 << 13),
    SuffixITY = (1 << 14) | Noun,
    SuffixCapable = (1 << 15),
    SuffixNCE = (1 << 16),
    SuffixNT = (1 << 17),
    SuffixION = (1 << 18),
    SuffixAL = (1 << 19) | Adjective,
    SuffixIC = (1 << 20) | Adjective,
    SuffixIVE = (1 << 21),
    SuffixOUS = (1 << 22) | Adjective,
    PrefixOver = (1 << 23),
    PrefixUnder = (1 << 24)
  };

  bool isAbbreviation(Word *w) override;
};
