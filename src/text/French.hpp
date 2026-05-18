#pragma once

#include "Language.hpp"
#include "Word.hpp"

class French : public Language {
private:
  static constexpr int numAbbrev = 2;
  const char *abbreviations[numAbbrev] = {"m", "mm"};

public:
  enum Flags {
      Adjective = (1 << 2), Plural = (1 << 3)
  };
  bool  isAbbreviation(Word *w) override;
};
