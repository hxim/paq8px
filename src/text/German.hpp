#pragma once

#include "Language.hpp"
#include "Word.hpp"

class German : public Language {
private:
  static constexpr int numAbbrev = 3;
  const char *abbreviations[numAbbrev] = {"fr", "hr", "hrn"};

public:
  enum Flags {
      Adjective = (1 << 2), Plural = (1 << 3), Female = (1 << 4)
  };

  bool  isAbbreviation(Word *w) override;
};
