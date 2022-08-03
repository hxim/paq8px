#pragma once

#include "Word.hpp"

class Language {
public:
  enum Flags {
      Verb = (1 << 0), Noun = (1 << 1)
  };
  enum Ids {
      Unknown, English, French, German, Count
  };

  virtual ~Language() = default;
  virtual bool isAbbreviation(Word *w) = 0;
};
