#pragma once

#include "Word.hpp"

class Language {
public:
    enum Flags {
        Verb = (1U << 0U), Noun = (1U << 1U)
    };
    enum Ids {
        Unknown, English, French, German, Count
    };

    virtual ~Language() = default;
    virtual auto isAbbreviation(Word *w) -> bool = 0;
};
