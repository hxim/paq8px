#pragma once

#include "Language.hpp"
#include "Word.hpp"

class French : public Language {
private:
    static constexpr int numAbbrev = 2;
    const char *abbreviations[numAbbrev] = {"m", "mm"};

public:
    enum Flags {
        Adjective = (1U << 2U), Plural = (1U << 3U)
    };
    auto isAbbreviation(Word *w) -> bool override;
};
