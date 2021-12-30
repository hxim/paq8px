#pragma once

#include "Language.hpp"
#include "Word.hpp"

class German : public Language {
private:
    static constexpr int numAbbrev = 3;
    const char *abbreviations[numAbbrev] = {"fr", "hr", "hrn"};

public:
    enum Flags {
        Adjective = (1U << 2U), Plural = (1U << 3U), Female = (1U << 4U)
    };

    auto isAbbreviation(Word *w) -> bool override;
};
