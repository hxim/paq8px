#pragma once

#include "Language.hpp"
#include "Word.hpp"

class English : public Language {
private:
    static constexpr int numAbbrev = 6;
    const char *abbreviations[numAbbrev] = {"mr", "mrs", "ms", "dr", "st", "jr"};

public:
    enum Flags {
        Adjective = (1U << 2U),
        Plural = (1U << 3U),
        Male = (1U << 4U),
        Female = (1U << 5U),
        Negation = (1U << 6U),
        PastTense = (1U << 7U) | Verb,
        PresentParticiple = (1U << 8U) | Verb,
        AdjectiveSuperlative = (1U << 9U) | Adjective,
        AdjectiveWithout = (1U << 10U) | Adjective,
        AdjectiveFull = (1U << 11U) | Adjective,
        AdverbOfManner = (1U << 12U),
        SuffixNESS = (1U << 13U),
        SuffixITY = (1U << 14U) | Noun,
        SuffixCapable = (1U << 15U),
        SuffixNCE = (1U << 16U),
        SuffixNT = (1U << 17U),
        SuffixION = (1U << 18U),
        SuffixAL = (1U << 19U) | Adjective,
        SuffixIC = (1U << 20U) | Adjective,
        SuffixIVE = (1U << 21U),
        SuffixOUS = (1U << 22U) | Adjective,
        PrefixOver = (1U << 23U),
        PrefixUnder = (1U << 24U)
    };

    auto isAbbreviation(Word *w) -> bool override;
};
