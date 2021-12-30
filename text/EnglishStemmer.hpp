#pragma once

#include "English.hpp"
#include "Stemmer.hpp"
#include <cctype>
#include <cstdint>

/**
 * English affix stemmer, based on the Porter2 stemmer.
 */
class EnglishStemmer : public Stemmer {
private:
    static constexpr int numVowels = 6;
    static constexpr char vowels[numVowels] = {'a', 'e', 'i', 'o', 'u', 'y'};
    static constexpr int numDoubles = 9;
    static constexpr char doubles[numDoubles] = {'b', 'd', 'f', 'g', 'm', 'n', 'p', 'r', 't'};
    static constexpr int numLiEndings = 10;
    static constexpr char liEndings[numLiEndings] = {'c', 'd', 'e', 'g', 'h', 'k', 'm', 'n', 'r', 't'};
    static constexpr int numNonShortConsonants = 3;
    static constexpr char nonShortConsonants[numNonShortConsonants] = {'w', 'x', 'Y'};
    static constexpr int numMaleWords = 9;
    const char *maleWords[numMaleWords] = {"he", "him", "his", "himself", "man", "men", "boy", "husband", "actor"};
    static constexpr int numFemaleWords = 8;
    const char *femaleWords[numFemaleWords] = {"she", "her", "herself", "woman", "women", "girl", "wife", "actress"};
    static constexpr int numCommonWords = 12;
    const char *commonWords[numCommonWords] = {"the", "be", "to", "of", "and", "in", "that", "you", "have", "with", "from", "but"};
    static constexpr int numSuffixesStep0 = 3;
    const char *suffixesStep0[numSuffixesStep0] = {"'s'", "'s", "'"};
    static constexpr int numSuffixesStep1B = 6;
    const char *suffixesStep1B[numSuffixesStep1B] = {"eedly", "eed", "ed", "edly", "ing", "ingly"};
    static constexpr uint32_t typesStep1B[numSuffixesStep1B] = {English::AdverbOfManner, 0, English::PastTense,
                                                                English::AdverbOfManner | English::PastTense, English::PresentParticiple,
                                                                English::AdverbOfManner | English::PresentParticiple};
    static constexpr int numSuffixesStep2 = 22;
    const char *(suffixesStep2[numSuffixesStep2])[2] = {{"ization", "ize"},
                                                        {"ational", "ate"},
                                                        {"ousness", "ous"},
                                                        {"iveness", "ive"},
                                                        {"fulness", "ful"},
                                                        {"tional",  "tion"},
                                                        {"lessli",  "less"},
                                                        {"biliti",  "ble"},
                                                        {"entli",   "ent"},
                                                        {"ation",   "ate"},
                                                        {"alism",   "al"},
                                                        {"aliti",   "al"},
                                                        {"fulli",   "ful"},
                                                        {"ousli",   "ous"},
                                                        {"iviti",   "ive"},
                                                        {"enci",    "ence"},
                                                        {"anci",    "ance"},
                                                        {"abli",    "able"},
                                                        {"izer",    "ize"},
                                                        {"ator",    "ate"},
                                                        {"alli",    "al"},
                                                        {"bli",     "ble"}};
    const uint32_t typesStep2[numSuffixesStep2] = {English::SuffixION, English::SuffixION | English::SuffixAL, English::SuffixNESS,
                                                   English::SuffixNESS, English::SuffixNESS, English::SuffixION | English::SuffixAL,
                                                   English::AdverbOfManner, English::AdverbOfManner | English::SuffixITY,
                                                   English::AdverbOfManner, English::SuffixION, 0, English::SuffixITY,
                                                   English::AdverbOfManner, English::AdverbOfManner, English::SuffixITY, 0, 0,
                                                   English::AdverbOfManner, 0, 0, English::AdverbOfManner, English::AdverbOfManner};
    static constexpr int numSuffixesStep3 = 8;
    const char *(suffixesStep3[numSuffixesStep3])[2] = {{"ational", "ate"},
                                                        {"tional",  "tion"},
                                                        {"alize",   "al"},
                                                        {"icate",   "ic"},
                                                        {"iciti",   "ic"},
                                                        {"ical",    "ic"},
                                                        {"ful",     ""},
                                                        {"ness",    ""}};
    static constexpr uint32_t typesStep3[numSuffixesStep3] = {English::SuffixION | English::SuffixAL,
                                                              English::SuffixION | English::SuffixAL, 0, 0, English::SuffixITY,
                                                              English::SuffixAL, English::AdjectiveFull, English::SuffixNESS};
    static constexpr int numSuffixesStep4 = 20;
    const char *suffixesStep4[numSuffixesStep4] = {"al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ou",
                                                   "ism", "ate", "iti", "ous", "ive", "ize", "sion", "tion"};
    static constexpr uint32_t typesStep4[numSuffixesStep4] = {English::SuffixAL, English::SuffixNCE, English::SuffixNCE, 0,
                                                              English::SuffixIC, English::SuffixCapable, English::SuffixCapable,
                                                              English::SuffixNT, 0, 0, English::SuffixNT, 0, 0, 0, English::SuffixITY,
                                                              English::SuffixOUS, English::SuffixIVE, 0, English::SuffixION,
                                                              English::SuffixION};
    static constexpr int numExceptionRegion1 = 3;
    const char *exceptionsRegion1[numExceptionRegion1] = {"gener", "arsen", "commun"};
    static constexpr int numExceptions1 = 19;
    const char *(exceptions1[numExceptions1])[2] = {{"skis",   "ski"},
                                                    {"skies",  "sky"},
                                                    {"dying",  "die"},
                                                    {"lying",  "lie"},
                                                    {"tying",  "tie"},
                                                    {"idly",   "idl"},
                                                    {"gently", "gentl"},
                                                    {"ugly",   "ugli"},
                                                    {"early",  "earli"},
                                                    {"only",   "onli"},
                                                    {"singly", "singl"},
                                                    {"sky",    "sky"},
                                                    {"news",   "news"},
                                                    {"howe",   "howe"},
                                                    {"atlas",  "atlas"},
                                                    {"cosmos", "cosmos"},
                                                    {"bias",   "bias"},
                                                    {"andes",  "andes"},
                                                    {"texas",  "texas"}};
    static constexpr uint32_t typesExceptions1[numExceptions1] = {English::Noun | English::Plural,
                                                                  English::Noun | English::Plural | English::Verb,
                                                                  English::PresentParticiple, English::PresentParticiple,
                                                                  English::PresentParticiple, English::AdverbOfManner,
                                                                  English::AdverbOfManner, English::Adjective,
                                                                  English::Adjective | English::AdverbOfManner, 0, English::AdverbOfManner,
                                                                  English::Noun, English::Noun, 0, English::Noun, English::Noun,
                                                                  English::Noun, English::Noun | English::Plural, English::Noun};
    static constexpr int numExceptions2 = 8;
    const char *exceptions2[numExceptions2] = {"inning", "outing", "canning", "herring", "earring", "proceed", "exceed", "succeed"};
    static constexpr uint32_t typesExceptions2[numExceptions2] = {English::Noun, English::Noun, English::Noun, English::Noun, English::Noun,
                                                                  English::Verb, English::Verb, English::Verb};

    inline auto isConsonant(char c) -> bool;
    static inline auto isShortConsonant(char c) -> bool;
    static inline auto isDouble(char c) -> bool;
    static inline auto isLiEnding(char c) -> bool;
    auto getRegion1(const Word *w) -> uint32_t;
    auto endsInShortSyllable(const Word *w) -> bool;
    auto isShortWord(const Word *w) -> bool;
    inline auto hasVowels(const Word *w) -> bool;
    static auto trimApostrophes(Word *w) -> bool;
    void markYsAsConsonants(Word *w);
    static auto processPrefixes(Word *w) -> bool;
    auto processSuperlatives(Word *w) -> bool;

    /**
     * Search for the longest among the suffixes, 's' or 's or ' and remove if found.
     * Examples: Children's toys / Vice presidents' duties
     * @param w
     * @return
     */
    auto step0(Word *w) -> bool;

    /**
     * Search for the longest among the following suffixes, and perform the action indicated.
     * @param w
     * @return
     */
    auto step1A(Word *w) -> bool;

    /**
     * Search for the longest among the following suffixes, and perform the action indicated.
     * @param w
     * @param r1
     * @return
     */
    auto step1B(Word *w, uint32_t r1) -> bool;

    auto step1C(Word *w) -> bool;

    auto step2(Word *w, uint32_t r1) -> bool;

    auto step3(Word *w, uint32_t r1, uint32_t r2) -> bool;

    /**
     * Search for the longest among the following suffixes, and, if found and in r2, perform the action indicated.
     * @param w
     * @param r2
     * @return
     */
    auto step4(Word *w, uint32_t r2) -> bool;

    /**
     * Search for the the following suffixes, and, if found, perform the action indicated.
     * @param w
     * @param r1
     * @param r2
     * @return
     */
    auto step5(Word *w, uint32_t r1, uint32_t r2) -> bool;

public:
    auto isVowel(char c) -> bool final;
    auto stem(Word *w) -> bool override;
};
