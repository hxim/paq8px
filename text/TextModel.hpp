#pragma once

#include "../ContextMap2.hpp"
#include "../RingBuffer.hpp"
#include "../Shared.hpp"
#include "Cache.hpp"
#include "English.hpp"
#include "EnglishStemmer.hpp"
#include "French.hpp"
#include "FrenchStemmer.hpp"
#include "German.hpp"
#include "GermanStemmer.hpp"
#include "Language.hpp"
#include "Paragraph.hpp"
#include "Segment.hpp"
#include "Sentence.hpp"
#include "Stemmer.hpp"
#include "Word.hpp"
#include "WordEmbeddingDictionary.hpp"
#include <cassert>
#include <cctype>
#include <cstdint>

#ifndef DISABLE_TEXTMODEL

class TextModel {
private:
    static constexpr int nCM2 = 28;
    static constexpr uint8_t asciiGroup[128] = {0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                                                5, 6, 7, 8, 17, 17, 9, 17, 10, 11, 12, 17, 17, 13, 14, 15, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                18, 19, 20, 23, 21, 22, 23, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                                2, 2, 2, 2, 2, 24, 27, 25, 27, 26, 27, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                                                3, 3, 3, 3, 3, 3, 3, 3, 28, 30, 29, 30, 30};
    Shared * const shared;
    static constexpr uint32_t MIN_RECOGNIZED_WORDS = 4;
    ContextMap2 cm;
    Array<Stemmer *> stemmers;
    Array<Language *> languages;
    Array<WordEmbeddingDictionary *> dictionaries;
    Cache<Word, 8> words[Language::Count];
    Cache<Segment, 4> segments;
    Cache<Sentence, 4> sentences;
    Cache<Paragraph, 2> paragraphs;
    Array<uint32_t> wordPos;
    uint32_t bytePos[256] {};
    struct WordDistance {
        uint32_t distance[4];
        uint32_t closest;
    };
    Cache<WordDistance, 4> wordDistances;
    Word *cWord, *pWord; // current word, previous word
    Segment *cSegment; // current segment
    Sentence *cSentence; // current sentence
    Paragraph *cParagraph; // current paragraph
    enum Parse {
        Unknown, ReadingWord, PossibleHyphenation, WasAbbreviation, AfterComma, AfterQuote, AfterAbbreviation, ExpectDigit
    } State, pState;
    struct {
        uint32_t count[Language::Count - 1]; // number of recognized words of each language in the last 64 words seen
        uint64_t mask[Language::Count - 1]; // binary mask with the recognition status of the last 64 words for each language
        int id; // current detected language
        int pId; // detected language of the previous word
    } Lang;
    struct {
        uint64_t numbers[2]; // last 2 numbers seen
        uint64_t numHashes[2]; // hashes of the last 2 numbers seen
        uint8_t numLength[2]; // digit length of last 2 numbers seen
        uint32_t numMask; // binary mask of the results of the arithmetic comparisons between the numbers seen
        uint32_t numDiff; // log2 of the consecutive differences between the last 16 numbers seen, clipped to 2 bits per difference
        uint32_t lastUpper; // distance to last uppercase letter
        uint32_t maskUpper; // binary mask of uppercase letters seen (in the last 32 bytes)
        uint32_t lastLetter; // distance to last letter
        uint32_t lastDigit; // distance to last digit
        uint32_t lastPunctuation; // distance to last punctuation character
        uint32_t lastNewLine; // distance to last new line character
        uint32_t prevNewLine; // distance to penultimate new line character
        uint32_t wordGap; // distance between the last words
        uint32_t spaces; // binary mask of whitespace characters seen (in the last 32 bytes)
        uint32_t spaceCount; // count of whitespace characters seen (in the last 32 bytes)
        uint32_t commas; // number of commas seen in this line (not this segment/sentence)
        uint32_t quoteLength; // length (in words) of current quote
        uint32_t maskPunctuation; // mask of relative position of last comma related to other punctuation
        uint32_t nestHash; // hash representing current nesting state
        uint32_t lastNest; // distance to last nesting character
        uint32_t masks[5], wordLength[2];
        int UTF8Remaining; // remaining bytes for current UTF8-encoded Unicode code point (-1 if invalid byte found)
        uint8_t firstLetter; // first letter of current word
        uint8_t firstChar; // first character of current line
        uint8_t expectedDigit; // next expected digit of detected numerical sequence
        uint8_t prevPunctuation; // most recent punctuation character seen
        Word topicDescriptor; // last word before ':'
    } Info;
    uint64_t parseCtx; // state of parser + relevant features used as a context (hash)
    void setContexts();

public:
    static constexpr int MIXERINPUTS =
            nCM2 * (ContextMap2::MIXERINPUTS + ContextMap2::MIXERINPUTS_RUN_STATS + ContextMap2::MIXERINPUTS_BYTE_HISTORY); // 196
    static constexpr int MIXERCONTEXTS = 2048 + 2048 + 4096 + 4096 + 2048 + 2048 + 4096 + 8192 + 2048 + (16 * 8); //30848
    static constexpr int MIXERCONTEXTSETS = 10;

    TextModel(Shared* const sh, uint64_t size);
    ~TextModel();
    void update();
    void setParam(int cmScale);
    void mix(Mixer &m);
};

#else
class TextModel {
public:
    static constexpr int MIXERINPUTS = 0;
    static constexpr int MIXERCONTEXTS = 0;
    static constexpr int MIXERCONTEXTSETS = 0;
};

#endif //DISABLE_TEXTMODEL
