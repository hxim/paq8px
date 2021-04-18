#ifndef PAQ8PX_INFO_HPP
#define PAQ8PX_INFO_HPP


#include "../Shared.hpp"
#include "../ContextMap2.hpp"
#include <cstdint>
#include <cctype>

/**
 * Used by @ref WordModel.
 */
class Info {
private:
    static constexpr uint32_t wPosBits = 16;
    static constexpr int maxWordLen = 45;
    static constexpr int maxLineMatch = 16;
    static constexpr int maxLastUpper = 63;
    static constexpr int maxLastLetter = 16;
    static constexpr int nCM1 = 17; /**< pdf / non_pdf contexts */
    Shared * const shared;
    ContextMap2 &cm;
    Array<uint32_t> wordPositions {1U << wPosBits}; /**< last positions of whole words/numbers */
    Array<uint16_t> checksums {1U << wPosBits}; /**< checksums for whole words/numbers */
    uint32_t c4 {}; /**< last 4 processed characters */
    uint8_t c {}; /**< last char */
    uint8_t pC {}; /**< previous char */
    uint8_t ppC {}; /**< char before the previous char (converted to lower case) */
    bool isLetter {}, isLetterPc {}, isLetterPpC {};
    uint8_t opened {}; /**< "(", "[", "{", "<", opening QUOTE, opening APOSTROPHE (or 0 when none of them) */
    uint8_t wordLen0 {}; /**< length of word0 */
    uint8_t wordLen1 {}; /**< length of word1 */
    uint8_t exprLen0 {}; /**< length of expr0 */
    uint64_t line0 {};
    uint64_t firstWord {}; /**< hash of line content, hash of first word on line */
    uint64_t word0 {}, word1 {}, word2 {}, word3 {}, word4 {}; /**< wordToken hashes, word0 is the partially processed ("current") word */
    uint64_t expr0 {}, expr1 {}, expr2 {}, expr3 {}, expr4 {}; /**< wordToken hashes for expressions */
    uint64_t keyword0 {}, gapToken0 {}, gapToken1 {}; /**< hashes */
    uint16_t w {}; /**< finalized hash of last partially processed word */
    uint16_t chk {}; /**< checksum of last partially processed word */
    int firstChar {}; /**< category of first character of the current line (or -1 when in first column) */
    int lineMatch {}; /**< the length of match of the current line vs the previous line */
    uint32_t nl1 {}, nl2 {}; /**< current newline position, previous newline position */
    uint64_t groups {}; /**< 8 last character categories */
    uint64_t text0 {}; /**< uninterrupted stream of letters */
    uint32_t lastLetter {}, lastUpper {}, wordGap {};
    uint32_t mask {}, expr0Chars {}, mask2 {}, f4 {};

public:
    Info(Shared* const sh, ContextMap2 &contextmap);

    /**
     * Zero the contents.
     */
    void reset();
    void shiftWords();
    void killWords();

    /**
     * \note: The pdf model also uses this function, therefore only @ref c1 may be referenced, @ref c4, @ref c8, etc. should not
     * @param isExtendedChar
     */
    void processChar(bool isExtendedChar);
    void lineModelPredict();
    void lineModelSkip();
    void predict(uint8_t pdfTextParserState);
};

#endif //PAQ8PX_INFO_HPP
