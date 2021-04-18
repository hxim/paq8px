#ifndef PAQ8PX_TEXTPARSERSTATEINFO_HPP
#define PAQ8PX_TEXTPARSERSTATEINFO_HPP

#include "../Array.hpp"
#include <cstdint>

#define TEXT_MIN_SIZE 1024 /**< size of minimum allowed text block (in bytes) */
#define TEXT_MAX_MISSES 8 /**< threshold: max allowed number of invalid UTF8 sequences seen recently before reporting "fail" */
#define TEXT_ADAPT_RATE 1024 /**< smaller (like 32) = illegal sequences are allowed to come more often, larger (like 1024) = more rigorous detection */

struct TextParserStateInfo {
public:
    Array<uint64_t> _start;
    Array<uint64_t> _end; /**< position of last char with a valid UTF8 state: marks the end of the detected TEXT block */
    Array<uint32_t> _EOLType; /**< 0: none or CR-only;   1: CRLF-only (applicable to EOL transform);   2: mixed or LF-only */
    uint32_t invalidCount {}; /**< adaptive count of invalid UTF8 sequences seen recently */
    uint32_t UTF8State {}; /**< state of utf8 parser; 0: valid;  12: invalid;  any other value: yet uncertain (more bytes must be read) */
    constexpr static int utf8Accept = 0;
    constexpr static int utf8Reject = 12;

    /**
     * UTF8 validator
     * Based on: http://bjoern.hoehrmann.de/utf-8/decoder/dfa/
     * Control characters (0x00-0x1f) and 0x7f are not allowed (except for 0/tab/cr/lf)
     */
    constexpr static uint8_t utf8StateTable[] = {
            // byte -> character class
            // character_class = utf8StateTable[byte]
            1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // 00..1f
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 20..3f
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 40..5f
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, // 60..7f
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, // 80..9f
            7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, // a0..bf
            8, 8, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, // c0..df
            10, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3, 11, 6, 6, 6, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // e0..ff
            // validating automaton
            // new_state = utf8StateTable[256*old_state + character_class]
            0, 12, 24, 36, 60, 96, 84, 12, 12, 12, 48, 72, // state  0-11
            12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, // state 12-23
            12, 0, 12, 12, 12, 12, 12, 0, 12, 0, 12, 12, // state 24-35
            12, 24, 12, 12, 12, 12, 12, 24, 12, 24, 12, 12, // state 36-47
            12, 12, 12, 12, 12, 12, 12, 24, 12, 12, 12, 12, // state 48-59
            12, 24, 12, 12, 12, 12, 12, 12, 12, 24, 12, 12, // state 60-71
            12, 12, 12, 12, 12, 12, 12, 36, 12, 36, 12, 12, // state 72-83
            12, 36, 12, 12, 12, 12, 12, 36, 12, 36, 12, 12, // state 84-95
            12, 36, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12 // state 96-108
    };
    static TextParserStateInfo& getInstance();
    void reset(uint64_t startPos);
    auto start() -> uint64_t;
    auto end() -> uint64_t;
    void setEnd(uint64_t end);
    auto eolType() -> uint32_t;
    void setEolType(uint32_t eolType);
    auto validLength() -> uint64_t;
    void next(uint64_t startPos);
    void removeFirst();
private:
    TextParserStateInfo() : _start(1), _end(1), _EOLType(1) {}

    /**
     * Copy constructor is private so that it cannot be called
     */
    TextParserStateInfo(TextParserStateInfo const &) : _start(1), _end(1), _EOLType(1) {}

    /**
     * Assignment operator is private so that it cannot be called
     */
    TextParserStateInfo &operator=(TextParserStateInfo const &) { return *this; }
  
};

#endif //PAQ8PX_TEXTPARSERSTATEINFO_HPP