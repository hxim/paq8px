#pragma once

#ifndef DISABLE_TEXTMODEL

#include "../ContextMap2.hpp"
#include "../RingBuffer.hpp"
#include "../Shared.hpp"
#include "Info.hpp"
#include <cctype>

/**
 * Model words, expressions, numbers, paragraphs/lines, etc.
 * simple processing of pdf text
 * simple modeling of some binary content
 */
class WordModel {
private:
    static constexpr int nCM1 = Info::nCM1; // pdf / non_pdf contexts
    static constexpr int nCM2_TEXT = Info::nCM2_TEXT; // 58
    static constexpr int nCM2_BIN = Info::nCM2_BIN; // 49
public:
    static constexpr int MIXERINPUTS_TEXT = (nCM1 + nCM2_TEXT) *(ContextMap2::MIXERINPUTS + ContextMap2::MIXERINPUTS_RUN_STATS + ContextMap2::MIXERINPUTS_BYTE_HISTORY); // 406
    static constexpr int MIXERINPUTS_BIN = (nCM1 + nCM2_BIN) * (ContextMap2::MIXERINPUTS + ContextMap2::MIXERINPUTS_RUN_STATS + ContextMap2::MIXERINPUTS_BYTE_HISTORY); // 343
    static constexpr int MIXERCONTEXTS = 16 * 8;
    static constexpr int MIXERCONTEXTSETS = 1;

private:
    Shared * const shared;
    ContextMap2 cm;
    Info infoNormal; //used for general content
    Info infoPdf; //used only in case of pdf text - in place of infoNormal
    uint8_t pdfTextParserState; // 0..7
public:
    WordModel(Shared* const sh, uint64_t size);
    void reset();
    void setParam(int cmScale);
    void mix(Mixer &m);
};

#else
class WordModel {
public:
    static constexpr int MIXERINPUTS = 0;
    static constexpr int MIXERCONTEXTS = 0;
    static constexpr int MIXERCONTEXTSETS = 0;
};
#endif //DISABLE_TEXTMODEL
