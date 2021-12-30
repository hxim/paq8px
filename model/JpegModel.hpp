#pragma once

#include "../BH.hpp"
#include "../APM.hpp"
#include "../Hash.hpp"
#include "../Ilog.hpp"
#include "../IndirectMap.hpp"
#include "../Mixer.hpp"
#include "../MixerFactory.hpp"
#include "../Random.hpp"
#include "../RingBuffer.hpp"
#include "../Shared.hpp"
#include "../StateMap.hpp"
#include <cstdint>

// Print a JPEG segment at buf[p...] for debugging
/*
void dump(const char* msg, int p) {
  printf("%s:", msg);
  int len=buf[p+2]*256+buf[p+3];
  for (int i=0; i<len+2; ++i)
    printf(" %02X", buf[p+i]);
  printf("\n");
}
*/

#define FINISH(success) \
  { \
    uint32_t length = pos - images[idx].offset; \
    /*if (success && idx && pos-lastPos==1)*/ \
    /*printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\bEmbedded JPEG at offset %d, size: %d bytes, level %d\nCompressing... ", images[idx].offset-pos+blPos, length, idx), fflush(stdout);*/ \
    memset(&images[idx], 0, sizeof(JPEGImage)); \
    mcusize = 0, dqt_state = -1; \
    idx -= (idx > 0); \
    if (images[idx].app <= length) \
      images[idx].app = 0; \
    else \
      images[idx].app -= length; \
  }

// Detect invalid JPEG data.  The proper response is to silently
// fall back to a non-JPEG model.
#define JASSERT(x) \
  if (! (x)) { \
    /*  printf("JPEG error at %d, line %d: %s\n", pos, __LINE__, #x); */ \
    if (idx > 0) \
      FINISH(false) else images[idx].jpeg = 0; \
    return images[idx].nextJpeg; \
  }

struct HUF {
    uint32_t min, max;
    int val;
}; // Huffman decode tables
// huf[Tc][Th][m] is the minimum, maximum+1, and pointer to codes for
// coefficient type Tc (0=DC, 1=AC), table Th (0-3), length m+1 (m=0-15)

struct JPEGImage {
    uint32_t offset, // offset of SOI marker
            jpeg, // 1 if JPEG is header detected, 2 if image data
            nextJpeg, // updated with jpeg on next byte boundary
            app, // Bytes remaining to skip in this marker
            sof, sos, data, // pointers to buf
            htSize; // number of pointers in ht
    int ht[8]; // pointers to Huffman table headers
    uint8_t qTable[256]; // table
    int qMap[10]; // block -> table number
};

/**
 * Model JPEG. Return 1 if a JPEG file is detected or else 0.
 * Only the baseline and 8 bit extended Huffman coded DCT modes are
 * supported.  The model partially decodes the JPEG image to provide
 * context for the Huffman coded symbols.
 */
class JpegModel {
private:
    static constexpr int N = 49; // number of contexts
public:
    static constexpr int MIXERINPUTS = 1 + 2 * N + 17;
    static constexpr int MIXERCONTEXTS = (1 + 8) + (1 + 1024) + (1 + 1024) + (1 + 1024);
    static constexpr int MIXERCONTEXTSETS = 4;

private:
    Random rnd;
    // state of parser
    enum {
        SOF0 = 0xc0, SOF1, SOF2, SOF3, DHT, RST0 = 0xd0, SOI = 0xd8, EOI, SOS, DQT, DNL, DRI, APP0 = 0xe0, COM = 0xfe, FF
    }; // Second byte of 2 byte codes
    static const int maxEmbeddedLevel = 3;
    JPEGImage images[maxEmbeddedLevel] {};
    int idx = -1;
    uint32_t lastPos = 0;

    // Huffman decode state
    uint32_t huffcode = 0; // Current Huffman code including extra bits
    int huffBits = 0; // Number of valid bits in huffcode
    int huffSize = 0; // Number of bits without extra bits
    int rs = -1; // Decoded huffcode without extra bits.  It represents
    // 2 packed 4-bit numbers, r=run of zeros, s=number of extra bits for
    // first nonzero code.  huffcode is complete when rs >= 0.
    // rs is -1 prior to decoding incomplete huffcode.
    int hbCount = 2;

    int mcuPos = 0; // position in MCU (0-639).  The low 6 bits mark
    // the coefficient in zigzag scan order (0=DC, 1-63=AC).  The high
    // bits mark the block within the MCU, used to select Huffman tables.

    // decoding tables
    Array<HUF> huf {128}; // Tc*64+Th*16+m -> min, max, val
    int mcusize = 0; // number of coefficients in an MCU
    int hufSel[2][10] {{0}}; // DC/AC, mcuPos/64 -> huf decode table
    Array<uint8_t> hBuf {2048}; // Tc*1024+Th*256+hufCode -> RS

    // Image state
    Array<int> color {10}; // block -> component (0-3)
    Array<int> pred {4}; // component -> last DC value
    int dc = 0; // DC value of the current block
    int width = 0; // Image width in MCU
    int row = 0, column = 0; // in MCU (column 0 to width-1)
    RingBuffer<uint8_t> coefficientBuffer {0x20000}; // Rotating buffer of coefficients, coded as:
    // DC: level shifted absolute value, low 4 bits discarded, i.e.
    //   [-1023...1024] -> [0...255].
    // AC: as an RS code: a run of R (0-15) zeros followed by an s (0-15)
    //   bit number, or 00 for end of block (in zigzag order).
    //   However if R=0, then the format is ssss11xx where ssss is s,
    //   xx is the first 2 extra bits, and the last 2 bits are 1 (since
    //   this never occurs in a valid RS code).
    int cPos = 0; // position in coefficientBuffer
    int rs1 = 0; // last RS code
    int resetPos = 0, resetLen = 0; // reset position
    int sSum = 0, sSum1 = 0, sSum2 = 0, sSum3 = 0;
    // sum of s in RS codes in block and sum of s in first component

    RingBuffer<int> cBuf2 {0x20000};
    Array<int> advPred {4}, sumU {8}, sumV {8}, runPred {6};
    int prevCoef = 0, prevCoef2 = 0, prevCoefRs = 0;
    Array<int> ls {10}; // block -> distance to previous block
    Array<int> blockW {10}, blockN {10}, samplingFactors {4};
    Array<int> lcp {7}, zPos {64};

    //for parsing Quantization tables
    int dqt_state = -1;
    uint32_t dqtEnd = 0, qNum = 0;

    // context model
    BH<9> t; // context hash -> bit history
    // As a cache optimization, the context does not include the last 1-2
    // bits of huffcode if the length (huffBits) is not a multiple of 3.
    // The 7 mapped values are for context+{"", 0, 00, 01, 1, 10, 11}.
    Array<uint64_t> cxt {N}; // context hashes
    Array<uint8_t *> cp {N}; // context pointers
    IndirectMap MJPEGMap;
    StateMap sm;
    Mixer *m1;
    APM apm1, apm2, apm3, apm4, apm5, apm6, apm7, apm8, apm9, apm10, apm11, apm12, apm13, apm14;
    Ilog *ilog = &Ilog::getInstance();
    Shared * const shared;

public:
    explicit JpegModel(Shared* const sh, const MixerFactory* const mf, uint64_t size);
    ~JpegModel();
    auto mix(Mixer &m) -> int;
};
