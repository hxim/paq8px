#pragma once

#include "../LargeStationaryMap.hpp"
#include "../Shared.hpp"
#include "../StateMap.hpp"
#include "../Mixer.hpp"
#include "../Random.hpp"

/**
 * Model for 1-bit (bilevel) images: scanned pages, faxes, line art, dithered photographs.
 *
 * The image is coded pixel by pixel, row by row. For each pixel the model gives the mixer:
 *
 *  - indexed contexts: small templates of the pixels around the current one, each looked up in a
 *    table of bit histories;
 *  - hashed contexts: templates too wide or too deep to index, looked up in a hash table;
 *  - a few direct predictors: how much ink is around, and how long the runs are along the row, up
 *    the column and along the diagonals;
 *  - four 2D match models, which find where the current neighbourhood appeared earlier in the image
 *    and predict what followed it there.
 *
 * It also selects the mixer's weight sets, and hands a summary of the neighbourhood to the SSE stage.
 *
 * Reading the template diagrams
 * -----------------------------
 * The templates in the .cpp are shown as pictures of the pixels they read:
 *
 *      3  ...xxx..          the number on the left: how many rows up (0 = the current row)
 *      2  ..xxxxx.          '?' : the pixel being coded (nothing right of it is known yet)
 *      1  ..xxxxxxx         'x' : a pixel the template reads
 *      0  xxxxx?            '.' : a pixel it does not read
 *
 * Other letters and digits are explained next to their picture.
 */
class Image1BitModel
{
private:

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Sizes
  //////////////////////////////////////////////////////////////////////////////////////////////////

  static constexpr int N = 41;      /**< number of indexed contexts */
  static constexpr int nMatch = 4;  /**< number of match models */

  /**< The number of slots of each indexed context, in the order setIndexedContexts() sets them. All
       contexts share one table (t), and each one owns the slice of it named here. */
  static constexpr int C =
    //the small templates around the current pixel
    (1 << 2) + (1 << 4) + (1 << 6) + (1 << 8) + (1 << 8) + (1 << 8) + (1 << 8) + (1 << 9) +
    (1 << 10) + (1 << 12) + (1 << 12) +
    //how much ink is around, at three scales
    5 + 13 + 41 +
    //the dither phase, and two more small templates
    (1 << 8) + (1 << 10) + (1 << 13) +
    //the adaptive templates
    (1 << 7) + (1 << 8) +
    //the 4 nearest pixels and the one two rows up
    32 +
    //the column, the sparse template, the dither phase over 16 rows, and the wide templates
    (1 << 16) + (1 << 16) + (1 << 11) + (1 << 14) + (1 << 14) + (1 << 15) + (1 << 15) + (1 << 15) +
    (1 << 15) + (1 << 14) + (1 << 14) + (1 << 15) + (1 << 16) + (1 << 16) + (1 << 17) +
    //the vertical runs
    (1 << 14) + (1 << 15) +
    //the slanted directions
    (1 << 12) + (1 << 16) +
    //the shape of the ink above
    (1 << 14) +
    //the reference line
    (1 << 12);

  /**< Most indexed contexts are also read through a hashed map, which learns differently. These
       ones are not, as for them the second reading does not help. (bit i = indexed context i) */
  static constexpr uint64_t noMirror =
    1ull << 6 | 1ull << 18 | 1ull << 20 | 1ull << 23 | 1ull << 25 | 1ull << 27 | 1ull << 29 |
    1ull << 32 | 1ull << 34 | 1ull << 36 | 1ull << 38;

  /**< number of hashed contexts: 57 of their own, and the indexed ones not in noMirror */
  static constexpr int nLSM = 55 + N - 11 + 2;

public:
  static constexpr int MIXERINPUTS =
    2 * N +                 // the indexed contexts
    9 * 2 +                 // the direct predictors
    nMatch +                // the match models
    nLSM * LargeStationaryMap::MIXERINPUTS;
  static constexpr int MIXERCONTEXTS = 16 + 64 + 41 + 32 + 64 + 16 + 64 + 512 + 256 + 256 + 49 + 256 + 33; // 1659
  static constexpr int MIXERCONTEXTSETS = 13;

  Image1BitModel(Shared* const sh);
  Image1BitModel(const Image1BitModel&) = delete;
  Image1BitModel& operator=(const Image1BitModel&) = delete;
  void setParam(int widthInBytes); /**< the row width; a change of width starts a new image */
  void update();
  void mix(Mixer& m);

private:
  Shared* const shared;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Where we are, and the rows above
  //////////////////////////////////////////////////////////////////////////////////////////////////

  uint32_t w{};                   /**< row width in bytes */
  uint32_t xByte = 0, yRow = 0;   /**< position of the current pixel: its byte in the row (the bit is bpos), and its row */

  /**< The pixels already coded, one register per row, the newest pixel in bit 0:
         r0         the current row: bit k is k+1 pixels to the left
         r1..r16    the rows 1..16 above: bit 8 is the current column, bit 8-dx is dx columns to the
                    right (so bits 0..7 are the 8 columns ahead, bits 9.. the columns behind)
         r1b..r8b   the rows 1..8 above, one byte further ahead: bit j is column 16-j, which brings
                    the columns +9..+16 into reach
       They are 64 bits wide, so templates and the adaptive template can reach far to the left. */
  uint64_t r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0, r8 = 0;
  uint64_t r9 = 0, r10 = 0, r11 = 0, r12 = 0, r13 = 0, r14 = 0, r15 = 0, r16 = 0;
  uint64_t r1b = 0, r2b = 0, r3b = 0, r4b = 0, r5b = 0, r6b = 0, r7b = 0, r8b = 0;

  /**< the pixel dy rows up (0..16) and dx columns to the right (up to +8; below 0 on the current row) */
  uint32_t at(int dy, int dx) const;
  /**< the n pixels of a sparse template, the first one in the highest bit */
  uint32_t sparse(const int8_t(*tpl)[2], int n) const;
  /**< the pixel dy rows up and dx columns to the right, read from the buffer; 0 if not coded yet */
  template<typename B> int px(const B& buf, int bpos, int dy, int dx) const;
  /**< n (up to 16) pixels starting at pixel q of the image, the first one in the highest bit; pixels
       not coded yet, or before the image, read as 0 */
  template<typename B> uint32_t pixelsAt(const B& buf, int64_t q, int n, uint32_t curByteIdx) const;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Runs, and the reference line
  //////////////////////////////////////////////////////////////////////////////////////////////////

  uint32_t runLength = 0;              /**< identical pixels ending just left of the current one (0 at the start of a row) */
  uint32_t runValue = 0;               /**< the colour of that run */
  uint32_t prevRun1 = 0, prevRun2 = 0; /**< the lengths of the two runs before it (0 at the start of a row) */

  static constexpr uint32_t maxRowBytes = 8192;   /**< widest row analysed (65536 pixels) */

  /**< The row above, indexed by where it changes colour (see buildRefLine): nc0[x] / nc1[x] is the
       first pixel at or after x where the row turns white / black. */
  Array<uint8_t> refBytes{ maxRowBytes };
  Array<uint32_t> nc0{ maxRowBytes * 8 + 2 };
  Array<uint32_t> nc1{ maxRowBytes * 8 + 2 };
  bool refValid = false;    /**< false on the first row of an image, and for rows too wide to index */
  template<typename B> void buildRefLine(const B& buf);

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Repeat periods, for the adaptive templates
  //////////////////////////////////////////////////////////////////////////////////////////////////

  static constexpr uint32_t minRowBytes = 16;     /**< narrower rows are not analysed */
  static constexpr uint32_t maxHPeriod = 48;      /**< horizontal periods searched: 2..48 pixels */
  static constexpr uint32_t minVPeriod = 9;       /**< vertical periods searched: 9..64 rows (the rows 1..8 above are read directly anyway) */
  static constexpr uint32_t maxVPeriod = 64;
  static constexpr int minProminence = 64;        /**< a peak smaller than this is not a period */

  Array<uint32_t> hAcc{ maxHPeriod + 1 };          /**< score per horizontal period, smoothed over the last rows */
  Array<uint32_t> vAcc{ maxVPeriod + 1 };          /**< score per vertical period, likewise */
  Array<uint8_t> rowBytes{ maxRowBytes };          /**< the row just finished */
  Array<uint64_t> rowWords{ maxRowBytes / 8 + 2 }; /**< the same, packed 64 pixels to a word */
  uint32_t hPeriod = 0, vPeriod = 0;               /**< the periods found; 0 = none */
  uint32_t qualH = 0, qualV = 0;                   /**< how well the image repeats at them, 0..3 */
  template<typename B> void updateRepeatPeriods(const B& buf);

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Match models
  //
  // Where the current neighbourhood was seen before, and what came next there. This finds repeated
  // letters and repeated halftone cells anywhere earlier in the image. Each of the four looks the
  // neighbourhood up under a key of a different shape (see runMatchModels).
  //////////////////////////////////////////////////////////////////////////////////////////////////

  static constexpr int matchBits = 22;             /**< log2 of the table size per key, in 4-byte words (2^22 words = 16 MB) */
  static constexpr int matchSlots = 3;             /**< positions remembered per bucket */
  static constexpr int bucketBits = matchBits - 2; /**< a bucket is 16 bytes */
  static constexpr uint32_t maxResidual = 16;      /**< a candidate differing from here in more of the 40 compared pixels is not taken */

  /**< the last places a key was seen, most recent first */
  struct MatchBucket
  {
    uint32_t pos[matchSlots];  /**< bpos << 29 | byte index within the image; 0 = empty */
    uint8_t check[matchSlots]; /**< 8 more bits of the key, to tell apart keys that share the bucket */
    uint8_t unused;
  };
  static_assert(sizeof(MatchBucket) == 16, "a match bucket should be 16 bytes");
  Array<MatchBucket> matchTable{ static_cast<uint64_t>(nMatch) << bucketBits };

  uint32_t matchHashIdx[nMatch] = {};  /**< the bucket of the current key */
  uint8_t matchCheck[nMatch] = {};     /**< the check byte of the current key */
  uint32_t matchByteIdx[nMatch] = {}, matchBpos[nMatch] = {}; /**< the source: the pixel the match predicts from */
  uint32_t matchLen[nMatch] = {};      /**< pixels predicted correctly since the match was taken; 0 = no match */
  uint32_t matchSrcX[nMatch] = {};     /**< the source's byte column within its row */
  uint32_t matchResid[nMatch] = {};    /**< how closely the source resembled the current neighbourhood when taken, 0..3 */
  int predictedBit[nMatch] = { -1, -1, -1, -1 }; /**< the pixel each match expects; -1 = none */
  uint32_t lastByteIdx = 0, lastBpos = 0;        /**< the pixel the current keys belong to */

  /**< how many of the 40 pixels around a candidate source differ from those around the current pixel */
  template<typename B> uint32_t residual(const B& buf, uint32_t byteIdx, uint32_t bp, uint32_t curByteIdx) const;

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Contexts and predictors
  //////////////////////////////////////////////////////////////////////////////////////////////////

  LargeStationaryMap mapL;  /**< the hashed contexts */
  StateMap stateMap;        /**< bit history -> probability, one set per indexed context */
  StateMap matchStateMap;   /**< the match models' situation -> probability, one set per key */
  Random rnd;
  Array<uint32_t> cxt{ N };     /**< the slot each indexed context selected for the current pixel */
  Array<uint8_t> t{ C };        /**< bit history per slot */
  Array<uint16_t> counts{ C };  /**< recent counts of 0s (high byte) and 1s (low byte) per slot */

  /**< What mix() reads off the neighbourhood of the current pixel, for the contexts, the mixer and
       the SSE stage. Each value is described where it is computed. */
  struct Features
  {
    //the pixels right around the current one: readNeighbourhood()
    uint32_t surrounding4, mCtx6, surrounding12, surrounding24, frame16, dil6, ero6;
    int bitcount04, bitcount12, bitcount24, bitcount40;
    //the runs and the reference line: trackRuns()
    uint32_t runIdx, curCol, vb1, vb2;
    //the columns above: readColumns()
    uint32_t col8, col16, col8L, col8R;
    int vrun, colCount;
    uint32_t vrunL, vrunR, vrun16, vrun16L, vrun16R, vrunLL, vrunRR, vrunLLL, vrunRRR;
    //the slanted directions: readDirections()
    uint32_t diagNW, diagNE, drunNW, drunNE, diagNW2, diagNE2, diagNW16, drunNW16;
    uint32_t steepNW, steepNE, flatNW, flatNE, srunNW, srunNE, frunNW;
    uint32_t vsteepNW, vsteepNE, vflatNW, vsrunNW, vfrunNW;
    uint32_t steepNW16, flatNW16, srunNW16, frunNW16;
    //the adaptive template pixels: readAdaptiveTemplate()
    uint32_t atH0, atH1, atH2, atV0, atV1, atV2, atV3;
    //the match models: runMatchModels()
    uint32_t lenBucketA, lenBucketB, matchVote, matchVote4, matchMode;
  } f{};

  //the steps of mix(), in the order it runs them
  template<typename B> void shiftRows(const B& buf, int y, int bpos);
  void trackRuns(int y, bool rowStart, int bpos);
  void readColumns();
  void readDirections();
  template<typename B> void readAdaptiveTemplate(const B& buf, int bpos);
  template<typename B> void runMatchModels(const B& buf, int bpos, bool rowStart);
  void readNeighbourhood(int y);
  void setIndexedContexts(int y, int bpos);
  void setHashedContexts(int y, int bpos);
  void addInputs(Mixer& m);
  void setMixerContexts(Mixer& m, int y, int bpos);
  void publishToSSE();

  /**< a direct predictor: turns a pair of counts into two mixer inputs */
  void add(Mixer& m, uint32_t n0, uint32_t n1);
};
