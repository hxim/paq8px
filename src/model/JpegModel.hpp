#pragma once

#include "../APM.hpp"
#include "../BH.hpp"
#include "../Hash.hpp"
#include "../Ilog.hpp"
#include "../IndirectMap.hpp"
#include "../Mixer.hpp"
#include "../MixerFactory.hpp"
#include "../Random.hpp"
#include "../RingBuffer.hpp"
#include "../Shared.hpp"
#include "../StateMap.hpp"
#include "../StationaryMapForJpegModel.hpp"
#include <cstdint>

// Check that the data is valid JPEG. If it is not, silently stop decoding the
// image: an embedded image hands over to the image that contains it, the
// top-level image to the other models.
// Only for use in JpegModel::update() (it needs pos, and returns from it),
// which never touches a Mixer.
#define JASSERT(x) \
  if (!(x)) { \
    if (idx > 0) { \
      finishImage(pos); \
    } \
    else { \
      images[idx].jpeg = 0; \
    } \
    return bail(); \
  }

// One entry of a Huffman decoding table: the codes of one length.
// huf[Tc][Th][m] holds the smallest code of length m+1, the largest code of
// that length + 1, and where their symbols start in hBuf.
// Tc: 0 = DC, 1 = AC. Th: table number 0-3. m: 0-15.
struct HUF
{
  uint32_t min, max;
  int val;
};

struct JPEGImage
{
  uint32_t offset, // position of the SOI marker
    jpeg, // 0: no JPEG, 1: in the headers, 2: SOS seen and data starts
    nextJpeg, // jpeg > 1, updated at every byte boundary
    app, // bytes to skip before parsing resumes: the rest of the current marker
    // segment plus the next marker's 4 byte header, so that the parser sees
    // that header complete in buf(4)..buf(1)
    sof, sos, data, // positions in buf of the SOF marker, the SOS marker and the image data
    dri, // position in buf of the DRI marker, 0 if none
    htSize; // number of entries in ht
  int ht[8]; // positions in buf of the DHT markers
  uint8_t qTable[256]; // the 4 quantization tables, 64 entries each, every entry stored minus 1
  int qMap[10]; // block of the MCU -> quantization table number
};

/**
 * The result of JpegModel::update() for the current bit. It tells the caller
 * which mixer to use, and so which models give inputs to it. Each mixer always
 * gets the same sequence of add() and set() calls.
 */
enum class JpegResult
{
  NoJpeg,      // not in a JPEG stream (any more): use the general mixer and the general models
  Special,     // in a JPEG stream, but JpegModel does not model this bit (stuffing, marker, restart
  // padding, or the rest of a byte after decoding stopped): add specialInput() to the
  // mixer instead of calling mix()
  EntropyCoded // in the entropy coded image data: call mix()
};

/**
 * Model for JPEG images. update() finds JPEG streams and tells whether the
 * current bit is entropy coded; mix() adds the mixer inputs for such bits.
 * The model partly decodes the image, and uses the decoded coefficients as
 * context for the Huffman coded symbols.
 * Only baseline and 8 bit extended Huffman coded DCT images are supported.
 */
class JpegModel
{
private:
  static constexpr int N = 42; // number of bit history contexts
  static constexpr int nSM = 86; // number of stationary maps

public:
  // 2 per bit history context + 1 per stationary map + 17 at the end of mix()
  // (smx, 2 from the inner mixer, 14 APM corrections)
  static constexpr int MIXERINPUTS = 2 * N + nSM + 17;
  // Must match the m.set() calls at the end of mix(), in number and in size.
  static constexpr int MIXERCONTEXTSETS = 13;
  static constexpr int MIXERCONTEXTS =
    8192 + // symbol and coefficient
    8192 + // symbol and frequency
    1024 + // predicted size of the coefficient
    4096 + // size of the in-block neighbours
    512 +  // where the next larger coefficient is expected
    24 +   // number of missing in-block neighbours
    48 +   // position inside the symbol
    64 +   // does the prediction reach the announced size category
    192 +  // coefficient the decoder is at
    48 +   // how many contexts are still unseen
    192 +  // coefficient this bit belongs to
    192 +  // frequency and position inside the symbol
    256;   // coefficient this bit belongs to, and component

  // What specialInput() and specialContext() give to the caller's Special mixer.
  // The context is c0 (the partial byte, 1..255) plus an offset for the kind of bit:
  //   0: byte stuffing and marker bits
  //   256: restart interval padding
  //   512: the rest of a byte after decoding stopped (see bail())
  static constexpr int SPECIALMIXERINPUTS = 1;
  static constexpr int SPECIALMIXERCONTEXTS = 768;
  static constexpr int SPECIALMIXERCONTEXTSETS = 1;

private:
  //////////////////////////////////////////////////////////////////////////////
  // Parser state
  //////////////////////////////////////////////////////////////////////////////

  Random rnd;
  enum
  {
    SOF0 = 0xc0, SOF1, SOF2, SOF3, DHT, RST0 = 0xd0, SOI = 0xd8, EOI, SOS, DQT, DNL, DRI, APP0 = 0xe0, COM = 0xfe, FF
  }; // the second byte of the 2 byte markers
  static const int maxEmbeddedLevel = 3; // a JPEG may contain a thumbnail, which may contain another one
  JPEGImage images[maxEmbeddedLevel]{};
  int idx = -1; // the image in images[] being decoded; -1 means "not initialized yet"
  uint32_t lastPos = 0; // pos at the last byte boundary; a jump means other data came in between

  // for parsing quantization tables
  int dqtState = -1;
  uint32_t dqtEnd = 0, qNum = 0, qNum16 = 0, qNum16b = 0;

  // zigzag position -> u (horizontal frequency)
  static constexpr uint8_t zzu[64] = {
        0, 1, 0, 0, 1, 2, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7,
        7, 6, 5, 4, 3, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 5, 6, 7, 7, 6, 7 };
  // zigzag position -> v (vertical frequency)
  static constexpr uint8_t zzv[64] = {
        0, 0, 1, 2, 1, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1,
        2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 4, 5, 6, 7, 7, 6, 5, 6, 7, 7 };

  //////////////////////////////////////////////////////////////////////////////
  // Huffman decoder state
  //////////////////////////////////////////////////////////////////////////////

  uint32_t huffCode = 0; // the bits of the current symbol so far: Huffman code and extra bits
  uint32_t huffBits = 0; // number of bits in huffCode
  uint32_t huffSize = 0; // length of the Huffman code alone, without the extra bits
  // The decoded symbol (RS), or -1 while its Huffman code is still being read.
  // AC: the high 4 bits are r, the number of zeros before the next nonzero
  // coefficient; the low 4 bits are s, the number of extra bits after the
  // code, which give the value of that coefficient. ZRL (F0) is 16 zeros.
  // DC: rs is s alone.
  int rs = -1;
  int hbCount = 2; // which of the 3 bits of the current bit history group we are on (see mix()); init value is 2 so that mix() does a lookup

  // Position in the MCU (0-639). The low 6 bits are the coefficient in zigzag
  // order (0 = DC, 1-63 = AC); the higher bits are the block within the MCU,
  // which selects the Huffman tables.
  uint32_t mcuPos = 0;

  Array<HUF> huf{ 128 }; // Tc*64 + Th*16 + m -> min, max, val (see HUF)
  int mcuSize = 0; // number of coefficients in an MCU
  int hufSel[2][10]{ {0} }; // DC/AC, block of the MCU -> Huffman table number
  Array<uint8_t> hBuf{ 2048 }; // Tc*1024 + Th*256 + index -> symbol (RS)

  //////////////////////////////////////////////////////////////////////////////
  // Image geometry and decoded coefficients
  //////////////////////////////////////////////////////////////////////////////

  Array<uint32_t> color{ 10 }; // block of the MCU -> component (0-3)
  Array<int> pred{ 4 }; // component -> DC value of its last block (DC values are coded as differences from it)
  int dc = 0; // DC value of the current block
  int width = 0; // image width in MCUs
  int row = 0, column = 0; // current MCU (column 0 to width-1)
  Array<int> ls{ 10 }; // block of the MCU -> distance in coefficients to the previous block of the same component
  // block of the MCU -> distance in coefficients to the west / north block of the same component
  Array<int> blockW{ 10 }, blockN{ 10 };
  Array<int> samplingFactors{ 4 }; // component, as its index in the scan (SOS) header -> sampling factors (H in the high 4 bits, V in the low 4 bits)
  Array<int> zPos{ 64 }; // u + 8v -> zigzag position

  // Circular buffer of the decoded coefficients, in a compact 8 bit form:
  // DC: (dc + 1023) >> 3, i.e. [-1023..1024] -> [0..255].
  // AC: the zeros of a run: (zeros left in the run, counting down to 1) << 4 | s.
  //     The coefficient after them: s << 4 | 0b1100 | its first 2 extra bits.
  //     It is nonzero, except after ZRL (F0): then it is the 16th zero, stored
  //     as 0b1100. Apart from that, 0b11 in bits 2-3 only occurs for a nonzero
  //     coefficient, because s <= 10.
  RingBuffer<uint8_t> coefficientBuffer{ 0x20000 };
  RingBuffer<int> cBuf2{ 0x20000 }; // the same coefficients as plain values
  int cPos = 0; // write position in coefficientBuffer and cBuf2
  int rs1 = 0; // the last completed symbol (RS)
  int resetPos = 0; // MCU index where the current restart interval started
  int resetLen = 0; // restart interval in MCUs: from DRI, then measured at each RSTn; 0 if unknown
  // Sums of s (the extra bit counts, a rough measure of how many bits a block takes):
  int sSum = 0;  // of the current block so far
  int sSum1 = 0; // of the luma blocks of this MCU completed so far
  int sSum2 = 0; // sSum1, or at the first block of an MCU, sSum3
  int sSum3 = 0; // of the last completed luma block

  //////////////////////////////////////////////////////////////////////////////
  // Predictions for the coefficient being decoded
  //
  // All of these are set by updatePredictions(), when a symbol completes, and
  // describe the next coefficient.
  //////////////////////////////////////////////////////////////////////////////

  // The north (sumU) and west (sumV) blocks projected onto this one, minus the
  // coefficients of this block decoded so far (each is removed as soon as it
  // is known). Indexed by u (sumU) and v (sumV).
  Array<int> sumU{ 8 }, sumV{ 8 };
  // What the neighbour blocks say this coefficient should be:
  // 0 from the west, 2 from the north, 1 and 3 from both.
  Array<int> advPred{ 4 };
  // How many zigzag positions ahead a coefficient larger than this one is
  // expected (* 2, + 1 if positive); 0 if none within 9 positions.
  // 0..2 use the same neighbours as advPred[0..2]; 3..5 the same with a larger threshold.
  // At the image border only runPred[0..2] (and advPred[0..2]) are replaced;
  // runPred[3..5] and advPred1/2/3[] are not.
  Array<int> runPred{ 6 };
  // The same as advPred[], one, two and three zigzag positions ahead. They are
  // computed by the runPred[] loop anyway, and are read by the stationary maps.
  Array<int> advPred1{ 3 }, advPred2{ 3 }, advPred3{ 3 };
  // The already decoded neighbours of this coefficient within the block, each
  // rescaled to this coefficient's quantization step, as a signed log. A
  // nonzero value gets +17 more, so that after the coarse divisions in the
  // contexts (lcp / 22 etc.) small values do not fall into the same bucket as 0.
  // 65535 when that neighbour would be outside the block.
  // With (u, v) the frequencies of this coefficient:
  //   lcp[0] (u-1, v)    lcp[1] (u, v-1)
  //   lcp[2] (u-2, v)    lcp[3] (u, v-2)
  //   lcp[4] (u-1, v-1)  lcp[5] (0, v)    lcp[6] (u, 0)
  // lcp[4..6] are 65535 unless u > 0 and v > 0.
  Array<int> lcp{ 7 };
  // The same coefficient in the blocks of this MCU before this one, from the
  // other components. For DC, the coded difference where it is known.
  //   prevCoef:   average of the previous component's blocks (Y for Cb, Cb
  //               for Cr) as a signed log, plus the number of blocks averaged
  //               << 20. The count is 0 for luma, which has no previous
  //               component, and 1 or more with subsampling; in the high bits
  //               it survives the divisions in the contexts.
  //   prevCoef2:  average of the luma blocks, as a signed log. Only for
  //               components 2 and up; for component 1 prevCoef already is luma.
  //   prevCoefRs: the coefficientBuffer entry of the previous component's
  //               blocks, r and s averaged separately; without such blocks,
  //               the entry 64 positions back.
  int prevCoef = 0, prevCoef2 = 0, prevCoefRs = 0;

  // Which neighbour blocks of the current block are inside the image.
  // bit 0: west (sumV[] is built from it), bit 1: north (sumU[]).
  // 3 everywhere except at the left and top border.
  uint32_t nbAvail = 3;

  // Masks of the predictions that carry no information. They are used as
  // mixer contexts, so the mixer can use different weights when those inputs
  // are empty.
  uint32_t lma = 0; // bit i set when lcp[i] is missing (== 65535); 7 bits
  uint32_t ama = 0; // bit i set when advPred[i] is zero; 3 bits

  // advPred[] and lcp[] for the coefficient the current bit really belongs to.
  // They differ from the arrays above only during the extra bits of an AC
  // symbol with a run of zeros: mcuPos still points at the first zero, but the
  // extra bits describe the nonzero coefficient r positions later.
  // alignPredictors() moves the predictions there as soon as rs is decoded.
  // Also, alAdv[3] leaves out a neighbour block outside the image; advPred[3]
  // does not.
  // Read by the APM keys and some mixer contexts. cxt[] and the stationary maps
  // still read advPred[] and lcp[].
  int alAdv[4]{};
  int alLcp[7]{}; // Same layout as lcp[]

  //////////////////////////////////////////////////////////////////////////////
  // Context model
  //////////////////////////////////////////////////////////////////////////////

  // Context hash -> bit histories. To save hash lookups, a context is looked
  // up only once every 3 bits (see hbCount). The slot it returns holds 7 bit
  // histories, one for each node of the 3 bit tree: context + {"", 0, 00, 01, 1, 10, 11}.
  BH<9> t;
  Array<uint64_t> cxt{ N }; // context hashes
  Array<uint8_t*> cp{ N }; // pointers to the bit history of each context for the current bit
  // Keyed by the position in the image (MCU, row, column) and the code so far.
  // Pays off for MJPEG, where the next frame often repeats the same position.
  IndirectMap MJPEGMap;
  StationaryMapForJpegModel sMap;
  StateMap sm; // bit history -> probability, one per context in cxt[]
  // A direct probability for the symbol state and the coefficient, next to
  // the inner mixer; one input of the outer mixer.
  StateMap smx;
  Mixer* m1; // the inner mixer
  APM apm1, apm2, apm3, apm4, apm5, apm6, apm7, apm8, apm9, apm10, apm11, apm12, apm13, apm14;
  Ilog* ilog = &Ilog::getInstance();
  Shared* const shared;

  //////////////////////////////////////////////////////////////////////////////
  // Values computed by update() for the current bit and used by mix()
  //
  // Only valid when the last update() returned JpegResult::EntropyCoded.
  //////////////////////////////////////////////////////////////////////////////

  uint32_t comp = 0; // color component of the current block
  uint32_t coef = 0; // zigzag position | component << 6
  uint32_t hc = 0; // the Huffman code so far, plus the DC/AC and luma flags
  uint32_t symKey = 0; // like hc, but shorter once the symbol is known (see update())
  int zu = 0, zv = 0; // horizontal / vertical frequency of the current coefficient
  bool firstCol = false; // no block to the west
  uint32_t phase = 0; // 0: Huffman code, 1: sign bit, 2: magnitude bits
  uint32_t predCat = 0; // predicted size category of the coefficient, 0..15
  int alZz = 0; // zigzag position of the coefficient the current bit belongs to
  int coldCount = 0; // how many of the N bit history contexts have never been seen

  int specialPrediction = 0; // mixer input to use when update() returns Special
  int specialCtx = 0; // mixer context to use when update() returns Special

  // True when cp[] points at the bit histories used for the previous
  // prediction. The next update() call must train them with the actual bit.
  // Note: if a non-JPEG block comes between two JPEG blocks, we are not called
  // in between, so the first update() of the second block trains cp[] with an
  // unrelated bit. That is only a few wrong updates per stream, and not worth
  // a check on every bit. A block-level reset would fix it, if one is ever added.
  bool predicted = false;

  JpegResult lastResult = JpegResult::NoJpeg; // what update() returned for the current bit

  JpegResult setResult(const JpegResult result) {
    lastResult = result;
    return result;
  }

  // Called by update() when JpegModel does not model this bit. If the current
  // byte started in the image data, stay on the Special mixer until the end of
  // the byte, so the mixer only changes at a byte boundary. Otherwise hand the
  // bit to the general models.
  JpegResult bail() {
    specialCtx = 512 + shared->State.c0;
    return setResult(images[idx].nextJpeg != 0 ? JpegResult::Special : JpegResult::NoJpeg);
  }

  /** Stop decoding the current image and go back to the image that contains
   * it (or to the general models when there is none). The containing image
   * then skips the bytes of this one as part of the marker segment they are in.
   */
  void finishImage(uint32_t pos);

  /** Load the standard Huffman tables, for an image that has none of its own. */
  void loadStandardHuffmanTables();

  /** Sign-preserving log: sign(x) * ilog(|x| + 1). */
  int signedLog(const int x) const {
    return (x < 0 ? -1 : +1) * ilog->log(abs(x) + 1);
  }

  /** Like signedLog(), with 17 more for a nonzero value (see lcp[]). */
  int neighbourLog(const int x) const {
    return (x < 0 ? -1 : +1) * (ilog->log(abs(x) + 1) + (x != 0 ? 17 : 0));
  }

  /**
   * The extrapolation from both neighbour blocks (advPred[3]) for zigzag
   * position zz2 of the current block, before the log scaling and before the
   * DC correction. Only uses the neighbour blocks that avail marks (same bits
   * as nbAvail): with only one, it uses that one doubled, like advPred[0] and
   * advPred[2]; with none, it returns 0. sumU[] and sumV[] must be up to date.
   */
  int extrapolateDct(int zz2, int q, uint32_t avail) const;

  /**
   * Fill out[0..6] with the in-block neighbours of zigzag position zz2 in the
   * layout of lcp[], while the decoder is at position zz (zz <= zz2). The
   * neighbours from zz on are inside a run of zeros: they are 0, and cBuf2
   * does not hold them yet.
   */
  void inBlockNeighbours(int zz, int zz2, int q, int* out) const;

  /**
   * Compute everything the model knows about the next coefficient: from the
   * neighbour blocks, from the coefficients of this block decoded so far, and
   * from the same coefficient in the other color components. Call it when a
   * symbol completes.
   */
  void updatePredictions();

  /**
   * Recompute alAdv[] and alLcp[] for the nonzero coefficient after a run of
   * r zeros starting at mcuPos. Call it as soon as rs is known, before its
   * extra bits are predicted.
   */
  void alignPredictors(int r);

  /**
   * Add the inputs of bit history context i: 2 to m and 1 to m1, always,
   * whatever the state is. Also counts the contexts never seen before.
   * cp[i] must already point at the bit history for this bit.
   */
  void addContextInputs(Mixer& m, const int i) {
    const uint8_t s = *cp[i];
    if (s == 0) {
      // Never seen before. Add 0 instead of a stretched probability near 2048:
      // a context that has not learned anything yet adds more noise than
      // information. This is what makes the higher order contexts worth having
      // on small files.
      sm.skip(i);
      m.add(0);
      m.add(0);
      m1->add(0);
      ++coldCount;
    }
    else {
      const int p = sm.p2(i, s);
      m.add((p - 2048) >> 3);
      // A state that has seen only one bit gets half the usual weight.
      int st = stretch(p);
      st >>= (1 + static_cast<int>(s <= 2));
      m.add(st);
      m1->add(st);
    }
  }

public:
  explicit JpegModel(Shared* const sh, const MixerFactory* const mf, uint64_t size);
  ~JpegModel();
  JpegModel(const JpegModel&) = delete; // owns m1
  JpegModel& operator=(const JpegModel&) = delete;

  /**
   * Parse and Huffman-decode the JPEG stream up to the current bit, train the
   * bit histories used for the previous prediction, and set
   * shared->State.JPEG.state. Never touches a Mixer.
   */
  JpegResult update();

  /**
   * Add the mixer inputs and contexts for an entropy coded bit. Only valid
   * right after update() returned JpegResult::EntropyCoded. Always makes
   * exactly MIXERINPUTS add() calls and MIXERCONTEXTSETS set() calls.
   */
  void mix(Mixer& m);

  /** The mixer input to use when update() returned Special. */
  int specialInput() const {
    assert(lastResult == JpegResult::Special);
    return specialPrediction;
  }

  /** The mixer context to use when update() returned Special, in 0..SPECIALMIXERCONTEXTS-1. */
  int specialContext() const {
    assert(lastResult == JpegResult::Special);
    return specialCtx;
  }
};
