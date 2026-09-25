#include "JpegModel.hpp"
#include "../BitCount.hpp"
#include <algorithm>

namespace
{

  // The standard Huffman tables (JPEG standard, section K.3). Used when an
  // image has no Huffman tables of its own.
  // IMPORTANT: only valid for 8 bit sample precision.
  constexpr uint8_t bitsDcLuminance[16] = { 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 };
  constexpr uint8_t valuesDcLuminance[12] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

  constexpr uint8_t bitsDcChrominance[16] = { 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 };
  constexpr uint8_t valuesDcChrominance[12] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };

  constexpr uint8_t bitsAcLuminance[16] = { 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d };
  constexpr uint8_t valuesAcLuminance[162] =
  { 0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51,
    0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1,
    0x15, 0x52, 0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18,
    0x19, 0x1a, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
    0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57,
    0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75,
    0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92,
    0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8,
    0xd9, 0xda, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2,
    0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa
  };

  constexpr uint8_t bitsAcChrominance[16] = { 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77 };
  constexpr uint8_t valuesAcChrominance[162] =
  { 0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07,
    0x61, 0x71, 0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09,
    0x23, 0x33, 0x52, 0xf0, 0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25,
    0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
    0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56,
    0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74,
    0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
    0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba,
    0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6,
    0xd7, 0xd8, 0xd9, 0xda, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2,
    0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa
  };

  // The standard tables by Tc * 2 + Th.
  constexpr const uint8_t* standardBits[4] = { bitsDcLuminance, bitsDcChrominance, bitsAcLuminance, bitsAcChrominance };
  constexpr const uint8_t* standardValues[4] = { valuesDcLuminance, valuesDcChrominance, valuesAcLuminance, valuesAcChrominance };

  // SOI followed by the start of a valid marker: FF D8 FF, then SOF0/SOF1, DHT,
  // or DB..FE (DQT, DNL, DRI, ..., APPx, COM).
  bool isJpegStart(const RingBuffer<uint8_t>& buf) {
    return buf(4) == 0xFF && buf(3) == 0xD8 && buf(2) == 0xFF &&
      ((buf(1) & 0xFE) == 0xC0 || buf(1) == 0xC4 || (buf(1) >= 0xDB && buf(1) <= 0xFE));
  }

} // namespace

JpegModel::JpegModel(Shared* const sh, const MixerFactory* const mf, const uint64_t size) :
  t(size),
  MJPEGMap(sh, 21, 3, 128, 127), /* BitsOfContext, InputBits, Scale, Limit */
  sMap(sh, nSM, 27, 3, 86), /* NumContexts, BitsOfMemory (2^27 bytes = 128 MiB), InputBits, Scale */
  sm(sh, N, 256, 1023, StateMapType::BitHistory),
  smx(sh, 1, 256 * 256, 1023, StateMapType::Generic),
  apm1(sh, 0x20000, 18, 1023),
  apm2(sh, 0x4000, 20, 1023), apm3(sh, 0x4000, 21, 1023), apm4(sh, 0x4000, 22, 1023), apm5(sh, 0x4000, 23, 1023),
  apm6(sh, 0x4000, 20, 1023), apm7(sh, 0x4000, 21, 1023), apm8(sh, 0x4000, 22, 1023), apm9(sh, 0x4000, 23, 1023), apm10(sh, 0x4000, 24, 1023),
  apm11(sh, 0x8000, 22, 1023), apm12(sh, 0x8000, 22, 1023), apm13(sh, 0x8000, 22, 1023), apm14(sh, 0x8000, 22, 1023),
  shared(sh) {
  // The inner mixer. It combines the bit history contexts and the stationary
  // maps into one prediction, which the APMs in mix() then refine.
  // Must match the m1->set() calls in mix(), in number and in size.
  constexpr int innerMixerContexts =
    1024 + // horizontal position in the image
    2 +    // no block to the west
    1024 + // coefficient and bits of the Huffman code so far
    1024 + // Huffman code so far and frequency
    8192 + // Huffman code so far (hashed)
    8 +    // which extrapolations are zero
    24;    // position inside the symbol
  m1 = mf->createMixer(
    N + 1 /*bias*/ + IndirectMap::MIXERINPUTS + nSM * StationaryMapForJpegModel::MIXERINPUTS,
    innerMixerContexts, 7, 0);
  m1->setScaleFactor(960, 320, 128);
}

JpegModel::~JpegModel() {
  delete m1;
}

void JpegModel::finishImage(const uint32_t pos) {
  const uint32_t length = pos - images[idx].offset;
  memset(&images[idx], 0, sizeof(JPEGImage));
  mcuSize = 0;
  dqtState = -1;
  idx -= static_cast<int>(idx > 0);
  if (images[idx].app <= length) {
    images[idx].app = 0;
  }
  else {
    images[idx].app -= length;
  }
}

void JpegModel::loadStandardHuffmanTables() {
  for (int t = 0; t < 4; ++t) { // t = Tc * 2 + Th
    const int tc = t >> 1;
    const int th = t & 1;
    HUF* const h = &huf[tc * 64 + th * 16];
    const int start = tc * 1024 + th * 256; // where the symbols go in hBuf
    int hval = start;
    int code = 0;
    int count = 0; // number of symbols
    for (int i = 0; i < 16; ++i) {
      const int x = standardBits[t][i]; // number of codes of length i+1
      h[i].min = code;
      h[i].max = (code += x);
      h[i].val = hval;
      hval += x;
      code += code;
      count += x;
    }
    for (int i = 0; i < count; ++i) {
      hBuf[start + i] = standardValues[t][i];
    }
  }
  images[idx].htSize = 4;
}

int JpegModel::extrapolateDct(const int zz2, const int q, const uint32_t avail) const {
  const int u = zzu[zz2];
  const int v = zzv[zz2];
  int64_t tailU = 0;
  int64_t tailV = 0;
  for (int i = u + 1; i < 8; ++i) {
    tailU += sumU[i];
  }
  for (int i = v + 1; i < 8; ++i) {
    tailV += sumV[i];
  }
  const int64_t fromN = static_cast<int64_t>(sumU[u]) * (2 + u) - 2 * tailU; // sumU[] comes from the north block
  const int64_t fromW = static_cast<int64_t>(sumV[v]) * (2 + v) - 2 * tailV; // sumV[] comes from the west block
  int64_t ex = 0;
  switch (avail) {
  case 3: ex = fromN + fromW; break;
  case 2: ex = 2 * fromN; break; // no west block
  case 1: ex = 2 * fromW; break; // no north block
  default: return 0;
  }
  ex = ex * 4 / (u + v + 16);
  return static_cast<int>(ex / ((images[idx].qTable[q + zz2] + 1) * 185));
}

void JpegModel::inBlockNeighbours(const int zz, const int zz2, const int q, int* const out) const {
  const uint8_t* const qt = images[idx].qTable;
  const int cposDc = cPos - zz;
  const int u = zzu[zz2];
  const int v = zzv[zz2];

  // Every in-block neighbour of zz2 comes before it in zigzag order. Either it
  // is already decoded (< zz), or it is inside the run. Then it is zero, and
  // its cBuf2 entry has not been written yet, so it must not be read.
  const auto scaled = [&](const int k) {
    if (k >= zz) {
      return 0;
    }
    return neighbourLog((qt[q + k] + 1) * cBuf2[cposDc + k] / (qt[q + zz2] + 1));
    };
  for (int i = 0; i < 4; ++i) {
    const int a = (i & 1) != 0 ? v : u;
    const int b = (i & 2) != 0 ? 2 : 1;
    out[i] = a < b ? 65535 : scaled(zPos[u + 8 * v - ((i & 1) != 0 ? 8 : 1) * b]);
  }
  if (u * v != 0) {
    out[4] = scaled(zPos[u + 8 * v - 9]);
    out[5] = scaled(zPos[8 * v]);
    out[6] = scaled(zPos[u]);
  }
  else {
    out[4] = out[5] = out[6] = 65535;
  }
}

void JpegModel::alignPredictors(const int r) {
  const int zz = mcuPos & 63;
  const int zz2 = zz + r; // the nonzero coefficient the extra bits belong to
  if (zz2 > 63) {
    return; // invalid run; the decoder stops when the symbol completes
  }
  const int q = 64 * images[idx].qMap[mcuPos >> 6];
  const uint8_t* const qt = images[idx].qTable;
  const int u = zzu[zz2];
  const int v = zzv[zz2];

  // alAdv[0..2]: sumU[] and sumV[] hold the neighbour projections minus the
  // coefficients of this block before zz. The zeros of the run would not change
  // them, so this gives exactly what updatePredictions() would compute for
  // advPred[] at zz2 (zz2 > 0, so there is no DC correction).
  for (int i = 0; i < 3; ++i) {
    int p = sumU[u] * i + sumV[v] * (2 - i);
    p /= (qt[q + zz2] + 1) * 185 * (16 + v) * (16 + u) / 128;
    alAdv[i] = signedLog(p);
  }
  // the same border replacement as in updatePredictions()
  if ((nbAvail & 1) == 0) {
    alAdv[1] = alAdv[2];
    alAdv[0] = 0;
  }
  if ((nbAvail & 2) == 0) {
    alAdv[1] = alAdv[0];
    alAdv[2] = 0;
  }
  alAdv[3] = signedLog(extrapolateDct(zz2, q, nbAvail));

  inBlockNeighbours(zz, zz2, q, alLcp);
}

void JpegModel::updatePredictions() {
  const int aComp = mcuPos >> 6;
  const int q = 64 * images[idx].qMap[aComp];
  const int zz = mcuPos & 63U;
  const int cposDc = cPos - zz;

  const int mcuIndex = column + row * width;
  // True when the MCU that just ended is the last one of a restart
  // interval: the RSTn marker that resets pred[] comes next, but has
  // not been parsed yet, so resetPos is out of date. The same test as
  // the restart padding check in update().
  // resetLen comes from the DRI segment. Without one it is only known
  // after the first RSTn marker, so in the first interval neither test
  // works: the first DC after it is predicted as a difference, and the
  // padding before it is modelled as ordinary bits.
  const bool restartAhead = aComp == 0 && resetLen > 0 && mcuIndex - resetPos == resetLen;
  const bool noReset = resetPos != mcuIndex && !restartAhead;
  // Is this block's DC coded as a difference from cBuf2[cposDc - ls[aComp]]?
  // After a restart this is only true when ls == 64, i.e. when that
  // block is in the same MCU.
  const bool dcRelative = noReset || (ls[aComp] == 64 && aComp != 0);

  // Is the neighbour block (same component) inside the image? It is
  // not when it would be in the previous MCU column at the left
  // border, or in the previous MCU row at the top border.
  const bool westOk = !(column == 0 && blockW[aComp] > 64 * aComp);
  const bool northOk = !(row == 0 && blockN[aComp] > 64 * aComp);
  nbAvail = static_cast<uint32_t>(westOk) | static_cast<uint32_t>(northOk) << 1;

  if (zz == 0) {
    for (int i = 0; i < 8; ++i) {
      sumU[i] = sumV[i] = 0;
    }
    // position in the buffer of the DC coefficient of the west block
    // of the same component (not necessarily in this MCU)
    const int offsetDcW = cposDc - blockW[aComp];
    // position in the buffer of the DC coefficient of the north block
    // of the same component (not necessarily in this MCU)
    const int offsetDcN = cposDc - blockN[aComp];
    for (int i = 0; i < 64; ++i) {
      sumU[zzu[i]] += ((zzv[i] & 1) != 0 ? -1 : 1) * (zzv[i] != 0 ? 16 * (16 + zzv[i]) : 185) * (images[idx].qTable[q + i] + 1) *
        cBuf2[offsetDcN + i];
      sumV[zzv[i]] += ((zzu[i] & 1) != 0 ? -1 : 1) * (zzu[i] != 0 ? 16 * (16 + zzu[i]) : 185) * (images[idx].qTable[q + i] + 1) *
        cBuf2[offsetDcW + i];
    }
  }
  else {
    // the previous coefficient of this block is now known: remove it
    sumU[zzu[zz - 1]] -=
      (zzv[zz - 1] != 0 ? 16 * (16 + zzv[zz - 1]) : 185) * (images[idx].qTable[q + zz - 1] + 1) * cBuf2[cPos - 1];
    sumV[zzv[zz - 1]] -=
      (zzu[zz - 1] != 0 ? 16 * (16 + zzu[zz - 1]) : 185) * (images[idx].qTable[q + zz - 1] + 1) * cBuf2[cPos - 1];
  }

  // advPred[0..2] for this coefficient. The loop also looks up to 9
  // positions ahead: where the next larger coefficient is expected
  // (runPred[]), and the predictions 1-3 positions ahead (advPred1/2/3[]).
  for (int i = 0; i < 3; ++i) {
    runPred[i] = runPred[i + 3] = 0;
    advPred1[i] = advPred2[i] = advPred3[i] = 0;
    for (int st = 0; st < 10 && zz + st < 64; ++st) {
      const int zz2 = zz + st;
      int p = sumU[zzu[zz2]] * i + sumV[zzv[zz2]] * (2 - i);
      p /= (images[idx].qTable[q + zz2] + 1) * 185 * (16 + zzv[zz2]) * (16 + zzu[zz2]) / 128;
      if (zz2 == 0 && dcRelative) {
        p -= cBuf2[cposDc - ls[aComp]];
      }
      p = signedLog(p);
      if (st == 1) {
        advPred1[i] = p;
      }
      else if (st == 2) {
        advPred2[i] = p;
      }
      else if (st == 3) {
        advPred3[i] = p;
      }
      if (st == 0) {
        advPred[i] = p;
      }
      else if (abs(p) > abs(advPred[i]) + 2 && abs(advPred[i]) < 210) {
        if (runPred[i] == 0) {
          runPred[i] = st * 2 + static_cast<int>(p > 0);
        }
        if (abs(p) > abs(advPred[i]) + 21 && runPred[i + 3] == 0) {
          runPred[i + 3] = st * 2 + static_cast<int>(p > 0);
        }
      }
    }
  }

  // advPred[3]: from both neighbour blocks together, even when one of them
  // is outside the image (alAdv[3] below leaves that one out)
  int ex = extrapolateDct(zz, q, 3);
  if (zz == 0 && dcRelative) {
    ex -= cBuf2[cposDc - ls[aComp]];
  }
  advPred[3] = signedLog(ex);

  // lcp[]: the already decoded neighbours of this coefficient in this
  // block, scaled to this coefficient's quantization step
  inBlockNeighbours(zz, zz, q, &lcp[0]);

  lma = 0;
  for (int i = 0; i < 7; ++i) {
    lma |= static_cast<uint32_t>(lcp[i] == 65535) << i;
  }

  // prevCoef, prevCoef2, prevCoefRs: the same coefficient in the
  // blocks of this MCU decoded before this one (other components)
  int prev1 = 0;
  int prev2 = 0;
  int cnt1 = 0;
  int cnt2 = 0;
  int r = 0;
  int s = 0;
  prevCoefRs = coefficientBuffer[cPos - 64];
  for (int i = 0; i < aComp; i++) {
    ex = cBuf2[cPos - (aComp - i) * 64];
    if (zz == 0 && (noReset || ls[i] == 64)) {
      ex -= cBuf2[cposDc - (aComp - i) * 64 - ls[i]];
    }
    if (color[i] == color[aComp] - 1) {
      prev1 += ex;
      cnt1++;
      r += coefficientBuffer[cPos - (aComp - i) * 64] >> 4;
      s += coefficientBuffer[cPos - (aComp - i) * 64] & 0x0F;
    }
    if (color[aComp] > 1 && color[i] == color[0]) {
      prev2 += ex;
      cnt2++;
    }
  }
  if (cnt1 > 0) {
    prev1 /= cnt1;
    r /= cnt1;
    s /= cnt1;
    prevCoefRs = (r << 4) | s;
  }
  if (cnt2 > 0) {
    prev2 /= cnt2;
  }
  prevCoef = signedLog(11 * prev1) + (cnt1 << 20);
  prevCoef2 = signedLog(11 * prev2);

  // Image border: replace the predictions that would read a neighbour
  // block outside the image
  if (!westOk) {
    runPred[1] = runPred[2];
    runPred[0] = 0;
    advPred[1] = advPred[2];
    advPred[0] = 0;
  }
  if (!northOk) {
    runPred[1] = runPred[0];
    runPred[2] = 0;
    advPred[1] = advPred[0];
    advPred[2] = 0;
  }

  // Computed after the border replacement above, so it also marks the
  // advPred[] entries that were set to 0 at the left and top border
  ama = 0;
  for (int i = 0; i < 3; ++i) {
    ama |= static_cast<uint32_t>(advPred[i] == 0) << i;
  }

  // The aligned copies start at this coefficient. alignPredictors()
  // moves them to the end of a run of zeros once the next symbol's rs
  // is known. alAdv[3] is advPred[3] without a neighbour that is
  // outside the image.
  for (int i = 0; i < 3; ++i) {
    alAdv[i] = advPred[i];
  }
  if (nbAvail == 0) {
    alAdv[3] = 0; // no neighbour: no extrapolation, and a DC difference of 0
  }
  else {
    int e3 = extrapolateDct(zz, q, nbAvail);
    if (zz == 0 && dcRelative) {
      e3 -= cBuf2[cposDc - ls[aComp]];
    }
    alAdv[3] = signedLog(e3);
  }
  for (int i = 0; i < 7; ++i) {
    alLcp[i] = lcp[i];
  }
}

JpegResult JpegModel::update() {

  // Train the bit histories used for the previous prediction with the actual
  // bit. This must happen on every bit after a prediction, even when this bit
  // is not predicted (byte stuffing, markers, decoding errors). Otherwise cp[]
  // would later be trained with an unrelated bit.
  if (predicted) {
    INJECT_SHARED_y
    for (int i = 0; i < N; ++i) {
      StateTable::update(cp[i], y, rnd);
    }
    predicted = false;
  }

  specialPrediction = 0;
  INJECT_SHARED_c0
  // The default context for Special bits (byte stuffing and marker bits).
  // bail() and the restart padding below replace it.
  specialCtx = c0;

  INJECT_SHARED_pos
  if (idx < 0) {
    memset(&images[0], 0, sizeof(images));
    idx = 0;
    lastPos = pos;
  }
  shared->State.JPEG.state = 0;

  // Only stop on a byte boundary
  INJECT_SHARED_buf
  INJECT_SHARED_bpos
  if (bpos == 0) {
    // The rest of the SOS header, after jpeg is set to 2, is not image data yet.
    images[idx].nextJpeg = static_cast<uint32_t>(images[idx].jpeg > 1 && pos >= images[idx].data);
  }
  if (bpos != 0 && (images[idx].jpeg == 0)) {
    return bail();
  }
  if (bpos == 0 && images[idx].app > 0) {
    --images[idx].app;
    if (idx < maxEmbeddedLevel - 1 && isJpegStart(buf)) { // an embedded image, e.g. a thumbnail
      memset(&images[++idx], 0, sizeof(JPEGImage));
    }
  }
  if (images[idx].app > 0) {
    return bail();
  }

  //////////////////////////////////////////////////////////////////////////////
  // Parse the headers, one byte at a time
  //////////////////////////////////////////////////////////////////////////////

  if (bpos == 0) {
    // Baseline Huffman coded DCT JPEG syntax:
    //   SOI APPx... misc... SOF0 DHT... SOS data EOI
    //
    // SOI (FF D8): start of image.
    // APPx (FF Ex) len ...: application data, ignored. There may be several.
    //   len is always a 2 byte big-endian length that includes itself but not
    //   the 2 byte marker before it. The other segments below use the same len.
    // misc: DQT, DNL, DRI, COM (COM is ignored).
    // SOF0 (FF C0) len 08 height width Nf [c HV Tq]...
    //   height and width are in pixels, 2 bytes each. Nf (1 byte) is the number
    //   of components. For each component: c is its identifier, HV its
    //   horizontal and vertical sampling factors (high and low 4 bits), and Tq
    //   its quantization table number. An MCU (minimum coded unit) holds H*V
    //   blocks of 64 DCT coefficients for each component.
    // DHT (FF C4) len [TcTh L1..L16 V1,1..V1,L1 ... V16,1..V16,L16]...
    //   defines Huffman table Th (0-3) for Tc (0 = DC, the first coefficient
    //   of a block; 1 = AC, the other 63). L1..L16 are the number of codes of
    //   each length 1-16, and the V values are the symbols, 1 byte each.
    //   An AC symbol RS means: a run of R (0-15) zeros, then a nonzero value
    //   given by the S (0-15) extra bits after the code; the value is negative
    //   if the first extra bit is 0. For example, symbol 0x63 followed by the
    //   bits 1,0,1 gives 7 coefficients: 0, 0, 0, 0, 0, 0, 5.
    //   Symbol 00 means end of block: the remaining AC coefficients are 0.
    //   Symbol F0 (ZRL) means 16 zeros: a run of 15, then a zero value
    //   (S = 0, no extra bits).
    // SOS (FF DA) len Ns [Cs TdTa]... 0 3F 00
    //   start of scan. For each of the Ns (1-4) components: Cs matches a c in
    //   SOF0, and TdTa gives its DC and AC Huffman table numbers (0-3), packed
    //   into one byte.
    // EOI (FF D9): end of image.
    // The Huffman coded data follows SOS. Markers may appear inside it:
    // RST0-RST7 (FF D0 to FF D7) mark the start of an independently coded part.
    // DNL (FF DC) 04 00 height may appear at the end of the scan (ignored).
    // FF 00 in the data stands for FF (so it is not mistaken for a marker).

    // Detect a JPEG: SOI followed by a valid marker
    if ((images[idx].jpeg == 0) && isJpegStart(buf)) {
      images[idx].jpeg = 1;
      images[idx].offset = pos - 4;
      images[idx].sos = images[idx].sof = images[idx].htSize = images[idx].data = images[idx].dri = 0;
      // SOI followed by APPx: skip its 2 length bytes first; then the APPx
      // detection below skips the rest of the segment.
      images[idx].app = static_cast<int>(buf(1) >> 4 == 0xE) * 2;
      mcuSize = huffCode = huffBits = huffSize = mcuPos = cPos = 0;
      rs = -1;
      memset(&huf[0], 0, huf.size() * sizeof(HUF));
      memset(&pred[0], 0, pred.size() * sizeof(int));
      resetPos = resetLen = 0;
    }

    // Detect the end of the JPEG: a marker in the data other than RSTx or
    // byte stuffing (00), or a jump in position since the last byte
    if ((images[idx].jpeg != 0) && (images[idx].data != 0) &&
      ((buf(2) == FF && (buf(1) != 0) && (buf(1) & 0xf8) != RST0) || (pos - lastPos > 1))) {
      JASSERT((buf(1) == EOI) || (pos - lastPos > 1))
      finishImage(pos);
    }
    lastPos = pos;
    if (images[idx].jpeg == 0) {
      return bail();
    }

    // Detect APPx, COM and other markers, so they can be skipped.
    // skippableSegment: C1-CF except DHT (SOF1-SOF15, JPG, DAC) and DC-FE
    // (DNL, DRI, DHP, EXP, APPx, JPGn, COM), only in the top-level image.
    // Skipping SOFn does not lose its position, which is saved below.
    bool skippableSegment = ((((buf(3) >= 0xC1) && (buf(3) <= 0xCF) && (buf(3) != DHT)) || ((buf(3) >= 0xDC) && (buf(3) <= 0xFE))) && idx == 0);
    if ((images[idx].data == 0) && (images[idx].app == 0) && buf(4) == FF &&
      (buf(3) >> 4 == 0xe || buf(3) == COM || skippableSegment)) {
      images[idx].app = buf(2) * 256 + buf(1) + 2;
      if (idx > 0) {
        JASSERT(pos + images[idx].app < images[idx].offset + images[idx - 1].app)
      }
    }

    // Save the positions of SOF, DHT, SOS and the image data
    if (buf(5) == FF && buf(4) == SOS) {
      int len = buf(3) * 256 + buf(2);
      if (len == 6 + 2 * buf(1) && (buf(1) != 0) && buf(1) <= 4) { // buf(1) is Ns
        images[idx].sos = pos - 5;
        images[idx].data = images[idx].sos + len + 2;
        images[idx].jpeg = 2;
      }
    }
    if (buf(4) == FF && buf(3) == DHT && images[idx].htSize < 8) {
      images[idx].ht[images[idx].htSize++] = pos - 4;
    }
    if (buf(4) == FF && buf(3) == DRI) {
      images[idx].dri = pos - 4;
    }
    if (buf(4) == FF && (buf(3) & 0xFE) == SOF0) {
      images[idx].sof = pos - 4;
    }
    // SOF3 is lossless JPEG: its data is not DCT coefficients, so decoding it
    // as baseline would only give useless contexts. Stop decoding this stream
    // and let the general models handle it.
    if (buf(4) == FF && buf(3) == SOF3) {
      images[idx].jpeg = images[idx].nextJpeg = 0;
    }

    // Parse the quantization tables. Each table is a header byte (precision and
    // table number) and 64 entries. An entry is 1 byte, or 2 bytes when the
    // header says 16 bit precision; then only the low byte is kept.
    if (buf(4) == FF && buf(3) == DQT) {
      dqtEnd = pos + buf(2) * 256 + buf(1) - 1;
      dqtState = 0;
      qNum16b = 0;
      qNum16 = 0;
    }
    else if (dqtState >= 0) {
      if (pos >= dqtEnd) {
        dqtState = -1;
      }
      else if (qNum16 != 0) { // 16 bit entries
        if (dqtState % 65 == 0) {
          qNum = buf(1);
          qNum16b = 0;
          qNum16 = qNum >> 4;
          qNum = qNum & 0xf;
        }
        else if ((qNum16b & 1) == 0) {
          JASSERT(buf(1) > 0)
          JASSERT(qNum < 4)
          images[idx].qTable[qNum * 64 + ((dqtState % 65) - 1)] = buf(1) - 1;
        }
        if ((qNum16b & 1) == 0) {
          dqtState++;
        }
        qNum16b++;
      }
      else { // 8 bit entries
        if (dqtState % 65 == 0) {
          qNum = buf(1);
          qNum16b = 0;
          qNum16 = qNum >> 4;
          qNum = qNum & 0xf;
        }
        else {
          JASSERT(buf(1) > 0)
          JASSERT(qNum < 4)
          images[idx].qTable[qNum * 64 + ((dqtState % 65) - 1)] = buf(1) - 1;
        }
        dqtState++;
        qNum16b++;
      }
    }

    // Restart marker: the coded data starts over after it
    if (buf(2) == FF && (buf(1) & 0xf8) == RST0) {
      huffCode = huffBits = huffSize = mcuPos = 0;
      rs = -1;
      memset(&pred[0], 0, pred.size() * sizeof(int));
      resetLen = column + row * width - resetPos;
      resetPos = column + row * width;
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // The first bit of the image data has just been coded: build the decoding tables
  //
  // That is bpos == 1 in the first data byte. The decoder below runs in the
  // same call and adds y, the bit just coded, so this is the first call where
  // y is image data. At bpos == 0, y would still be the last bit of the SOS
  // header.
  //////////////////////////////////////////////////////////////////////////////

  {
    if (pos == images[idx].data && bpos == 1) {
      // Build the Huffman tables.
      // huf[Tc][Th][m] = smallest code of length m+1, largest + 1, position of the symbols
      for (uint32_t i = 0; i < images[idx].htSize; ++i) {
        uint32_t p = images[idx].ht[i] + 4; // position of the current table, after the length field
        uint32_t end = p + buf[p - 2] * 256 + buf[p - 1] - 2; // end of the Huffman table
        uint32_t count = 0; // limits the number of tables, in case the data is invalid
        while (p < end && end < pos && end < p + 2100 && ++count < 10) {
          int tc = buf[p] >> 4;
          int th = buf[p] & 15;
          if (tc >= 2 || th >= 4) {
            break; // invalid table: the JASSERT(p == end) below stops decoding
          }
          HUF* h = &huf[tc * 64 + th * 16]; // [tc][th][0];
          int val = p + 17; // position of the symbols
          int hval = tc * 1024 + th * 256; // where the symbols go in hBuf
          int j = 0;
          for (j = 0; j < 256; ++j) { // copy the symbols
            hBuf[hval + j] = buf[val + j];
          }
          int code = 0;
          for (j = 0; j < 16; ++j) {
            h[j].min = code;
            h[j].max = code += buf[p + j + 1];
            h[j].val = hval;
            val += buf[p + j + 1];
            hval += buf[p + j + 1];
            code *= 2;
          }
          p = val;
          JASSERT(hval - (tc * 1024 + th * 256) <= 256) // at most 256 symbols, within this table's part of hBuf
        }
        JASSERT(p == end)
      }
      huffCode = huffBits = huffSize = 0;
      rs = -1;

      if (images[idx].htSize == 0) {
        loadStandardHuffmanTables();
      }

      // No SOF0/SOF1 before the scan (e.g. progressive SOF2, or arithmetic
      // coding): the data cannot be decoded as baseline, so stop decoding this
      // stream, the same way as for invalid data.
      JASSERT(images[idx].sof != 0)

      // Build the table that selects the Huffman table for each block of the
      // MCU, and get the image width.
      int ns = buf[images[idx].sos + 4];
      int nf = buf[images[idx].sof + 9];
      JASSERT(ns <= 4 && nf <= 4)
      mcuSize = 0; // blocks per MCU
      int hmax = 0; // largest horizontal sampling factor of the components in the scan
      int hmaxFrame = 0; // largest horizontal sampling factor of all components
      for (int j = 0; j < nf; ++j) {
        hmaxFrame = max(hmaxFrame, buf[images[idx].sof + 3 * j + 11] >> 4);
      }
      for (int i = 0; i < ns; ++i) {
        for (int j = 0; j < nf; ++j) {
          if (buf[images[idx].sos + 2 * i + 5] == buf[images[idx].sof + 3 * j + 10]) { // Cs == c ?
            int hv = buf[images[idx].sof + 3 * j + 11]; // sampling factors H and V, packed
            if (hv >> 4 > hmax) {
              hmax = hv >> 4;
            }
            // A non-interleaved scan (Ns = 1) has one block per MCU, whatever
            // the sampling factors are (JPEG standard, A.2.2).
            if (ns == 1) {
              hv = 0x11;
            }
            samplingFactors[i] = hv;
            hv = (hv & 15U) * (hv >> 4); // number of blocks of this component in an MCU
            JASSERT(hv >= 1 && hv + mcuSize <= 10)
            while (hv != 0) {
              JASSERT(mcuSize < 10)
              hufSel[0][mcuSize] = buf[images[idx].sos + 2 * i + 6] >> 4 & 15;
              hufSel[1][mcuSize] = buf[images[idx].sos + 2 * i + 6] & 15;
              JASSERT(hufSel[0][mcuSize] < 4 && hufSel[1][mcuSize] < 4)
              color[mcuSize] = i;
              int tq = buf[images[idx].sof + 3 * j + 12]; // quantization table number (0..3)
              JASSERT(tq >= 0 && tq < 4)
              images[idx].qMap[mcuSize] = tq;
              --hv;
              ++mcuSize;
            }
          }
        }
      }
      JASSERT(hmax >= 1 && hmax <= 10 && hmaxFrame >= hmax)
      int j = 0;
      for (j = 0; j < mcuSize; ++j) {
        ls[j] = 0;
        for (int i = 1; i < mcuSize; ++i) {
          if (color[(j + i) % mcuSize] == color[j]) {
            ls[j] = i;
          }
        }
        ls[j] = (mcuSize - ls[j]) << 6;
      }
      for (j = 0; j < 64; ++j) {
        zPos[zzu[j] + 8 * zzv[j]] = j;
      }
      width = buf[images[idx].sof + 7] * 256 + buf[images[idx].sof + 8]; // in pixels
      if (ns == 1) {
        // One block per MCU: the width is the component's own width in blocks.
        // The component is hmax / hmaxFrame of the image width (rounded up).
        const int componentWidth = (width * hmax + hmaxFrame - 1) / hmaxFrame; // in pixels
        width = (componentWidth + 7) / 8; // in MCUs
      }
      else {
        width = (width - 1) / (hmax * 8) + 1; // in MCUs
      }
      JASSERT(width > 0)

      // The restart interval from DRI, so the restart tests work from the start
      // (see updatePredictions()). Each RSTn marker then measures it again.
      if (images[idx].dri != 0) {
        resetLen = buf[images[idx].dri + 4] * 256 + buf[images[idx].dri + 5];
      }
      mcuSize *= 64; // coefficients per MCU
      row = column = 0;

      // Nothing has been decoded from this image yet, so these predictions left
      // over from the previous image must not be used. This matters for files
      // with many small images.
      nbAvail = 0;
      for (int i = 0; i < 4; ++i) {
        alAdv[i] = 0;
      }
      for (int i = 0; i < 7; ++i) {
        alLcp[i] = 65535;
      }

      // With subsampling a component has several blocks in an MCU. Compute for
      // each block the distance to its west and north neighbour of the same
      // component.
      int x = 0;
      int y = 0;
      for (j = 0; j < (mcuSize >> 6); j++) {
        int i = color[j];
        int w = samplingFactors[i] >> 4;
        int h = samplingFactors[i] & 0x0f;
        blockW[j] = x == 0 ? mcuSize - 64 * (w - 1) : 64;
        blockN[j] = y == 0 ? mcuSize * width - 64 * w * (h - 1) : w * 64;
        x++;
        if (x >= w) {
          x = 0;
          y++;
        }
        if (y >= h) {
          x = 0;
          y = 0;
        }
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Huffman decoding: add the last bit to the current symbol. When the symbol
  // is complete, store its coefficients and compute the predictions for the
  // next coefficient.
  //////////////////////////////////////////////////////////////////////////////

  {
    if ((mcuSize != 0) && buf(1 + static_cast<int>(bpos == 0)) != FF) { // skip the stuffed 00 after FF
      JASSERT(huffBits <= 32)
      INJECT_SHARED_y
      huffCode += huffCode + y;
      ++huffBits;
      if (rs < 0) {
        JASSERT(huffBits >= 1 && huffBits <= 16)
        const int ac = static_cast<int>((mcuPos & 63U) > 0);
        JASSERT((mcuPos >> 6) < 10)
        const int sel = hufSel[ac][mcuPos >> 6];
        JASSERT(sel >= 0 && sel < 4)
        const int i = huffBits - 1;
        const HUF* h = &huf[ac * 64 + sel * 16]; // [ac][sel];
        JASSERT(h[i].min <= h[i].max && h[i].val < 2048)
        if (huffCode < h[i].max) {
          JASSERT(huffCode >= h[i].min)
          int k = h[i].val + huffCode - h[i].min;
          JASSERT(k >= 0 && k < 2048)
          rs = hBuf[k];
          huffSize = huffBits;
          // An AC symbol with a run of zeros and extra bits: the extra bits
          // belong to the coefficient r positions after mcuPos, so move the
          // predictions there. (With s == 0 the symbol is already complete: it
          // is handled right below, and updatePredictions() resets the predictions.)
          if (ac != 0 && (rs >> 4) != 0 && (rs & 15) != 0) {
            alignPredictors(rs >> 4);
          }
        }
      }
      if (rs >= 0) {
        if (huffSize + (rs & 15) == huffBits) { // the symbol is complete
          rs1 = rs;
          int ex = 0; // the value given by the extra bits
          if ((mcuPos & 63) != 0) { // AC
            if (rs == 0) { // end of block
              mcuPos = (mcuPos + 63) & 0xFFFFFFC0;
              JASSERT(mcuPos <= static_cast<uint32_t>(mcuSize) && mcuPos <= 640)
              bool first = true; // the first remaining coefficient is stored as 0
              while ((cPos & 63) != 0) {
                cBuf2.set(cPos, 0);
                coefficientBuffer.set(cPos, first ? 0 : (63 - (cPos & 63U)) << 4);
                cPos++;
                first = false;
              }
            }
            else { // r zeros, then a value given by s extra bits (nonzero, except for ZRL)
              // The value is negative if the first extra bit is 0.
              JASSERT((rs & 15) <= 10)
              const int r = rs >> 4;
              const int s = rs & 15;
              JASSERT(mcuPos >> 6 == (mcuPos + r) >> 6)
              mcuPos += r + 1;
              ex = huffCode & ((1U << s) - 1);
              if (s != 0 && (ex >> (s - 1)) == 0) {
                ex -= (1u << s) - 1;
              }
              for (int i = r; i >= 1; --i) {
                cBuf2.set(cPos, 0);
                coefficientBuffer.set(cPos, (i << 4) | s);
                cPos++;
              }
              cBuf2.set(cPos, ex);
              coefficientBuffer.set(cPos, (s << 4) | (huffCode << 2 >> s & 3U) | 12);
              cPos++;
              sSum += s;
            }
          }
          else { // DC: the symbol is s alone (rs < 12)
            JASSERT(rs < 12)
            ++mcuPos;
            ex = huffCode & ((1U << rs) - 1);
            if (rs != 0 && (ex >> (rs - 1)) == 0) {
              ex -= (1U << rs) - 1;
            }
            JASSERT(mcuPos >> 6 < 10)
            const uint32_t dcComp = color[mcuPos >> 6];
            JASSERT(dcComp < 4)
            dc = pred[dcComp] += ex;

            // realign to a block boundary; needed for some thumbnails in phone images
            while ((cPos & 63) != 0) {
              cPos++;
            }

            cBuf2.set(cPos, dc);
            coefficientBuffer.set(cPos, (dc + 1023) >> 3);
            cPos++;
            if ((mcuPos >> 6) == 0) {
              sSum1 = 0;
              sSum2 = sSum3;
            }
            else {
              if (color[(mcuPos >> 6) - 1] == color[0]) {
                sSum1 += (sSum3 = sSum);
              }
              sSum2 = sSum1;
            }
            sSum = rs;
          }
          JASSERT(mcuPos <= static_cast<uint32_t>(mcuSize))
          if (mcuPos >= static_cast<uint32_t>(mcuSize)) {
            mcuPos = 0;
            if (++column == width) {
              column = 0;
              ++row;
            }
          }
          huffCode = huffSize = huffBits = 0;
          rs = -1;
          updatePredictions();
        }
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Is this bit entropy coded, i.e. should JpegModel predict it?
  //////////////////////////////////////////////////////////////////////////////

  if ((images[idx].jpeg == 0) || (images[idx].data == 0) || pos < images[idx].data) { // not yet in the image data
    return bail();
  }
  // Only predict the bits that the Huffman decoder above adds to huffCode, so
  // that every prediction matches one decoder step, and hbCount and cp[] stay
  // in step with the decoder.
  // Each call predicts the current bit and adds the previous bit to the
  // decoder. When bpos > 0 both bits are in the same byte, so both checks look
  // at the byte before it, buf(1). When bpos == 0 they are in different bytes:
  // the previous bit is the last bit of buf(1), which the decoder checks
  // against buf(2); the current bit is the first bit of a new byte, which the
  // check below tests against buf(1). So the two checks only differ when
  // bpos == 0.
  if (buf(1) == FF) { // byte stuffing or marker
    return setResult(JpegResult::Special);
  }
  if (resetLen > 0 && resetLen == column + row * width - resetPos && mcuPos == 0 && huffCode == (1u << huffBits) - 1u) {
    // Padding before a restart marker: all its bits are 1
    specialPrediction = 2047;
    specialCtx = 256 + c0;
    return setResult(JpegResult::Special);
  }

  //////////////////////////////////////////////////////////////////////////////
  // Context values for this bit, used by mix() and by the SSE stage
  //////////////////////////////////////////////////////////////////////////////

  comp = color[mcuPos >> 6];
  coef = (mcuPos & 63) | comp << 6;
  hc = (huffCode * 4 + static_cast<uint32_t>((mcuPos & 63) == 0) * 2 + static_cast<uint32_t>(comp == 0)) | 1u << (huffBits + 2);
  firstCol = column == 0 && static_cast<uint32_t>(blockW[mcuPos >> 6]) > mcuPos;
  zu = zzu[mcuPos & 63];
  zv = zzv[mcuPos & 63];

  // How far we are into the current symbol. While the Huffman code is being
  // read, huffSize is 0, so e is the code length so far; after that, e counts
  // the extra bits.
  const uint32_t e = huffBits - huffSize;

  // While the Huffman code is being read there is one context per node of the
  // code tree. Once the symbol is known, the code itself does not matter any
  // more, only: which symbol it was, which extra bit we are on, and the sign
  // (the first extra bit). Keeping all the extra bits would double the number
  // of contexts with each bit, without any gain.
  symKey = rs < 0 ? hc
    : 0x80000000u | rs << 8 | e << 4 | (e != 0 ? (huffCode >> (e - 1)) & 1 : 0) << 2
    | static_cast<uint32_t>((mcuPos & 63) == 0) << 1 | static_cast<uint32_t>(comp == 0);

  // During a run of zeros, the bits belong to the coefficient r positions after
  // mcuPos, and alAdv[] and alLcp[] describe that one.
  alZz = static_cast<int>(mcuPos & 63) +
    ((rs >= 0 && (mcuPos & 63) != 0) ? min(63 - static_cast<int>(mcuPos & 63), rs >> 4) : 0);
  // How large the model expects this coefficient to be, as a JPEG size
  // category (the number of extra bits, SSSS). Exact below |p| = 32; above
  // that, it can be one too high just above a power of two, because of how
  // ilog rounds.
  const int predMag = abs(alAdv[1]); // 16 * log2(|p| + 1)
  predCat = min(15, (predMag + 15) >> 4);

  // A short summary of this bit for the SSE stage: which kind of bit it is and
  // where in the symbol (symClass), and what our own prediction says about it
  // (predBits). There are three kinds of bits: Huffman code, sign, magnitude.
  // 8 bits. Codes of 7 or more bits lose their leading-1 length marker, and
  // then share values with shorter codes.
  uint32_t symClass = 0;
  uint32_t predBits = 0; // 2 bits
  phase = 0;
  if (rs < 0) { // Huffman code
    // the code read so far, with a leading 1 to mark its length
    symClass = ((((1u << e) | (huffCode & ((1u << e) - 1))) << 1) | 0u) & 0xFF; // bit 0 is 0 for Huffman code bits
    predBits = min(3, static_cast<int>(predCat));
  }
  else {
    const uint32_t s = rs & 15;
    const uint32_t positive = e != 0 ? (huffCode >> (e - 1)) & 1 : 0;
    symClass = 1 | s << 1 | std::min(e, 3u) << 5 | positive << 7;
    if (e == 0) { // sign bit
      phase = 1;
      predBits = static_cast<uint32_t>(alAdv[1] > 0) << 1 |
        static_cast<uint32_t>(static_cast<int>(predCat) >= static_cast<int>(s)); // the prediction reaches the size category
    }
    else { // magnitude bits: each bit halves the range of values still possible
      phase = 2;
      const uint32_t x = huffCode & ((1u << e) - 1);       // the extra bits so far
      const uint32_t mid = (2 * x + 1) << (s - e - 1);     // the smallest coded value where the next bit is 1
      // d > 0: the prediction says the next bit is 1. Negative values are coded
      // as v + 2^s - 1, so for them the bit is 1 when |v| <= 2^s - 1 - mid.
      const int d = positive != 0
        ? predMag - ilog->log(mid + 1)
        : ilog->log((1u << s) - mid) - predMag;
      predBits = static_cast<uint32_t>(d >= 0) << 1 | static_cast<uint32_t>(abs(d) >= 8); // which side, and by at least half an octave
    }
  }

  shared->State.JPEG.state = 1 + (
    predBits << 10 | // 2 bits
    symClass << 2 |  // 8 bits, also tells the phase
    static_cast<uint32_t>(comp == 0) << 1 |
    static_cast<uint32_t>(zu + zv < 5)
    ); // 1 + (0..4095)

  return setResult(JpegResult::EntropyCoded);
}

void JpegModel::mix(Mixer& m) {
  assert(lastResult == JpegResult::EntropyCoded);

  //////////////////////////////////////////////////////////////////////////////
  // Bit history contexts
  //
  // The contexts are looked up at the start of every symbol and then once
  // every 3 bits. For the bits in between, cp[] just moves to the next node of
  // the 7 bit histories in the slot, which saves a hash lookup per bit.
  //
  // The //strong, //medium and //weak notes on the contexts, maps and mixer
  // contexts below record how much each one helped in tuning (see "For
  // tuning"); commented out lines were not worth keeping.
  //////////////////////////////////////////////////////////////////////////////

  if (++hbCount > 2 || huffBits == 0) {
    hbCount = 0;
  }
  if (hbCount == 0) {
    uint64_t n = static_cast<uint64_t>(hc) * 64;
    int i = 0;

    // n + 24 (three contexts) and n + 32 (two contexts) are shared on purpose:
    // these contexts are similar, so sharing their statistics helps.

    // For tuning:
    //int skip = (int)shared->tuning_param;
    //int k = 0;
    //k++; if (k == skip); else cxt[i++] = ...

    cxt[i++] = hash(n + 1, hc);
    cxt[i++] = hash(n + 2, coef, advPred[2] / 12 + (runPred[2] << 8), sSum2 >> 6, prevCoef / 72);
    cxt[i++] = hash(n + 3, coef, advPred[0] / 12 + (runPred[0] << 8), sSum2 >> 6, prevCoef / 72);
    //cxt[i++] = hash(n + 4, coef, advPred[1] / 11 + (runPred[1] << 8), sSum2 >> 6); //weak
    //cxt[i++] = hash(n + 5, rs1, advPred[2] / 7, runPred[5] / 2, prevCoef / 10); //weak
    cxt[i++] = hash(n + 6, rs1, advPred[0] / 7, runPred[3] / 2, prevCoef / 10); //weak
    cxt[i++] = hash(n + 7, rs1, advPred[1] / 11, runPred[4]);
    cxt[i++] = hash(n + 8, advPred[2] / 14, runPred[2], advPred[0] / 14, runPred[0]);
    cxt[i++] = hash(n + 9, coefficientBuffer[cPos - blockN[mcuPos >> 6]] >> 4, advPred[3] / 17, runPred[1], runPred[5]); //weak
    cxt[i++] = hash(n + 10, coefficientBuffer[cPos - blockW[mcuPos >> 6]] >> 4, advPred[3] / 17, runPred[1], runPred[3]);
    cxt[i++] = hash(n + 11, lcp[0] / 22, lcp[1] / 22, advPred[1] / 7, runPred[1]);
    cxt[i++] = hash(n + 12, lcp[0] / 22, lcp[1] / 22, mcuPos & 63, lcp[4] / 30);
    cxt[i++] = hash(n + 13, zu / 2, lcp[0] / 13, lcp[2] / 30, prevCoef / 40 + ((prevCoef2 / 28) << 20));
    cxt[i++] = hash(n + 14, zv / 2, lcp[1] / 13, lcp[3] / 30, prevCoef / 40 + ((prevCoef2 / 28) << 20));
    cxt[i++] = hash(n + 15, rs1, prevCoef / 42, prevCoef2 / 34, lcp[0] / 60, lcp[2] / 14, lcp[1] / 60, lcp[3] / 14); // strong
    cxt[i++] = hash(n + 16, mcuPos & 63, column >> 1);  // strong
    cxt[i++] = hash(n + 17, column >> 3, min(5 + 2 * (comp == 0), zu + zv), lcp[0] / 10, lcp[2] / 40, lcp[1] / 10, lcp[3] / 40);  // strong
    cxt[i++] = hash(n + 18, sSum >> 3, mcuPos & 63);
    cxt[i++] = hash(n + 19, rs1, mcuPos & 63, runPred[1]);
    cxt[i++] = hash(n + 20, coef, sSum2 >> 5, advPred[3] / 30, comp != 0 ? hash(n + prevCoef / 22, prevCoef2 / 50) : sSum / ((mcuPos & 0x3F) + 1));
    cxt[i++] = hash(n + 21, lcp[0] / 40, lcp[1] / 40, advPred[1] / 28, comp != 0 ? prevCoef / 40 + ((prevCoef2 / 40) << 20) : lcp[4] / 22, min(7, zu + zv), sSum / (2 * (zu + zv) + 1)); // very strong
    cxt[i++] = hash(n + 22, zv, coefficientBuffer[cPos - blockN[mcuPos >> 6]], advPred[2] / 28, runPred[2]); // strong
    cxt[i++] = hash(n + 23, zu, coefficientBuffer[cPos - blockW[mcuPos >> 6]], advPred[0] / 28, runPred[0]); // strong
    cxt[i++] = hash(n + 24, advPred[2] / 7, runPred[2]);//weak
    cxt[i++] = hash(n + 24, advPred[0] / 7, runPred[0]);
    cxt[i++] = hash(n + 24, advPred[1] / 7, runPred[1]);//weak
    cxt[i++] = hash(n + 25, zv, lcp[1] / 14, advPred[2] / 16, runPred[5]); // strong
    cxt[i++] = hash(n + 26, zu, lcp[0] / 14, advPred[0] / 16, runPred[3]); // strong
    cxt[i++] = hash(n + 27, lcp[0] / 14, lcp[1] / 14, advPred[3] / 16); //weak
    cxt[i++] = hash(n + 28, coef, prevCoef / 10, prevCoef2 / 20);
    cxt[i++] = hash(n + 29, coef, sSum >> 2, prevCoefRs);
    cxt[i++] = hash(n + 30, coef, advPred[1] / 17, lcp[(zu < zv)] / 24, lcp[2] / 20, lcp[3] / 24); // very strong also for MJPEG files
    cxt[i++] = hash(n + 31, coef, advPred[3] / 11, lcp[(zu < zv)] / 50, lcp[2 + 3 * (zu * zv > 1)] / 50, lcp[3 + 3 * static_cast<uint64_t>(zu * zv > 1)] / 50);
    cxt[i++] = hash(n + 32, hc, advPred[3] / 13, prevCoef / 11, (zu + zv < 4));//weak
    cxt[i++] = hash(n + 32, hc, advPred[2] / 13, prevCoef / 11, (zu + zv < 4));//weak
    //cxt[i++] = hash(n + 32, hc, advPred[1] / 13, prevCoef / 11, static_cast<int>(zu + zv < 4));//weak
    //cxt[i++] = hash(n + 33, hc, advPred[1] / 13, prevCoef2 / 20, prevCoefRs);//weak
    cxt[i++] = hash(n + 34, hc, advPred[2] / 13, prevCoef / 40 + ((prevCoef2 / 40) << 20), prevCoefRs);
    cxt[i++] = hash(n + 35, hc, advPred[0] / 13, prevCoef / 40 + ((prevCoef2 / 40) << 20), (zu + zv < 4));//weak
    cxt[i++] = hash(n + 36, prevCoef2, prevCoefRs);//weak except for MJPEGs
    cxt[i++] = hash(n + 37, hc, runPred[0] / 13, prevCoef / 11, (zu + zv < 4));
    //cxt[i++] = hash(n + 38, hc, advPred[2] / 7, runPred[2]); //weak
    cxt[i++] = hash(n + 39, hc, advPred[2] / 12, runPred[2], sSum2 >> 6, prevCoef / 12); //weak?
    //cxt[i++] = hash(n + 40, hc, advPred[1] / 11, runPred[1], sSum2 >> 6); //weak
    //cxt[i++] = hash(n + 41, hc, rs1, advPred[0] / 7, prevCoef2 / 10); //weak
    cxt[i++] = hash(n + 42, hc, lcp[0] / 12, lcp[1] / 12, mcuPos & 63, lcp[4] / 10); // strong for MJPEG files
    cxt[i++] = hash(n + 43, hc, zu / 2, prevCoef / 40 + ((prevCoef2 / 28) << 20));
    cxt[i++] = hash(n + 44, hc, zv / 2, prevCoef / 40 + ((prevCoef2 / 28) << 20));
    cxt[i++] = hash(n + 45, hc, column >> 3, min(5 + 2 * (comp == 0), zu + zv), prevCoef / 40 + ((prevCoef2 / 28) << 20));

    assert(i == N);
  }

  coldCount = 0; // counted by addContextInputs() below; used as a mixer context

  m1->add(128); // bias
  sm.subscribe();
  assert(hbCount <= 2);
  switch (hbCount) {
  case 0: {
    for (int i = 0; i < N; ++i) {
      cp[i] = t[cxt[i]];
      addContextInputs(m, i);
    }
    break;
  }
  case 1: {
    const size_t hcOffset = 1 + (huffCode & 1) * 3;
    for (int i = 0; i < N; ++i) {
      cp[i] += hcOffset;
      addContextInputs(m, i);
    }
    break;
  }
  default: {
    const size_t hcOffset = 1 + (huffCode & 1);
    for (int i = 0; i < N; ++i) {
      cp[i] += hcOffset;
      addContextInputs(m, i);
    }
    break;
  }
  }
  // cp[] now points at the bit histories used for this prediction. The next
  // update() call must train them, whatever else it does.
  predicted = true;

  //////////////////////////////////////////////////////////////////////////////
  // Stationary maps
  //////////////////////////////////////////////////////////////////////////////

  if (hbCount == 0) {
    const uint64_t h = hc >> 2; // drop the DC/AC and luma flags
    int j = 0;

    // For tuning:
    //int skip = (int)shared->tuning_param;
    //int k = 0;
    //k++; if (k == skip); else sMap[j++].set ...

    // === DCT extrapolation x coefficient position ===
    sMap.set(j++, hash(h, coef, advPred[0] / 11));
    //sMap.set(j++, hash(h, coef, advPred[1] / 11)); //weak
    sMap.set(j++, hash(h, coef, advPred[2] / 11));
    //sMap.set(j++, hash(h, coef, advPred[3] / 11)); //weak

    // === extrapolation of the next positions x coefficient position ===
    sMap.set(j++, hash(h, coef, advPred1[0] / 11));
    sMap.set(j++, hash(h, coef, advPred1[1] / 11));
    sMap.set(j++, hash(h, coef, advPred1[2] / 11));
    sMap.set(j++, hash(h, coef, advPred2[0] / 11)); //strong
    //sMap.set(j++, hash(h, coef, advPred2[1] / 11)); //weak
    sMap.set(j++, hash(h, coef, advPred2[2] / 11));
    //sMap.set(j++, hash(h, coef, advPred3[2] / 11)); //weak
    sMap.set(j++, hash(h, coef, advPred1[2] / 11, advPred2[2] / 11));
    sMap.set(j++, hash(h, advPred[2] / 11, advPred1[2] / 11));

    // === in-block neighbours x coefficient position ===
    sMap.set(j++, hash(h, coef, lcp[0] / 7));
    sMap.set(j++, hash(h, coef, lcp[1] / 7));
    sMap.set(j++, hash(h, coef, lcp[2] / 7));
    sMap.set(j++, hash(h, coef, lcp[3] / 7));
    sMap.set(j++, hash(h, coef, lcp[4] / 7));
    sMap.set(j++, hash(h, coef, lcp[5] / 7));
    sMap.set(j++, hash(h, coef, lcp[6] / 7));

    // === pairs of in-block neighbours ===
    sMap.set(j++, hash(h, lcp[0] / 14, lcp[1] / 14));
    sMap.set(j++, hash(h, lcp[2] / 14, lcp[3] / 14));
    sMap.set(j++, hash(h, lcp[3] / 14, lcp[4] / 14));
    sMap.set(j++, hash(h, coef, lcp[0] / 10));
    sMap.set(j++, hash(h, coef, lcp[2] / 10, lcp[3] / 10));
    sMap.set(j++, hash(h, coef, lcp[5] / 14, lcp[6] / 14)); //strong
    sMap.set(j++, hash(h, lcp[0] / 14, lcp[2] / 14));
    sMap.set(j++, hash(h, lcp[1] / 14, lcp[3] / 14)); //strong
    sMap.set(j++, hash(h, lcp[4] / 14, lcp[5] / 14));

    // === run prediction x coefficient position ===
    sMap.set(j++, hash(h, coef, runPred[0]));
    //sMap.set(j++, hash(h, coef, runPred[1])); //weak
    sMap.set(j++, hash(h, coef, runPred[2]));
    //sMap.set(j++, hash(h, coef, runPred[3])); //weak
    //sMap.set(j++, hash(h, coef, runPred[4])); //weak
    //sMap.set(j++, hash(h, coef, runPred[5])); //weak
    //sMap.set(j++, hash(h, coef, mcuPos & 63, runPred[1])); //weak
    sMap.set(j++, hash(h, coef, runPred[1], runPred[4]));

    // === extrapolation x run prediction ===
    sMap.set(j++, hash(h, advPred[3] / 17, runPred[1], runPred[5]));
    sMap.set(j++, hash(h, advPred[1] / 16, runPred[3]));
    sMap.set(j++, hash(h, advPred[0] / 16, runPred[1])); //strong
    sMap.set(j++, hash(h, advPred[2] / 17, runPred[0], runPred[4])); //strong
    sMap.set(j++, hash(h, advPred1[2] / 17, runPred[0], runPred[2])); //strong
    sMap.set(j++, hash(h, advPred[0] / 17, runPred[1], runPred[3]));
    sMap.set(j++, hash(h, advPred[2] / 16, runPred[2]));
    sMap.set(j++, hash(h, advPred2[2] / 17, runPred[0], runPred[2]));

    // === extrapolation x in-block neighbour ===
    sMap.set(j++, hash(h, advPred[1] / 12, lcp[0] / 12)); //strong
    sMap.set(j++, hash(h, advPred[0] / 12, lcp[1] / 12)); //strong
    sMap.set(j++, hash(h, advPred[2] / 12, lcp[0] / 12));
    sMap.set(j++, hash(h, advPred[2] / 12, lcp[1] / 12)); //strong
    //sMap.set(j++, hash(h, advPred[3] / 12, lcp[0] / 12)); //weak
    sMap.set(j++, hash(h, advPred[1] / 12, lcp[2] / 12));
    sMap.set(j++, hash(h, advPred[0] / 12, lcp[3] / 12));

    // === extrapolation x the same coefficient in other components ===
    sMap.set(j++, hash(h, advPred[0] / 16, prevCoef / 42)); //strong
    sMap.set(j++, hash(h, advPred[1] / 16, prevCoef / 42)); //strong
    sMap.set(j++, hash(h, advPred[2] / 16, prevCoef / 42)); //strong
    sMap.set(j++, hash(h, advPred[3] / 16, prevCoef / 42)); //strong
    sMap.set(j++, hash(h, coef, advPred1[0] / 16, prevCoef / 42));
    //sMap.set(j++, hash(h, coef, advPred1[1] / 16, prevCoef / 42)); //weak
    //sMap.set(j++, hash(h, coef, advPred1[2] / 16, prevCoef / 42)); //weak
    //sMap.set(j++, hash(h, coef, advPred2[0] / 16, prevCoef / 42)); //weak
    //sMap.set(j++, hash(h, coef, advPred2[1] / 16, prevCoef / 42)); //weak
    //sMap.set(j++, hash(h, coef, advPred3[0] / 16, prevCoef / 42)); //weak
    sMap.set(j++, hash(h, coef, advPred3[1] / 16, prevCoef / 42));  //strong
    sMap.set(j++, hash(h, coef, advPred[3] / 13, prevCoef / 11));
    //sMap.set(j++, hash(h, advPred[0] / 16, prevCoef2 / 42)); //weak
    //sMap.set(j++, hash(h, advPred[2] / 16, prevCoef2 / 42)); //weak
    sMap.set(j++, hash(h, advPred[0] / 16, advPred[2] / 16, prevCoef / 42));

    // === previously decoded values x coefficient position ===
    sMap.set(j++, hash(h, coef, rs1));
    sMap.set(j++, hash(h, coef, prevCoef));
    //sMap.set(j++, hash(h, coef, prevCoef2)); //weak
    sMap.set(j++, hash(h, coef, prevCoefRs));
    sMap.set(j++, hash(h, coef, prevCoef / 42));
    sMap.set(j++, hash(h, coef, prevCoef / 22, prevCoef2 / 50));
    sMap.set(j++, hash(h, coef, abs(prevCoef) / 16, abs(prevCoef2) / 16));
    sMap.set(j++, hash(h, prevCoefRs, prevCoef / 42));

    // === the last symbol (rs1) combined with other values ===
    sMap.set(j++, hash(h, rs1, mcuPos & 63));
    sMap.set(j++, hash(h, rs, rs1));
    sMap.set(j++, hash(h, rs1, runPred[0]));
    sMap.set(j++, hash(h, rs1, runPred[1]));
    sMap.set(j++, hash(h, rs1, runPred[2]));
    //sMap.set(j++, hash(h, rs1, runPred[3])); //weak
    sMap.set(j++, hash(h, rs1, prevCoef / 22)); //strong
    sMap.set(j++, hash(h, rs1, prevCoefRs));
    sMap.set(j++, hash(h, advPred[0] / 7, rs1));
    sMap.set(j++, hash(h, advPred[2] / 7, rs1));
    //sMap.set(j++, hash(h, rs1, coef, comp)); //weak
    //sMap.set(j++, hash(h, rs1, prevCoef2 / 22)); //weak
    sMap.set(j++, hash(h, advPred[3] / 7, rs1));
    sMap.set(j++, hash(h, advPred1[2] / 7, rs1));
    //sMap.set(j++, hash(h, rs1, runPred[4], runPred[5])); //weak

    // === extra bit sums (sSum and related) ===
    sMap.set(j++, hash(h, coef, sSum >> 2));
    //sMap.set(j++, hash(h, coef, sSum2 >> 2)); //weak
    sMap.set(j++, hash(h, coef, sSum1 >> 2));
    //sMap.set(j++, hash(h, coef, sSum3 >> 2)); //weak
    sMap.set(j++, hash(h, coef, min(63, sSum)));
    sMap.set(j++, hash(h, sSum >> 3, sSum2 >> 3));
    sMap.set(j++, hash(h, sSum2 >> 2, prevCoef2 / 42));
    sMap.set(j++, hash(h, sSum >> 1, prevCoef2 / 10));
    sMap.set(j++, hash(h, sSum3 >> 2, prevCoef / 42));
    sMap.set(j++, hash(h, sSum1 >> 2, prevCoef / 42));
    sMap.set(j++, hash(h, sSum >> 2, rs1));

    // === neighbour blocks (north and west, same component) ===
    sMap.set(j++, hash(h, coefficientBuffer[cPos - blockN[mcuPos >> 6]], coefficientBuffer[cPos - blockW[mcuPos >> 6]]));
    sMap.set(j++, hash(h, coefficientBuffer[cPos - blockN[mcuPos >> 6]] >> 4, advPred[3] / 17));
    sMap.set(j++, hash(h, coefficientBuffer[cPos - blockW[mcuPos >> 6]] >> 4, advPred[3] / 17));
    sMap.set(j++, hash(h, coefficientBuffer[cPos - blockN[mcuPos >> 6]] >> 4, advPred[2] / 17));
    sMap.set(j++, hash(h, coefficientBuffer[cPos - blockW[mcuPos >> 6]] >> 4, advPred[0] / 17));

    // === frequency and position ===
    sMap.set(j++, hash(h, coef));
    //sMap.set(j++, hash(h, coef, zu, zv));  //weak
    sMap.set(j++, hash(h, coef, column >> 3));
    //sMap.set(j++, hash(h, mcuPos & 63, column >> 1)); //weak

    // === masks of the missing predictions ===
    sMap.set(j++, hash(h, ama, coef));
    //sMap.set(j++, hash(h, lma, coef)); //weak
    //sMap.set(j++, hash(h, lma, ama, coef)); //weak
    sMap.set(j++, hash(h, lma, advPred[1] / 12, lcp[0] / 12));

    assert(j == nSM);

    MJPEGMap.set(hash(mcuPos, column, row, hc >> 2));
  }

  for (int j = 0; j < nSM; ++j) {
    int st = sMap.mix(*m1, j);
    m.add(st);
  }
  MJPEGMap.mix(*m1);

  //////////////////////////////////////////////////////////////////////////////
  // Where we are inside the current symbol. A Huffman code bit and a magnitude
  // bit need different weights, so both mixers use this as a context.
  //////////////////////////////////////////////////////////////////////////////

  const uint32_t e = huffBits - huffSize;
  const int posInSym =
    phase == 0 ? min(15, static_cast<int>(huffBits)) : // 0..15: depth in the Huffman code tree
    phase == 1 ? 16 :                                  // the sign bit
    17 + min(6, static_cast<int>(e) - 1);              // 17..23: which magnitude bit

  //////////////////////////////////////////////////////////////////////////////
  // Inner mixer contexts
  //////////////////////////////////////////////////////////////////////////////

  // Horizontal position in MCUs. Images wider than 1024 MCUs: divided by
  // width / 1024 (rounded down), then clamped to 1023.
  int colCtx = (width > 1024) ? (min(1023, column / max(1, width / 1024))) : column;
  m1->set(colCtx, 1024);
  m1->set(static_cast<uint32_t>(firstCol), 2); // no block to the west
  m1->set(coef | (min(3, huffBits) << 8), 1024);
  m1->set(((hc & 0x1FE) << 1) | min(3, ilog2(zu + zv)), 1024);
  m1->set(finalize64(hash(hc), 13), 8192); // the Huffman code so far
  m1->set(ama, 8); // which extrapolations are zero; strong for MJPEG files
  m1->set(posInSym, 24);

  // A direct prediction from the symbol state and the coefficient alone.
  m.add(stretch(smx.p1(finalize64(hash(symKey, coef), 16))));

  //////////////////////////////////////////////////////////////////////////////
  // The inner mixer's prediction, refined by the APMs
  //
  // apm1 adjusts the inner mixer's prediction for the current symbol and
  // coefficient. The APM chains below then adjust it further, one feature per
  // APM. The outer mixer does not get the APM outputs as probabilities, but as
  // the change each APM made to its input (in the stretched domain). The
  // changes are the new information, and they are much less correlated with
  // each other than the probabilities.
  //////////////////////////////////////////////////////////////////////////////

  // Adds to the outer mixer how much pr differs from sParent (both stretched),
  // and returns stretch(pr), so it can be the parent of the next APM.
  const auto refine = [&m](const int pr, const int sParent) {
    const int s = stretch(pr);
    m.add(std::clamp((s - sParent) * 2, -2047, 2047));
    return s;
    };

  // APM key parts. alAdv[] and alLcp[] describe the coefficient the current
  // bit belongs to, also after a run of zeros.
  // The sign of a prediction is only used on the sign bit (the first extra
  // bit): the Huffman symbol only depends on the size, and on the later extra
  // bits the actual sign is already in symKey.
  const bool signBit = rs >= 0 && e == 0;
  const auto bucket = [signBit](const int v) {
    const int b = abs(v) / 12;
    return signBit ? b * 3 + 1 + static_cast<int>(v > 0) - static_cast<int>(v < 0) : b;
    };
  const int kA0 = bucket(alAdv[0]);
  const int kA1 = bucket(alAdv[1]);
  const int kA2 = bucket(alAdv[2]);
  const int kA3 = bucket(alAdv[3]);
  const int kL0 = bucket(alLcp[0]);
  const int kL1 = bucket(alLcp[1]);
  const int kL2 = bucket(alLcp[2]);
  const int kL3 = bucket(alLcp[3]);
  const int kL4 = bucket(alLcp[4]);
  // The keys based on neighbour blocks also include nbAvail, so that the
  // replaced or zeroed predictions at the image border get their own entries.
  // nbAvail is 3 inside the image, so there it only changes the hash, and does
  // not split the entries. lcp[] only reads inside the block, so its keys do
  // not need it.

  const int pr0 = m1->p();
  const int s0 = stretch(pr0);
  m.add(s0 >> 1);
  m.add((pr0 - 2048) >> 3);

  const int pr1 = apm1.p(pr0, finalize64(hash(symKey, coef), 17));
  const int s1 = refine(pr1, s0);

  // chain: extrapolations from both neighbours, then from the west
  const int pr2 = apm2.p(pr1, finalize64(hash(symKey, kA1, nbAvail), 14));
  const int s2 = refine(pr2, s1);
  const int pr3 = apm3.p(pr2, finalize64(hash(symKey, kA0, nbAvail), 14));
  refine(pr3, s2);

  // chain: extrapolation from the north, then from both (advPred[3] style)
  const int pr4 = apm4.p(pr1, finalize64(hash(symKey, kA2, nbAvail), 14));
  const int s4 = refine(pr4, s1);
  const int pr5 = apm5.p(pr4, finalize64(hash(symKey, kA3, nbAvail), 14));
  refine(pr5, s4);

  // chain: the in-block neighbours
  const int pr6 = apm6.p(pr1, finalize64(hash(symKey, kL0), 14));
  const int s6 = refine(pr6, s1);
  const int pr7 = apm7.p(pr6, finalize64(hash(symKey, kL1), 14));
  const int s7 = refine(pr7, s6);
  const int pr8 = apm8.p(pr7, finalize64(hash(symKey, kL2), 14));
  const int s8 = refine(pr8, s7);
  const int pr9 = apm9.p(pr8, finalize64(hash(symKey, kL3), 14));
  const int s9 = refine(pr9, s8);
  const int pr10 = apm10.p(pr9, finalize64(hash(symKey, kL4), 14));
  refine(pr10, s9);

  // Pairs of APMs on pr0, each on two features. The second APM also refines
  // pr0, but the mixer gets its difference from the first APM's output instead
  // of from pr0, so the two inputs overlap less.
  const int pr11 = apm11.p(pr0, finalize64(hash(symKey, kL0, kL1), 15));
  const int s11 = refine(pr11, s0);
  const int pr12 = apm12.p(pr0, finalize64(hash(symKey, kA0, kA1, nbAvail), 15));
  refine(pr12, s11);

  const int pr13 = apm13.p(pr0, finalize64(hash(symKey, kA1, kA2, nbAvail), 15));
  const int s13 = refine(pr13, s0);
  const int pr14 = apm14.p(pr0, finalize64(hash(symKey, kA2, kA3, nbAvail), 15));
  refine(pr14, s13);

  //////////////////////////////////////////////////////////////////////////////
  // Outer mixer contexts
  //
  // Each context set selects its own weights, so the mixer can weight the
  // inputs above differently depending on what is being decoded and how well
  // the model knows this part of the image.
  //////////////////////////////////////////////////////////////////////////////

  // For tuning:
  //int skip = (int)shared->tuning_param;
  //int k = 0;
  //k++; if (k == skip); else m.set(...

  // What is being decoded: the symbol, and the coefficient or frequency. The
  // most specific sets, so they only pay off on large files.
  m.set(finalize64(hash(symKey, coef), 13), 8192); // strong only for large files
  m.set(finalize64(hash(symKey, min(3, (zu + zv) / 3)), 13), 8192); // strong for large files
  //m.set(static_cast<uint32_t>(zu + zv < 5) | (static_cast<uint32_t>(huffBits > 8) << 1) | (static_cast<uint32_t>(firstCol) << 2), 8); //2*2*2 //weak
  //m.set((min(3, huffBits / 2) << 8) | coef, 1024); //256*4 //weak

  // How large the neighbour blocks say this coefficient should be, and how
  // large its already decoded neighbours in the block are.
  m.set((predCat << 6) | (min(15, abs(alAdv[0]) / 16) << 2) | min(3, huffBits), 1024); // 16*16*4
  m.set((min(15, abs(lcp[2]) / 16) << 8) | (min(15, abs(lcp[1]) / 16) << 4) | min(15, abs(lcp[0]) / 16), 4096); // 16*16*16 //strong for MJPEG

  // Where the next larger coefficient is expected.
  m.set((min(7, runPred[1]) << 6) | (min(7, runPred[0]) << 3) | min(7, runPred[2]), 512); //8*8*8

  // How many of the in-block neighbours do not exist at this frequency.
  m.set(bitCount(lma) * 3 + static_cast<int>(phase), 24); //0..7, 0..2 //mostly for small files

  // A Huffman code bit needs different weights than a magnitude bit.
  m.set(posInSym * 2 + static_cast<int>(comp == 0), 48); //strong for smaller files

  // Does the predicted size reach the size category of the decoded symbol?
  // The clearest available sign of whether the model is right about this
  // coefficient.
  const int sCategory = rs < 0 ? 0 : (rs & 15);
  const int catCtx = rs < 0
    ? min(15, static_cast<int>(predCat))
    : 16 + (min(3, max(-3, static_cast<int>(predCat) - sCategory)) + 3) * 4 + min(3, static_cast<int>(e));
  m.set(catCtx, 64); //medium, stronger for small files

  // Which coefficient of the block the decoder is at.
  m.set(static_cast<int>(mcuPos & 63) * 3 + static_cast<int>(phase), 192); //medium

  // How many contexts have never been seen. Early in an image most contexts
  // are still empty, and the mixer should rely on the maps and the APMs more.
  m.set(min(15, coldCount) * 3 + static_cast<int>(phase), 48); //strong, especially for smaller files

  // Which coefficient this bit belongs to. Differs from mcuPos during the extra
  // bits after a run of zeros.
  m.set(alZz * 3 + static_cast<int>(phase), 192); //strong
  m.set(min(7, zzu[alZz] + zzv[alZz]) * 24 + posInSym, 192);
  m.set(alZz * 4 + min(3, static_cast<int>(comp)), 256); //weak
}
