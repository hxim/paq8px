#include "Image1BitModel.hpp"
#include "../Stretch.hpp"
#include "../BitCount.hpp"
#include "../Hash.hpp"
#include <algorithm>

//////////////////////////////////////////////////////////////////////////////////////////////////
// Small helpers
//////////////////////////////////////////////////////////////////////////////////////////////////

static inline uint32_t bitCount64(uint64_t v) {
  return (uint32_t)(bitCount((uint32_t)v) + bitCount((uint32_t)(v >> 32)));
}

/**
 * A run length in 16 steps: exact for short runs, coarser for long ones. 0 = start of a row.
 */
static uint32_t runBucket(uint32_t runLength) {
  return runLength < 8 ? runLength :
    runLength < 10 ? 8 :
    runLength < 12 ? 9 :
    runLength < 16 ? 10 :
    runLength < 24 ? 11 :
    runLength < 32 ? 12 :
    runLength < 48 ? 13 :
    runLength < 64 ? 14 : 15;
}

/**
 * A match length in 8 steps. 0 = no match.
 */
static uint32_t matchBucket(uint32_t matchLength) {
  return matchLength == 0 ? 0 :
    matchLength < 4 ? 1 :
    matchLength < 8 ? 2 :
    matchLength < 16 ? 3 :
    matchLength < 32 ? 4 :
    matchLength < 64 ? 5 :
    matchLength < 256 ? 6 : 7;
}

/**
 * The length of an active match (1 or more) in 16 steps: exact up to 8, coarser above. The first
 * few pixels of a match are where its reliability changes the most.
 */
static uint32_t matchBucketFine(uint32_t matchLength) {
  return matchLength <= 8 ? matchLength - 1 :
    matchLength < 12 ? 8 :
    matchLength < 16 ? 9 :
    matchLength < 24 ? 10 :
    matchLength < 32 ? 11 :
    matchLength < 64 ? 12 :
    matchLength < 128 ? 13 :
    matchLength < 512 ? 14 : 15;
}

/**
 * How many pixels a match source differed in when it was taken, in 4 steps: none, 1-2, 3-6, more.
 */
static uint32_t residBucket(uint32_t r) {
  return r == 0 ? 0 : r <= 2 ? 1 : r <= 6 ? 2 : 3;
}

/**
 * The signed distance from the current pixel to a colour change on the row above, in 32 steps:
 * one step per pixel up to 8 either way, coarser beyond. 31 = no colour change left on that row.
 */
static uint32_t transBucket(int d) {
  if (d == INT32_MAX) return 31;
  if (d >= -8 && d <= 8) return static_cast<uint32_t>(d + 12); //4..20
  if (d < 0) {
    return d >= -16 ? 3 : d >= -32 ? 2 : d >= -64 ? 1 : 0;
  }
  return d <= 16 ? 21 : d <= 32 ? 22 : d <= 64 ? 23 : 24;
}

/**
 * How many pixels of a line of pixels (bit 0 = nearest) have the same colour as the nearest one.
 * The sentinel bit, one above the last pixel, stands for "all of them".
 */
static uint32_t runUp(const uint32_t col, const uint32_t sentinel) {
  const uint32_t d = ((col & 1) != 0 ? ~col : col) | sentinel;
  return static_cast<uint32_t>(bitCount((d & (0u - d)) - 1));
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// Sparse templates
//
// These three were not designed by hand but found by a search: pixels were added one at a time,
// each time the one that improved the model's predictions the most. The result is not a block but
// a scatter: the nearest pixels at about a stroke's width apart, and single pixels spread over the
// height of a letter, up to 14 rows up. Read together they give a rough picture of the letter
// around the current pixel. {dy, dx} as in at().
//////////////////////////////////////////////////////////////////////////////////////////////////

//the 15 pixels that predict the best on their own
//  6  ........x....
//  5  .............
//  4  .............
//  3  .......xxx...
//  2  ......x.x...x
//  1  ...x...xxxx..
//  0  x...x..x?
static const int8_t T15[15][2] = { {1,0},{0,-1},{1,1},{2,-2},{0,-4},{2,4},{1,2},{0,-8},{1,-5},{2,0},{3,1},{3,0},{6,0},{3,-1},{1,-1} };

//grown from the first 12 pixels of T15, by what they add to the rest of the model
// 11  x...............
// 10  ...x.....x....x.
//  9  ................
//  8  ............x...
//  7  ................
//  6  ......x.x....x.x
//  5  ................
//  4  ................
//  3  ........xx......
//  2  ......x.x...x.x.
//  1  ...x....xxx.....
//  0  x...x..x?
static const int8_t G22[22][2] = { {1,0},{0,-1},{1,1},{2,-2},{0,-4},{2,4},{1,2},{0,-8},{1,-5},{2,0},{3,1},{3,0},
  {11,-8},{6,0},{6,5},{10,1},{10,-5},{2,6},{6,-2},{6,7},{10,6},{8,4} };

//grown from the first 8 pixels of T15, by what they add to the rest of the model including G22
// 14  ..........x.....
// 13  ................
// 12  .....x.......x..
// 11  ................
// 10  .x....x.x.......
//  9  ................
//  8  ................
//  7  ................
//  6  ..x.....x.x...x.
//  5  ................
//  4  ......x.x...x...
//  3  ................
//  2  ......x..x..x..x
//  1  ........xxx.....
//  0  x...x..x?
static const int8_t H23[23][2] = { {1,0},{0,-1},{1,1},{2,-2},{0,-4},{2,4},{1,2},{0,-8},
  {10,-7},{2,1},{12,-3},{6,6},{10,0},{6,-6},{12,5},{2,7},{4,4},{4,0},{10,-2},{4,-2},{6,0},{6,2},{14,2} };

//////////////////////////////////////////////////////////////////////////////////////////////////
// Construction
//////////////////////////////////////////////////////////////////////////////////////////////////

Image1BitModel::Image1BitModel(Shared* const sh) :
  shared(sh),
  mapL(sh, nLSM, 20, 128), //contexts, hash bits, input scale
  stateMap(sh, N, 256, 1023, StateMapType::BitHistory),
  matchStateMap(sh, nMatch, 1 << 7, 1023, StateMapType::Generic) {
}

void Image1BitModel::setParam(int widthInBytes) {
  if (w != static_cast<uint32_t>(widthInBytes)) { //the width changes: a new image begins
    w = widthInBytes;
    xByte = yRow = 0;
    runLength = runValue = 0;
    hPeriod = vPeriod = qualH = qualV = 0;
    for (uint32_t d = 0; d <= maxHPeriod; ++d) hAcc[d] = 0;
    for (uint32_t k = 0; k <= maxVPeriod; ++k) vAcc[k] = 0;
    for (int k = 0; k < nMatch; ++k) { matchLen[k] = 0; predictedBit[k] = -1; }
    refValid = false;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// Reading pixels
//////////////////////////////////////////////////////////////////////////////////////////////////

uint32_t Image1BitModel::at(const int dy, const int dx) const {
  const uint64_t row =
    dy == 0 ? r0 : dy == 1 ? r1 : dy == 2 ? r2 : dy == 3 ? r3 : dy == 4 ? r4 : dy == 5 ? r5 : dy == 6 ? r6 :
    dy == 7 ? r7 : dy == 8 ? r8 : dy == 9 ? r9 : dy == 10 ? r10 : dy == 11 ? r11 : dy == 12 ? r12 :
    dy == 13 ? r13 : dy == 14 ? r14 : dy == 15 ? r15 : r16;
  return static_cast<uint32_t>(dy == 0 ? row >> (-dx - 1) : row >> (8 - dx)) & 1;
}

uint32_t Image1BitModel::sparse(const int8_t(*tpl)[2], const int n) const {
  uint32_t v = 0;
  for (int k = 0; k < n; ++k) v = v << 1 | at(tpl[k][0], tpl[k][1]);
  return v;
}

template<typename B>
int Image1BitModel::px(const B& buf, int bpos, int dy, int dx) const {
  const int off = bpos + dx + 64; //offset by 64 so that >> and & work for dx down to -64
  const int dist = (int)(dy * w) - (off >> 3) + 8;
  if (dist < 1) return 0;
  return (buf(dist) >> (7 - (off & 7))) & 1;
}

//Three bytes are read: the one pixel q is in, and the two after it.
template<typename B>
uint32_t Image1BitModel::pixelsAt(const B& buf, const int64_t q, const int n, const uint32_t curByteIdx) const {
  const int64_t b0 = q >= 0 ? q >> 3 : -((-q + 7) >> 3); //floor(q / 8)
  uint32_t v = 0;
  for (int j = 0; j < 3; ++j) {
    const int64_t b = b0 + j;
    uint32_t byte = 0;
    if (b >= 0 && b < static_cast<int64_t>(curByteIdx)) byte = buf(static_cast<uint32_t>(static_cast<int64_t>(curByteIdx) - b));
    v = v << 8 | byte;
  }
  const int shift = 24 - static_cast<int>(q - b0 * 8) - n;
  return (v >> shift) & ((1u << n) - 1);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// At the start of each row: analyse the row just finished
//////////////////////////////////////////////////////////////////////////////////////////////////

//Index the row just finished by where it changes colour, so that the next colour change after any
//position is a single lookup (see trackRuns, which uses it).
//
//A row is a sequence of white and black runs, and what matters is where a new run starts. For
//every position x:
//
//    nc1[x] = where the row next turns black, at or after x
//    nc0[x] = where the row next turns white, at or after x
//
//    x:        0  1  2  3  4  5  6  7  8  9
//    row:      0  0  1  1  1  0  0  1  0  0
//    nc1[x]:   2  2  2  7  7  7  7  7  -  -
//    nc0[x]:   5  5  5  5  5  5  8  8  8  -        ( - = none left, stored as the row length )
//
//Both are built in one pass from right to left. The row is taken to start on white, so if its first
//pixel is black, that is a colour change too.
template<typename B>
void Image1BitModel::buildRefLine(const B& buf) {
  if (w == 0 || w > maxRowBytes) { refValid = false; return; }
  for (uint32_t j = 0; j < w; ++j) refBytes[j] = buf(w - j);
  const uint32_t pixels = w * 8;
  nc0[pixels] = nc1[pixels] = pixels; //no colour change from here on
  uint32_t next0 = pixels, next1 = pixels;
  uint32_t cur = refBytes[(pixels - 1) >> 3] & 1; //the last pixel of the row
  for (uint32_t x = pixels; x-- > 0;) {
    const uint32_t left = x > 0 ? ((refBytes[(x - 1) >> 3] >> (7 - ((x - 1) & 7))) & 1) : 0;
    if (cur != left) {
      if (cur != 0) next1 = x; else next0 = x;
    }
    nc0[x] = next0;
    nc1[x] = next1;
    cur = left;
  }
  refValid = true;
}

//Find the distances at which the image repeats itself, horizontally (along the row) and vertically
//(from row to row). These give the adaptive template pixels (see readAdaptiveTemplate): a dither
//or halftone screen repeats every few pixels, and lines of text repeat every few rows.
//
//The row just finished is compared with itself shifted by d pixels, and with the rows v rows above
//it. The score of a comparison counts only the ink, not the paper:
//
//     score = 1024 * (pixels set in both) / (pixels set in either)     (1024 = identical)
//
//Simply counting the pixels that agree would not work: a mostly white page agrees with itself at
//any distance. And the distance chosen is not the one with the best score - nearby pixels are
//always similar, so that would always be the smallest distance - but the one whose score stands out
//the most from its neighbours' (its prominence).
template<typename B>
void Image1BitModel::updateRepeatPeriods(const B& buf) {
  if (w < minRowBytes || w > maxRowBytes) { hPeriod = vPeriod = qualH = qualV = 0; return; }

  //unpack the row just finished: buf(w) is its first byte, buf(1) its last
  for (uint32_t j = 0; j < w; ++j) rowBytes[j] = buf(w - j);
  const uint32_t nWords = (w + 7) / 8;
  for (uint32_t k = 0; k < nWords; ++k) {
    uint64_t v = 0;
    for (uint32_t b = 0; b < 8; ++b) {
      const uint32_t j = k * 8 + b;
      v = v << 8 | (j < w ? rowBytes[j] : 0);
    }
    rowWords[k] = v; //bit 63 of word 0 is the leftmost pixel of the row
  }

  const auto prominence = [](const uint32_t* acc, const uint32_t k) {
    return 2 * static_cast<int>(acc[k]) - static_cast<int>(acc[k - 1]) - static_cast<int>(acc[k + 1]);
    };

  //horizontal: the row against itself shifted by d pixels. The first word is skipped, as the bits
  //shifted into it would come from outside the row.
  uint32_t scoreAtH[maxHPeriod + 2] = {};
  for (uint32_t d = 2; d <= maxHPeriod; ++d) {
    uint32_t inter = 0, uni = 0;
    for (uint32_t k = 1; k < nWords; ++k) {
      const uint64_t shifted = (rowWords[k] >> d) | (rowWords[k - 1] << (64 - d));
      inter += bitCount64(rowWords[k] & shifted);
      uni += bitCount64(rowWords[k] | shifted);
    }
    scoreAtH[d] = uni == 0 ? 0 : (inter << 10) / uni;
    hAcc[d] = hAcc[d] - (hAcc[d] >> 3) + scoreAtH[d]; //smoothed over about 8 rows, so the choice is stable
  }
  uint32_t bestD = 0;
  int bestPromH = minProminence;
  for (uint32_t d = 3; d < maxHPeriod; ++d) {
    const int prom = prominence(&hAcc[0], d);
    if (prom > bestPromH) { bestPromH = prom; bestD = d; }
  }
  hPeriod = bestD;
  //how well it repeats, from this row alone, so it follows changes in the image at once
  const uint32_t scoreH = bestD == 0 ? 0 : scoreAtH[bestD];
  qualH = bestD == 0 ? 0 : scoreH < 400 ? 0 : scoreH < 600 ? 1 : scoreH < 800 ? 2 : 3;

  //vertical: the row against the rows minVPeriod..maxVPeriod above it
  uint32_t scoreAtV[maxVPeriod + 2] = {};
  uint32_t lastK = 0;
  for (uint32_t k = minVPeriod; k <= maxVPeriod; ++k) {
    if (yRow < k + 1) break;
    uint32_t inter = 0, uni = 0;
    const uint32_t base = (k + 1) * w;
    for (uint32_t j = 0; j < w; ++j) {
      const uint32_t a = rowBytes[j], b = buf(base - j);
      inter += bitCount(a & b);
      uni += bitCount(a | b);
    }
    scoreAtV[k] = uni == 0 ? 0 : (inter << 10) / uni;
    vAcc[k] = vAcc[k] - (vAcc[k] >> 3) + scoreAtV[k];
    lastK = k;
  }
  uint32_t bestV = 0;
  int bestPromV = minProminence;
  for (uint32_t k = minVPeriod + 1; k + 1 <= lastK; ++k) {
    const int prom = prominence(&vAcc[0], k);
    if (prom > bestPromV) { bestPromV = prom; bestV = k; }
  }
  vPeriod = bestV;
  const uint32_t scoreV = bestV == 0 ? 0 : scoreAtV[bestV];
  qualV = bestV == 0 ? 0 : scoreV < 400 ? 0 : scoreV < 600 ? 1 : scoreV < 800 ? 2 : 3;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// Match model helpers
//////////////////////////////////////////////////////////////////////////////////////////////////

//The 40 pixels compared: the 8 to the left on the current row, and 16 on each of the two rows
//above (7 to the left of the column and 8 to the right). The candidate's pixels are read from the
//buffer in the same layout as r0..r2, so they can be compared bit by bit.
//  2  .xxxxxxxxxxxxxxxx
//  1  .xxxxxxxxxxxxxxxx
//  0  xxxxxxxx?
template<typename B>
uint32_t Image1BitModel::residual(const B& buf, const uint32_t byteIdx, const uint32_t bp, const uint32_t curByteIdx) const {
  const int64_t s = static_cast<int64_t>(byteIdx) * 8 + bp;
  const int64_t rowPixels = static_cast<int64_t>(w) * 8;
  const uint32_t s0 = pixelsAt(buf, s - 8, 8, curByteIdx);
  const uint32_t s1 = pixelsAt(buf, s - rowPixels - 7, 16, curByteIdx);
  const uint32_t s2 = pixelsAt(buf, s - 2 * rowPixels - 7, 16, curByteIdx);
  return static_cast<uint32_t>(
    bitCount((s0 ^ static_cast<uint32_t>(r0)) & 0xff) +
    bitCount((s1 ^ static_cast<uint32_t>(r1)) & 0xffff) +
    bitCount((s2 ^ static_cast<uint32_t>(r2)) & 0xffff));
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// update(): learn from the pixel just coded
//////////////////////////////////////////////////////////////////////////////////////////////////

void Image1BitModel::update() {
  INJECT_SHARED_y

  //the indexed contexts: their bit histories and their counts
  for (int i = 0; i < N; ++i) {
    StateTable::update(&t[cxt[i]], y, rnd);

    uint32_t c = counts[cxt[i]];
    int n0 = (c >> 8) & 255;
    int n1 = c & 255;

    if (y == 0) n0++; else n1++;
    if (n0 + n1 >= 64) { //halve both, so the counts follow recent pixels
      n0 >>= 1;
      n1 >>= 1;
    }

    counts[cxt[i]] = n0 << 8 | n1;
  }

  //the match models: if the prediction was right, the match moves on to the next pixel, otherwise
  //it ends (the next pixel looks the neighbourhood up again)
  for (int k = 0; k < nMatch; ++k) {
    if (matchLen[k] > 0) {
      if (predictedBit[k] == y) {
        if (matchLen[k] < 65535) matchLen[k]++;
        if (++matchBpos[k] == 8) {
          matchBpos[k] = 0;
          matchByteIdx[k]++;
          //the source has reached the end of its row: what follows it is another row
          if (++matchSrcX[k] >= w) matchLen[k] = 0;
        }
      }
      else matchLen[k] = 0;
    }
    //remember where this key was seen: here
    MatchBucket& bucket = matchTable[(static_cast<size_t>(k) << bucketBits) + matchHashIdx[k]];
    for (int s = matchSlots - 1; s > 0; --s) {
      bucket.pos[s] = bucket.pos[s - 1];
      bucket.check[s] = bucket.check[s - 1];
    }
    bucket.pos[0] = lastBpos << 29 | (lastByteIdx & 0x1fffffff);
    bucket.check[0] = matchCheck[k];
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
// mix(): predict the next pixel
//////////////////////////////////////////////////////////////////////////////////////////////////

void Image1BitModel::mix(Mixer& m) {
  update();

  INJECT_SHARED_y
  INJECT_SHARED_buf
  INJECT_SHARED_bpos

  shiftRows(buf, y, bpos);

  //at the start of a row, the row above is complete: analyse it
  const bool rowStart = (bpos == 0 && xByte == 0);
  if (rowStart) {
    if (yRow > 0) { updateRepeatPeriods(buf); buildRefLine(buf); }
    else refValid = false;
  }

  //read the neighbourhood
  trackRuns(y, rowStart, bpos);
  readColumns();
  readDirections();
  readAdaptiveTemplate(buf, bpos);
  runMatchModels(buf, bpos, rowStart);
  readNeighbourhood(y);

  //and turn it into predictions
  setIndexedContexts(y, bpos);
  setHashedContexts(y, bpos);
  mapL.mix(m);
  addInputs(m);
  setMixerContexts(m, y, bpos);
  publishToSSE();

  //move on to the next pixel
  if (bpos == 7 && w != 0) {
    if (++xByte >= w) { xByte = 0; yRow++; }
  }
}

//Shift the pixel just coded into r0, and the pixel above the next one into each of the rows above
//(see the row registers in the header).
template<typename B>
void Image1BitModel::shiftRows(const B& buf, const int y, const int bpos) {
  r0 += r0 + y;
  r1 += r1 + ((buf(1 * w - 1) >> (7 - bpos)) & 1);
  r2 += r2 + ((buf(2 * w - 1) >> (7 - bpos)) & 1);
  r3 += r3 + ((buf(3 * w - 1) >> (7 - bpos)) & 1);
  r4 += r4 + ((buf(4 * w - 1) >> (7 - bpos)) & 1);
  r5 += r5 + ((buf(5 * w - 1) >> (7 - bpos)) & 1);
  r6 += r6 + ((buf(6 * w - 1) >> (7 - bpos)) & 1);
  r7 += r7 + ((buf(7 * w - 1) >> (7 - bpos)) & 1);
  r8 += r8 + ((buf(8 * w - 1) >> (7 - bpos)) & 1);
  r9 += r9 + ((buf(9 * w - 1) >> (7 - bpos)) & 1);
  r10 += r10 + ((buf(10 * w - 1) >> (7 - bpos)) & 1);
  r11 += r11 + ((buf(11 * w - 1) >> (7 - bpos)) & 1);
  r12 += r12 + ((buf(12 * w - 1) >> (7 - bpos)) & 1);
  r13 += r13 + ((buf(13 * w - 1) >> (7 - bpos)) & 1);
  r14 += r14 + ((buf(14 * w - 1) >> (7 - bpos)) & 1);
  r15 += r15 + ((buf(15 * w - 1) >> (7 - bpos)) & 1);
  r16 += r16 + ((buf(16 * w - 1) >> (7 - bpos)) & 1);

  //the same rows one byte further ahead; a row narrower than three bytes has no such byte
  if (w >= 3) {
    r1b += r1b + ((buf(1 * w - 2) >> (7 - bpos)) & 1);
    r2b += r2b + ((buf(2 * w - 2) >> (7 - bpos)) & 1);
    r3b += r3b + ((buf(3 * w - 2) >> (7 - bpos)) & 1);
    r4b += r4b + ((buf(4 * w - 2) >> (7 - bpos)) & 1);
    r5b += r5b + ((buf(5 * w - 2) >> (7 - bpos)) & 1);
    r6b += r6b + ((buf(6 * w - 2) >> (7 - bpos)) & 1);
    r7b += r7b + ((buf(7 * w - 2) >> (7 - bpos)) & 1);
    r8b += r8b + ((buf(8 * w - 2) >> (7 - bpos)) & 1);
  }
}

//The run of identical pixels that ends just left of the current pixel, and where the row above (the
//reference line) changes colour next.
//
//A bilevel image is made of runs, and a run usually ends where the run above it ended: the edge of
//a letter or a line continues from one row to the next. This is the idea behind the "vertical
//mode" of CCITT fax coding (T.4/T.6), and the names follow that standard:
//
//  a0        x                        <- the current run starts at a0; x is the pixel being coded
//  |---------?
//  ....|..........|.....              <- the row above: its next two colour changes, b1 and b2
//      b1         b2
//
//b1 is the first colour change on the row above, right of a0, to the colour the current run would
//change to; b2 is the change after it. When x reaches b1 the current run is likely to end. When x
//is past b2, the run above has already ended inside the current one.
void Image1BitModel::trackRuns(const int y, const bool rowStart, const int bpos) {
  //runs do not continue from one row to the next
  const uint32_t uy = static_cast<uint32_t>(y);
  if (rowStart) { runLength = 0; runValue = uy; prevRun1 = prevRun2 = 0; }
  else if (uy == runValue) { if (runLength < 65535) runLength++; }
  else { prevRun2 = prevRun1; prevRun1 = runLength; runValue = uy; runLength = 1; }
  f.runIdx = runBucket(runLength); //0..15

  f.curCol = runLength == 0 ? 0 : runValue; //the colour of the current run; a row starts on white
  int dB1 = INT32_MAX, dB2 = INT32_MAX;
  if (refValid) {
    const uint32_t pixels = w * 8;
    const uint32_t x = xByte * 8 + static_cast<uint32_t>(bpos);
    const uint32_t from = runLength == 0 ? 0 : x - runLength + 1; //just right of a0
    const uint32_t b1 = f.curCol == 0 ? nc1[from] : nc0[from];
    if (b1 < pixels) {
      dB1 = static_cast<int>(x) - static_cast<int>(b1);
      const uint32_t b2 = f.curCol == 0 ? nc0[b1 + 1] : nc1[b1 + 1];
      if (b2 < pixels) dB2 = static_cast<int>(x) - static_cast<int>(b2);
    }
  }
  f.vb1 = transBucket(dB1); //x - b1, 0..31
  f.vb2 = transBucket(dB2); //x - b2
}

//The columns above the current pixel, and how far the ink (or the paper) runs up each of them.
//
//The edge of a vertical stroke - the stem of a letter, a table rule, the side of a frame - is where
//the vertical runs next to each other end, just as the reference line does for horizontal runs.
//Seven columns side by side give the profile of a stroke across its width.
//
// 16  ..LcR..
// 15  ..LcR..
// 14  ..LcR..
// 13  ..LcR..
// 12  ..LcR..
// 11  ..LcR..
// 10  ..LcR..
//  9  ..LcR..
//  8  32LcR23
//  7  32LcR23
//  6  32LcR23
//  5  32LcR23
//  4  32LcR23
//  3  32LcR23
//  2  32LcR23
//  1  32LcR23
//  0  ...?
//  c: col8 (rows 1..8) and col16 (rows 9..16), bit 0 nearest
//  L, R: col8L and col8R, the columns either side (rows 9..16 are used for their runs only)
//  2, 3: the columns two and three away, used for their runs only
void Image1BitModel::readColumns() {
  const auto column8 = [&](const int shift) -> uint32_t {
    return static_cast<uint32_t>((r1 >> shift & 1) | (r2 >> shift & 1) << 1 | (r3 >> shift & 1) << 2 | (r4 >> shift & 1) << 3 |
      (r5 >> shift & 1) << 4 | (r6 >> shift & 1) << 5 | (r7 >> shift & 1) << 6 | (r8 >> shift & 1) << 7);
  };
  const auto column16 = [&](const int shift) -> uint32_t {
    return static_cast<uint32_t>((r9 >> shift & 1) | (r10 >> shift & 1) << 1 | (r11 >> shift & 1) << 2 | (r12 >> shift & 1) << 3 |
      (r13 >> shift & 1) << 4 | (r14 >> shift & 1) << 5 | (r15 >> shift & 1) << 6 | (r16 >> shift & 1) << 7);
  };
  f.col8 = column8(8);
  f.col16 = column16(8);
  f.vrun = static_cast<int>(runUp(f.col8, 0x100u)); //1..8
  f.colCount = bitCount(f.col8); //0..8: how much of the column is ink

  f.col8L = column8(9);
  f.col8R = column8(7);
  f.vrunL = runUp(f.col8L, 0x100u);                        //1..8
  f.vrunR = runUp(f.col8R, 0x100u);                        //1..8
  f.vrun16 = runUp(f.col8 | f.col16 << 8, 0x10000u);       //1..16
  f.vrun16L = runUp(f.col8L | column16(9) << 8, 0x10000u); //1..16
  f.vrun16R = runUp(f.col8R | column16(7) << 8, 0x10000u); //1..16
  f.vrunLL = runUp(column8(10), 0x100u);   //1..8
  f.vrunRR = runUp(column8(6), 0x100u);    //1..8
  f.vrunLLL = runUp(column8(11), 0x100u);  //1..8
  f.vrunRRR = runUp(column8(5), 0x100u);   //1..8
}

//The pixels above, read along slanted lines: the two diagonals and the slopes between them and the
//column. Each line is kept as 8 bits (bit 0 nearest), and as how far the colour of its nearest
//pixel runs along it.
//
//A rectangular template sees a slanted stroke - the leg of a 'v', a sloping line in a drawing, an
//italic letter - as a staircase, and has to learn every step of it separately. Read along its own
//direction, the stroke is just a run.
//
//The diagonals diagNW and diagNE (the digits are the order of the pixels, 1 = nearest):
//  8  8...............8
//  7  .7.............7.
//  6  ..6...........6..
//  5  ...5.........5...
//  4  ....4.......4....
//  3  .....3.....3.....
//  2  ......2...2......
//  1  .......1.1.......
//  0  ........?
//diagNW2 and diagNE2 are the same lines one column further out. diagNW16 continues diagNW from
//row 9 to row 16.
//
//The steep slopes steepNW and steepNE (one column per two rows):
//  8  8.......8
//  7  7.......7
//  6  .6.....6.
//  5  .5.....5.
//  4  ..4...4..
//  3  ..3...3..
//  2  ...2.2...
//  1  ...1.1...
//  0  ....?
//steepNW16 continues steepNW from row 9 to row 16.
//
//The flat slopes flatNW and flatNE (two columns per row):
//  8  8...............................8
//  7  ..7...........................7..
//  6  ....6.......................6....
//  5  ......5...................5......
//  4  ........4...............4........
//  3  ..........3...........3..........
//  2  ............2.......2............
//  1  ..............1...1..............
//  0  ................?
//flatNW16 continues flatNW from row 9 to row 16.
//
//The very steep slopes vsteepNW and vsteepNE (one column per three rows):
//  8  8.....8
//  7  7.....7
//  6  .6...6.
//  5  .5...5.
//  4  .4...4.
//  3  ..3.3..
//  2  ..2.2..
//  1  ..1.1..
//  0  ...?
//
//And the very flat slope vflatNW (three columns per row):
//  8  8........................
//  7  ...7.....................
//  6  ......6..................
//  5  .........5...............
//  4  ............4............
//  3  ...............3.........
//  2  ..................2......
//  1  .....................1...
//  0  ........................?
void Image1BitModel::readDirections() {
  f.diagNW = (r1 >> 9 & 1) | (r2 >> 10 & 1) << 1 | (r3 >> 11 & 1) << 2 | (r4 >> 12 & 1) << 3 |
    (r5 >> 13 & 1) << 4 | (r6 >> 14 & 1) << 5 | (r7 >> 15 & 1) << 6 | (r8 >> 16 & 1) << 7;
  f.diagNE = (r1 >> 7 & 1) | (r2 >> 6 & 1) << 1 | (r3 >> 5 & 1) << 2 | (r4 >> 4 & 1) << 3 |
    (r5 >> 3 & 1) << 4 | (r6 >> 2 & 1) << 5 | (r7 >> 1 & 1) << 6 | (r8 & 1) << 7;
  f.drunNW = runUp(f.diagNW, 0x100u); //1..8
  f.drunNE = runUp(f.diagNE, 0x100u); //1..8
  f.diagNW2 = (r1 >> 10 & 1) | (r2 >> 11 & 1) << 1 | (r3 >> 12 & 1) << 2 | (r4 >> 13 & 1) << 3 |
    (r5 >> 14 & 1) << 4 | (r6 >> 15 & 1) << 5 | (r7 >> 16 & 1) << 6 | (r8 >> 17 & 1) << 7;
  f.diagNE2 = (r1 >> 6 & 1) | (r2 >> 5 & 1) << 1 | (r3 >> 4 & 1) << 2 | (r4 >> 3 & 1) << 3 |
    (r5 >> 2 & 1) << 4 | (r6 >> 1 & 1) << 5 | (r7 & 1) << 6 | (r8b >> 7 & 1) << 7;
  f.diagNW16 = (r9 >> 17 & 1) | (r10 >> 18 & 1) << 1 | (r11 >> 19 & 1) << 2 | (r12 >> 20 & 1) << 3 |
    (r13 >> 21 & 1) << 4 | (r14 >> 22 & 1) << 5 | (r15 >> 23 & 1) << 6 | (r16 >> 24 & 1) << 7;
  f.drunNW16 = runUp(f.diagNW | f.diagNW16 << 8, 0x10000u); //1..16

  f.steepNW = (r1 >> 9 & 1) | (r2 >> 9 & 1) << 1 | (r3 >> 10 & 1) << 2 | (r4 >> 10 & 1) << 3 |
    (r5 >> 11 & 1) << 4 | (r6 >> 11 & 1) << 5 | (r7 >> 12 & 1) << 6 | (r8 >> 12 & 1) << 7;
  f.steepNE = (r1 >> 7 & 1) | (r2 >> 7 & 1) << 1 | (r3 >> 6 & 1) << 2 | (r4 >> 6 & 1) << 3 |
    (r5 >> 5 & 1) << 4 | (r6 >> 5 & 1) << 5 | (r7 >> 4 & 1) << 6 | (r8 >> 4 & 1) << 7;
  f.flatNW = (r1 >> 10 & 1) | (r2 >> 12 & 1) << 1 | (r3 >> 14 & 1) << 2 | (r4 >> 16 & 1) << 3 |
    (r5 >> 18 & 1) << 4 | (r6 >> 20 & 1) << 5 | (r7 >> 22 & 1) << 6 | (r8 >> 24 & 1) << 7;
  f.flatNE = (r1 >> 6 & 1) | (r2 >> 4 & 1) << 1 | (r3 >> 2 & 1) << 2 | (r4 & 1) << 3 |
    (r5b >> 6 & 1) << 4 | (r6b >> 4 & 1) << 5 | (r7b >> 2 & 1) << 6 | (r8b & 1) << 7;
  f.srunNW = runUp(f.steepNW, 0x100u);
  f.srunNE = runUp(f.steepNE, 0x100u);
  f.frunNW = runUp(f.flatNW, 0x100u);

  f.vsteepNW = (r1 >> 9 & 1) | (r2 >> 9 & 1) << 1 | (r3 >> 9 & 1) << 2 | (r4 >> 10 & 1) << 3 |
    (r5 >> 10 & 1) << 4 | (r6 >> 10 & 1) << 5 | (r7 >> 11 & 1) << 6 | (r8 >> 11 & 1) << 7;
  f.vsteepNE = (r1 >> 7 & 1) | (r2 >> 7 & 1) << 1 | (r3 >> 7 & 1) << 2 | (r4 >> 6 & 1) << 3 |
    (r5 >> 6 & 1) << 4 | (r6 >> 6 & 1) << 5 | (r7 >> 5 & 1) << 6 | (r8 >> 5 & 1) << 7;
  f.vflatNW = (r1 >> 11 & 1) | (r2 >> 14 & 1) << 1 | (r3 >> 17 & 1) << 2 | (r4 >> 20 & 1) << 3 |
    (r5 >> 23 & 1) << 4 | (r6 >> 26 & 1) << 5 | (r7 >> 29 & 1) << 6 | (r8 >> 32 & 1) << 7;
  f.vsrunNW = runUp(f.vsteepNW, 0x100u);
  f.vfrunNW = runUp(f.vflatNW, 0x100u);

  f.steepNW16 = (r9 >> 13 & 1) | (r10 >> 13 & 1) << 1 | (r11 >> 14 & 1) << 2 | (r12 >> 14 & 1) << 3 |
    (r13 >> 15 & 1) << 4 | (r14 >> 15 & 1) << 5 | (r15 >> 16 & 1) << 6 | (r16 >> 16 & 1) << 7;
  f.flatNW16 = (r9 >> 26 & 1) | (r10 >> 28 & 1) << 1 | (r11 >> 30 & 1) << 2 | (r12 >> 32 & 1) << 3 |
    (r13 >> 34 & 1) << 4 | (r14 >> 36 & 1) << 5 | (r15 >> 38 & 1) << 6 | (r16 >> 40 & 1) << 7;
  f.srunNW16 = runUp(f.steepNW | f.steepNW16 << 8, 0x10000u); //1..16
  f.frunNW16 = runUp(f.flatNW | f.flatNW16 << 8, 0x10000u);   //1..16
}

//The adaptive template pixels: the pixels one repeat period away (see updateRepeatPeriods), d pixels
//along the row and v rows up.
//
//        V1 V0 V2         <- v rows up: columns -1, 0, +1
//           V3            <- v+1 rows up
//              ...
//  H1.....x.....H2        <- the row above: columns -d and +d
//  H0.....?               <- the current row: column -d
template<typename B>
void Image1BitModel::readAdaptiveTemplate(const B& buf, const int bpos) {
  f.atH0 = f.atH1 = f.atH2 = f.atV0 = f.atV1 = f.atV2 = f.atV3 = 0;
  if (hPeriod >= 2) {
    f.atH0 = static_cast<uint32_t>(r0 >> (hPeriod - 1)) & 1;
    f.atH1 = static_cast<uint32_t>(r1 >> (8 + hPeriod)) & 1;
    f.atH2 = static_cast<uint32_t>(px(buf, bpos, 1, static_cast<int>(hPeriod)));
  }
  if (vPeriod >= minVPeriod && yRow > vPeriod) {
    f.atV0 = static_cast<uint32_t>(px(buf, bpos, static_cast<int>(vPeriod), 0));
    f.atV1 = static_cast<uint32_t>(px(buf, bpos, static_cast<int>(vPeriod), -1));
    f.atV2 = static_cast<uint32_t>(px(buf, bpos, static_cast<int>(vPeriod), 1));
    f.atV3 = static_cast<uint32_t>(px(buf, bpos, static_cast<int>(vPeriod) + 1, 0));
  }
}

//The match models: where was this neighbourhood seen before, and what came next there?
//
//Each model hashes the neighbourhood into a key of its own shape, and looks the key up in its
//table, which remembers the last three places the key was seen. Of those, the one whose 40
//surrounding pixels look the most like the current ones is taken (see residual), unless even that
//one differs in too many. From then on the match predicts that the pixel after the source is the
//same as the pixel after the current one, until it is wrong, or until either of them reaches the
//end of its row.
//
//Four keys, because they fail at different moments, and because several of them agreeing is worth
//more than any one of them.
template<typename B>
void Image1BitModel::runMatchModels(const B& buf, const int bpos, const bool rowStart) {
  const uint32_t curByteIdx = yRow * w + xByte;

  //A: a diamond, wide and shallow. The rows above, read wide, say more about which letter this is
  //than a narrow block reaching the same number of pixels further up.
  //  4  ....xxxxxxxxxxxx....
  //  3  ...xxxxxxxxxxxxxx...
  //  2  ..xxxxxxxxxxxxxxxx..
  //  1  xxxxxxxxxxxxxxxxxxxx
  //  0  .xxxxxxxxxxxx?
  const uint64_t keyA = hash(r0 & 0xfff, (r1 >> 2) & 0xfffff, (r2 >> 4) & 0xffff, (r3 >> 5) & 0x3fff, (r4 >> 6) & 0xfff);

  //B: a column narrowing over twelve rows: the same letter, wherever it is on the line
  // 12  .....xxx..
  // 11  .....xxx..
  // 10  .....xxx..
  //  9  .....xxx..
  //  8  ....xxxxx.
  //  7  ....xxxxx.
  //  6  ....xxxxx.
  //  5  ....xxxxx.
  //  4  ...xxxxxxx
  //  3  ...xxxxxxx
  //  2  ...xxxxxxx
  //  1  ...xxxxxxx
  //  0  xxxxxx?
  const uint64_t keyB = hash(r0 & 0x3f,
    (r1 >> 5 & 0x7f) | (r2 >> 5 & 0x7f) << 7 | (r3 >> 5 & 0x7full) << 14 | (r4 >> 5 & 0x7full) << 21,
    (r5 >> 6 & 0x1f) | (r6 >> 6 & 0x1f) << 5 | (r7 >> 6 & 0x1full) << 10 | (r8 >> 6 & 0x1full) << 15,
    (r9 >> 7 & 7) | (r10 >> 7 & 7) << 3 | (r11 >> 7 & 7ull) << 6 | (r12 >> 7 & 7ull) << 9);

  //C and D: the same letter, blurred. Two prints of the same letter differ in a pixel here and
  //there, and an exact key would treat them as different. These keys merge the rows above in pairs,
  //so they only ask for the same shape: C keeps a pixel where both rows of the pair have ink (AND),
  //D where either has (OR). One forgives a missing pixel, the other an extra one.
  //In rows 1..8, 'a' and 'b' are the two rows of a pair:
  //  8  ..bbbbbbbbbbbb
  //  7  ..aaaaaaaaaaaa
  //  6  ..bbbbbbbbbbbb
  //  5  ..aaaaaaaaaaaa
  //  4  ..bbbbbbbbbbbb
  //  3  ..aaaaaaaaaaaa
  //  2  ..bbbbbbbbbbbb
  //  1  ..aaaaaaaaaaaa
  //  0  xxxxxxxx?
  const uint64_t a12 = r1 & r2, a34 = r3 & r4, a56 = r5 & r6, a78 = r7 & r8;
  const uint64_t keyC = hash(r0 & 0xff,
    (a12 >> 3 & 0xfff) | (a34 >> 3 & 0xfffull) << 12,
    (a56 >> 3 & 0xfff) | (a78 >> 3 & 0xfffull) << 12);
  const uint64_t keyD = hash(r0 & 0xff,
    ((r1 | r2) >> 3 & 0xfff) | ((r3 | r4) >> 3 & 0xfffull) << 12,
    ((r5 | r6) >> 3 & 0xfff) | ((r7 | r8) >> 3 & 0xfffull) << 12);

  //the high bits of a key select its bucket, the byte below them is its check byte
  const uint64_t keys[nMatch] = { keyA, keyB, keyC, keyD };
  for (int k = 0; k < nMatch; ++k) {
    matchHashIdx[k] = finalize64(keys[k], bucketBits);
    matchCheck[k] = static_cast<uint8_t>(keys[k] >> (56 - bucketBits));
  }
  //a match does not continue into the next row
  if (rowStart) {
    for (int k = 0; k < nMatch; ++k) matchLen[k] = 0;
  }
  for (int k = 0; k < nMatch; ++k) {
    if (matchLen[k] == 0 && w != 0) {
      //of the places this key was seen, take the one that looks the most like here
      const MatchBucket& bucket = matchTable[(static_cast<size_t>(k) << bucketBits) + matchHashIdx[k]];
      uint32_t bestRes = maxResidual + 1;
      for (int s = 0; s < matchSlots; ++s) {
        const uint32_t e = bucket.pos[s];
        if (e == 0) break;
        if (bucket.check[s] != matchCheck[k]) continue;
        const uint32_t cand = e & 0x1fffffff;
        const uint32_t candBpos = e >> 29;
        if (cand >= curByteIdx) continue;
        const uint32_t res = residual(buf, cand, candBpos, curByteIdx);
        if (res < bestRes) { bestRes = res; matchByteIdx[k] = cand; matchBpos[k] = candBpos; }
      }
      if (bestRes <= maxResidual) {
        matchLen[k] = 1;
        matchSrcX[k] = matchByteIdx[k] % w;
        matchResid[k] = residBucket(bestRes);
      }
    }
    predictedBit[k] = -1;
    if (matchLen[k] > 0) {
      const uint32_t dist = curByteIdx - matchByteIdx[k];
      if (dist >= 1) predictedBit[k] = (buf(dist) >> (7 - matchBpos[k])) & 1;
      else matchLen[k] = 0;
    }
  }
  lastByteIdx = curByteIdx;
  lastBpos = static_cast<uint32_t>(bpos);

  //what they say together: for each, 0 = no match, 1 = expects paper, 2 = expects ink
  f.lenBucketA = matchBucket(matchLen[0]);
  f.lenBucketB = matchBucket(matchLen[1]);
  f.matchVote = (predictedBit[0] + 1) * 3 + (predictedBit[1] + 1); //A and B: 0..8
  f.matchVote4 = (f.matchVote * 3 + (predictedBit[2] + 1)) * 3 + (predictedBit[3] + 1); //all four: 0..80

  //The best of the active matches is the longest one, and of equally long ones the one that
  //resembled the current neighbourhood the most. The matches disagree when any two of them expect
  //different pixels.
  bool expects0 = false, expects1 = false;
  int bestMatch = -1;
  for (int k = 0; k < nMatch; ++k) {
    if (predictedBit[k] < 0) continue;
    if (predictedBit[k] == 0) expects0 = true; else expects1 = true;
    if (bestMatch < 0 || matchLen[k] > matchLen[bestMatch] ||
      (matchLen[k] == matchLen[bestMatch] && matchResid[k] < matchResid[bestMatch]))
      bestMatch = k;
  }
  const uint32_t matchUncertain = static_cast<uint32_t>(expects0 && expects1);
  //the verdict, for the mixer and the SSE stage: 0 = no match, otherwise 1 + these bits:
  //  bits 3-4: the length of the best match (1-3, 4-15, 16-63, 64 or more)
  //  bit 2:    the pixel it expects
  //  bit 1:    the matches disagree
  //  bit 0:    the best match's source was identical to the current neighbourhood
  f.matchMode = 0;
  if (bestMatch >= 0) {
    const uint32_t bl = matchLen[bestMatch];
    const uint32_t lenQ = bl < 4 ? 0 : bl < 16 ? 1 : bl < 64 ? 2 : 3;
    f.matchMode = 1 + (lenQ << 3 | static_cast<uint32_t>(predictedBit[bestMatch]) << 2 | matchUncertain << 1 |
      static_cast<uint32_t>(matchResid[bestMatch] == 0));
  }
}

//The pixels right around the current one, used by several contexts, the direct predictors, the
//mixer and the SSE stage.
void Image1BitModel::readNeighbourhood(const int y) {
  //surrounding4
  //  1  xxx
  //  0  x?
  f.surrounding4 = y | (r1 >> 7 & 7) << 1;

  //mCtx6
  //  3  ...x
  //  2  ...x
  //  1  ...x
  //  0  xxx?
  f.mCtx6 = ((r0 & 7) | (r1 >> 8 & 1) << 3 | (r2 >> 8 & 1) << 4 | (r3 >> 8 & 1) << 5);

  //surrounding12
  //  2  xxxxx
  //  1  xxxxx
  //  0  xx?
  f.surrounding12 = (r0 & 3) | (r1 >> 6 & 0x1f) << 2 | (r2 >> 6 & 0x1f) << 7;

  //surrounding24
  //  3  xxxxxxx
  //  2  xxxxxxx
  //  1  xxxxxxx
  //  0  xxx?
  f.surrounding24 = (r0 & 7) | (r1 >> 5 & 0x7f) << 3 | (r2 >> 5 & 0x7f) << 10 | (r3 >> 5 & 0x7f) << 17;

  //frame16: a frame around surrounding24; the two together cover 40 pixels
  //  4  xxxxxxxxx
  //  3  x.......x
  //  2  x.......x
  //  1  x.......x
  //  0  x...?
  f.frame16 = (r0 >> 3 & 1) | (r1 >> 12 & 1) << 1 | (r2 >> 12 & 1) << 2 | (r3 >> 12 & 1) << 3 | (r4 >> 4 & 0x1ff) << 4 | (r3 >> 4 & 1) << 13 | (r2 >> 4 & 1) << 14 | (r1 >> 4 & 1) << 15;

  //how many of the 4, 12, 24 and 40 pixels around are ink: the grey level, for dithered images
  f.bitcount04 = bitCount(f.surrounding4);
  f.bitcount12 = bitCount(f.surrounding12);
  f.bitcount24 = bitCount(f.surrounding24);
  f.bitcount40 = bitCount(f.frame16) + f.bitcount24;

  //the shape of the ink just above, blurred: the two rows above are merged, and each pixel with
  //the one to its left, so each of the 6 bits covers 2x2 pixels. dil6 is set where any of the 4 is
  //ink (the ink dilated), ero6 where all 4 are (the ink eroded). The 6 cells cover:
  //  2  xxxxxxx
  //  1  xxxxxxx
  //  0  ....?
  const uint64_t mo12 = (r1 | r2), ma12 = (r1 & r2);
  f.dil6 = static_cast<uint32_t>((mo12 | mo12 >> 1) >> 6) & 0x3f;
  f.ero6 = static_cast<uint32_t>((ma12 & ma12 >> 1) >> 6) & 0x3f;
}

//The indexed contexts: small enough to address their slots directly. Each context uses its own part
//of the table t (see C in the header).
void Image1BitModel::setIndexedContexts(const int y, const int bpos) {
  int c = 0; //where the current context's slots start
  size_t i = 0; //context number

  //////////////////////////////////////////////////////////////////////////////
  // Small templates: the pixels right around the current one, from all sides
  //////////////////////////////////////////////////////////////////////////////

  //  1  .x
  //  0  x?
  cxt[i++] = c + (y | (r1 >> 8 & 1) << 1);
  c += 1 << 2;

  //surrounding4
  //  1  xxx
  //  0  x?
  cxt[i++] = c + f.surrounding4;
  c += 1 << 4;

  //mCtx6
  //  3  ...x
  //  2  ...x
  //  1  ...x
  //  0  xxx?
  cxt[i++] = c + f.mCtx6;
  c += 1 << 6;

  //  2  .xx..
  //  1  ..xxx
  //  0  xxx?
  cxt[i++] = c + ((r0 & 7) | (r1 >> 7 & 7) << 3 | (r2 >> 9 & 3) << 6);
  c += 1 << 8;

  //  3  .x...
  //  2  .x...
  //  1  xxxxx
  //  0  x?
  cxt[i++] = c + (y | (r1 >> 5 & 0x1f) << 1 | (r2 >> 8 & 1) << 6 | (r3 >> 8 & 1) << 7);
  c += 1 << 8;

  //  3  .xx.
  //  2  .xx.
  //  1  .xxx
  //  0  x?
  cxt[i++] = c + (y | (r1 >> 6 & 7) << 1 | (r2 >> 7 & 3) << 4 | (r3 >> 7 & 3) << 6);
  c += 1 << 8;

  //the column above, 8 rows
  //  8  x
  //  7  x
  //  6  x
  //  5  x
  //  4  x
  //  3  x
  //  2  x
  //  1  x
  //  0  ?
  cxt[i++] = c + ((r1 >> 8 & 1) << 7 | (r2 >> 8 & 1) << 6 | (r3 >> 8 & 1) << 5 | (r4 >> 8 & 1) << 4 | (r5 >> 8 & 1) << 3 | (r6 >> 8 & 1) << 2 | (r7 >> 8 & 1) << 1 | (r8 >> 8 & 1));
  c += 1 << 8;

  //  4  xx
  //  3  xx
  //  2  xx
  //  1  xx
  //  0  x?
  cxt[i++] = c + (y | (r1 >> 8 & 3) << 1 | (r2 >> 8 & 3) << 3 | (r3 >> 8 & 3) << 5 | (r4 >> 8 & 3) << 7);
  c += 1 << 9;

  //  1  xxxxxx
  //  0  .xxxx?
  cxt[i++] = c + ((r0 & 0x0f) | (r1 >> 8 & 0x3f) << 4);
  c += 1 << 10;

  //  2  .....xx.
  //  1  .xxxxxxx
  //  0  xxx?
  cxt[i++] = c + ((r0 & 7) | (r1 >> 4 & 0x7f) << 3 | ((r2 >> 5) & 3) << 10);
  c += 1 << 12;

  //surrounding12
  //  2  xxxxx
  //  1  xxxxx
  //  0  xx?
  cxt[i++] = c + f.surrounding12;
  c += 1 << 12;

  //how many of the 4, the 12 and the 40 pixels around are ink
  cxt[i++] = c + f.bitcount04;
  c += 5;

  cxt[i++] = c + f.bitcount12;
  c += 13;

  cxt[i++] = c + f.bitcount40;
  c += 41;

  //the dither phase (x mod 4, y mod 4), and surrounding4. An ordered dither repeats every few
  //pixels, so where we are within its cell tells a lot about the next pixel.
  cxt[i++] = c + ((bpos & 3) | (yRow & 3) << 2 | f.surrounding4 << 4);
  c += 1 << 8;

  //  1  .xxxxx
  //  0  xxxxx?
  cxt[i++] = c + ((r0 & 0x1f) | (r1 >> 8 & 0x1f) << 5);
  c += 1 << 10;

  //  4  xxx
  //  3  xxx
  //  2  xxx
  //  1  xxx
  //  0  x?
  cxt[i++] = c + (y | (r1 >> 7 & 7) << 1 | (r2 >> 7 & 7) << 4 | (r3 >> 7 & 7) << 7 | (r4 >> 7 & 7) << 10);
  c += 1 << 13;

  //the horizontal adaptive template (see readAdaptiveTemplate): W, N, and the pixels one period
  //along the row away, with how well the image repeats at that period
  cxt[i++] = c + (y | (r1 >> 8 & 1) << 1 | f.atH0 << 2 | f.atH1 << 3 | f.atH2 << 4 | qualH << 5);
  c += 1 << 7;

  //the vertical adaptive template: W, N, and the pixels one period up, with how well the image
  //repeats at that period
  cxt[i++] = c + (y | (r1 >> 8 & 1) << 1 | f.atV0 << 2 | f.atV1 << 3 | f.atV2 << 4 | f.atV3 << 5 | qualV << 6);
  c += 1 << 8;

  //surrounding4, and the pixel two rows up
  //  2  .x.
  //  1  xxx
  //  0  x?
  cxt[i++] = c + (f.surrounding4 | (r2 >> 8 & 1) << 4);
  c += 32;

  //////////////////////////////////////////////////////////////////////////////
  // Wide, shallow templates, and the column.
  //
  // Beyond the nearest pixels, the templates below read one or two rows as wide
  // as possible, at every depth down to 13 rows up, instead of compact blocks.
  // A row of a letter, of a line or of a dither pattern is a horizontal feature,
  // so a wide reading of a row says more about the current pixel than a block of
  // the same number of pixels. Each depth is a separate look at the same letter,
  // and the mixer learns how far up it is still worth looking.
  //////////////////////////////////////////////////////////////////////////////

  //the column above, 16 rows: vertical lines, frame borders, and the spacing of lines of text
  // 16  x
  // 15  x
  // 14  x
  // 13  x
  // 12  x
  // 11  x
  // 10  x
  //  9  x
  //  8  x
  //  7  x
  //  6  x
  //  5  x
  //  4  x
  //  3  x
  //  2  x
  //  1  x
  //  0  ?
  cxt[i++] = c + (f.col8 | f.col16 << 8);
  c += 1 << 16;

  //the sparse template T15
  //  6  ........x....
  //  5  .............
  //  4  .............
  //  3  .......xxx...
  //  2  ......x.x...x
  //  1  ...x...xxxx..
  //  0  x...x..x?
  cxt[i++] = c + sparse(T15, 15);
  c += 1 << 16;

  //the dither phase over 16 rows (x mod 8, y mod 16), and surrounding4
  cxt[i++] = c + (bpos | (yRow & 15) << 3 | f.surrounding4 << 7);
  c += 1 << 11;

  //  1  xxxxxxxxxxxxx
  //  0  .....x?
  cxt[i++] = c + (y | (r1 >> 2 & 0x1fff) << 1);
  c += 1 << 14;

  //  2  xxxxxxxxxxxxx
  //  1  .............
  //  0  .....x?
  cxt[i++] = c + (y | (r2 >> 2 & 0x1fff) << 1);
  c += 1 << 14;

  //  1  xxxxxxxxxxxxxxx
  //  0  .......?
  cxt[i++] = c + (r1 >> 1 & 0x7fff);
  c += 1 << 15;

  //  4  xxxxxxx
  //  3  xxxxxxx
  //  2  .......
  //  1  .......
  //  0  ..x?
  cxt[i++] = c + (y | (r3 >> 5 & 0x7f) << 1 | (r4 >> 5 & 0x7f) << 8);
  c += 1 << 15;

  //  2  xxxxxxxxxxxxxxx
  //  1  ...............
  //  0  .......?
  cxt[i++] = c + (r2 >> 1 & 0x7fff);
  c += 1 << 15;

  //  6  xxxxxxx
  //  5  xxxxxxx
  //   (rows 4..1 up: none)
  //  0  ..x?
  cxt[i++] = c + (y | (r5 >> 5 & 0x7f) << 1 | (r6 >> 5 & 0x7f) << 8);
  c += 1 << 15;

  //  4  xxxxxxxxxxxxx
  //  3  .............
  //  2  .............
  //  1  .............
  //  0  .....x?
  cxt[i++] = c + (y | (r4 >> 2 & 0x1fff) << 1);
  c += 1 << 14;

  //  3  xxxxxxxxxxxxx
  //  2  .............
  //  1  .............
  //  0  .....x?
  cxt[i++] = c + (y | (r3 >> 2 & 0x1fff) << 1);
  c += 1 << 14;

  // 10  xxxxxxx
  //  9  xxxxxxx
  //   (rows 8..1 up: none)
  //  0  ..x?
  cxt[i++] = c + (y | (r9 >> 5 & 0x7f) << 1 | (r10 >> 5 & 0x7f) << 8);
  c += 1 << 15;

  // 13  xxxxx
  // 12  xxxxx
  // 11  xxxxx
  //   (rows 10..1 up: none)
  //  0  .x?
  cxt[i++] = c + (y | (r11 >> 6 & 0x1f) << 1 | (r12 >> 6 & 0x1f) << 6 | (r13 >> 6 & 0x1f) << 11);
  c += 1 << 16;

  //  2  ..xxxxxx..
  //  1  xxxxxxxxxx
  //  0  ....?
  cxt[i++] = c + ((r1 >> 3 & 0x3ff) | (r2 >> 5 & 0x3f) << 10);
  c += 1 << 16;

  //  2  xxxxxxxxxxxxxxxxx
  //  1  .................
  //  0  ........?
  cxt[i++] = c + (r2 & 0x1ffff);
  c += 1 << 17;

  //////////////////////////////////////////////////////////////////////////////
  // Runs and directions
  //////////////////////////////////////////////////////////////////////////////

  //how far the ink runs up the column and the two either side, over 16 rows (the columns 'L',
  //'c' and 'R' of readColumns): how a vertical stroke is ending across its width. And N and W.
  cxt[i++] = c + ((f.vrun16 - 1) | (f.vrun16L - 1) << 4 | (f.vrun16R - 1) << 8 | (f.col8 & 1) << 12 | y << 13);
  c += 1 << 14;

  //how far the ink runs up the five columns -2..+2, over 8 rows: the profile of a stroke
  cxt[i++] = c + ((f.vrunLL - 1) | (f.vrunL - 1) << 3 | (f.vrun - 1) << 6 | (f.vrunR - 1) << 9 | (f.vrunRR - 1) << 12);
  c += 1 << 15;

  //how far the ink runs along the two steep slopes and the flat one to the left (see
  //readDirections), the nearest pixel of each steep slope, and W
  cxt[i++] = c + ((f.srunNW - 1) | (f.srunNE - 1) << 3 | (f.frunNW - 1) << 6 | (f.steepNW & 1) << 9 | (f.steepNE & 1) << 10 | y << 11);
  c += 1 << 12;

  //the two diagonals (see readDirections)
  //  8  x...............x
  //  7  .x.............x.
  //  6  ..x...........x..
  //  5  ...x.........x...
  //  4  ....x.......x....
  //  3  .....x.....x.....
  //  2  ......x...x......
  //  1  .......x.x.......
  //  0  ........?
  cxt[i++] = c + (f.diagNW | f.diagNE << 8);
  c += 1 << 16;

  //the blurred shape of the ink just above (dil6 and ero6, see readNeighbourhood), W and N
  cxt[i++] = c + (f.dil6 | f.ero6 << 6 | y << 12 | (r1 >> 8 & 1) << 13);
  c += 1 << 14;

  //the next two colour changes on the row above (b1 and b2, see trackRuns), the colour of the
  //current run, and N
  cxt[i++] = c + (f.vb1 | f.vb2 << 5 | f.curCol << 10 | (r1 >> 8 & 1) << 11);
  c += 1 << 12;

  assert(i == N);
  assert(c == C);
}

//The hashed contexts: templates too wide, too deep or too blurred to index directly.
void Image1BitModel::setHashedContexts(const int y, const int bpos) {
  //most indexed contexts are read here a second time (see noMirror in the header)
  size_t i;
  for (i = 0; i < N; ++i) {
    if ((noMirror >> i & 1) == 0) mapL.set(hash(i, cxt[i]));
  }

  //////////////////////////////////////////////////////////////////////////////
  // Blocks around the current pixel
  //////////////////////////////////////////////////////////////////////////////

  //  3  ....xxxxx.....
  //  2  ...xxxxxxx....
  //  1  .xxxxxxxxxxxxx
  //  0  xxxxxx?
  mapL.set(hash(i++, (r0 & 0x3f), (r1 >> 1 & 0x1fff), (r2 >> 5 & 0x7f), (r3 >> 6 & 0x1f)));

  //  3  ........xx
  //  2  ........xx
  //  1  ..xxxxxxxx
  //  0  xxxxxxxx?
  mapL.set(hash(i++, (r0 & 0xff), (r1 >> 7 & 0xff), (r2 >> 7 & 3) << 2 | (r3 >> 7 & 3)));

  //  8  .xx
  //  7  .xx
  //  6  .xx
  //  5  .xx
  //  4  .xx
  //  3  .xx
  //  2  .xx
  //  1  .xx
  //  0  x?
  mapL.set(hash(i++, y | (r1 >> 7 & 3) << 1, (r2 >> 7 & 3) | (r3 >> 7 & 3) << 2, (r4 >> 7 & 3) | (r5 >> 7 & 3) << 2, (r6 >> 7 & 3) | (r7 >> 7 & 3) << 2, (r8 >> 7 & 3)));

  //surrounding24
  //  3  xxxxxxx
  //  2  xxxxxxx
  //  1  xxxxxxx
  //  0  xxx?
  mapL.set(hash(i++, f.surrounding24));

  //surrounding24 and frame16
  //  4  xxxxxxxxx
  //  3  xxxxxxxxx
  //  2  xxxxxxxxx
  //  1  xxxxxxxxx
  //  0  xxxx?
  mapL.set(hash(i++, f.surrounding24, f.frame16));

  //  1  .........xxxxxxxxxxxxxxxx
  //  0  xxxxxxxxxxxxxxxx?
  mapL.set(hash(i++, r0 & 0xffff, r1 & 0xffff));

  //  8  xxx
  //  7  xxx
  //  6  xxx
  //  5  xxx
  //  4  xxx
  //  3  xxx
  //  2  xxx
  //  1  xxx
  //  0  x?
  mapL.set(hash(i++, y | (r1 >> 7 & 7) << 1,
    (r2 >> 7 & 7) | (r3 >> 7 & 7) << 3 | (r4 >> 7 & 7) << 6,
    (r5 >> 7 & 7) | (r6 >> 7 & 7) << 3 | (r7 >> 7 & 7) << 6 | (r8 >> 7 & 7) << 9));

  //surrounding12, and the dither phase (x mod 8, y mod 8)
  mapL.set(hash(i++, f.surrounding12, bpos | (yRow & 7) << 3));

  //  4  .........................xxxxxxxx
  //  3  .................xxxxxxxxxxxxxxxx
  //  2  .................xxxxxxxxxxxxxxxx
  //  1  .........xxxxxxxxxxxxxxxxxxxxxxxx
  //  0  xxxxxxxxxxxxxxxxxxxxxxxx?
  mapL.set(hash(i++, r0 & 0xffffff, r1 & 0xffffff, r2 & 0xffff, r3 & 0xffff, r4 & 0xff));

  //a diamond: narrower further away, where each pixel says less
  //  4  ..........xxxxx....
  //  3  .........xxxxxxx...
  //  2  ........xxxxxxxxx..
  //  1  ........xxxxxxxxxxx
  //  0  xxxxxxxxxxxx?
  mapL.set(hash(i++, r0 & 0xfff, (r1 >> 2) & 0x7ff, (r2 >> 4) & 0x1ff, (r3 >> 5) & 0x7f, (r4 >> 6) & 0x1f));

  //the rows above, and of the current row only W: what is left at the start of a row, and what
  //carries a vertical stroke across a gap in the current row
  //  5  xxxxx
  //  4  xxxxx
  //  3  xxxxx
  //  2  xxxxx
  //  1  xxxxx
  //  0  .x?
  mapL.set(hash(i++, y | (r1 >> 6 & 0x1f) << 1, (r2 >> 6 & 0x1f) | (r3 >> 6 & 0x1f) << 5, (r4 >> 6 & 0x1f) | (r5 >> 6 & 0x1f) << 5));

  //the column above (col8), and the nearest pixels
  //  8  ......x..
  //  7  ......x..
  //  6  ......x..
  //  5  ......x..
  //  4  ......x..
  //  3  ......x..
  //  2  .....xxx.
  //  1  ....xxxxx
  //  0  xxxxxx?
  mapL.set(hash(i++, f.col8, r0 & 0x3f, (r1 >> 6 & 0x1f) | (r2 >> 7 & 7) << 5));

  //  8  xxxx
  //  7  xxxx
  //  6  xxxx
  //  5  xxxx
  //  4  xxxx
  //  3  xxxx
  //  2  xxxx
  //  1  xxxx
  //  0  .x?
  mapL.set(hash(i++, y | (r1 >> 7 & 0xf) << 1,
    (r2 >> 7 & 0xf) | (r3 >> 7 & 0xf) << 4 | (r4 >> 7 & 0xf) << 8,
    (r5 >> 7 & 0xf) | (r6 >> 7 & 0xf) << 4 | (r7 >> 7 & 0xf) << 8 | (r8 >> 7 & 0xf) << 12));

  //  3  ......xxxxxxxxx....
  //  2  ....xxxxxxxxxxxxx..
  //  1  ..xxxxxxxxxxxxxxxxx
  //  0  xxxxxxxxxx?
  mapL.set(hash(i++, r0 & 0x3ff, r1 & 0x1ffff, (r2 >> 2) & 0x1fff, (r3 >> 4) & 0x1ff));

  //  4  xxxxxxx
  //  3  xxxxxxx
  //  2  xxxxxxx
  //  1  xxxxxxx
  //  0  ..x?
  mapL.set(hash(i++, y | (r1 >> 5 & 0x7f) << 1, (r2 >> 5 & 0x7f) | (r3 >> 5 & 0x7f) << 7, (r4 >> 5 & 0x7f)));

  //how much ink is around, over 12, 24 and 40 pixels, with surrounding4 and the ink in the column above
  mapL.set(hash(i++, static_cast<uint32_t>(f.bitcount40) | static_cast<uint32_t>(f.bitcount24) << 6 |
    static_cast<uint32_t>(f.bitcount12) << 11, f.surrounding4, static_cast<uint32_t>(f.colCount)));

  //surrounding24, and the dither phase (x mod 8, y mod 8)
  mapL.set(hash(i++, f.surrounding24, bpos | (yRow & 7) << 3));

  //////////////////////////////////////////////////////////////////////////////
  // Deep and wide: whole rows, and rows up to 16 above
  //////////////////////////////////////////////////////////////////////////////

  //the column above, 16 rows, with W and WW
  // 16  ..x
  // 15  ..x
  // 14  ..x
  // 13  ..x
  // 12  ..x
  // 11  ..x
  // 10  ..x
  //  9  ..x
  //  8  ..x
  //  7  ..x
  //  6  ..x
  //  5  ..x
  //  4  ..x
  //  3  ..x
  //  2  ..x
  //  1  ..x
  //  0  xx?
  mapL.set(hash(i++, y | (r0 & 3) << 1, f.col8 | f.col16 << 8));

  // 16  xxx
  // 15  xxx
  // 14  xxx
  // 13  xxx
  // 12  xxx
  // 11  xxx
  // 10  xxx
  //  9  xxx
  //   (rows 8..3 up: none)
  //  2  xxx
  //  1  xxx
  //  0  x?
  mapL.set(hash(i++, y | (r1 >> 7 & 7) << 1 | (r2 >> 7 & 7) << 4,
    (r9 >> 7 & 7) | (r10 >> 7 & 7) << 3 | (r11 >> 7 & 7) << 6 | (r12 >> 7 & 7) << 9,
    (r13 >> 7 & 7) | (r14 >> 7 & 7) << 3 | (r15 >> 7 & 7) << 6 | (r16 >> 7 & 7) << 9));

  // 16  ..................x....
  // 15  ..................x....
  // 14  ..................x....
  // 13  ..................x....
  // 12  ..................x....
  // 11  ..................x....
  // 10  ..................x....
  //  9  ..................x....
  //   (rows 8..2 up: none)
  //  1  .............xxxxxxxxxx
  //  0  xxxxxxxxxxxxxxxxxx?
  mapL.set(hash(i++, r0 & 0x3ffff, r1 >> 4 & 0x3ff, f.col16));

  //surrounding24, the current run (length and colour) and the run up the column
  mapL.set(hash(i++, f.surrounding24, f.runIdx | runValue << 4, static_cast<uint32_t>(f.vrun)));

  //  3  ........xxxxxxxxxxxxxxxx........
  //  2  xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  //  1  xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  //  0  ...............?
  mapL.set(hash(i++, r1 & 0xffffff, r2 & 0xffffff, r3 & 0xffff, (r1b & 0xff) | (r2b & 0xffull) << 8));

  //  3  xxxxxxxxx
  //  2  xxxxxxxxx
  //  1  xxxxxxxxx
  //  0  ...x?
  mapL.set(hash(i++, y | (r1 >> 4 & 0x1ff) << 1, (r2 >> 4 & 0x1ff) | (r3 >> 4 & 0x1ff) << 9));

  //  8  xxxxxxxxxxxxxxxx
  //  7  xxxxxxxxxxxxxxxx
  //  6  xxxxxxxxxxxxxxxx
  //  5  xxxxxxxxxxxxxxxx
  //   (rows 4..1 up: none)
  //  0  ......x?
  mapL.set(hash(i++, r5 & 0xffff, r6 & 0xffff, r7 & 0xffff, r8 & 0xffff, y));

  // 12  xxxxxxxxxxxxxxxx
  // 11  xxxxxxxxxxxxxxxx
  // 10  xxxxxxxxxxxxxxxx
  //  9  xxxxxxxxxxxxxxxx
  //   (rows 8..1 up: none)
  //  0  ......x?
  mapL.set(hash(i++, r9 & 0xffff, r10 & 0xffff, r11 & 0xffff, r12 & 0xffff, y));

  //  3  xxxxxxxxxxxxxxx
  //  2  xxxxxxxxxxxxxxx
  //  1  xxxxxxxxxxxxxxx
  //  0  ......x?
  mapL.set(hash(i++, r1 >> 1 & 0x7fff, r2 >> 1 & 0x7fff, r3 >> 1 & 0x7fff, y));

  //  1  xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  //  0  ......................x?
  mapL.set(hash(i++, r1 & 0xffffffffull, r1b & 0xff, y));

  //  3  ................xxxxxxxxxxxxxxxxxxxxxxxx
  //  2  xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  //  1  ........................................
  //  0  ......................x?
  mapL.set(hash(i++, r2 & 0xffffffffull, (r2b & 0xff) | (r3b & 0xffull) << 8, r3 & 0xffff, y));

  // 16  ........x........
  // 15  ........x........
  // 14  ........x........
  // 13  ........x........
  // 12  ........x........
  // 11  ........x........
  // 10  ........x........
  //  9  ........x........
  //  8  ........x........
  //  7  ........x........
  //  6  ........x........
  //  5  ........x........
  //  4  ........x........
  //  3  ........x........
  //  2  ........x........
  //  1  ........xxxxxxxxx
  //  0  xxxxxxxx?
  mapL.set(hash(i++, f.col8 | f.col16 << 8, r0 & 0xff, r1 & 0xff));

  //  0  xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx?
  mapL.set(hash(i++, r0 & 0xffffffffull, y));

  // 16  ..x..
  // 15  ..x..
  // 14  ..x..
  // 13  ..x..
  // 12  ..x..
  // 11  ..x..
  // 10  ..x..
  //  9  ..x..
  //  8  ..x..
  //  7  ..x..
  //  6  ..x..
  //  5  ..x..
  //  4  ..x..
  //  3  ..x..
  //  2  xxxxx
  //  1  xxxxx
  //  0  xx?
  mapL.set(hash(i++, f.col8 | f.col16 << 8, f.surrounding12));

  // 16  .x.
  // 15  .x.
  // 14  .x.
  // 13  .x.
  // 12  .x.
  // 11  .x.
  // 10  .xx
  //  9  .xx
  //  8  .x.
  //  7  .x.
  //  6  .x.
  //  5  .x.
  //  4  .x.
  //  3  .x.
  //  2  .xx
  //  1  .xx
  //  0  x?
  mapL.set(hash(i++, (r1 >> 7 & 3) | (r2 >> 7 & 3) << 2 | (r9 >> 7 & 3) << 4 | (r10 >> 7 & 3) << 6, f.col8 | f.col16 << 8, y));

  //  4  xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  //  3  xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  //  2  ................................
  //  1  ................................
  //  0  ......................x?
  mapL.set(hash(i++, r3 & 0xffffffffull, r4 & 0xffffffffull, y));

  //the row above, read further ahead: from 7 behind to 16 ahead
  //  1  xxxxxxxxxxxxxxx.xxxxxxxx
  //  0  ......x?
  mapL.set(hash(i++, y | (r1 >> 1 & 0x7fff) << 1, r1b & 0xff));

  //  2  ......xxxxx....xxxxxxxx
  //  1  ......xxxxx....xxxxxxxx
  //  0  xxxxxx?
  mapL.set(hash(i++, y | (r0 & 0x3f) << 1, (r1 >> 4 & 0x1f) | (r1b & 0xff) << 5, (r2 >> 4 & 0x1f) | (r2b & 0xff) << 5));

  //what is coming on the three rows above, beyond the reach of the other templates
  //  3  ..........xxxxxxxx
  //  2  ..........xxxxxxxx
  //  1  xxx.......xxxxxxxx
  //  0  x?
  mapL.set(hash(i++, y | (r1 >> 7 & 7) << 1, (r1b & 0xff) | (r2b & 0xff) << 8 | (r3b & 0xffull) << 16));

  //////////////////////////////////////////////////////////////////////////////
  // Match models, runs and directions
  //////////////////////////////////////////////////////////////////////////////

  //the pixels the four match models expect, and surrounding12
  mapL.set(hash(i++, f.matchVote4, f.surrounding12));

  //how far the ink runs up the seven columns -3..+3 (see readColumns), with W and WW
  mapL.set(hash(i++, (f.vrunLLL - 1) | (f.vrunLL - 1) << 3 | (f.vrunL - 1) << 6 | (f.vrun - 1) << 9 |
    (f.vrunR - 1) << 12 | (f.vrunRR - 1) << 15 | (f.vrunRRR - 1) << 18, y | (r0 & 3) << 1));

  //the same for the three middle columns over 16 rows and the two beside them over 8, with
  //surrounding12 and the ink in the column above
  mapL.set(hash(i++, (f.vrun16 - 1) | (f.vrun16L - 1) << 4 | (f.vrun16R - 1) << 8 | (f.vrunLL - 1) << 12 | (f.vrunRR - 1) << 15,
    f.surrounding12, f.colCount));

  //the lengths of the current run and the two before it, the colour of the current run,
  //surrounding12, and the next colour change on the row above (b1, see trackRuns)
  mapL.set(hash(i++, f.runIdx | runBucket(prevRun1) << 4 | runBucket(prevRun2) << 8 | runValue << 12, f.surrounding12, f.vb1));

  //the diagonals and surrounding12: a slanted stroke, or a corner?
  //  8  x...............x
  //  7  .x.............x.
  //  6  ..x...........x..
  //  5  ...x.........x...
  //  4  ....x.......x....
  //  3  .....x.....x.....
  //  2  ......xxxxx......
  //  1  ......xxxxx......
  //  0  ......xx?
  mapL.set(hash(i++, f.diagNW | f.diagNE << 8, f.surrounding12));

  //each diagonal two pixels wide, which gives the width of a slanted stroke, not only its edge;
  //with W, WW and WWW
  //  8  xx...............xx
  //  7  .xx.............xx.
  //  6  ..xx...........xx..
  //  5  ...xx.........xx...
  //  4  ....xx.......xx....
  //  3  .....xx.....xx.....
  //  2  ......xx...xx......
  //  1  .......xx.xx.......
  //  0  .......xx?
  mapL.set(hash(i++, f.diagNW | f.diagNW2 << 8 | f.diagNE << 16 | f.diagNE2 << 24, y | (r0 & 3) << 1));

  //the very steep and very flat slopes (see readDirections), with W, WW, WWW and the three pixels above
  mapL.set(hash(i++, f.vsteepNW | f.vsteepNE << 8 | f.vflatNW << 16, y | (r0 & 3) << 1 | (r1 >> 7 & 7) << 3));

  //the steep and flat slopes, with W, WW and WWW
  mapL.set(hash(i++, f.steepNW | f.steepNE << 8, f.flatNW | f.flatNE << 8, y | (r0 & 3) << 1));

  //how far the ink runs in every direction: the diagonals, the slopes, the columns -1..+1 and the
  //current row; with surrounding4 and W
  mapL.set(hash(i++,
    (f.drunNW - 1) | (f.drunNE - 1) << 3 | (f.srunNW - 1) << 6 | (f.srunNE - 1) << 9 |
    (f.frunNW - 1) << 12 | (f.vrun - 1) << 15 | f.runIdx << 18,
    (f.vsrunNW - 1) | (f.vfrunNW - 1) << 3 | (f.vrunL - 1) << 6 | (f.vrunR - 1) << 9, f.surrounding4, y));

  //////////////////////////////////////////////////////////////////////////////
  // The shape of the ink, blurred
  //
  // Two prints of the same letter differ in a pixel here and there, and an exact
  // template treats them as two different contexts. These templates merge
  // neighbouring pixels first, so both prints give the same context: rows are
  // merged in pairs, and each pixel with the one to its left. A merged cell is
  // ink if any of its pixels is (the ink dilated), or only if all of them are
  // (the ink eroded).
  //
  // The diagrams show all the pixels that go into the cells.
  //////////////////////////////////////////////////////////////////////////////
  {
    const uint64_t o12 = (r1 | r2), o34 = (r3 | r4), o56 = (r5 | r6);
    const uint64_t a12 = (r1 & r2), a34 = (r3 & r4), a56 = (r5 & r6);
    //rows 1..6, ten cells across, dilated; then the same eroded; with W, WW and WWW
    //  6  xxxxxxxxxxx
    //  5  xxxxxxxxxxx
    //  4  xxxxxxxxxxx
    //  3  xxxxxxxxxxx
    //  2  xxxxxxxxxxx
    //  1  xxxxxxxxxxx
    //  0  ...xxx?
    mapL.set(hash(i++, ((o12 | o12 >> 1) >> 4 & 0x3ff) | ((o34 | o34 >> 1) >> 4 & 0x3ffull) << 10 | ((o56 | o56 >> 1) >> 4 & 0x3ffull) << 20,
      y | (r0 & 7) << 1));
    mapL.set(hash(i++, ((a12 & a12 >> 1) >> 4 & 0x3ff) | ((a34 & a34 >> 1) >> 4 & 0x3ffull) << 10 | ((a56 & a56 >> 1) >> 4 & 0x3ffull) << 20,
      y | (r0 & 7) << 1));

    //rows 1 and 2, thirteen cells across, both dilated and eroded: where there certainly is ink,
    //where there certainly is not, and the uncertain edge between them; with surrounding4
    //  2  xxxxxxxxxxxxxx
    //  1  xxxxxxxxxxxxxx
    //  0  .......x?
    mapL.set(hash(i++, (o12 | o12 >> 1) >> 3 & 0x1fff, (a12 & a12 >> 1) >> 3 & 0x1fff, f.surrounding4, y));

    //rows 1..12, eight cells across, dilated: tall enough for a whole letter; with W and WW
    // 12  xxxxxxxxx
    // 11  xxxxxxxxx
    // 10  xxxxxxxxx
    //  9  xxxxxxxxx
    //  8  xxxxxxxxx
    //  7  xxxxxxxxx
    //  6  xxxxxxxxx
    //  5  xxxxxxxxx
    //  4  xxxxxxxxx
    //  3  xxxxxxxxx
    //  2  xxxxxxxxx
    //  1  xxxxxxxxx
    //  0  ...xx?
    const uint64_t o78 = (r7 | r8), o9a = (r9 | r10), obc = (r11 | r12);
    mapL.set(hash(i++, ((o12 | o12 >> 1) >> 5 & 0xff) | ((o34 | o34 >> 1) >> 5 & 0xffull) << 8 | ((o56 | o56 >> 1) >> 5 & 0xffull) << 16,
      ((o78 | o78 >> 1) >> 5 & 0xff) | ((o9a | o9a >> 1) >> 5 & 0xffull) << 8 | ((obc | obc >> 1) >> 5 & 0xffull) << 16,
      y | (r0 & 3) << 1));

    //rows 1..6, ten cells across, each cell three pixels wide, dilated: the coarsest shape that
    //still tells letters apart; with surrounding4
    //  6  xxxxxxxxxxxx
    //  5  xxxxxxxxxxxx
    //  4  xxxxxxxxxxxx
    //  3  xxxxxxxxxxxx
    //  2  xxxxxxxxxxxx
    //  1  xxxxxxxxxxxx
    //  0  ......x?
    const uint64_t t12 = o12 | o12 >> 1 | o12 >> 2, t34 = o34 | o34 >> 1 | o34 >> 2, t56 = o56 | o56 >> 1 | o56 >> 2;
    mapL.set(hash(i++, (t12 >> 4 & 0x3ff) | (t34 >> 4 & 0x3ffull) << 10 | (t56 >> 4 & 0x3ffull) << 20, f.surrounding4, y));
  }

  //////////////////////////////////////////////////////////////////////////////
  // Long lines, the reference line, and the sparse templates
  //////////////////////////////////////////////////////////////////////////////

  //the steep and the flat slope to the left over 16 rows (see readDirections), how far the ink
  //runs along each, and W
  mapL.set(hash(i++, f.steepNW | f.steepNW16 << 8 | f.flatNW << 16 | f.flatNW16 << 24, (f.srunNW16 - 1) | (f.frunNW16 - 1) << 4, y));

  //the diagonal to the left over 16 rows, how far the ink runs along it, and surrounding4
  // 16  x.................
  // 15  .x................
  // 14  ..x...............
  // 13  ...x..............
  // 12  ....x.............
  // 11  .....x............
  // 10  ......x...........
  //  9  .......x..........
  //  8  ........x.........
  //  7  .........x........
  //  6  ..........x.......
  //  5  ...........x......
  //  4  ............x.....
  //  3  .............x....
  //  2  ..............x...
  //  1  ...............xxx
  //  0  ...............x?
  mapL.set(hash(i++, f.diagNW | f.diagNW16 << 8, (f.drunNW16 - 1) | f.surrounding4 << 4));

  //how far the ink runs up the columns -1..+1 (and the middle one over 16 rows), the columns
  //either side, and surrounding4
  mapL.set(hash(i++, (f.vrun - 1) | (f.vrunL - 1) << 3 | (f.vrunR - 1) << 6 | (f.vrun16 - 1) << 9,
    f.col8L | f.col8R << 8, f.surrounding4));

  //the next two colour changes on the row above (b1 and b2, see trackRuns), the current run and
  //surrounding12: where the run above ends, and what the pixels look like as we get there
  mapL.set(hash(i++, f.vb1 | f.vb2 << 5 | f.curCol << 10, f.runIdx, f.surrounding12));

  //b1 and the current run, with the two rows above, read wide
  //  2  .xxxxxxx.
  //  1  xxxxxxxxx
  //  0  ....?
  mapL.set(hash(i++, f.vb1 | f.curCol << 5 | f.runIdx << 6, r1 >> 4 & 0x1ff, r2 >> 5 & 0x7f));

  //the sparse templates G22 and H23
  // 11  x...............
  // 10  ...x.....x....x.
  //  9  ................
  //  8  ............x...
  //  7  ................
  //  6  ......x.x....x.x
  //  5  ................
  //  4  ................
  //  3  ........xx......
  //  2  ......x.x...x.x.
  //  1  ...x....xxx.....
  //  0  x...x..x?
  mapL.set(hash(i++, sparse(G22, 22)));
  // 14  ..........x.....
  // 13  ................
  // 12  .....x.......x..
  // 11  ................
  // 10  .x....x.x.......
  //  9  ................
  //  8  ................
  //  7  ................
  //  6  ..x.....x.x...x.
  //  5  ................
  //  4  ......x.x...x...
  //  3  ................
  //  2  ......x..x..x..x
  //  1  ........xxx.....
  //  0  x...x..x?
  mapL.set(hash(i++, sparse(H23, 23)));
}

//The mixer inputs that do not come from the hashed contexts.
void Image1BitModel::addInputs(Mixer& m) {
  //the indexed contexts: a prediction from the bit history, and one from the counts
  stateMap.subscribe();
  for (int i = 0; i < N; ++i) {
    const uint8_t state = t[cxt[i]];
    if (state == 0) {
      stateMap.skip(i);
      m.add(0);
      m.add(0);
    }
    else {
      int p1 = stateMap.p2(i, state);
      int st = stretch(p1) >> 1;
      m.add(st);
      int n0 = counts[cxt[i]] >> 8;
      int n1 = counts[cxt[i]] & 255;
      n0++;
      n1++;

      p1 = (n1 << 12) / (n0 + n1);
      st = stretch(p1) >> 1;
      m.add(st);
    }
  }

  //how much ink is around. The counts are passed the other way round (ink as 0s): in a dithered
  //image, much ink around predicts paper next, and the mixer learns the sign it needs.
  add(m, f.bitcount04, 04 - f.bitcount04);
  add(m, f.bitcount12, 12 - f.bitcount12);
  add(m, f.bitcount24, 24 - f.bitcount24);
  add(m, f.bitcount40, 40 - f.bitcount40);
  add(m, f.colCount, 8 - f.colCount);

  //a long run predicts more of the same. At the start of a row runLength is 0: no prediction.
  const uint32_t rl = runLength < 24 ? runLength : 24;
  add(m, runValue != 0 ? 0 : rl, runValue != 0 ? rl : 0);

  //the same for the runs up the column and along the two diagonals
  {
    const uint32_t vl = std::min(f.vrun16, 16u);
    add(m, (f.col8 & 1) != 0 ? 0 : vl, (f.col8 & 1) != 0 ? vl : 0);
    const uint32_t dl = std::min(f.drunNW16, 16u);
    add(m, (f.diagNW & 1) != 0 ? 0 : dl, (f.diagNW & 1) != 0 ? dl : 0);
    const uint32_t el = std::min(f.drunNE, 8u);
    add(m, (f.diagNE & 1) != 0 ? 0 : el, (f.diagNE & 1) != 0 ? el : 0);
  }

  //the match models: how far each one can be trusted is learned from its length, the pixel it
  //expects, and how closely its source resembled the current neighbourhood
  matchStateMap.subscribe();
  for (int k = 0; k < nMatch; ++k) {
    if (predictedBit[k] < 0) {
      matchStateMap.skip(k);
      m.add(0);
      continue;
    }
    const uint32_t exp = static_cast<uint32_t>(predictedBit[k]);
    const uint32_t ctx = matchBucketFine(matchLen[k]) | exp << 4 | matchResid[k] << 5; //7 bits
    m.add(stretch(matchStateMap.p2(k, ctx)) >> 1);
  }
}

//A direct predictor: the probability that the next pixel is 1, from how many 0s (n0) and 1s (n1)
//speak for each, given to the mixer in two forms.
void Image1BitModel::add(Mixer& m, uint32_t n0, uint32_t n1) {
  int p1 = ((n1 + 1) << 12) / ((n0 + n1) + 2);
  const int a = stretch(p1) >> 1;
  const int b = (p1 - 2048) >> 2;
  m.add(a); m.add(b);
}

//The mixer weight sets: each line selects one set of weights by what it describes.
void Image1BitModel::setMixerContexts(Mixer& m, const int y, const int bpos) {
  m.set(f.surrounding4, 16);
  m.set(f.mCtx6, 64);
  m.set(f.bitcount40, 41);                                                      //how much ink is around
  m.set(f.runIdx | runValue << 4, 32);                                          //the current run
  m.set((f.vrun - 1) | (f.col8 & 1) << 3 | y << 4 | (r1 >> 7 & 1) << 5, 64);    //the run up the column, N, W and NE
  m.set((bpos & 3) | (yRow & 3) << 2, 16);                                      //the dither phase
  m.set(f.matchVote | (f.lenBucketA >> 2) << 4 | (f.lenBucketB >> 2) << 5, 64); //match models A and B: the pixel they expect, and whether their match is long
  m.set((f.vrun - 1) | (f.vrunL - 1) << 3 | (f.vrunR - 1) << 6, 512);           //the runs up the columns -1..+1
  m.set((f.drunNW - 1) | (f.drunNE - 1) << 3 | (f.diagNW & 1) << 6 | (f.diagNE & 1) << 7, 256); //the runs along the diagonals
  m.set(f.runIdx | runBucket(prevRun1) << 4, 256);                              //the current run and the one before it
  m.set(bitCount(f.dil6) * 7 + bitCount(f.ero6), 49);                           //how much of the blurred shape above is ink
  //the nearest pixels:
  //  1  ......xx
  //  0  xxxxxx?
  m.set(static_cast<uint32_t>((r0 & 0x3f) | (r1 >> 7 & 3) << 6), 256);
  m.set(f.matchMode, 33);                                                       //the match models' verdict
}

//Hand the SSE stage a summary of the neighbourhood (see Shared::State.Image1).
void Image1BitModel::publishToSSE() {
  auto& im = shared->State.Image1;
  im.ctx12 = f.surrounding12;
  im.shape = (f.drunNW - 1) | (f.drunNE - 1) << 3 | (f.vrun - 1) << 6 | (f.srunNW - 1) << 9 |
    (f.srunNE - 1) << 12 | (f.frunNW - 1) << 15 | (f.vrun16 - 1) << 18;
  im.column = f.col8 | f.col16 << 8;
  im.rowAbove = static_cast<uint32_t>(r1 >> 1 & 0x7fff);
  im.run = static_cast<uint8_t>(f.runIdx | runValue << 4);
  im.ref = static_cast<uint8_t>(f.vb1 | f.curCol << 5);
  im.ink = static_cast<uint8_t>(f.bitcount40);
  im.match = static_cast<uint8_t>(f.matchMode);
}
