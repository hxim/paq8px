#ifndef PAQ8PX_ZLIB_HPP
#define PAQ8PX_ZLIB_HPP

#include "Filters.hpp"
#include "../utils.hpp"
#include <zlib.h>

static auto parseZlibHeader(int header) -> int {
  switch( header ) {
    case 0x2815:
      return 0;
    case 0x2853:
      return 1;
    case 0x2891:
      return 2;
    case 0x28cf:
      return 3;
    case 0x3811:
      return 4;
    case 0x384f:
      return 5;
    case 0x388d:
      return 6;
    case 0x38cb:
      return 7;
    case 0x480d:
      return 8;
    case 0x484b:
      return 9;
    case 0x4889:
      return 10;
    case 0x48c7:
      return 11;
    case 0x5809:
      return 12;
    case 0x5847:
      return 13;
    case 0x5885:
      return 14;
    case 0x58c3:
      return 15;
    case 0x6805:
      return 16;
    case 0x6843:
      return 17;
    case 0x6881:
      return 18;
    case 0x68de:
      return 19;
    case 0x7801:
      return 20;
    case 0x785e:
      return 21;
    case 0x789c:
      return 22;
    case 0x78da:
      return 23;
    default:
      return -1;
  }
}

static auto zlibInflateInit(z_streamp strm, int zh) -> int {
  if( zh == -1 ) {
    return inflateInit2(strm, -MAX_WBITS);
  }
  return inflateInit(strm);
}

MTFList mtf(81);

static auto encodeZlib(File *in, File *out, uint64_t len, int &headerSize) -> int {
  const int block = 1U << 16U;
  const int limit = 128;
  uint8_t zin[block * 2];
  uint8_t zOut[block];
  uint8_t zRec[block * 2];
  uint8_t diffByte[81 * limit];
  uint64_t diffPos[81 * limit];

  // Step 1 - parse offset type form zlib stream header
  uint64_t posBackup = in->curPos();
  uint32_t h1 = in->getchar();
  uint32_t h2 = in->getchar();
  in->setpos(posBackup);
  int zh = parseZlibHeader(h1 * 256 + h2);
  int memLevel = 0;
  int cLevel = 0;
  int cType = zh % 4;
  int window = zh == -1 ? 0 : MAX_WBITS + 10 + zh / 4;
  int minCLevel = window == 0 ? 1 : cType == 3 ? 7 : cType == 2 ? 6 : cType == 1 ? 2 : 1;
  int maxCLevel = window == 0 ? 9 : cType == 3 ? 9 : cType == 2 ? 6 : cType == 1 ? 5 : 1;
  int index = -1;
  int nTrials = 0;
  bool found = false;

  // Step 2 - check recompressibility, determine parameters and save differences
  z_stream mainStrm;
  z_stream recStrm[81];
  int diffCount[81];
  int recPos[81];
  int mainRet = Z_STREAM_END;
  mainStrm.zalloc = Z_NULL;
  mainStrm.zfree = Z_NULL;
  mainStrm.opaque = Z_NULL;
  mainStrm.next_in = Z_NULL;
  mainStrm.avail_in = 0;
  if( zlibInflateInit(&mainStrm, zh) != Z_OK ) {
    return 0;
  }
  for( int i = 0; i < 81; i++ ) {
    cLevel = (i / 9) + 1;
    // Early skip if invalid parameter
    if( cLevel < minCLevel || cLevel > maxCLevel ) {
      diffCount[i] = limit;
      continue;
    }
    memLevel = (i % 9) + 1;
    recStrm[i].zalloc = Z_NULL;
    recStrm[i].zfree = Z_NULL;
    recStrm[i].opaque = Z_NULL;
    recStrm[i].next_in = Z_NULL;
    recStrm[i].avail_in = 0;
    int ret = deflateInit2(&recStrm[i], cLevel, Z_DEFLATED, window - MAX_WBITS, memLevel, Z_DEFAULT_STRATEGY);
    diffCount[i] = (ret == Z_OK) ? 0 : limit;
    recPos[i] = block * 2;
    diffPos[i * limit] = 0xFFFFFFFFFFFFFFFF;
    diffByte[i * limit] = 0;
  }

  for( uint64_t i = 0; i < len; i += block ) {
    uint32_t blSize = min(uint32_t(len - i), block);
    nTrials = 0;
    for( int j = 0; j < 81; j++ ) {
      if( diffCount[j] >= limit ) {
        continue;
      }
      nTrials++;
      if( recPos[j] >= block ) {
        recPos[j] -= block;
      }
    }
    // early break if nothing left to test
    if( nTrials == 0 ) {
      break;
    }
    memmove(&zRec[0], &zRec[block], block);
    memmove(&zin[0], &zin[block], block);
    in->blockRead(&zin[block], blSize); // Read block from input file

    // Decompress/inflate block
    mainStrm.next_in = &zin[block];
    mainStrm.avail_in = blSize;
    do {
      mainStrm.next_out = &zOut[0];
      mainStrm.avail_out = block;
      mainRet = inflate(&mainStrm, Z_FINISH);
      nTrials = 0;

      // Recompress/deflate block with all possible parameters
      for( int j = mtf.getFirst(); j >= 0; j = mtf.getNext()) {
        if( diffCount[j] >= limit ) {
          continue;
        }
        nTrials++;
        recStrm[j].next_in = &zOut[0];
        recStrm[j].avail_in = block - mainStrm.avail_out;
        recStrm[j].next_out = &zRec[recPos[j]];
        recStrm[j].avail_out = block * 2 - recPos[j];
        int ret = deflate(&recStrm[j], mainStrm.total_in == len ? Z_FINISH : Z_NO_FLUSH);
        if( ret != Z_BUF_ERROR && ret != Z_STREAM_END && ret != Z_OK ) {
          diffCount[j] = limit;
          continue;
        }

        // Compare
        int end = 2 * block - static_cast<int>(recStrm[j].avail_out);
        int tail = max(mainRet == Z_STREAM_END ? static_cast<int>(len) - static_cast<int>(recStrm[j].total_out) : 0, 0);
        for( int k = recPos[j]; k < end + tail; k++ ) {
          if((k < end && i + k - block < len && zRec[k] != zin[k]) || k >= end ) {
            if( ++diffCount[j] < limit ) {
              const int p = j * limit + diffCount[j];
              diffPos[p] = i + k - block;
              assert(k < int(sizeof(zin) / sizeof(*zin)));
              diffByte[p] = zin[k];
            }
          }
        }
        // Early break on perfect match
        if( mainRet == Z_STREAM_END && diffCount[j] == 0 ) {
          index = j;
          found = true;
          break;
        }
        recPos[j] = 2U * block - recStrm[j].avail_out;
      }
    } while( mainStrm.avail_out == 0 && mainRet == Z_BUF_ERROR && nTrials > 0 );
    if((mainRet != Z_BUF_ERROR && mainRet != Z_STREAM_END) || nTrials == 0 ) {
      break;
    }
  }
  int minCount = (found) ? 0 : limit;
  for( int i = 80; i >= 0; i-- ) {
    cLevel = (i / 9) + 1;
    if( cLevel >= minCLevel && cLevel <= maxCLevel ) {
      deflateEnd(&recStrm[i]);
    }
    if( !found && diffCount[i] < minCount ) {
      minCount = diffCount[index = i];
    }
  }
  inflateEnd(&mainStrm);
  if( minCount == limit ) {
    return 0;
  }
  mtf.moveToFront(index);

  // Step 3 - write parameters, differences and precompressed (inflated) data
  out->putChar(diffCount[index]);
  out->putChar(window);
  out->putChar(index);
  for( int i = 0; i <= diffCount[index]; i++ ) {
    const int v = i == diffCount[index] ? int(len - diffPos[index * limit + i]) :
                  int(diffPos[index * limit + i + 1] - diffPos[index * limit + i]) - 1;
    out->put32(v);
  }
  for( int i = 0; i < diffCount[index]; i++ ) {
    out->putChar(diffByte[index * limit + i + 1]);
  }

  in->setpos(posBackup);
  mainStrm.zalloc = Z_NULL;
  mainStrm.zfree = Z_NULL;
  mainStrm.opaque = Z_NULL;
  mainStrm.next_in = Z_NULL;
  mainStrm.avail_in = 0;
  if( zlibInflateInit(&mainStrm, zh) != Z_OK ) {
    return 0;
  }
  for( uint64_t i = 0; i < len; i += block ) {
    uint32_t blSize = min(uint32_t(len - i), block);
    in->blockRead(&zin[0], blSize);
    mainStrm.next_in = &zin[0];
    mainStrm.avail_in = blSize;
    do {
      mainStrm.next_out = &zOut[0];
      mainStrm.avail_out = block;
      mainRet = inflate(&mainStrm, Z_FINISH);
      out->blockWrite(&zOut[0], block - mainStrm.avail_out);
    } while( mainStrm.avail_out == 0 && mainRet == Z_BUF_ERROR);
    if( mainRet != Z_BUF_ERROR && mainRet != Z_STREAM_END ) {
      break;
    }
  }
  inflateEnd(&mainStrm);
  headerSize = diffCount[index] * 5 + 7;
  return static_cast<int>(mainRet == Z_STREAM_END);
}

static auto decodeZlib(File *in, uint64_t size, File *out, FMode mode, uint64_t &diffFound) -> int {
  const int block = 1U << 16U;
  const int limit = 128;
  uint8_t zin[block];
  uint8_t zOut[block];
  int diffCount = min(in->getchar(), limit - 1);
  int window = in->getchar() - MAX_WBITS;
  int index = in->getchar();
  int memLevel = (index % 9) + 1;
  int cLevel = (index / 9) + 1;
  int len = 0;
  int diffPos[limit];
  diffPos[0] = -1;
  for( int i = 0; i <= diffCount; i++ ) {
    int v = in->get32();
    if( i == diffCount ) {
      len = v + diffPos[i];
    } else {
      diffPos[i + 1] = v + diffPos[i] + 1;
    }
  }
  uint8_t diffByte[limit];
  diffByte[0] = 0;
  for( int i = 0; i < diffCount; i++ ) {
    diffByte[i + 1] = in->getchar();
  }
  size -= 7 + 5 * diffCount;

  z_stream recStrm;
  int diffIndex = 1;
  int recPos = 0;
  recStrm.zalloc = Z_NULL;
  recStrm.zfree = Z_NULL;
  recStrm.opaque = Z_NULL;
  recStrm.next_in = Z_NULL;
  recStrm.avail_in = 0;
  int ret = deflateInit2(&recStrm, cLevel, Z_DEFLATED, window, memLevel, Z_DEFAULT_STRATEGY);
  if( ret != Z_OK ) {
    return 0;
  }
  for( uint64_t i = 0; i < size; i += block ) {
    uint32_t blSize = min(uint32_t(size - i), block);
    in->blockRead(&zin[0], blSize);
    recStrm.next_in = &zin[0];
    recStrm.avail_in = blSize;
    do {
      recStrm.next_out = &zOut[0];
      recStrm.avail_out = block;
      ret = deflate(&recStrm, i + blSize == size ? Z_FINISH : Z_NO_FLUSH);
      if( ret != Z_BUF_ERROR && ret != Z_STREAM_END && ret != Z_OK ) {
        break;
      }
      const int have = min(block - recStrm.avail_out, len - recPos);
      while( diffIndex <= diffCount && diffPos[diffIndex] >= recPos && diffPos[diffIndex] < recPos + have ) {
        zOut[diffPos[diffIndex] - recPos] = diffByte[diffIndex];
        diffIndex++;
      }
      if( mode == FDECOMPRESS ) {
        out->blockWrite(&zOut[0], have);
      } else if( mode == FCOMPARE ) {
        for( int j = 0; j < have; j++ ) {
          if( zOut[j] != out->getchar() && (diffFound == 0u)) {
            diffFound = recPos + j + 1;
          }
        }
      }
      recPos += have;

    } while( recStrm.avail_out == 0 );
  }
  while( diffIndex <= diffCount ) {
    if( mode == FDECOMPRESS ) {
      out->putChar(diffByte[diffIndex]);
    } else if( mode == FCOMPARE ) {
      if( diffByte[diffIndex] != out->getchar() && (diffFound == 0u)) {
        diffFound = recPos + 1;
      }
    }
    diffIndex++;
    recPos++;
  }
  deflateEnd(&recStrm);
  return recPos == len ? len : 0;
}

#endif //PAQ8PX_ZLIB_HPP
