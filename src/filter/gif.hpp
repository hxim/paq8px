#pragma once

#include "../Array.hpp"
#include "Filter.hpp"
#include <cstdint>


#define LZW_TABLE_SIZE 9221

#define LZW_FIND(k) \
  { \
    offset     = ( int ) finalize64((k) * PHI64, 13); \
    int stride = (offset > 0) ? LZW_TABLE_SIZE - offset : 1; \
    while (true) { \
      if ((index = table[offset]) < 0) { \
        index = -offset - 1; \
        break; \
      } else if (dict[index] == int(k)) { \
        break; \
      } \
      offset -= stride; \
      if (offset < 0) \
        offset += LZW_TABLE_SIZE; \
    } \
  }

#define LZW_RESET \
  { \
    for (int i = 0; i < LZW_TABLE_SIZE; table[i] = -1, i++) \
      ; \
  }

static int encodeGif(File *in, File *out, uint64_t len, int &headerSize) {
  int codeSize = in->getchar();
  int diffPos = 0;
  int clearPos = 0;
  int bsize = 0;
  int code = 0;
  int offset = 0;
  uint64_t beginIn = in->curPos();
  uint64_t beginOut = out->curPos();
  Array<uint8_t> output(4096);
  headerSize = 6;
  out->putChar(headerSize >> 8);
  out->putChar(headerSize & 255);
  out->putChar(bsize);
  out->putChar(clearPos >> 8);
  out->putChar(clearPos & 255);
  out->putChar(codeSize);
  Array<int> table(LZW_TABLE_SIZE);
  for( int phase = 0; phase < 2; phase++ ) {
    in->setpos(beginIn);
    int bits = codeSize + 1;
    int shift = 0;
    int buffer = 0;
    int blockSize = 0;
    int maxcode = (1u << codeSize) + 1;
    int last = -1;
    Array<int> dict(4096);
    LZW_RESET
    bool end = false;
    while((blockSize = in->getchar()) > 0 && in->curPos() - beginIn < len && !end ) {
      for( int i = 0; i < blockSize; i++ ) {
        buffer |= in->getchar() << shift;
        shift += 8;
        while( shift >= bits && !end ) {
          code = buffer & ((1u << bits) - 1);
          buffer >>= bits;
          shift -= bits;
          if((bsize == 0) && code != (1u << codeSize)) {
            headerSize += 4;
            out->put32(0);
          }
          if( bsize == 0 ) {
            bsize = blockSize;
          }
          if( code == (1 << codeSize)) {
            if( maxcode > (1 << codeSize) + 1 ) {
              if((clearPos != 0) && clearPos != 69631 - maxcode ) {
                return 0;
              }
              clearPos = 69631 - maxcode;
            }
            bits = codeSize + 1, maxcode = (1u << codeSize) + 1, last = -1;
            LZW_RESET
          } else if( code == (1u << codeSize) + 1 ) {
            end = true;
          } else if( code > maxcode + 1 ) {
            return 0;
          } else {
            int j = (code <= maxcode ? code : last);
            int size = 1;
            while( j >= (1 << codeSize)) {
              output[4096 - (size++)] = dict[j] & 255;
              j = dict[j] >> 8;
            }
            output[4096 - size] = j;
            if( phase == 1 ) {
              out->blockWrite(&output[4096 - size], size);
            } else {
              diffPos += size;
            }
            if( code == maxcode + 1 ) {
              if( phase == 1 ) {
                out->putChar(j);
              } else {
                diffPos++;
              }
            }
            if( last != -1 ) {
              if( ++maxcode >= 8191 ) {
                return 0;
              }
              if( maxcode <= 4095 ) {
                int key = (static_cast<uint32_t>(last) << 8) + j;
                int index = -1;
                LZW_FIND(key)
                dict[maxcode] = key;
                table[(index < 0) ? -index - 1 : offset] = maxcode;
                if( phase == 0 && index > 0 ) {
                  headerSize += 4;
                  j = diffPos - size - static_cast<int>(code == maxcode);
                  out->put32(j);
                  diffPos = size + static_cast<int>(code == maxcode);
                }
              }
              if( maxcode >= ((1 << bits) - 1) && bits < 12 ) {
                bits++;
              }
            }
            last = code;
          }
        }
      }
    }
  }
  diffPos = static_cast<int>(out->curPos());
  out->setpos(beginOut);
  out->putChar(headerSize >> 8);
  out->putChar(headerSize & 255);
  out->putChar(255 - bsize);
  out->putChar((clearPos >> 8) & 255);
  out->putChar(clearPos & 255);
  out->setpos(diffPos);
  return static_cast<int>(in->curPos() - beginIn == len - 1);
}

#define GIF_WRITE_BLOCK(count) \
  { \
    output[0] = (count); \
    if (mode == FMode::FDECOMPRESS) \
      out->blockWrite(&output[0], (count) + 1); \
    else if (mode == FMode::FCOMPARE) \
      for (int j = 0; j < (count) + 1; j++) \
        if (output[j] != out->getchar() && ! diffFound) { \
          diffFound = outsize + j + 1; \
          return 1; \
        } \
    outsize += (count) + 1; \
    blockSize = 0; \
  }

#define GIF_WRITE_CODE(c) \
  { \
    buffer += (c) << shift; \
    shift += bits; \
    while (shift >= 8) { \
      output[++blockSize] = buffer & 255; \
      buffer >>= 8; \
      shift -= 8; \
      if (blockSize == bsize) GIF_WRITE_BLOCK(bsize); \
    } \
  }

static int decodeGif(File *in, uint64_t size, File *out, FMode mode, uint64_t &diffFound) {
  int diffCount = in->getchar();
  int curDiff = 0;
  Array<int> diffPos(4096);
  diffCount = ((diffCount << 8) + in->getchar() - 6) / 4;
  int bsize = 255 - in->getchar();
  int clearPos = in->getchar();
  clearPos = (clearPos << 8) + in->getchar();
  clearPos = (69631 - clearPos) & 0xffff;
  int codesize = in->getchar();
  int bits = codesize + 1;
  int shift = 0;
  int buffer = 0;
  int blockSize = 0;
  if( diffCount > 4096 || clearPos <= (1 << codesize) + 2 ) {
    return 1;
  }
  int maxcode = (1u << codesize) + 1;
  int input = 0;
  int code = 0;
  int offset = 0;
  Array<int> dict(4096);
  Array<int> table(LZW_TABLE_SIZE);
  LZW_RESET
  for( int i = 0; i < diffCount; i++ ) {
    diffPos[i] = in->getchar();
    diffPos[i] = (diffPos[i] << 8) + in->getchar();
    diffPos[i] = (diffPos[i] << 8) + in->getchar();
    diffPos[i] = (diffPos[i] << 8) + in->getchar();
    if( i > 0 ) {
      diffPos[i] += diffPos[i - 1];
    }
  }
  Array<uint8_t> output(256);
  size -= 6 + diffCount * 4;
  int last = in->getchar();
  int total = static_cast<int>(size) + 1;
  int outsize = 1;
  if( mode == FMode::FDECOMPRESS ) {
    out->putChar(codesize);
  } else if( mode == FMode::FCOMPARE ) {
    if( codesize != out->getchar() && (diffFound == 0)) {
      diffFound = 1;
    }
  }
  if( diffCount == 0 || diffPos[0] != 0 ) GIF_WRITE_CODE(1u << codesize) else {
    curDiff++;
  }
  while( size != 0 && (input = in->getchar()) != EOF) {
    size--;
    int key = (static_cast<uint32_t>(last) << 8) + input;
    int index = (code = -1);
    if( last < 0 ) {
      index = input;
    } else LZW_FIND(key)
    code = index;
    if( curDiff < diffCount && total - static_cast<int>(size) > diffPos[curDiff] ) {
      curDiff++, code = -1;
    }
    if( code < 0 ) {
      GIF_WRITE_CODE(last)
      if( maxcode == clearPos ) {
        GIF_WRITE_CODE(1u << codesize)
        bits = codesize + 1, maxcode = (1u << codesize) + 1;
        LZW_RESET
      } else {
        ++maxcode;
        if( maxcode <= 4095 ) {
          dict[maxcode] = key;
          table[(index < 0) ? -index - 1 : offset] = maxcode;
        }
        if( maxcode >= (1 << bits) && bits < 12 ) {
          bits++;
        }
      }
      code = input;
    }
    last = code;
  }
  GIF_WRITE_CODE(last)
  GIF_WRITE_CODE((1u << codesize) + 1)
  if( shift > 0 ) {
    output[++blockSize] = buffer & 255;
    if( blockSize == bsize ) GIF_WRITE_BLOCK(bsize)
  }
  if( blockSize > 0 ) GIF_WRITE_BLOCK(blockSize)
  if( mode == FMode::FDECOMPRESS ) {
    out->putChar(0);
  } else if( mode == FMode::FCOMPARE ) {
    if( 0 != out->getchar() && (diffFound == 0)) {
      diffFound = outsize + 1;
    }
  }
  return outsize + 1;
}
