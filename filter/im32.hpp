#pragma once

#include "../Encoder.hpp"
#include "../file/File.hpp"
#include "Filter.hpp"
#include <cstdint>

// 32-bit image
static void encodeIm32(File *in, File *out, uint64_t len, int width, bool skipRgb) {
  int r = 0;
  int g = 0;
  int b = 0;
  int a = 0;
  for( int i = 0; i < static_cast<int>(len / width); i++ ) {
    for( int j = 0; j < width / 4; j++ ) {
      b = in->getchar();
      g = in->getchar();
      r = in->getchar();
      a = in->getchar();
      out->putChar(g);
      out->putChar(skipRgb ? r : g - r);
      out->putChar(skipRgb ? b : g - b);
      out->putChar(a);
    }
    for( int j = 0; j < width % 4; j++ ) {
      out->putChar(in->getchar());
    }
  }
  for( int i = len % width; i > 0; i-- ) {
    out->putChar(in->getchar());
  }
}

static auto decodeIm32(Encoder &en, uint64_t size, int width, File *out, FMode mode, uint64_t &diffFound, bool skipRgb) -> uint64_t {
  int r = 0;
  int g = 0;
  int b = 0;
  int a = 0;
  int p = 0;
  bool rgb = (width & (1U << 31U)) > 0;
  if( rgb ) {
    width ^= (1U << 31U);
  }
  for( int i = 0; i < static_cast<int>(size / width); i++ ) {
    p = i * width;
    for( int j = 0; j < width / 4; j++ ) {
      b = en.decompressByte(&en.predictorMain), g = en.decompressByte(&en.predictorMain), r = en.decompressByte(&en.predictorMain), a = en.decompressByte(&en.predictorMain);
      if( mode == FDECOMPRESS ) {
        out->putChar(skipRgb ? r : b - r);
        out->putChar(b);
        out->putChar(skipRgb ? g : b - g);
        out->putChar(a);
        if((j == 0) && ((i & 0xf) == 0)) {
          en.printStatus();
        }
      } else if( mode == FCOMPARE ) {
        if(((skipRgb ? r : b - r) & 255U) != out->getchar() && (diffFound == 0u)) {
          diffFound = p + 1;
        }
        if( b != out->getchar() && (diffFound == 0u)) {
          diffFound = p + 2;
        }
        if(((skipRgb ? g : b - g) & 255U) != out->getchar() && (diffFound == 0u)) {
          diffFound = p + 3;
        }
        if(((a) & 255) != out->getchar() && (diffFound == 0u)) {
          diffFound = p + 4;
        }
        p += 4;
      }
    }
    for( int j = 0; j < width % 4; j++ ) {
      if( mode == FDECOMPRESS ) {
        out->putChar(en.decompressByte(&en.predictorMain));
      } else if( mode == FCOMPARE ) {
        if( en.decompressByte(&en.predictorMain) != out->getchar() && (diffFound == 0u)) {
          diffFound = p + j + 1;
        }
      }
    }
  }
  for( int i = size % width; i > 0; i-- ) {
    if( mode == FDECOMPRESS ) {
      out->putChar(en.decompressByte(&en.predictorMain));
    } else if( mode == FCOMPARE ) {
      if( en.decompressByte(&en.predictorMain) != out->getchar() && (diffFound == 0u)) {
        diffFound = size - i;
        break;
      }
    }
  }
  return size;
}
