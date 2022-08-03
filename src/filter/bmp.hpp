#pragma once

#include "Filter.hpp"
#include "../file/File.hpp"
#include <cstdint>

/**
 * 24/32-bit image data transforms, controlled by OPTION_SKIPRGB:
 *   - simple color transform (b, g, r) -> (g, g-r, g-b)
 *   - channel reorder only (b, g, r) -> (g, r, b)
 * Detects RGB565 to RGB888 conversions.
 */
class BmpFilter : public Filter {
private:
  int stride = 3; //3: RGB or BGR, 4: RGBA or BGRA
  int width = 0;
  bool skipRgb = false;
  bool isPossibleRgb565 = true;
  uint32_t rgb565Run = 0;
  static constexpr int rgb565MinRun = 63;
public:

  void setWidth(int w) {
    this->width = w;
  }
  void setSkipRgb(bool skipRgb) {
    this->skipRgb = skipRgb;
  }
  void setHasApha() {
    this->stride = 4;
    this->isPossibleRgb565 = false; //to fix false positives (and to keep compatibility with previous version)
  }

  void encode(File *in, File *out, uint64_t size, int width, int & /*headerSize*/) override {
    uint32_t r = 0;
    uint32_t g = 0;
    uint32_t b = 0;
    for( int i = 0; i < static_cast<int>(size / width); i++ ) {
      for( int j = 0; j < width / stride; j++ ) {
        b = in->getchar();
        g = in->getchar();
        r = in->getchar();
        if( isPossibleRgb565 ) {
          int rgb565RunPrevious = rgb565Run;
          rgb565Run = min(rgb565Run + 1, 0xFFFF) *
                  static_cast<int>((b & 7) == ((b & 8) - ((b >> 3) & 1)) && (g & 3) == ((g & 4) - ((g >> 2) & 1)) &&
                                    (r & 7) == ((r & 8) - ((r >> 3) & 1)));
          if( rgb565Run > rgb565MinRun || rgb565RunPrevious >= rgb565MinRun ) {
            b ^= (b & 8) - ((b >> 3) & 1);
            g ^= (g & 4) - ((g >> 2) & 1);
            r ^= (r & 8) - ((r >> 3) & 1);
          }
          isPossibleRgb565 = rgb565Run > 0;
        }
        out->putChar(g);
        out->putChar(skipRgb ? r : g - r);
        out->putChar(skipRgb ? b : g - b);
        if (stride == 4) {
          out->putChar(in->getchar());
        }
      }
      for( int j = 0; j < width % stride; j++ ) {
        out->putChar(in->getchar());
      }
    }
    for( int i = size % width; i > 0; i-- ) {
      out->putChar(in->getchar());
    }
  }

  uint64_t decode(File * /*in*/, File *out, FMode fMode, uint64_t size, uint64_t &diffFound) override {
    uint32_t r = 0;
    uint32_t g = 0;
    uint32_t b = 0;
    uint32_t a = 0;
    uint32_t p = 0;
    for( int i = 0; i < static_cast<int>(size / width); i++ ) {
      p = i * width;
      for( int j = 0; j < width / stride; j++ ) {
        g = encoder->decompressByte(&encoder->predictorMain);
        r = encoder->decompressByte(&encoder->predictorMain);
        b = encoder->decompressByte(&encoder->predictorMain);
        if (stride == 4) {
          a = encoder->decompressByte(&encoder->predictorMain);
        }
        if( !skipRgb ) {
          r = g - r, b = g - b;
        }
        if( isPossibleRgb565 ) {
          if( rgb565Run >= rgb565MinRun ) {
            b ^= (b & 8) - ((b >> 3) & 1);
            g ^= (g & 4) - ((g >> 2) & 1);
            r ^= (r & 8) - ((r >> 3) & 1);
          }
          rgb565Run = min(rgb565Run + 1, 0xFFFF) *
                  static_cast<uint32_t>((b & 7) == ((b & 8) - ((b >> 3) & 1)) && (g & 3) == ((g & 4) - ((g >> 2) & 1)) &&
                                        (r & 7) == ((r & 8) - ((r >> 3) & 1)));
          isPossibleRgb565 = rgb565Run > 0;
        }
        if( fMode == FMode::FDECOMPRESS ) {
          out->putChar(b);
          out->putChar(g);
          out->putChar(r);
          if (stride == 4) {
            out->putChar(a);
          }
          if((j == 0) && ((i & 0xF) == 0)) {
            encoder->printStatus();
          }
        } else if( fMode == FMode::FCOMPARE ) {
          if((b & 255) != out->getchar() && (diffFound == 0)) {
            diffFound = p + 1;
          }
          if( g != out->getchar() && (diffFound == 0)) {
            diffFound = p + 2;
          }
          if((r & 255) != out->getchar() && (diffFound == 0)) {
            diffFound = p + 3;
          }
          if (stride == 4) {
            if ((a & 255) != out->getchar() && (diffFound == 0)) {
              diffFound = p + 4;
            }
          }
          p += stride;
        }
      }
      for( int j = 0; j < width % stride; j++ ) {
        if( fMode == FMode::FDECOMPRESS ) {
          out->putChar(encoder->decompressByte(&encoder->predictorMain));
        } else if( fMode == FMode::FCOMPARE ) {
          if( encoder->decompressByte(&encoder->predictorMain) != out->getchar() && (diffFound == 0)) {
            diffFound = p + j + 1;
          }
        }
      }
    }
    for( int i = size % width; i > 0; i-- ) {
      if( fMode == FMode::FDECOMPRESS ) {
        out->putChar(encoder->decompressByte(&encoder->predictorMain));
      } else if( fMode == FMode::FCOMPARE ) {
        if( encoder->decompressByte(&encoder->predictorMain) != out->getchar() && (diffFound == 0)) {
          diffFound = size - i;
          break;
        }
      }
    }
    return size;
  }
};
