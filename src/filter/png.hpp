#pragma once

#include "Filter.hpp"
#include "../file/File.hpp"
#include <cstdint>

/**
 * 8/24/32-bit png image data encode/decode
 * filter bytes from individual lines go to a separate header
 */
class PngFilter : public Filter {
private:
  int stride = 3; //1: Gray/Indexed, 3: RGB, 4: RGBA
  int width = 0;
public:

  void setWidth(int w) {
    this->width = w;
  }
  void setStride(int stride) {
    this->stride = stride;
  }

  void encode(File *in, File *out, uint64_t size, int width, int & headerSize) override {
    int lineWidth = width + 1; //including filter byte
    headerSize = static_cast<int>(size / lineWidth); // = number of rows
    RingBuffer<uint8_t> filterBuffer(nextPowerOf2(headerSize));
    RingBuffer<uint8_t> pixelBuffer(nextPowerOf2(size - headerSize));
    assert(filterBuffer.size() >= headerSize);
    assert(pixelBuffer.size() >= size - headerSize);
    for( int line = 0; line < headerSize; line++ ) {
      uint8_t filter = in->getchar();
      filterBuffer.add(filter);
      for (int x = 0; x < width; x++) {
        uint8_t c1 = in->getchar();
        switch (filter) {
          case 0: {
            break;
          }
          case 1: {
            c1=(static_cast<uint8_t>(c1 + (x < stride ? 0 : pixelBuffer(stride))));
            break;
          }
          case 2: {
            c1=(static_cast<uint8_t>(c1 + (line == 0 ? 0 : pixelBuffer(width))));
            break;
          }
          case 3: {
            c1 = (static_cast<uint8_t>(c1 + (((line == 0 ? 0 : pixelBuffer(width)) + (x < stride ? 0 : pixelBuffer(stride))) >> 1)));
            break;
          }
          case 4: {
            c1 = (static_cast<uint8_t>(c1 + paeth(
              x < stride ? 0 : pixelBuffer(stride),
              line == 0 ? 0 : pixelBuffer(width),
              line == 0 || x < stride ? 0 : pixelBuffer(width + stride))));
            break;
          }
          default:
            //fail: unexpected filter code.
            return;
        }
        pixelBuffer.add(c1);
      }
    }
    uint32_t len1 = filterBuffer.getpos();
    uint32_t len2 = pixelBuffer.getpos();
    for (uint32_t i = 0; i < len1; i++)
      out->putChar(filterBuffer[i]);
    for (uint32_t i = 0; i < len2; i++)
      out->putChar(pixelBuffer[i]);
  }

  uint64_t decode(File *in, File *out, FMode fMode, uint64_t size, uint64_t &diffFound) override {
    int lineWidth = width + 1; //including filter byte
    int headerSize = static_cast<int>(size / lineWidth); // = number of rows
    RingBuffer<uint8_t> filterBuffer(nextPowerOf2(headerSize));
    RingBuffer<uint8_t> pixelBuffer(nextPowerOf2(size - headerSize));
    assert(filterBuffer.size() >= headerSize);
    assert(pixelBuffer.size() >= size - headerSize);
    for (int line = 0; line < headerSize; line++) {
      uint8_t filter = in->getchar();
      filterBuffer.add(filter);
    }
    uint32_t p = 0;
    for (int line = 0; line < headerSize; line++) {
      uint8_t filter = filterBuffer[line];
      if (fMode == FMode::FDECOMPRESS) {
        out->putChar(filter);
      }
      else if (fMode == FMode::FCOMPARE) {
        p++;
        if (filter != out->getchar() && (diffFound == 0)) {
          diffFound = p;
        }
      }
      for (int x = 0; x < width; x++) {
        uint8_t c1 = in->getchar();
        uint8_t c = c1;
        switch (filter) {
          case 0: {
            break;
          }
          case 1: {
            c1 = (static_cast<uint8_t>(c1 - (x < stride ? 0 : pixelBuffer(stride))));
            break;
          }
          case 2: {
            c1 = (static_cast<uint8_t>(c1 - (line == 0 ? 0 : pixelBuffer(width))));
            break;
          }
          case 3: {
            c1 = (static_cast<uint8_t>(c1 - (((line == 0 ? 0 : pixelBuffer(width)) + (x < stride ? 0 : pixelBuffer(stride))) >> 1)));
            break;
          }
          case 4: {
            c1 = (static_cast<uint8_t>(c1 - paeth(
              x < stride ? 0 : pixelBuffer(stride),
              line == 0 ? 0 : pixelBuffer(width),
              line == 0 || x < stride ? 0 : pixelBuffer(width + stride))));
            break;
          }
          default:
            //fail: unexpected filter code.
            break;
        }
        pixelBuffer.add(c);
        if (fMode == FMode::FDECOMPRESS) {
          out->putChar(c1);
        }
        else if (fMode == FMode::FCOMPARE) {
          p++;
          if (c1 != out->getchar() && (diffFound == 0)) {
            diffFound = p;
          }
        }
      }
    }
    return size;
  }
};
