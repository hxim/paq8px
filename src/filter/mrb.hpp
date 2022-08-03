#pragma once

#include "Filter.hpp"

class MrbRleFilter : Filter {
private:
  int width;
  int height;

  int encodeRLE(uint8_t* dst, uint8_t* ptr, int src_end, int maxlen) {
    int i = 0;
    int ind = 0;
    for (ind = 0; ind < src_end; ) {
      if (i > maxlen) return i;
      if (ptr[ind + 0] != ptr[ind + 1] || ptr[ind + 1] != ptr[ind + 2]) {
        // Guess how many non repeating bytes we have
        int j = 0;
        for (j = ind + 1; j < (src_end); j++)
          if ((ptr[j + 0] == ptr[j + 1] && ptr[j + 2] == ptr[j + 0]) || ((j - ind) >= 127)) break;
        int pixels = j - ind;
        if (j + 1 == src_end && pixels < 8)pixels++;
        dst[i++] = 0x80 | pixels;
        for (int cnt = 0; cnt < pixels; cnt++) {
          dst[i++] = ptr[ind + cnt];
          if (i > maxlen) return i;
        }
        ind = ind + pixels;
      }
      else {
        // Get the number of repeating bytes
        int j = 0;
        for (j = ind + 1; j < (src_end); j++)
          if (ptr[j + 0] != ptr[j + 1]) break;
        int pixels = j - ind + 1;
        if (j == src_end && pixels < 4) {
          pixels--;
          dst[i] = uint8_t(0x80 | pixels);
          i++;
          if (i > maxlen) return i;
          for (int cnt = 0; cnt < pixels; cnt++) {
            dst[i] = ptr[ind + cnt];
            i++;
            if (i > maxlen) return i;
          }
          ind = ind + pixels;
        }
        else {
          j = pixels;
          while (pixels > 127) {
            dst[i++] = 127;
            dst[i++] = ptr[ind];
            if (i > maxlen) return i;
            pixels = pixels - 127;
          }
          if (pixels > 0) {
            if (j == src_end) pixels--;
            dst[i++] = pixels;
            dst[i++] = ptr[ind];
            if (i > maxlen) return i;
          }
          ind = ind + j;
        }
      }
    }
    return i;
  }

public:
  void setInfo(int width, int height) {
    this->width = width;
    this->height = height;
  }
  void encode(File* in, File* out, uint64_t size, int info, int& /*headerSize*/) override {
    uint64_t savepos = in->curPos();
    int totalSize = (width)*height;
    Array<uint8_t, 1> ptrin(totalSize + 4);
    Array<uint8_t, 1> ptr(size + 4);
    Array<uint32_t> diffpos(4096);
    uint32_t count = 0;
    uint8_t value = 0;
    int diffcount = 0;
    // decode RLE
    for (int i = 0; i < totalSize; ++i) {
      if ((count & 0x7F) == 0) {
        count = in->getchar();
        value = in->getchar();
      }
      else if (count & 0x80) {
        value = in->getchar();
      }
      count--;
      ptrin[i] = value;
    }
    // encode RLE
    int a = encodeRLE(&ptr[0], &ptrin[0], totalSize, size);
    assert(a < (size + 4));
    // compare to original and output diff data
    in->setpos(savepos);
    for (int i = 0; i < size; i++) {
      uint8_t b = ptr[i], c = in->getchar();
      if (diffcount == 4095 || diffcount > (size / 2) || i > 0xFFFFFF) return; // fail
      if (b != c) {
        if (diffcount < 4095)diffpos[diffcount++] = c + (i << 8);
      }
    }
    out->putChar((diffcount >> 8) & 255); out->putChar(diffcount & 255);
    if (diffcount > 0)
      out->blockWrite((uint8_t*)&diffpos[0], diffcount * 4);
    out->put32(size);
    out->blockWrite(&ptrin[0], totalSize);
  }

  uint64_t decode(File* in, File* out, FMode fMode, uint64_t  size, uint64_t& diffFound) override {
    if (size == 0) {
      diffFound = 1;
      return 0;
    }
    Array<uint32_t> diffpos(4096);
    int diffcount = 0;
    diffcount = (in->getchar() << 8) + in->getchar();
    if (diffcount > 0) in->blockRead((uint8_t*)&diffpos[0], diffcount * 4);
    int len = in->get32();
    Array<uint8_t, 1> fptr(size + 4);
    Array<uint8_t, 1> ptr(size + 4);
    in->blockRead(&fptr[0], size);
    encodeRLE(&ptr[0], &fptr[0], size - 2 - 4 - diffcount * 4, len); //size - header
    //Write out or compare
    if (fMode == FMode::FDECOMPRESS) {
      int diffo = diffpos[0] >> 8;
      int diffp = 0;
      for (int i = 0; i < len; i++) {
        if (i == diffo && diffcount) {
          ptr[i] = diffpos[diffp] & 255, diffp++, diffo = diffpos[diffp] >> 8;
        }
      }
      out->blockWrite(&ptr[0], len);
    }
    else if (fMode == FMode::FCOMPARE) {
      int diffo = diffpos[0] >> 8;
      int diffp = 0;
      for (int i = 0; i < len; i++) {
        if (i == diffo && diffcount) {
          ptr[i] = diffpos[diffp] & 255, diffp++, diffo = diffpos[diffp] >> 8;
        }
        uint8_t b = ptr[i];
        if (b != out->getchar() && !diffFound) diffFound = out->curPos();
      }
    }
    assert(len < size);
    return len;
  }
};
