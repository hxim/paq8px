#pragma once

#include "Filter.hpp"
#include "../Array.hpp"
#include "../file/File.hpp"
#include "../CharacterNames.hpp"

constexpr int powers[5] = { 85 * 85 * 85 * 85, 85 * 85 * 85, 85 * 85, 85, 1 };

class Base85Filter : Filter {
public:
  void encode(File* in, File* out, uint64_t size, int  /*info*/, int& /*headerSize*/) override {
    int lfp = 0;
    int tlf = 0;
    int b85mem = (size >> 2) * 5 + 100;
    Array<uint8_t, 1> ptr(b85mem);
    int olen = 5;
    int c;
    int count = 0;
    uint32_t tuple = 0;
    for (int f = 0; f < size; f++) {
      c = in->getchar();
      if (olen + 10 > b85mem) {
        count = 0;
        break;
      }
      if (c == CARRIAGE_RETURN || c == NEW_LINE) {
        if (lfp == 0) {
          lfp = f;
          tlf = c;
        }
        if (tlf != c)
          tlf = 0;
        continue;
      }
      if (c == 'z' && count == 0) {
        if (olen + 10 > b85mem) {
          count = 0;
          break;
        }
        for (int i = 1; i < 5; i++)
          ptr[olen++] = 0;
        continue;
      }
      if (c == EOF) {
        if (olen + 10 > b85mem) {
          count = 0;
          break;
        }
        if (count > 0) {
          tuple += powers[count - 1];
          for (int i = 1; i < count; i++)
            ptr[olen++] = tuple >> ((4 - i) * 8);
        }
        break;
      }
      tuple += (c - '!') * powers[count++];
      if (count == 5) {
        if (olen > b85mem + 10) {
          count = 0;
          break;
        }
        for (int i = 1; i < count; i++)
          ptr[olen++] = tuple >> ((4 - i) * 8);
        tuple = 0;
        count = 0;
      }
    }
    if (count > 0) {
      tuple += powers[count - 1];
      for (int i = 1; i < count; i++)
        ptr[olen++] = tuple >> ((4 - i) * 8);
    }
    ptr[0] = lfp & 255; //nl lenght
    ptr[1] = size & 255;
    ptr[2] = size >> 8 & 255;
    ptr[3] = size >> 16 & 255;
    if (tlf != 0) {
      if (tlf == 10)
        ptr[4] = 128;
      else ptr[4] = 64;
    }
    else
      ptr[4] = size >> 24 & 63; //1100 0000
    out->blockWrite(&ptr[0], olen);
  }

  uint64_t decode(File* in, File* out, FMode fMode, uint64_t /*size*/, uint64_t& diffFound) override {
    int i;
    int fle = 0;
    int nlsize = 0;
    int outlen = 0;
    int tlf = 0;
    nlsize = in->getchar();
    outlen = in->getchar();
    outlen += (in->getchar() << 8);
    outlen += (in->getchar() << 16);
    tlf = (in->getchar());
    outlen += ((tlf & 63) << 24);
    Array<uint8_t, 1> ptr((outlen >> 2) * 5 + 10);
    tlf = (tlf & 192);
    if (tlf == 128)
      tlf = NEW_LINE;
    else if (tlf == 64)
      tlf = CARRIAGE_RETURN;
    else
      tlf = 0;
    int c;
    int count = 0;
    int lenlf = 0;
    uint32_t tuple = 0;

    while (fle < outlen) {
      c = in->getchar();
      if (c != EOF) {
        tuple |= ((uint32_t)c) << ((3 - count++) * 8);
        if (count < 4) continue;
      }
      else if (count == 0) break;
      int i;
      int lim;
      char out[5];
      if (tuple == 0 && count == 4) { // for 0x00000000
        if (nlsize && lenlf >= nlsize) {
          if (tlf)
            ptr[fle++] = (tlf);
          else {
            ptr[fle++] = CARRIAGE_RETURN;
            ptr[fle++] = NEW_LINE;
          }
          lenlf = 0;
        }
        ptr[fle++] = 'z';
      }
      else {
        for (i = 0; i < 5; i++) {
          out[i] = tuple % 85 + '!';
          tuple /= 85;
        }
        lim = 4 - count;
        for (i = 4; i >= lim; i--) {
          if (nlsize && lenlf >= nlsize && ((outlen - fle) >= 5)) {// skip nl if only 5 bytes left
            if (tlf)
              ptr[fle++] = (tlf);
            else {
              ptr[fle++] = CARRIAGE_RETURN;
              ptr[fle++] = NEW_LINE;
            }
            lenlf = 0;
          }
          ptr[fle++] = out[i];
          lenlf++;
        }
      }
      if (c == EOF) break;
      tuple = 0;
      count = 0;
    }
    if (fMode == FMode::FDECOMPRESS) {
      out->blockWrite(&ptr[0], outlen);
    }
    else if (fMode == FMode::FCOMPARE) {
      for (i = 0; i < outlen; i++) {
        uint8_t b = ptr[i];
        if (b != out->getchar() && !diffFound) diffFound = out->curPos();
      }
    }
    return outlen;
  }
};

