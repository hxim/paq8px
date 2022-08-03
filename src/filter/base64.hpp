#pragma once

#include "Filter.hpp"
#include "../Array.hpp"
#include "../file/File.hpp"

namespace base64 {
  constexpr bool isdigit(int8_t c) {
    return c >= '0' && c <= '9';
  }

  constexpr bool islower(int8_t c) {
    return c >= 'a' && c <= 'z';
  }

  constexpr bool isupper(int8_t c) {
    return c >= 'A' && c <= 'Z';
  }

  constexpr bool isalpha(int8_t c) {
    return islower(c) || isupper(c);
  }

  constexpr bool isalnum(int8_t c) {
    return isalpha(c) || isdigit(c);
  }

  static constexpr char table1[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
} // namespace base64

class Base64Filter : Filter {
private:
  static char valueB(char c) {
    const char *p = strchr(base64::table1, c);
    if( p != nullptr ) {
      return static_cast<char>(p - base64::table1);
    }
    return 0;
  }

  static bool isBase64(uint8_t c) {
    return (base64::isalnum(c) || (c == '+') || (c == '/') || (c == 10) || (c == 13));
  }

public:
  void encode(File *in, File *out, uint64_t size, int  /*info*/, int & /*headerSize*/) override {
    uint64_t inLen = 0;
    int i = 0;
    int lineSize = 0;
    uint8_t b = 0;
    uint8_t tlf = 0;
    uint8_t src[4];
    uint64_t b64Mem = (size >> 2) * 3 + 10;
    Array<uint8_t> ptr(b64Mem);
    int olen = 5;

    while( b = in->getchar(), inLen++, (b != '=') && isBase64(b) && inLen <= size ) {
      if( b == 13 || b == 10 ) {
        if(lineSize == 0 ) {
          lineSize = inLen;
          tlf = b;
        }
        if( tlf != b ) {
          tlf = 0;
        }
        continue;
      }
      src[i++] = b;
      if( i == 4 ) {
        for( int j = 0; j < 4; j++ ) {
          src[j] = valueB(src[j]);
        }
        src[0] = (src[0] << 2) + ((src[1] & 0x30) >> 4);
        src[1] = ((src[1] & 0xf) << 4) + ((src[2] & 0x3c) >> 2);
        src[2] = ((src[2] & 0x3) << 6) + src[3];

        ptr[olen++] = src[0];
        ptr[olen++] = src[1];
        ptr[olen++] = src[2];
        i = 0;
      }
    }

    if( i != 0 ) {
      for( int j = i; j < 4; j++ ) {
        src[j] = 0;
      }

      for( int j = 0; j < 4; j++ ) {
        src[j] = valueB(src[j]);
      }

      src[0] = (src[0] << 2) + ((src[1] & 0x30) >> 4);
      src[1] = ((src[1] & 0xf) << 4) + ((src[2] & 0x3c) >> 2);
      src[2] = ((src[2] & 0x3) << 6) + src[3];

      for( int j = 0; (j < i - 1); j++ ) {
        ptr[olen++] = src[j];
      }
    }
    ptr[0] = lineSize & 255;
    ptr[1] = size & 255;
    ptr[2] = (size >> 8) & 255;
    ptr[3] = (size >> 16) & 255;
    if( tlf != 0 ) {
      if( tlf == 10 ) {
        ptr[4] = 128;
      } else {
        ptr[4] = 64;
      }
    } else {
      ptr[4] = (size >> 24) & 63; //1100 0000
    }
    out->blockWrite(&ptr[0], olen);
  }

  uint64_t decode(File *in, File *out, FMode fMode, uint64_t /*size*/, uint64_t &diffFound) override {
    uint8_t inn[3];
    int i = 0;
    int len = 0;
    int blocksOut = 0;
    int fle = 0;
    int lineSize = in->getchar();
    int outLen = in->getchar() + (in->getchar() << 8) + (in->getchar() << 16);
    uint8_t tlf = (in->getchar());
    outLen += ((tlf & 63) << 24);
    Array<uint8_t> ptr((outLen >> 2) * 4 + 10);
    tlf = (tlf & 192);
    if( tlf == 128 ) {
      tlf = 10; // LF: 10
    } else if( tlf == 64 ) {
      tlf = 13; // LF: 13
    } else {
      tlf = 0;
    }

    while( fle < outLen ) {
      len = 0;
      for( i = 0; i < 3; i++ ) {
        int c = in->getchar();
        if( c != EOF) {
          inn[i] = static_cast<uint8_t>(c);
          len++;
        } else {
          inn[i] = 0;
        }
      }
      if( len != 0 ) {
        uint8_t in0 = inn[0];
        uint8_t in1 = inn[1];
        uint8_t in2 = inn[2];
        ptr[fle++] = (base64::table1[in0 >> 2]);
        ptr[fle++] = (base64::table1[((in0 & 0x03) << 4) | ((in1 & 0xf0) >> 4)]);
        ptr[fle++] = ((len > 1 ? base64::table1[((in1 & 0x0f) << 2) | ((in2 & 0xc0) >> 6)] : '='));
        ptr[fle++] = ((len > 2 ? base64::table1[in2 & 0x3f] : '='));
        blocksOut++;
      }
      else {
        if (fMode == FMode::FDECOMPRESS) {
          quit("Unexpected Base64 decoding state");
        }
        else if (fMode == FMode::FCOMPARE) {
          diffFound = fle;
          break; // give up
        }
      }
      if( blocksOut >= (lineSize / 4) && lineSize != 0 ) { //no lf if lineSize==0
        if((blocksOut != 0) && !in->eof() && fle <= outLen ) { //no lf if eof
          if( tlf != 0 ) {
            ptr[fle++] = tlf;
          } else {
            ptr[fle++] = 13;
            ptr[fle++] = 10;
          }
        }
        blocksOut = 0;
      }
    }
    //Write out or compare
    if( fMode == FMode::FDECOMPRESS ) {
      out->blockWrite(&ptr[0], outLen);
    } else if( fMode == FMode::FCOMPARE ) {
      for( i = 0; i < outLen; i++ ) {
        uint8_t b = ptr[i];
        if( b != out->getchar() && (diffFound == 0)) {
          diffFound = static_cast<int>(out->curPos());
        }
      }
    }
    return outLen;
  }
};
