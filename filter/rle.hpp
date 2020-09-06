#ifndef PAQ8PX_RLE_HPP
#define PAQ8PX_RLE_HPP

#include "Filter.hpp"
#include "../VLI.hpp"

#define RLE_OUTPUT_RUN \
  { \
    while (run > 128) { \
      *outPtr++ = 0xFF, *outPtr++ = byte; \
      run -= 128; \
    } \
    *outPtr++ = (uint8_t)(0x80U | (run - 1)), *outPtr++ = byte; \
  }

class RleFilter : Filter {
private:
    enum {
        BASE, LITERAL, RUN, LITERAL_RUN
    } state = BASE;

    void handleRun(uint8_t byte, uint8_t *&outPtr, uint8_t *&lastLiteral, int &run) {
      if( run > 1 ) {
        state = RUN;
        RLE_OUTPUT_RUN
      } else {
        lastLiteral = outPtr;
        *outPtr++ = 0, *outPtr++ = byte;
        state = LITERAL;
      }
    }

    void handleLiteral(uint8_t byte, uint8_t *&outPtr, uint8_t *lastLiteral, int &run) {
      if( run > 1 ) {
        state = LITERAL_RUN;
        RLE_OUTPUT_RUN
      } else {
        if( ++(*lastLiteral) == 127 ) {
          state = BASE;
        }
        *outPtr++ = byte;
      }
    }

    auto handleLiteralRun(uint8_t *outPtr, uint8_t *lastLiteral) -> uint8_t {
      uint8_t loop = 0;
      if( outPtr[-2] == 0x81 && *lastLiteral < (125)) {
        state = (((*lastLiteral) += 2) == 127) ? BASE : LITERAL;
        outPtr[-2] = outPtr[-1];
      } else {
        state = RUN;
      }
      loop = 1;
      return loop;
    }

public:
    void encode(File *in, File *out, uint64_t size, int info, int &headerSize) override {
      uint8_t b = 0;
      uint8_t c = in->getchar();
      int i = 1;
      int maxBlockSize = info & 0xFFFFFFU;
      out->putVLI(maxBlockSize);
      headerSize = VLICost(maxBlockSize);
      while( i < static_cast<int>(size)) {
        b = in->getchar(), i++;
        if( c == 0x80 ) {
          c = b;
          continue;
        }
        if( c > 0x7F ) {
          for( uint32_t j = 0; j <= (c & 0x7FU); j++ ) {
            out->putChar(b);
          }
          c = in->getchar(), i++;
        } else {
          for(uint32_t j = 0; j <= c; j++, i++ ) {
            out->putChar(b), b = in->getchar();
          }
          c = b;
        }
      }
    }

    auto decode(File *in, File *out, FMode fMode, uint64_t  /*size*/, uint64_t &diffFound) -> uint64_t override {
      uint8_t inBuffer[0x10000] = {0};
      uint8_t outBuffer[0x10200] = {0};
      uint64_t pos = 0;
      int maxBlockSize = static_cast<int>(in->getVLI());

      do {
        uint64_t remaining = in->blockRead(&inBuffer[0], maxBlockSize);
        auto *inPtr = (uint8_t *) inBuffer;
        auto *outPtr = (uint8_t *) outBuffer;
        uint8_t *lastLiteral = nullptr;
        state = BASE;
        while( remaining > 0 ) {
          uint8_t byte = *inPtr++;
          uint8_t loop = 0;
          int run = 1;
          for( remaining--; remaining > 0 && byte == *inPtr; remaining--, run++, inPtr++ ) {
          }
          do {
            loop = 0;
            switch( state ) {
              case BASE:
              case RUN: {
                handleRun(byte, outPtr, lastLiteral, run);
                break;
              }
              case LITERAL: {
                handleLiteral(byte, outPtr, lastLiteral, run);
                break;
              }
              case LITERAL_RUN: {
                loop = handleLiteralRun(outPtr, lastLiteral);
              }
            }
          } while( loop != 0u );
        }

        uint64_t length = outPtr - (&outBuffer[0]);
        if( fMode == FDECOMPRESS ) {
          out->blockWrite(&outBuffer[0], length);
        } else if( fMode == FCOMPARE ) {
          for(uint32_t j = 0; j < length; ++j ) {
            if( outBuffer[j] != out->getchar() && (diffFound == 0u)) {
              diffFound = pos + j + 1;
              break;
            }
          }
        }
        pos += length;
      } while( !in->eof() && (diffFound == 0u));
      return pos;
    }
};

#endif //PAQ8PX_RLE_HPP
