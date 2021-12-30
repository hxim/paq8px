#pragma once

#include "../Encoder.hpp"
#include "../file/File.hpp"
#include "Filter.hpp"
#include <cstdint>

class EndiannessFilter : public Filter {
public:
    void encode(File *in, File *out, uint64_t size, int  /*info*/, int & /*headerSize*/) override {
      for( uint64_t i = 0, l = size >> 1U; i < l; i++ ) {
        uint8_t b = in->getchar();
        out->putChar(in->getchar());
        out->putChar(b);
      }
      if((size & 1U) > 0 ) {
        out->putChar(in->getchar());
      }
    }

    auto decode(File * /*in*/, File *out, FMode fMode, uint64_t size, uint64_t &diffFound) -> uint64_t override {
      for( uint64_t i = 0, l = size >> 1U; i < l; i++ ) {
        uint8_t b1 = encoder->decompressByte(&encoder->predictorMain);
        uint8_t b2 = encoder->decompressByte(&encoder->predictorMain);
        if( fMode == FDECOMPRESS ) {
          out->putChar(b2);
          out->putChar(b1);
        } else if( fMode == FCOMPARE ) {
          bool ok = out->getchar() == b2;
          ok &= out->getchar() == b1;
          if( !ok && (diffFound == 0u)) {
            diffFound = size - i * 2;
            break;
          }
        }
        if( fMode == FDECOMPRESS && ((i & 0x7FFU) == 0u)) {
          encoder->printStatus();
        }
      }
      if((diffFound == 0u) && (size & 1U) > 0 ) {
        if( fMode == FDECOMPRESS ) {
          out->putChar(encoder->decompressByte(&encoder->predictorMain));
        } else if( fMode == FCOMPARE ) {
          if( out->getchar() != encoder->decompressByte(&encoder->predictorMain)) {
            diffFound = size - 1;
          }
        }
      }
      return size;
    }

};
