#pragma once

#include "../Encoder.hpp"
#include "../file/File.hpp"
#include "Filter.hpp"
#include <cstdint>

class EndiannessFilter : public Filter {
public:
  void encode(File *in, File *out, uint64_t size, int  /*info*/, int & /*headerSize*/) override {
    for( uint64_t i = 0, l = size >> 1; i < l; i++ ) {
      uint8_t b = in->getchar();
      out->putChar(in->getchar());
      out->putChar(b);
    }
    if((size & 1) > 0 ) {
      out->putChar(in->getchar());
    }
  }

  uint64_t decode(File * /*in*/, File *out, FMode fMode, uint64_t size, uint64_t &diffFound) override {
    for( uint64_t i = 0, l = size >> 1; i < l; i++ ) {
      uint8_t b1 = encoder->decompressByte(&encoder->predictorMain);
      uint8_t b2 = encoder->decompressByte(&encoder->predictorMain);
      if( fMode == FMode::FDECOMPRESS ) {
        out->putChar(b2);
        out->putChar(b1);
      } else if( fMode == FMode::FCOMPARE ) {
        bool ok = out->getchar() == b2;
        ok &= out->getchar() == b1;
        if( !ok && (diffFound == 0)) {
          diffFound = size - i * 2;
          break;
        }
      }
      if( fMode == FMode::FDECOMPRESS && ((i & 0x7FF) == 0)) {
        encoder->printStatus();
      }
    }
    if((diffFound == 0) && (size & 1) > 0 ) {
      if( fMode == FMode::FDECOMPRESS ) {
        out->putChar(encoder->decompressByte(&encoder->predictorMain));
      } else if( fMode == FMode::FCOMPARE ) {
        if( out->getchar() != encoder->decompressByte(&encoder->predictorMain)) {
          diffFound = size - 1;
        }
      }
    }
    return size;
  }

};
