#pragma once

#include "../file/File.hpp"
#include "../Encoder.hpp"
#include <cstdint>

enum class FMode {
    FDECOMPRESS, FCOMPARE, FDISCARD
};

class Filter {
protected:
  Encoder *encoder = nullptr;
public:
  virtual void encode(File *in, File *out, uint64_t size, int info, int &headerSize) = 0;
  virtual uint64_t decode(File *in, File *out, FMode fMode, uint64_t size, uint64_t &diffFound) = 0;

  void setEncoder(Encoder &en) {
    encoder = &en;
  }

  virtual ~Filter() = default;
};
