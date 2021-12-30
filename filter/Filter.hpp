#pragma once

#include "../file/File.hpp"
#include "../Encoder.hpp"
#include <cstdint>

typedef enum {
    FDECOMPRESS, FCOMPARE, FDISCARD
} FMode;

class Filter {
protected:
    Encoder *encoder = nullptr;
public:
    virtual void encode(File *in, File *out, uint64_t size, int info, int &headerSize) = 0;
    virtual auto decode(File *in, File *out, FMode fMode, uint64_t size, uint64_t &diffFound) -> uint64_t = 0;

    void setEncoder(Encoder &en) {
      encoder = &en;
    }

    virtual ~Filter() = default;
};
