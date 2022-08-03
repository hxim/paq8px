#pragma once

#include "../String.hpp"
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sys/stat.h>

/**
 * This is an abstract class for all the required file operations.
 * The main purpose of these classes is to keep temporary files in RAM as mush as possible. The default behaviour is to simply
 * pass function calls to the operating system - except in case of temporary files.
 */
class File {
public:
  virtual ~File() = default;
  virtual bool open(const char *filename, bool mustSucceed) = 0;
  virtual void create(const char *filename) = 0;
  virtual void close() = 0;
  virtual int getchar() = 0;
  virtual void putChar(uint8_t c) = 0;
  void append(const char *s);
  virtual uint64_t blockRead(uint8_t *ptr, uint64_t count) = 0;
  virtual void blockWrite(uint8_t *ptr, uint64_t count) = 0;
  uint32_t get32();
  void put32(uint32_t x);
  uint64_t getVLI();
  void putVLI(uint64_t i);
  virtual void setpos(uint64_t newPos) = 0;
  virtual void setEnd() = 0;
  virtual uint64_t curPos() = 0;
  virtual bool eof() = 0;
};
