#pragma once

#include "../file/File.hpp"
#include "LZWEntry.hpp"
#include <cstdint>

class LZWDictionary {
private:
  static constexpr int hashSize = 9221;
  LZWentry dictionary[4096] {};
  short table[hashSize] {};
  uint8_t buffer[4096] {};

public:
  int index;
  LZWDictionary();
  void reset();
  int findEntry(int prefix, int suffix);
  void addEntry(int prefix, int suffix, int offset = -1);
  int dumpEntry(File *f, int code);
};
