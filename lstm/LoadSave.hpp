#pragma once

#include <cstdio>
#include <cstdint>
#include <cstring>

class LoadSave {
public:
  explicit LoadSave(FILE* f);

  FILE* file;

  void WriteFloatArray(const float* data, size_t count);
  void ReadFloatArray(float* data, size_t count);

  void WriteTextLine(const char* str);
  bool ReadTextLine(char* buffer, size_t buffer_size);

  void WriteU32(uint32_t value);
  uint32_t ReadU32();
};
