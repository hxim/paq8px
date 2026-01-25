#include "LoadSave.hpp"
#include <cstdlib>

LoadSave::LoadSave(FILE* f) : file(f) {}

void LoadSave::WriteFloatArray(const float* data, size_t count) {
  if (fwrite(data, sizeof(float), count, file) != count) {
    fprintf(stderr, "Error writing float array to file\n");
    exit(1);
  }
}

void LoadSave::ReadFloatArray(float* data, size_t count) {
  if (fread(data, sizeof(float), count, file) != count) {
    fprintf(stderr, "Error reading float array from file\n");
    exit(1);
  }
}

void LoadSave::WriteTextLine(const char* str) {
  fprintf(file, "%s\n", str);
}

bool LoadSave::ReadTextLine(char* buffer, size_t buffer_size) {
  if (fgets(buffer, buffer_size, file) == nullptr) {
    return false;
  }
  // Remove trailing newline
  size_t len = strlen(buffer);
  if (len > 0 && buffer[len - 1] == '\n') {
    buffer[len - 1] = '\0';
  }
  return true;
}

void LoadSave::WriteU32(uint32_t value) {
  if (fwrite(&value, sizeof(uint32_t), 1, file) != 1) {
    fprintf(stderr, "Error writing uint32_t to file\n");
    exit(1);
  }
}

uint32_t LoadSave::ReadU32() {
  uint32_t value;
  if (fread(&value, sizeof(uint32_t), 1, file) != 1) {
    fprintf(stderr, "Error reading uint32_t from file\n");
    exit(1);
  }
  return value;
}
