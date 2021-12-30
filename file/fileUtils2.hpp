#pragma once

#include "FileDisk.hpp"

/**
 * Verify that the specified file exists and is readable, determine file size
 * @todo Large file support
 * @param filename
 * @return
 */
static auto getFileSize(const char *filename) -> uint64_t {
  FileDisk f;
  f.open(filename, true);
  f.setEnd();
  const auto fileSize = f.curPos();
  f.close();
  if((fileSize >> 31U) != 0 ) {
    quit("Large files not supported.");
  }
  return fileSize;
}

static void appendToFile(const char *filename, const char *s) {
  FILE *f = openFile(filename, APPEND);
  if( f == nullptr ) {
    printf("Warning: Could not log compression results to %s\n", filename);
  } else {
    fprintf(f, "%s", s);
    fclose(f);
  }
}
