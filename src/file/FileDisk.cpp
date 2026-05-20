#include "FileDisk.hpp"
#include "../SystemDefines.hpp"

FILE* FileDisk::makeTmpFile() {
#if defined(WINDOWS)
  wchar_t szTempFileName[MAX_PATH];
  const UINT uRetVal = GetTempFileNameW(L".", L"tmp", 0, szTempFileName);
  if (uRetVal == 0) return nullptr;
  return fopen(Utf8Str(szTempFileName).utf8_str, "w+bTD");
#else
  return tmpfile();
#endif
}

FileDisk::FileDisk() { file = nullptr; }

FileDisk::~FileDisk() { close(); }

bool FileDisk::open(const char *filename, bool mustSucceed) {
  assert(file == nullptr);
  file = openFile(filename, READ);
  const bool success = (file != nullptr);
  if( !success && mustSucceed ) {
    printf("Unable to open file %s (%s)", filename, strerror(errno));
    quit();
  }
  return success;
}

void FileDisk::create(const char *filename) {
  assert(file == nullptr);
  makeDirectories(filename);
  file = openFile(filename, WRITE);
  if( file == nullptr ) {
    printf("Unable to create file %s (%s)", filename, strerror(errno));
    quit();
  }
}

void FileDisk::createTmp() {
  assert(file == nullptr);
  file = makeTmpFile();
  if( file == nullptr ) {
    printf("Unable to create temporary file (%s)", strerror(errno));
    quit();
  }
}

void FileDisk::close() {
  if( file != nullptr ) {
    fclose(file);
  }
  file = nullptr;
}

int FileDisk::getchar() { return fgetc(file); }

void FileDisk::putChar(uint8_t c) { fputc(c, file); }

uint64_t FileDisk::blockRead(uint8_t *ptr, uint64_t count) { return fread(ptr, 1, count, file); }

void FileDisk::blockWrite(uint8_t *ptr, uint64_t count) { fwrite(ptr, 1, count, file); }

void FileDisk::setpos(uint64_t newPos) { fseeko(file, newPos, SEEK_SET); }

void FileDisk::setEnd() { fseeko(file, 0, SEEK_END); }

uint64_t FileDisk::curPos() { return ftello(file); }

bool FileDisk::eof() { return feof(file) != 0; }