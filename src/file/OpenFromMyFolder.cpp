#include "OpenFromMyFolder.hpp"

void OpenFromMyFolder::myself(FileDisk *f) {
#ifdef WINDOWS
  int i;
      Array<wchar_t> myFileName(MAX_PATH);
      if ((i = GetModuleFileNameW(nullptr, &myFileName[0], MAX_PATH)) && i < MAX_PATH && i != 0) {
        f->open(Utf8Str(&myFileName[0]).utf8_str, true);
      } else
        quit(ofmf::myPathError);
#endif
#ifdef __APPLE__
  char myFileName[PATH_MAX];
  uint32_t size = sizeof(myFileName);
  if( _NSGetExecutablePath(myFileName, &size) == 0 ) {
    f->open(&myFileName[0], true);
  } else {
    quit(ofmf::myPathError);
  }
#endif
#ifdef UNIX
#ifndef __APPLE__
  Array<char> myFileName(PATH_MAX + 1);
      if( readlink("/proc/self/exe", &myFileName[0], PATH_MAX) != -1 )
        f->open(&myFileName[0], true);
      else
        quit(ofmf::myPathError);
#endif
#endif
}

void OpenFromMyFolder::anotherFile(FileDisk *f, const char *filename) {
  const uint64_t fLength = strlen(filename) + 1;
#ifdef WINDOWS
  int i;
  Array<wchar_t> myFileNameW(MAX_PATH + fLength);
  if ((i = GetModuleFileNameW(nullptr, &myFileNameW[0], MAX_PATH)) && i < MAX_PATH && i != 0) {
    Array<char> myFileName(MAX_PATH + fLength);
    strcpy(&myFileName[0], Utf8Str(&myFileNameW[0]).utf8_str);
    char *endOfPath = strrchr(&myFileName[0], '\\');
#endif
#ifdef __APPLE__
  char myFileName[PATH_MAX + fLength];
  uint32_t size = sizeof(myFileName);
  if( _NSGetExecutablePath(myFileName, &size) == 0 ) {
    char *endOfPath = strrchr(&myFileName[0], '/');
#endif
#ifdef UNIX
#ifndef __APPLE__
    char myFileName[PATH_MAX + fLength];
        if( readlink("/proc/self/exe", myFileName, PATH_MAX) != -1 ) {
          char *endOfPath = strrchr(&myFileName[0], '/');
#endif
#endif
    if( endOfPath == nullptr ) {
      quit(ofmf::myPathError);
    }
    endOfPath++;
    strcpy(endOfPath, filename); //append filename to my path
    f->open(&myFileName[0], true);
  } else {
    quit(ofmf::myPathError);
  }
}
