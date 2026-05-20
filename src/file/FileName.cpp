#include "FileName.hpp"

FileName::FileName(const char *s) : String(s) {}

int FileName::lastSlashPos() const {
  int pos = findLast('/');
  if( pos < 0 ) {
    pos = findLast('\\');
  }
  return pos; //has no path when -1
}

void FileName::keepFilename() {
  const int pos = lastSlashPos();
  stripStart(pos + 1); //don't keep last slash
}

void FileName::keepPath() {
  const int pos = lastSlashPos();
  stripEnd(strsize() - pos - 1); //keep last slash
}

void FileName::replaceSlashes() { //
  const uint64_t sSize = strsize();
  for( uint64_t i = 0; i < sSize; i++ ) {
    if((*this)[i] == BADSLASH ) {
      (*this)[i] = GOODSLASH;
    }
  }
  chk_consistency();
}
