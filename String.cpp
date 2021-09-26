#include <string.h>
#include "utils.hpp"
#include "String.hpp"

#ifndef NDEBUG
  void String::chk_consistency() const {
    for (uint32_t i = 0; i < size() - 1; i++) {
      if ((*this)[i] == 0) {
        quit("Internal error - string consistency check failed (1).");
      }
    }
    if (((*this)[size() - 1]) != 0) {
      quit("Internal error - string consistency check failed (2).");
    }
  }
#endif

void String::appendIntRecursive(uint64_t x) {
  if( x <= 9 ) {
    pushBack('0' + char(x));
  } else {
    const uint64_t rem = x % 10;
    x = x / 10;
    if( x != 0 ) {
      appendIntRecursive(x);
    }
    pushBack('0' + char(rem));
  }
}

auto String::c_str() const -> const char * {
  return &(*this)[0];
}

auto String::strsize() const -> uint64_t {
  chk_consistency();
  assert(size() > 0);
  return size() - 1;
}

void String::operator=(const char *s) {
  resize(strlen(s) + 1);
  memcpy(&(*this)[0], s, size());
  chk_consistency();
}

void String::operator+=(const char *s) {
  const uint64_t pos = size();
  const uint64_t len = strlen(s);
  resize(pos + len);
  memcpy(&(*this)[pos - 1], s, len + 1);
  chk_consistency();
}

void String::operator+=(char c) {
  popBack(); //Remove NUL
  pushBack(c);
  pushBack(0); //Append NUL
  chk_consistency();
}

void String::operator+=(uint64_t x) {
  popBack(); //Remove NUL
  if( x == 0 ) {
    pushBack('0');
  } else {
    appendIntRecursive(x);
  }
  pushBack(0); //Append NUL
  chk_consistency();
}

auto String::endsWith(const char *ending) const -> bool {
  const uint64_t endingSize = strlen(ending);
  if( endingSize > strsize()) {
    return false;
  }
  const int cmp = memcmp(ending, &(*this)[strsize() - endingSize], endingSize);
  return (cmp == 0);
}

void String::stripEnd(uint64_t count) {
  assert(strsize() >= count);
  const uint64_t newSize = strsize() - count;
  resize(newSize);
  pushBack(0); //Append NUL
  chk_consistency();
}

auto String::beginsWith(const char *beginning) const -> bool {
  const uint64_t beginningSize = strlen(beginning);
  if( beginningSize > strsize()) {
    return false;
  }
  const int cmp = memcmp(beginning, &(*this)[0], beginningSize);
  return (cmp == 0);
}

void String::stripStart(uint64_t count) {
  assert(strsize() >= count);
  const uint64_t newSize = strsize() - count;
  memmove(&(*this)[0], &(*this)[count], newSize);
  resize(newSize);
  pushBack(0); //Append NUL
  chk_consistency();
}

auto String::findLast(char c) const -> int {
  uint64_t i = strsize();
  while( i-- > 0 ) {
    if((*this)[i] == c ) {
      return static_cast<int>(i);
    }
  }
  return -1; //not found
}

String::String(const char *s) : Array<char>(strlen(s) + 1) {
  memcpy(&(*this)[0], s, size());
  chk_consistency();
}
