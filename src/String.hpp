#pragma once

#include "Array.hpp"

/**
 * A specialized string class
 */
class String : public Array<char> {
private:
  void appendIntRecursive(uint64_t x);
protected:
#ifdef NDEBUG
  #define chk_consistency() ((void) 0)
#else
  void chk_consistency() const;
#endif
public:
  const char* c_str() const;
  /**
    * Does not include the NUL terminator
    * @return length of the string
    */
  uint64_t strsize() const;
  void operator=(const char *s);
  void operator+=(const char *s);
  void operator+=(char c);
  void operator+=(uint64_t x);
  bool endsWith(const char *ending) const;
  void stripEnd(uint64_t count);
  bool beginsWith(const char *beginning) const;
  void stripStart(uint64_t count);
  int findLast(char c) const;
  explicit String(const char *s = "");
};
