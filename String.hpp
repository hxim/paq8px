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
    [[nodiscard]] auto c_str() const -> const char *;
    /**
     * Does not include the NUL terminator
     * @return length of the string
     */
    [[nodiscard]] auto strsize() const -> uint64_t;
    void operator=(const char *s);
    void operator+=(const char *s);
    void operator+=(char c);
    void operator+=(uint64_t x);
    auto endsWith(const char *ending) const -> bool;
    void stripEnd(uint64_t count);
    auto beginsWith(const char *beginning) const -> bool;
    void stripStart(uint64_t count);
    [[nodiscard]] auto findLast(char c) const -> int;
    explicit String(const char *s = "");
};
