#pragma once

#include "Word.hpp"

class Stemmer {
protected:
    auto getRegion(const Word *w, uint32_t from) -> uint32_t;
    static auto suffixInRn(const Word *w, uint32_t rn, const char *suffix) -> bool;
    static auto charInArray(char c, const char a[], int len) -> bool;
public:
    virtual ~Stemmer() = default;
    virtual auto isVowel(char c) -> bool = 0;
    virtual auto stem(Word *w) -> bool = 0;
};
