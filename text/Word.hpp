#pragma once

#include "../Hash.hpp"
#include <cctype>
#include <cmath>
#include <cstdio>

class Word {
private:
    auto calculateHash() -> uint64_t;
public:
    constexpr static int maxWordSize = 64;
    constexpr static int wordEmbeddingSize = 3;
    uint8_t letters[maxWordSize] {};
    uint8_t start {};
    uint8_t end {};
    uint64_t Hash[2] {};
    uint32_t type {};
    uint32_t language {};
    uint32_t embedding {};
    Word();
    void reset();
    auto operator==(const char *s) const -> bool;
    auto operator!=(const char *s) const -> bool;
    void operator+=(char c);
    auto operator-(Word w) const -> uint32_t;
    auto operator+(Word w) const -> uint32_t;
    auto operator[](uint8_t i) const -> uint8_t;
    auto operator()(uint8_t i) const -> uint8_t;
    [[nodiscard]] auto length() const -> uint32_t;
    [[nodiscard]] auto distanceTo(Word w) const -> uint32_t;
    void calculateWordHash();

    /**
     * Called by a stemmer after stemming
     */
    void calculateStemHash();
    auto changeSuffix(const char *oldSuffix, const char *newSuffix) -> bool;
    auto matchesAny(const char **a, int count) -> bool;
    auto endsWith(const char *suffix) const -> bool;
    auto startsWith(const char *prefix) const -> bool;
    void print() const;
};
