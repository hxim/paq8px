#pragma once

#include <cassert>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>

/**
 * Track time and memory used.
 * @remark: only @ref Array<T> reports its memory usage, we don't know about other types
 */
class ProgramChecker {
private:
    uint64_t memUsed {};  /**< Bytes currently in use (all allocated minus all freed) */
    uint64_t maxMem {};   /**< Most bytes allocated ever */
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;

    /**
     * Constructor is private so that it cannot be called
     */
    ProgramChecker() {
      startTime = std::chrono::high_resolution_clock::now();
    }

    /**
     * Copy constructor is private so that it cannot be called
     */
    ProgramChecker(ProgramChecker const & /*unused*/) {}

    /**
     * Assignment operator is private so that it cannot be called
     */
    auto operator=(ProgramChecker const & /*unused*/) -> ProgramChecker & { return *this; }

    static ProgramChecker *instance;
public:
    static auto getInstance() -> ProgramChecker *;
    void alloc(uint64_t n);
    void free(uint64_t n);
    [[nodiscard]] auto getRuntime() const -> double;

    /**
     * Print elapsed time and used memory
     */
    void print() const;
    ~ProgramChecker();
};
