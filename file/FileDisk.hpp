#pragma once

#include "File.hpp"
#include "fileUtils.hpp"

/**
 * This class is responsible for files on disk.
 * It simply passes function calls to stdio.
 */
class FileDisk : public File {
private:
    /**
     * Helper function: create a temporary file
     *
     * On Windows when using tmpFile() the temporary file may be created
     * in the root directory causing access denied error when User Account Control (UAC) is on.
     * To avoid this issue with tmpFile() we simply use fopen() instead.
     * We create the temporary file in the directory where the executable is launched from.
     * Luckily the MS c runtime library provides two (MS specific) fopen() flags: "T"emporary and "d"elete.
     * @return
     */
    static auto makeTmpFile() -> FILE *;
protected:
    FILE *file;

public:
    FileDisk();
    ~FileDisk() override;
    auto open(const char *filename, bool mustSucceed) -> bool override;
    void create(const char *filename) override;
    void createTmp();
    void close() override;
    auto getchar() -> int override;
    void putChar(uint8_t c) override;
    auto blockRead(uint8_t *ptr, uint64_t count) -> uint64_t override;
    void blockWrite(uint8_t *ptr, uint64_t count) override;
    void setpos(uint64_t newPos) override;
    void setEnd() override;
    auto curPos() -> uint64_t override;
    auto eof() -> bool override;
};
