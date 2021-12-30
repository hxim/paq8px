#pragma once

#include "FileDisk.hpp"
#include "File.hpp"

/**
 * This class is responsible for temporary files in RAM or on disk
 * Initially it uses RAM for temporary file content.
 * In case of the content size in RAM grows too large, it is written to disk,
 * the RAM is freed and all subsequent file operations will use the file on disk.
 */
class FileTmp : public File {
private:
    Array<uint8_t> *contentInRam; /**< content of file */
    uint64_t filePos;
    uint64_t fileSize;
    FileDisk *fileOnDisk;

    void forgetContentInRam();
    void forgetFileOnDisk();

#ifdef NDEBUG
    static constexpr uint32_t MAX_RAM_FOR_TMP_CONTENT = 64 * 1024 * 1024; //64 MB (per file)
#else
    static constexpr uint32_t MAX_RAM_FOR_TMP_CONTENT = 64; // to trigger switching to disk earlier - for testing
#endif
    /**
     * switch: ram->disk
     */
    void ramToDisk();
public:
    FileTmp();
    ~FileTmp() override;

    /**
     *  This method is forbidden for temporary files.
     */
    auto open(const char * /*filename*/, bool /*mustSucceed*/) -> bool override;

    /**
     *  This method is forbidden for temporary files.
     */
    void create(const char * /*filename*/) override;
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
