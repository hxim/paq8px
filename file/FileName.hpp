#pragma once

#include "fileUtils.hpp"
#include "../String.hpp"

/**
 * A class to represent a filename.
 */
class FileName : public String {
public:
    explicit FileName(const char *s = "");
    [[nodiscard]] auto lastSlashPos() const -> int;
    void keepFilename();
    void keepPath();

    /**
     * Prepare path string for screen output.
     */
    void replaceSlashes();
};
