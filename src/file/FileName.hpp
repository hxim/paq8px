#pragma once

#include "fileUtils.hpp"
#include "../String.hpp"

/**
 * A class to represent a filename.
 */
class FileName : public String {
public:
  explicit FileName(const char *s = "");
  int lastSlashPos() const;
  void keepFilename();
  void keepPath();

  /**
    * Prepare path string for screen output.
    */
  void replaceSlashes();
};
