#pragma once

#include "FileDisk.hpp"

#ifdef __APPLE__

#include <mach-o/dyld.h>

#endif

namespace ofmf {
  static char myPathError[] = "Can't determine my path.";
} // namespace ofmf

/**
 * Helper class: open a file from the executable's folder
 */
class OpenFromMyFolder {
private:
public:
  /**
    * This static method will open the executable itself for reading
    * @param f The @ref FileDisk to read the contents of the file into.
    */
  static void myself(FileDisk *f);

  /**
    * This static method will open a file from the executable's folder.
    * Only ASCII file names are supported.
    * @param f The @ref FileDisk to read the contents of the file into.
    * @param filename The name of the file to open
    */
  static void anotherFile(FileDisk *f, const char *filename);
};
