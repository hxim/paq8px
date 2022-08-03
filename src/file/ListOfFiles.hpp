#pragma once

#include "FileName.hpp"
#include "../Array.hpp"
#include "fileUtils.hpp"
#include <cassert>

class ListOfFiles {
private:
  enum class ParseState {
      IN_HEADER, FINISHED_A_FILENAME, FINISHED_A_LINE, PROCESSING_FILENAME
  } ;
  ParseState state; /**< parsing state */
  FileName basePath;
  String listOfFiles; /**< path/file list in first column, columns separated by tabs, rows separated by newlines, with header in 1st row */
  Array<FileName *> names; /**< all file names parsed from listOfFiles */
public:
  ListOfFiles();
  ~ListOfFiles();
  void setBasePath(const char *s);
  void addChar(char c);
  int getCount();
  const char* getfilename(int i);
  String* getString();
};
