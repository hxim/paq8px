#include "ListOfFiles.hpp"
#include "../CharacterNames.hpp"

ListOfFiles::ListOfFiles() : state(ParseState::IN_HEADER), names(0) {}

ListOfFiles::~ListOfFiles() {
  for( int i = 0; i < static_cast<int>(names.size()); i++ ) {
    delete names[i];
  }
}

void ListOfFiles::setBasePath(const char *s) {
  assert(basePath.strsize() == 0);
  basePath += s;
}

void ListOfFiles::addChar(char c) {
  if( c != EOF) {
    listOfFiles += c;
  }
  if( c == 10 || c == 13 || c == EOF) { //got a newline
    state = ParseState::FINISHED_A_LINE; //empty lines / extra newlines (cr, crlf or lf) are ignored
  } else if( state == ParseState::IN_HEADER ) {
    return; //ignore anything in header
  } else if( c == TAB ) { //got a tab
    state = ParseState::FINISHED_A_FILENAME; //ignore the rest (other columns)
  } else { // got a character
    if( state == ParseState::FINISHED_A_FILENAME ) {
      return; //ignore the rest (other columns)
    }
    if( state == ParseState::FINISHED_A_LINE ) {
      names.pushBack(new FileName(basePath.c_str()));
      state = ParseState::PROCESSING_FILENAME;
      if( c == '/' || c == '\\' ) {
        quit("For security reasons absolute paths are not allowed in the file list.");
      }
      // TODO: prohibit parent folder references in path ('/../')
    }
    if( c == 0 || c == ':' || c == '?' || c == '*' ) {
      printf("\nIllegal character ('%c') in file list.", c);
      quit();
    }
    if( c == BADSLASH ) {
      c = GOODSLASH;
    }
    (*names[names.size() - 1]) += c;
  }
}

int ListOfFiles::getCount() { return static_cast<int>(names.size()); }

const char* ListOfFiles::getfilename(int i) { return names[i]->c_str(); }

String* ListOfFiles::getString() { return &listOfFiles; }
