#include "FileTmp.hpp"

void FileTmp::forgetContentInRam() {
  if( contentInRam != nullptr ) {
    delete contentInRam;
    contentInRam = nullptr;
    filePos = 0;
    fileSize = 0;
  }
}

void FileTmp::forgetFileOnDisk() {
  if( fileOnDisk != nullptr ) {
    fileOnDisk->close();
    delete fileOnDisk;
    fileOnDisk = nullptr;
  }
}

void FileTmp::ramToDisk() {
  assert(fileOnDisk == nullptr);
  fileOnDisk = new FileDisk();
  fileOnDisk->createTmp();
  if( fileSize > 0 ) {
    fileOnDisk->blockWrite(&((*contentInRam)[0]), fileSize);
  }
  fileOnDisk->setpos(filePos);
  forgetContentInRam();
}

FileTmp::FileTmp() {
  contentInRam = new Array<uint8_t>(0);
  filePos = 0;
  fileSize = 0;
  fileOnDisk = nullptr;
}

FileTmp::~FileTmp() { close(); }

bool FileTmp::open(const char * /*filename*/, bool /*mustSucceed*/) {
  assert(false);
  return false;
}

void FileTmp::create(const char * /*filename*/) { assert(false); }

void FileTmp::close() {
  forgetContentInRam();
  forgetFileOnDisk();
}

int FileTmp::getchar() {
  if( contentInRam != nullptr ) {
    if( filePos >= fileSize ) {
      return EOF;
    }
    const uint8_t c = (*contentInRam)[filePos];
    filePos++;
    return c;
  }
  return fileOnDisk->getchar();
}

void FileTmp::putChar(uint8_t c) {
  if( contentInRam != nullptr ) {
    if( filePos < MAX_RAM_FOR_TMP_CONTENT ) {
      if( filePos == fileSize ) {
        contentInRam->pushBack(c);
        fileSize++;
      } else {
        (*contentInRam)[filePos] = c;
      }
      filePos++;
      return;
    }
    ramToDisk();
  }
  fileOnDisk->putChar(c);
}

uint64_t FileTmp::blockRead(uint8_t *ptr, uint64_t count) {
  if( contentInRam != nullptr ) {
    const uint64_t available = fileSize - filePos;
    if( available < count ) {
      count = available;
    }
    if( count > 0 ) {
      memcpy(ptr, &((*contentInRam)[filePos]), count);
    }
    filePos += count;
    return count;
  }
  return fileOnDisk->blockRead(ptr, count);
}

void FileTmp::blockWrite(uint8_t *ptr, uint64_t count) {
  if( contentInRam != nullptr ) {
    if( filePos + count <= MAX_RAM_FOR_TMP_CONTENT ) {
      contentInRam->resize((filePos + count));
      if( count > 0 ) {
        memcpy(&((*contentInRam)[filePos]), ptr, count);
      }
      fileSize += count;
      filePos += count;
      return;
    }
    ramToDisk();
  }
  fileOnDisk->blockWrite(ptr, count);
}

void FileTmp::setpos(uint64_t newPos) {
  if( contentInRam != nullptr ) {
    if( newPos > fileSize ) {
      ramToDisk(); //panic: we don't support seeking past end of file (but stdio does) - let's switch to disk
    } else {
      filePos = newPos;
      return;
    }
  }
  fileOnDisk->setpos(newPos);
}

void FileTmp::setEnd() {
  if( contentInRam != nullptr ) {
    filePos = fileSize;
  } else {
    fileOnDisk->setEnd();
  }
}

uint64_t FileTmp::curPos() {
  if( contentInRam != nullptr ) {
    return filePos;
  }

  return fileOnDisk->curPos();
}

bool FileTmp::eof() {
  if( contentInRam != nullptr ) {
    return filePos >= fileSize;
  }

  return fileOnDisk->eof();
}
