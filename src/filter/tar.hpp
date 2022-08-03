#pragma once

#include "Filter.hpp"
#include "../file/File.hpp"
#include "../CharacterNames.hpp"
#include <cstdint>

/**
 * UStar (Unix Standard TAR) detection and transformation
 * 
 */
class TarFilter : public Filter {
private:

  struct TARheader { // 512 bytes
    char name[100];       //   0 | file name
    char mode[8];         // 100 | file mode (permissions)
    char uid[8];          // 108 | owner user id (octal)
    char gid[8];          // 116 | owner group id (octal)
    char size[12];        // 124 | file size in bytes (octal)
    char mtime[12];       // 136 | last modification time in numeric Unix time format (octal)
    char chksum[8];       // 148 | sum of unsigned characters in header block, filled with spaces while calculating (octal)
    char typeflag;        // 156 | link flag / file type
    char linkname[100];   // 157 | name of linked file
    char magic[8];        // 257 | "ustar\000" in calgary.tar created by windows 10 and gnu tar
                          //       "ustar  \0" in mozilla, samba, xml or in calgary.tar created by total commander or gnu tar
    char uname[32];       // 265 | owner user name (string)
    char gname[32];       // 297 | owner group name (string)
    char devmajor[8];     // 329 | device major number
    char devminor[8];     // 337 | device minor number
    char prefix[155];     // 345 | filename prefix
    char padding[12];     // 500 | padding

    int oct2bin(const char* p, int size) {
      while (*p == SPACE) { //skip leading spaces
        ++p;
        --size;
      }
      int i = 0;
      while (size > 0) {
        if (*p == 0/* && size == 1*/) //the last char must be a \0
          break;
        if (*p == SPACE)
          break;
        if (*p < '0' || *p > '7')
          return -1; //fail
        i *= 8;
        i += *p - '0';
        ++p;
        --size;
      }
      return i;
    }

    int calculateChecksum() {
      const char* p = &name[0];
      constexpr int chksumOffset = offsetof(TARheader, chksum);
      constexpr int chksumSize = 8;
      int u = 0;
      for (int n = 0; n < sizeof(TARheader); ++n) {
        if (n < chksumOffset || n >= chksumOffset + chksumSize) //exluding the checksum bytes
          u += ((uint8_t*)p)[n];
        else
          u += SPACE; // emulate spaces in place of the checksum bytes 
      }
      return u;
    }

    // checksum format examples: 
    // "0012201\0" (as in calgary.tar as created by total commander)
    // "012201\0 " (as in samba or calgary.tar created by windows 10 or gnu tar)
    // " 12201\0 " (as in mozilla and xml)

    void clearChecksum() {
      //look up the terminating \0 and fill it's left side with either spaces or '0's to preserve 
      //information about the format (see the checksum format examples above).
      char filler = chksum[0] == SPACE ? SPACE : '0';
      int i = 0;
      for (; i < sizeof(tarh.chksum); i++)
        if (chksum[i + 1] == 0) break;
      //now i points to the last digit of the checksum
      for (; i >= 0; i--)
        chksum[i] = filler;
    }

    void generateChecksum() {
      //look up the terminating \0 and fill it's left side with the checksum (octal)
      int checksum = calculateChecksum();
      int i = 0;
      for (; i < sizeof(tarh.chksum); i++)
        if (chksum[i + 1] == 0) break;
      //now i points to the last digit of the checksum
      for (; i >= 0; i--) {
        chksum[i] = (checksum & 7) + '0';
        checksum >>= 3;
        if (checksum == 0)
          break;
      }
    }

    bool verifyChecksum() {
      return calculateChecksum() == oct2bin(&chksum[0], 8);
    }

    bool isEmptySector() {
      const char* p = &name[0];
      for (int n = 511; n >= 0; --n)
        if (p[n] != 0)
          return false;
      return true;
    }

  } tarh;

  Array<uint64_t> detectedSectorStartPositions{ 0 };
  Array<uint64_t> detectedFileStartPositions{ 0 };
  Array<uint64_t> detectedFileLengths{ 0 };
  uint64_t detectedEmptySectorCount{ 0 };

  //detect tar content
  //a tar file is: hdr+filecontent + hdr+filecontent + etc...
  //this function figures out where each file starts and how long they are
  bool process(File* in, uint64_t maxFilePos) {
    uint64_t sectorStartPos = this->detectedStartPos;
    while (true) {
      if (sectorStartPos >= maxFilePos)
        return false; //fail
      in->setpos(sectorStartPos);
      int bytesRead = in->blockRead((uint8_t*)&tarh, sizeof(tarh));
      if (bytesRead != sizeof(tarh)) {
        return false; //fail
      }
      if (tarh.isEmptySector()) {
        do {
          detectedEmptySectorCount++;
          sectorStartPos += sizeof(tarh);
          int bytesRead = in->blockRead((uint8_t*)&tarh, sizeof(tarh));
          if (bytesRead != sizeof(tarh))
            break;
        } while (tarh.isEmptySector());
        this->detectedEndPos = sectorStartPos;
        return true;
      }
      bool ok0 = tarh.verifyChecksum();
      bool ok1 = tarh.oct2bin(tarh.uid, sizeof(tarh.uid)) >= 0;
      bool ok2 = tarh.oct2bin(tarh.gid, sizeof(tarh.gid)) >= 0;
      bool ok3 = tarh.oct2bin(tarh.size, sizeof(tarh.size)) >= 0;
      bool ok4 = tarh.oct2bin(tarh.size, sizeof(tarh.typeflag)) >= 0;
      if (!ok0 || !ok1 || !ok2 || !ok3 || !ok4) {
        return false; //fail
      }
      
      detectedSectorStartPositions.pushBack(sectorStartPos);

      int fileSize = tarh.oct2bin(tarh.size, 12);
      if (fileSize != 0) {
        detectedFileStartPositions.pushBack(sectorStartPos + sizeof(TARheader));
        detectedFileLengths.pushBack(fileSize);
      }

      int sectorsToJump = (fileSize + 511) >> 9;
      sectorStartPos += sizeof(TARheader) * (sectorsToJump + 1);
    }
  }

  void Print() {
    for (size_t i = 0; i < detectedSectorStartPositions.size(); i++) {
      printf("tar sector position: %d\n", (int)detectedSectorStartPositions[i]);
    }
    for (size_t i = 0; i < detectedFileStartPositions.size(); i++) {
      printf("file position: %d, length: %d\n", (int)detectedFileStartPositions[i], (int)detectedFileLengths[i]);
    }
    printf("empty sectors: %d, length: %d\n", (int)detectedEmptySectorCount, (int)(detectedEmptySectorCount * sizeof(tarh)));
  }

public:
  uint64_t detectedStartPos{};
  uint64_t detectedEndPos{};

  bool detect(File* in, uint64_t maxFilePos) {
    uint64_t userNamePos = in->curPos();
    this->detectedStartPos = userNamePos - offsetof(TARheader, uname);
    return process(in, maxFilePos);
  }

  void encode(File *in, File *out, uint64_t size, int width, int & headerSize) override {
    this->detectedStartPos = in->curPos();
    uint64_t maxFilePos = this->detectedStartPos + size;
    bool success = process(in, maxFilePos);
    if (!success)
      quit("Internal error in TAR detection.");

    //for debugging
    //Print();

    out->putVLI(detectedSectorStartPositions.size());
    out->putVLI(detectedEmptySectorCount);

    for (size_t i = 0; i < detectedSectorStartPositions.size(); i++) {
      in->setpos(detectedSectorStartPositions[i]);
      int bytesRead = in->blockRead((uint8_t*)&tarh, sizeof(tarh));
      if (bytesRead != sizeof(tarh)) {
        quit("Internal error in TAR transformation.");
      }
      tarh.clearChecksum();
      out->blockWrite((uint8_t*)&tarh, sizeof(tarh));
    }

    Array<uint8_t, 1> fileData{0};
    for (size_t i = 0; i < detectedFileStartPositions.size(); i++) {
      fileData.resize(detectedFileLengths[i]);
      in->setpos(detectedFileStartPositions[i]);
      in->blockRead(&fileData[0], detectedFileLengths[i]);
      out->blockWrite(&fileData[0], detectedFileLengths[i]);
    }

    return;
  }

  uint64_t decode(File* in, File* out, FMode fMode, uint64_t size, uint64_t& diffFound) override {
    size_t sectorCount = in->getVLI();
    size_t emptySectorCount = in->getVLI();
    size_t curPos = in->curPos();
    Array<TARheader, 1> headerData{ sectorCount };
    Array<uint8_t, 1> fileData{ 0 };
    uint64_t p = 0;
    uint64_t fileDataStartPos = curPos + sectorCount * sizeof(tarh);
    for (size_t i = 0; i < sectorCount; i++) {
      in->setpos(curPos + i * sizeof(tarh));
      int bytesRead = in->blockRead((uint8_t*)&headerData[i], sizeof(tarh));
      if (bytesRead != sizeof(tarh)) {
        if (fMode == FMode::FCOMPARE)
          diffFound = p + 1;
        return 0;
      }
      headerData[i].generateChecksum();

      if (fMode == FMode::FDECOMPRESS) {
        p += sizeof(tarh);
        out->blockWrite((uint8_t*)&headerData[i], sizeof(tarh));
      }
      else if (fMode == FMode::FCOMPARE) {
        for (int j = 0; j < sizeof(tarh); j++) {
          p++;
          int c1 = out->getchar();
          int c2 = ((uint8_t*)&headerData[i])[j];
          if (c1 != c2 && (diffFound == 0)) {
            diffFound = p;
          }
        }
      }

      int fileSize = tarh.oct2bin(headerData[i].size, 12);
      
      if (fileSize != 0) {
        in->setpos(fileDataStartPos);
        fileData.resize(fileSize);
        int bytesRead = in->blockRead(&fileData[0], fileSize);
        if (bytesRead != fileSize) {
          if (fMode == FMode::FCOMPARE)
            diffFound = p;
          return p;
        }
        if (fMode == FMode::FDECOMPRESS) {
          p += fileSize;
          out->blockWrite(&fileData[0], fileSize);
        }
        else if (fMode == FMode::FCOMPARE) {
          for (int j = 0; j < fileSize; j++) {
            p++;
            if (fileData[j] != out->getchar() && (diffFound == 0)) {
              diffFound = p;
            }
          }
        }
        fileDataStartPos += fileSize;
        //printf("filesize (%d): %d\n",int(i), int(fileSize));
        
        //padding sector with 0 when needed
        while ((fileSize & 511) != 0) {
          p++;
          fileSize++;
          if (fMode == FMode::FDECOMPRESS) {
            out->putChar(0);
          }
          else if (fMode == FMode::FCOMPARE) {
            if (out->getchar() != 0 && (diffFound == 0)) {
              diffFound = p;
            }
          }
        }
      }
    }

    //write empty sectors at the end
    for (size_t i = 0; i < emptySectorCount * sizeof(tarh); i++) {
      p++;
      if (fMode == FMode::FDECOMPRESS) {
        out->putChar(0);
      }
      else if (fMode == FMode::FCOMPARE) {
        if (out->getchar() != 0 && (diffFound == 0)) {
          diffFound = p;
        }
      }
    }

    return p;
  }

  void getFilePositions(File* in, Array<uint64_t,1> &filePositions) {
    assert(filePositions.size() == 0);
    size_t sectorCount = in->getVLI();
    size_t emptySectorCount = in->getVLI();
    size_t curPos = in->curPos();
    uint64_t fileDataStartPos = curPos + sectorCount * sizeof(tarh);
    filePositions.pushBack(fileDataStartPos); //the first entry is the first file position
    for (size_t i = 0; i < sectorCount; i++) {
      int bytesRead = in->blockRead((uint8_t*)&tarh, sizeof(tarh));
      assert(bytesRead == sizeof(tarh));

      int fileSize = tarh.oct2bin(tarh.size, 12);
      assert(fileSize >= 0);

      if (fileSize != 0) {
        fileDataStartPos += fileSize;
        filePositions.pushBack(fileDataStartPos); //the last entry is exactly the tempfile size (it points past to the last file)
      }
    }
  }

};
