#pragma once

#include "../Array.hpp"
#include "../BlockType.hpp"
#include "../CharacterNames.hpp"
#include "../Encoder.hpp"
#include "../TransformOptions.hpp"
#include "../file/File.hpp"
#include "../file/FileDisk.hpp"
#include "../file/FileTmp.hpp"
#include "../Utils.hpp"
#include "Filter.hpp"
#include "TextParserStateInfo.hpp"
#include "base64.hpp"
#include "base85.hpp"
#include "cd.hpp"
#include "ecc.hpp"
#include "gif.hpp"
#include "lzw.hpp"
#include "mrb.hpp"
#include "png.hpp"
#include "tar.hpp"
#include "DecAlphaFilter.hpp"
#include <cctype>
#include <cstdint>
#include <cstring>

/////////////////////////// Filters /////////////////////////////////
//@todo: Update this documentation
//
// Before compression, data is encoded in blocks with the following format:
//
//   <type> <size> <encoded-data>
//
// type is 1 byte (type BlockType): DEFAULT=0, JPEG, EXE
// size is 4 bytes in big-endian format.
// Encoded-data decodes to <size> bytes.  The encoded size might be
// different.  Encoded data is designed to be more compressible.
//
//   void encode(File *in, File *out, int n);
//
// Reads n bytes of in (open in "rb" mode) and encodes one or
// more blocks to temporary file out (open in "wb+" mode).
// The file pointer of in is advanced n bytes.  The file pointer of
// out is positioned after the last byte written.
//
//   en.setFile(File *out);
//   int decode(Encoder& en);
//
// Decodes and returns one byte.  Input is from en.decompressByte(), which
// reads from out if in COMPRESS mode.  During compression, n calls
// to decode() must exactly match n bytes of in, or else it is compressed
// as type 0 without encoding.
//
//   BlockType detect(File *in, int n, BlockType type);
//
// Reads n bytes of in, and detects when the type changes to
// something else.  If it does, then the file pointer is repositioned
// to the start of the change and the new type is returned.  If the type
// does not change, then it repositions the file pointer n bytes ahead
// and returns the old type.
//
// For each type X there are the following 2 functions:
//
//   void encode_X(File *in, File *out, int n, ...);
//
// encodes n bytes from in to out.
//
//   int decode_X(Encoder& en);
//
// decodes one byte from en and returns it.  decode() and decode_X()
// maintain state information using static variables.

#ifndef DISABLE_ZLIB

#include "zlib.hpp"

#endif //DISABLE_ZLIB

static bool isGrayscalePalette(File *in, int n = 256, int isRGBA = 0) {
  uint64_t offset = in->curPos();
  int stride = 3 + isRGBA;
  int res = (n > 0) << 8;
  int order = 1;
  for( int i = 0; (i < n * stride) && ((res >> 8) != 0); i++ ) {
    int b = in->getchar();
    if( b == EOF) {
      res = 0;
      break;
    }
    if (i == 0) {
      res = 0x100 | b;
      order = 1 - 2 * static_cast<int>(b > int(ilog2(n) / 4));
      continue;
    }
    //"j" is the index of the current byte in this color entry
    int j = i % stride;
    if (j == 0) {
      // load first component of this entry
      int k = (b - (res & 0xFF)) * order;
      res = res & (static_cast<int>(k >= 0 && k <= 8) << 8);
      res |= (res) != 0 ? b : 0;
    }
    else if (j == 3) {
    res &= (static_cast<int>((b == 0) || (b == 0xFF)) * 0x1FF); // alpha/attribute component must be zero or 0xFF
    }
    else {
    res &= (static_cast<int>(b == (res & 0xFF)) * 0x1FF);
    }
  }
  in->setpos(offset);
  return (res >> 8) > 0;
}

//for MRB detection:
//read compressed word,dword
uint16_t GetCWord(File* f) {
  uint8_t b = f->getchar();
  if (b & 1) return ((f->getchar() << 8) | b) >> 1;
  return b >> 1;
}
uint32_t GetCDWord(File* f) {
  uint16_t w = f->getchar();
  w = w | (f->getchar() << 8);
  if (w & 1) {
    uint16_t w1 = f->getchar();
    w1 = w1 | (f->getchar() << 8);
    return ((w1 << 16) | w) >> 1;
  }
  return w >> 1;
}

ALWAYS_INLINE
static bool is_base85(unsigned char c) {
  return (isalnum(c) || (c == 13) || (c == 10) || (c == 'y') || (c == 'z') || (c >= '!' && c <= 'u'));
}

struct DetectionInfo {
  uint64_t HeaderStart{};
  uint64_t HeaderLength{};
  uint64_t DataStart{};
  uint64_t DataLength{};
  BlockType Type{};
  int DataInfo{};

  void IMG_DET(uint64_t blockStart, BlockType type, uint64_t start_pos, uint32_t header_len, uint32_t width, uint32_t height) {
    Type = type;
    HeaderStart = blockStart + start_pos;
    HeaderLength = header_len;
    DataStart = HeaderStart + HeaderLength;
    DataLength = static_cast<uint64_t>(width) * height;
    DataInfo = width;
  }

  void AUD_DET(uint64_t blockStart, BlockType type, uint64_t start_pos, uint32_t header_len, uint64_t data_len, int wmode) {
    Type = type;
    HeaderStart = blockStart + start_pos;
    HeaderLength = header_len;
    DataStart = HeaderStart + HeaderLength;
    DataLength = data_len;
    DataInfo = wmode;
  }

  void MRB_DET(uint64_t blockStart, BlockType type, uint8_t packingMethod, uint16_t colorBits, uint64_t start_pos, uint32_t header_len, uint64_t data_len, int width, int height) {
    Type = type;
    HeaderStart = blockStart + start_pos;
    HeaderLength = header_len;
    DataStart = HeaderStart + HeaderLength;
    DataLength = data_len;
    DataInfo = (colorBits << 2 | packingMethod) << 24 | width << 12 | height;
  }

  void DBF_DET(uint64_t blockStart, BlockType type, uint64_t start_pos, uint32_t header_len, uint64_t data_len, int recordLength) {
    Type = type;
    HeaderStart = blockStart + start_pos;
    HeaderLength = header_len;
    DataStart = HeaderStart + HeaderLength;
    DataLength = data_len;
    DataInfo = recordLength;
  }

  bool SizeVerificationPassed(uint64_t nextBlockStart) {
    bool passed = DataStart + DataLength <= nextBlockStart;
    if (!passed)
      memset(this, 0, sizeof(DetectionInfo));
    return passed;
  }
};

struct TextDetectionInfo {
  uint64_t DataStart{};
  uint64_t DataLength{};
  BlockType Type{}; // DEFAULT / TEXT / TEXT_EOL
};

// Detect text blocks (TEXT/TEXT_EOL) inside a DEFAULT block
static TextDetectionInfo detectText(File* in, uint64_t blockStart, uint64_t blockSize) {
  TextDetectionInfo detectionInfo;

  TextParserStateInfo textParser;
  textParser.reset(0);

  in->setpos(blockStart);
  const uint64_t n = blockSize;
  uint32_t buf0 = 0;

  for (uint64_t i = 0; i < n; ++i) {
    int c = in->getchar();
    if (c == EOF) {
      quit("detectText(): Unexpected end of file");
    }
    uint8_t pc = buf0 & 0xff;
    buf0 = buf0 << 8 | c;

    uint32_t t = TextParserStateInfo::utf8StateTable[c];
    textParser.UTF8State = TextParserStateInfo::utf8StateTable[256 + textParser.UTF8State + t];

    //some exceptions we still accept
    if (textParser.UTF8State == TextParserStateInfo::utf8Reject) { // illegal state
      if (c == 0 && pc >= 32 && pc <= 127 /* asciiz */) {
        textParser.UTF8State = TextParserStateInfo::utf8Accept;
      }
    }

    if (c == NEW_LINE) {
      if (pc != CARRIAGE_RETURN) {
        textParser.EOLType = 2; // mixed or LF-only
      }
      else if (textParser.EOLType == 0) {
        textParser.EOLType = 1; // CRLF-only
      }
    }

    if (textParser.UTF8State == TextParserStateInfo::utf8Accept) {
      textParser.invalidCount = textParser.invalidCount * (TextParserStateInfo::TEXT_ADAPT_RATE - 1) / TextParserStateInfo::TEXT_ADAPT_RATE;
      if (textParser.invalidCount == 0) {
        textParser.End = i;
      }
    }

    if( textParser.UTF8State == TextParserStateInfo::utf8Reject ) { // illegal state
      textParser.invalidCount = textParser.invalidCount * (TextParserStateInfo::TEXT_ADAPT_RATE - 1) / TextParserStateInfo::TEXT_ADAPT_RATE + TextParserStateInfo::TEXT_ADAPT_RATE;
      textParser.UTF8State = TextParserStateInfo::utf8Accept; // reset state
      if( textParser.invalidCount >= TextParserStateInfo::TEXT_MAX_MISSES * TextParserStateInfo::TEXT_ADAPT_RATE ) {

        //end of text block
        //if we have a large enough valid textblock, get it
        if (textParser.Start == 0 && textParser.isLargeText()) {
          detectionInfo.Type = textParser.EOLType == 1 ? BlockType::TEXT_EOL : BlockType::TEXT;
          detectionInfo.DataStart = blockStart;
          detectionInfo.DataLength = textParser.End;
          return detectionInfo;
        }

        textParser.reset(i + 1); // it's not text (or not long enough) - start over
      }
    }

    //early stop
    //we have a large enough text block, but it's not the first block
    if (textParser.Start != 0 && textParser.isLargeText()) {
      detectionInfo.Type = BlockType::DEFAULT;
      detectionInfo.DataStart = blockStart;
      detectionInfo.DataLength = textParser.Start;
      return detectionInfo;
    }
  }

  //TEXT
  if (textParser.Start == 0 && (
    textParser.End + 1 == n || /*< the whole block is text, no matter how small */
    textParser.isSmallText())) /*< the whole block is a large enough text block with some garbage */
  {
    detectionInfo.Type = textParser.EOLType == 1 ? BlockType::TEXT_EOL : BlockType::TEXT;
    detectionInfo.DataStart = blockStart;
    detectionInfo.DataLength = n;
    return detectionInfo;
  }

  //DEFAULT (->TEXT)
  if (textParser.Start != 0 && textParser.isLargeText()) {
    detectionInfo.Type = BlockType::DEFAULT;
    detectionInfo.DataStart = blockStart;
    detectionInfo.DataLength = textParser.Start; //could overshoot depending on how the inalidcount decays
    return detectionInfo;
  }

  //no text found at all, or it is too small
  //DEFAULT
  detectionInfo.Type = BlockType::DEFAULT;
  detectionInfo.DataStart = blockStart;
  detectionInfo.DataLength = n;
  return detectionInfo;
}

struct dBASE {
  uint8_t Version{};
  uint32_t nRecords{};
  uint16_t RecordLength{};
  uint16_t HeaderLength{};
};

// Detect blocks
static DetectionInfo detect(File *in, uint64_t blockSize, const TransformOptions* const transformOptions) {

  DetectionInfo detectionInfo;

  int textParserState = 0;
  uint64_t blockHash = 0;

  // TODO: Large file support
  const uint64_t n = blockSize;

  // last 16 bytes
  uint32_t buf3 = 0;
  uint32_t buf2 = 0;
  uint32_t buf1 = 0;
  uint32_t buf0 = 0;
  
  uint64_t start = 0;

  start = in->curPos(); // start of the current block
  
  // For EXE detection
  Array<uint64_t> absPos(256); // CALL/JMP abs. address. low byte -> last offset
  Array<uint64_t> relPos(256); // CALL/JMP relative address. low byte -> last offset
  int e8e9count = 0; // number of consecutive CALL/JMPs
  uint64_t e8e9pos = 0; // offset of first CALL or JMP instruction
  uint64_t e8e9last = 0; // offset of most recent CALL or JMP
  
  // For DEC Alpha detection
  int decAlpha = 0;
  uint64_t decAlphaHeaderStart = 0;
  uint16_t decAlphaNumberOfSections = 0;
  uint64_t decAlphaNextExpectedOffset = 0;
  uint64_t decAlphaSectionStart = 0;
  uint64_t decAlphaSectionLen = 0;

  // For JPEG detection
  uint64_t soi = 0; // Start Of Image 
  uint64_t sof = 0; // Start Of Frame
  uint64_t sos = 0; // Start Of Scan
  uint64_t app = 0; // Application-specific marker

  // For WAVE detection
  uint64_t wavi = 0;
  int wavSize = 0; // filesize
  int wavch = 0; // number of channels: 1 or 2
  int wavbps = 0; // bits per sample: 8 or 16
  int wavm = 0;
  int wavType = 0; // 1 for WAV, 2 for SF2
  int wavLen = 0;
  uint64_t wavlist = 0; //position of a LIST info chunk (if present)
  
  // For AIFF detection
  uint64_t aiff = 0;
  int aiffm = 0;
  int aiffs = 0; 
  
  // For S3M detection
  uint64_t s3mi = 0;
  int s3Mno = 0;
  int s3Mni = 0; 
   
  // For MRB detection
  uint64_t mrb = 0; 
  uint8_t mrbPictureType = 0;
  uint8_t mrbPackingMethod = 0;
  uint16_t mrbmulti = 0; // number of pictures in multi-resolution-bitmap minus 1

  // For BMP detection
  uint64_t bmpi = 0;
  uint64_t dibi = 0;
  int imgbpp = 0;
  int bmpx = 0;
  int bmpy = 0;
  int bmpof = 0;
  int bmps = 0;
  int nColors = 0;
  
  // For RGB detection
  uint64_t rgbi = 0;
  int rgbx = 0;
  int rgby = 0; 
  
  // For TGA detection
  uint64_t tga = 0;
  int tgax = 0;
  int tgay = 0;
  int tgaz = 0;
  int tgat = 0;
  int tgaid = 0;
  int tgamap = 0; 
  
  // For PBM (Portable BitMap), PGM (Portable GrayMap), PPM (Portable PixMap), PAM (Portable Arbitrary Map) detection
  uint64_t pgm = 0;
  int pgmComment = 0; // flag for presence of a comment line
  int pgmw = 0; // width
  int pgmh = 0; // height
  int pgmc = 0;  // color depth
  int pgmn = 0; // format: 4: PBM, 5: PGM 6: PPM, 7: PAM
  int pamatr = 0; // currently processed PAM attribute
  int pamd = 0; // PAM color depth (number of color channes)
  uint64_t pgmdata = 0; // image data start
  uint64_t pgmDataSize = 0; // image data size in bytes
  int pgmPtr = 0; // index in pgmBuf
  char pgmBuf[32];
  
  // For CD sectors detection
  uint64_t cdi = 0;
  int cda = 0;
  int cdm = 0; 
  uint32_t cdf = 0;

  // For ZLIB stream detection
  uint8_t zBuf[256 + 32] = {0};
  uint8_t zin[1 << 16] = {0};
  uint8_t zout[1 << 16] = {0};

  // For DBF detection
  dBASE dbase{};
  uint64_t dbasei = 0;

  int zBufPos = 0;
  uint64_t zZipPos = UINT64_MAX;
  int histogram[256] = {0};
  int pdfIm = 0;
  int pdfImW = 0;
  int pdfImH = 0;
  int pdfImB = 0;
  int pdfGray = 0;
  uint64_t pdfimp = 0;

  // For base64 detection
  int b64S = 0;
  uint64_t b64I = 0;
  uint64_t b64Line = 0;
  uint64_t b64Nl = 0;
  uint64_t b64End = 0;

  // For base85 (ascii85) detection
  int b85state = 0; //0 or 1
  int b85linelength = 0;
  uint64_t base85start = 0;
  uint64_t base85end = 0;

  // For GIF detection
  uint64_t gifi = 0;
  uint64_t gifa = 0;
  int gif = 0;
  int gifw = 0;
  int gifc = 0;
  int gifb = 0;
  int gifplt = 0;
  static bool gifGray = false;
  
  // For PNG detection
  uint64_t png = 0;
  int pngw = 0;
  int pngh = 0;
  int pngbps = 0;
  int pngType = 0;
  int pnggray = 0;
  int lastChunk = 0;
  uint64_t nextChunk = 0;

  for (uint64_t i = 0; i < n; ++i) {
    int c = in->getchar();
    if (c == EOF) {
      quit("detect(): Unexpected end of file");
    }

    blockHash = hash(blockHash, c);

    buf3 = buf3 << 8 | buf2 >> 24;
    buf2 = buf2 << 8 | buf1 >> 24;
    buf1 = buf1 << 8 | buf0 >> 24;
    buf0 = buf0 << 8 | c;

    // helper for .pbm .pgm .ppm .pam detection
    uint32_t t = TextParserStateInfo::utf8StateTable[c];
    textParserState = TextParserStateInfo::utf8StateTable[256 + textParserState + t];

    // detect PNG images
    if ((png == 0) && buf3 == 0x89504E47 /*%PNG*/ && buf2 == 0x0D0A1A0A && buf1 == 0x0000000D && buf0 == 0x49484452) {
      png = i, pngType = -1, lastChunk = buf3;
    }
    if (png != 0) {
      const uint64_t p = i - png;
      if (p == 12) {
        pngw = buf2;
        pngh = buf1;
        pngbps = buf0 >> 24;
        pngType = static_cast<uint8_t>(buf0 >> 16); //Color type codes represent sums of the following values: 1 (palette used), 2 (color used), and 4 (alpha channel used). Valid values are 0, 2, 3, 4, and 6.
        pnggray = 0;
        png *= static_cast<int>((buf0 & 0xFFFF) == 0 && (pngw != 0) && (pngh != 0) && pngbps == 8 &&
          ((pngType == 0) || pngType == 2 || pngType == 3 || pngType == 4 || pngType == 6));
      }
      else if (p > 12 && pngType < 0) {
        png = 0;
      }
      else if (p == 17) {
        png *= static_cast<int>((buf1 & 0xFF) == 0);
        nextChunk = (png) != 0 ? i + 8 : 0;
      }
      else if (p > 17 && i == nextChunk) {
        nextChunk += buf1 + 4 /*CRC*/ + 8 /*Chunk length+id*/;
        lastChunk = buf0;
        png *= static_cast<int>(lastChunk != 0x49454E44 /*IEND*/);
        if (lastChunk == 0x504C5445 /*PLTE*/) {
          png *= static_cast<int>(buf1 % 3 == 0);
          pnggray = static_cast<int>((png != 0) && isGrayscalePalette(in, buf1 / 3));
        }
      }
    }

    // ZLIB stream detection
#ifndef DISABLE_ZLIB
    histogram[c]++;
    if (i >= 256)
      histogram[zBuf[zBufPos]]--;
    zBuf[zBufPos] = c;
    if (zBufPos < 32)
      zBuf[zBufPos + 256] = c;
    zBufPos = (zBufPos + 1) & 0xFF;

    int zh = parseZlibHeader(((int)zBuf[(zBufPos - 32) & 0xFF]) * 256 + (int)zBuf[(zBufPos - 32 + 1) & 0xFF]);
    bool valid = (i >= 31 && zh != -1);
    if (!valid && transformOptions->useBruteForceDeflateDetection && i >= 255) {
      uint8_t bType = (zBuf[zBufPos] & 7) >> 1;
      if ((valid = (bType == 1 || bType == 2))) {
        int maximum = 0, used = 0, offset = zBufPos;
        for (int i = 0; i < 4; i++, offset += 64) {
          for (int j = 0; j < 64; j++) {
            int freq = histogram[zBuf[(offset + j) & 0xFF]];
            used += (freq > 0);
            maximum += (freq > maximum);
          }
          if (maximum >= ((12 + i) << i) || used * (6 - i) < (i + 1) * 64) {
            valid = false;
            break;
          }
        }
      }
    }
    if (valid || zZipPos == i) {
      int streamLength = 0, ret = 0, brute = (zh == -1 && zZipPos != i);

      // Quick check possible stream by decompressing first 32 bytes
      z_stream strm;
      strm.zalloc = Z_NULL;
      strm.zfree = Z_NULL;
      strm.opaque = Z_NULL;
      strm.next_in = Z_NULL;
      strm.avail_in = 0;
      if (zlibInflateInit(&strm, zh) == Z_OK) {
        strm.next_in = &zBuf[(zBufPos - (brute ? 0 : 32)) & 0xFF];
        strm.avail_in = 32;
        strm.next_out = zout;
        strm.avail_out = 1 << 16;
        ret = inflate(&strm, Z_FINISH);
        ret = (inflateEnd(&strm) == Z_OK && (ret == Z_STREAM_END || ret == Z_BUF_ERROR) && strm.total_in >= 16);
      }
      if (ret) {
        // Verify valid stream and determine stream length
        const uint64_t savedpos = in->curPos();
        strm.zalloc = Z_NULL;
        strm.zfree = Z_NULL;
        strm.opaque = Z_NULL;
        strm.next_in = Z_NULL;
        strm.avail_in = 0;
        strm.total_in = strm.total_out = 0;
        if (zlibInflateInit(&strm, zh) == Z_OK) {
          uint64_t blstart = static_cast<uint64_t>(std::max<int64_t>(i - (brute ? 255 : 31), 0));
          for (uint64_t j = blstart; j < n; j += 1 << 16) {
            uint32_t blsize = static_cast<uint32_t>(min(n - j, UINT64_C(1) << 16));
            in->setpos(start + j);
            if (in->blockRead(zin, blsize) != blsize)
              break;
            strm.next_in = zin;
            strm.avail_in = blsize;
            do {
              strm.next_out = zout;
              strm.avail_out = 1 << 16;
              ret = inflate(&strm, Z_FINISH);
            } while (strm.avail_out == 0 && ret == Z_BUF_ERROR);
            if (ret == Z_STREAM_END)
              streamLength = strm.total_in;
            if (ret != Z_BUF_ERROR)
              break;
          }
          if (inflateEnd(&strm) != Z_OK)
            streamLength = 0;
        }
        in->setpos(savedpos);
      }
      if (streamLength > (brute << 7)) {
        int info = 0;
        if (pdfImW > 0 && pdfImW < 0x1000000 && pdfImH > 0) {
          if (pdfImB == 8 && (int)strm.total_out == pdfImW * pdfImH)
            info = ((pdfGray ? BlockType::IMAGE8GRAY : BlockType::IMAGE8) << 24) | pdfImW;
          if (pdfImB == 8 && (int)strm.total_out == pdfImW * pdfImH * 3)
            info = (BlockType::IMAGE24 << 24) | pdfImW * 3;
          if (pdfImB == 4 && (int)strm.total_out == ((pdfImW + 1) / 2) * pdfImH)
            info = (BlockType::IMAGE4 << 24) | ((pdfImW + 1) / 2);
          if (pdfImB == 1 && (int)strm.total_out == ((pdfImW + 7) / 8) * pdfImH)
            info = (BlockType::IMAGE1 << 24) | ((pdfImW + 7) / 8);
          pdfGray = 0;
        }
        else if (png && pngw < 0x1000000 && lastChunk == 0x49444154 /*IDAT*/) {
          if (pngbps == 8 && pngType == 2 /*color, no alpha*/ && (int)strm.total_out == (pngw * 3 + 1) * pngh)
            info = (BlockType::PNG24 << 24) | (pngw * 3), png = 0;
          else if (pngbps == 8 && pngType == 6 /*color, with alpha*/ && (int)strm.total_out == (pngw * 4 + 1) * pngh)
            info = (BlockType::PNG32 << 24) | (pngw * 4), png = 0;
          else if (pngbps == 8 && (pngType == 0 /*no color channels, no palette, i.e. grayscale*/ || pngType == 3 /*palette, color*/) && (int)strm.total_out == (pngw + 1) * pngh)
            info = (((pngType == 0 /*grayscale png*/ || pnggray) ? BlockType::PNG8GRAY : BlockType::PNG8) << 24) | (pngw), png = 0;
        }
        detectionInfo.Type = BlockType::ZLIB;
        detectionInfo.DataInfo = info;
        detectionInfo.DataLength = streamLength;
        detectionInfo.DataStart = start + i - (brute ? 255 : 31);
        return detectionInfo;
      }
    }
    if (zh == -1 && zBuf[(zBufPos - 32) & 0xFF] == 'P' && zBuf[(zBufPos - 32 + 1) & 0xFF] == 'K' &&
      zBuf[(zBufPos - 32 + 2) & 0xFF] == '\x3' && zBuf[(zBufPos - 32 + 3) & 0xFF] == '\x4' && zBuf[(zBufPos - 32 + 8) & 0xFF] == '\x8' &&
      zBuf[(zBufPos - 32 + 9) & 0xFF] == '\0') {
      int nlen = (int)zBuf[(zBufPos - 32 + 26) & 0xFF] + ((int)zBuf[(zBufPos - 32 + 27) & 0xFF]) * 256 +
        (int)zBuf[(zBufPos - 32 + 28) & 0xFF] + ((int)zBuf[(zBufPos - 32 + 29) & 0xFF]) * 256;
      if (nlen < 256 && i + 30 + nlen < n)
        zZipPos = i + 30 + nlen;
    }
#endif //DISABLE_ZLIB

    // dBASE VERSIONS
    //  '02' > FoxBase
    //  '03' > dBase III without memo file
    //  '04' > dBase IV without memo file
    //  '05' > dBase V without memo file
    //  '07' > Visual Objects 1.x
    //  '30' > Visual FoxPro
    //  '31' > Visual FoxPro with AutoIncrement field
    //  '43' > dBASE IV SQL table files, no memo
    //  '63' > dBASE IV SQL system files, no memo
    //  '7b' > dBase IV with memo file
    //  '83' > dBase III with memo file
    //  '87' > Visual Objects 1.x with memo file
    //  '8b' > dBase IV with memo file
    //  '8e' > dBase IV with SQL table
    //  'cb' > dBASE IV SQL table files, with memo
    //  'f5' > FoxPro with memo file - tested
    //  'fb' > FoxPro without memo file
    //
    if (dbasei == 0 && ((c & 7) == 3 /*dBase level 3-5*/ || (c & 7) == 4 /*dBase level 7*/ || (c >> 4) == 3 || c == 0xf5 || c == 0x30)) {
      dbasei = i + 1;
      dbase.Version = ((c >> 4) == 3) ? 3 : c & 7;
    }
    if (dbasei) {
      const int p = int(i - dbasei + 1);
      //1-2-3: Date of last update; in YYMMDD format
      if (p == 1) { if (c < 83) dbasei = 0; } //year (the DBF file type was introduced with dBASE II in 1983.)
      else if (p == 2) { if (!(c > 0 && c < 13)) dbasei = 0; } //month
      else if (p == 3) { if (!(c > 0 && c < 32)) dbasei = 0; }//day
      //4-7: Number of records in the table. (Least significant byte first.)
      else if (p == 7) { if (!((dbase.nRecords = bswap(buf0)) > 0 && dbase.nRecords < 0x40000000)) dbasei = 0; }
      //8-9: Number of bytes in the header. (Least significant byte first.)
      else if (p == 9) { if (!((dbase.HeaderLength = ((buf0 >> 8) & 0xff) | (c << 8)) > 32 && (((dbase.HeaderLength - 32 - 1) % 32) == 0 || (dbase.HeaderLength > 255 + 8 && (((dbase.HeaderLength -= 255 + 8) - 32 - 1) % 32) == 0)))) dbasei = 0; }
      //10-11: Number of bytes in the record. (Least significant byte first.)
      else if (p == 11) { if (!(((dbase.RecordLength = ((buf0 >> 8) & 0xff) | (c << 8))) > 8 && dbase.HeaderLength + dbase.nRecords * dbase.RecordLength < blockSize)) dbasei = 0; }
      //12-13: Reserved; filled with zeros.
      //14: Flag indicating incomplete dBASE IV transaction. 0 or 1.
      //15: dBASE IV encryption flag. 0 or 1.
      else if (p == 15) { if ((buf0 & 0xfffffefe) != 0) dbasei = 0; }
      //16-27: Reserved for multi - user processing.
      //28: Production .mdx file flag; 1 if there is a production .mdx file, 0 if not
      else if (p == 28) { if ((c & 0xfe) != 0) dbasei = 0; }
      //30-31: Reserved; filled with zeros.
      else if (p == 31) { if ((buf0 & 0xffff) != 0) dbasei = 0; }
      //32: for dBase III, IV and 5 the 'Field descriptor array' starts now and is n*32 bytes, for level 7 it starts at 68 and is n*48 bytes
      else if (p == 32) {
        uint64_t savedpos = in->curPos();
        in->setpos(savedpos - 34 + dbase.HeaderLength);
        uint8_t marker = in->getchar(); // field descriptor array terminator, it must be 0x0d
        if (marker != 0x0d) { 
          dbasei = 0; 
          in->setpos(savedpos); 
        }
        else {
          uint32_t endPos = dbase.nRecords * dbase.RecordLength;
          uint64_t seekpos = endPos + in->curPos();
          in->setpos(seekpos);
          marker = in->getchar(); // file end marker, it must be 0x1a
          if (marker != 0x1a) {
            dbasei = 0;
            in->setpos(savedpos);
          }
          else {
            //success
            in->setpos(savedpos);
            detectionInfo.DBF_DET(start, BlockType::DBF, dbasei - 1, dbase.HeaderLength, dbase.nRecords* dbase.RecordLength + 1, dbase.RecordLength);
            return detectionInfo;
          }
        }
      }
    }

    if (i - pdfimp > 1024) {
      pdfIm = pdfImW = pdfImH = pdfImB = pdfGray = 0; // fail
    }
    if (pdfIm > 1 && !((isspace(c) != 0) || (isdigit(c) != 0))) {
      pdfIm = 1;
    }
    if (pdfIm == 2 && (isdigit(c) != 0)) {
      pdfImW = pdfImW * 10 + (c - '0');
    }
    if (pdfIm == 3 && (isdigit(c) != 0)) {
      pdfImH = pdfImH * 10 + (c - '0');
    }
    if (pdfIm == 4 && (isdigit(c) != 0)) {
      pdfImB = pdfImB * 10 + (c - '0');
    }
    if ((buf0 & 0xffff) == 0x3c3c) {
      pdfimp = i, pdfIm = 1; // <<
    }
    if ((pdfIm != 0) && (buf1 & 0xffff) == 0x2f57 && buf0 == 0x69647468) {
      pdfIm = 2, pdfImW = 0; // /Width
    }
    if ((pdfIm != 0) && (buf1 & 0xffffff) == 0x2f4865 && buf0 == 0x69676874) {
      pdfIm = 3, pdfImH = 0; // /Height
    }
    if ((pdfIm != 0) && buf3 == 0x42697473 && buf2 == 0x50657243 && buf1 == 0x6f6d706f && buf0 == 0x6e656e74 &&
        zBuf[(zBufPos - 32 + 15) & 0xFF] == '/') {
      pdfIm = 4, pdfImB = 0; // /BitsPerComponent
    }
    if ((pdfIm != 0) && (buf2 & 0xFFFFFF) == 0x2F4465 && buf1 == 0x76696365 && buf0 == 0x47726179) {
      pdfGray = 1; // /DeviceGray
    }

    // CD sectors detection (mode 1 and mode 2 form 1+2 - 2352 bytes)
    if (buf1 == 0x00ffffff && buf0 == 0xffffffff && (cdi == 0)) {
      cdi = i, cda = -1, cdm = 0;
    }
    if ((cdi != 0) && i > cdi) {
      const int p = (i - cdi) % 2352;
      if (p == 8 && (buf1 != 0xffffff00 || ((buf0 & 0xff) != 1 && (buf0 & 0xff) != 2))) {
        cdi = 0;
      }
      else if (p == 16 && i + 2336 < n) {
        uint8_t data[2352];
        const uint64_t savedPos = in->curPos();
        in->setpos(start + i - 23);
        in->blockRead(data, 2352);
        in->setpos(savedPos);
        int t = CdFilter::expandCdSector(data, cda, 1);
        if (t != cdm) {
          cdm = t * static_cast<int>(i - cdi < 2352);
        }
        if ((cdm != 0) && cda != 10 && (cdm == 1 || buf0 == buf1)) {
          if (detectionInfo.Type != BlockType::CD) {
            detectionInfo.Type = BlockType::CD;
            detectionInfo.DataStart = start + cdi - 7;
            detectionInfo.DataInfo = cdm;
          }
          cda = (data[12] << 16) + (data[13] << 8) + data[14];
          if (cdm != 1 && i - cdi > 2352 && buf0 != cdf) {
            cda = 10;
          }
          if (cdm != 1) {
            cdf = buf0;
          }
        }
        else {
          cdi = 0;
        }
      }
      if ((i + 1 == n || cdi == 0) && detectionInfo.Type == BlockType::CD) {
        detectionInfo.DataLength = (start + i - p - 7) - detectionInfo.DataStart;
        return detectionInfo;
      }
    }
    if (detectionInfo.Type == BlockType::CD) {
      continue;
    }

    // Detect JPEG by code SOI APPx (FF D8 FF Ex) followed by
    // SOF0 (FF C0 xx xx 08) and SOS (FF DA) within a reasonable distance.
    // Detect end by any code other than RST0-RST7 (FF D9-D7) or
    // a byte stuff (FF 00).

    if ((soi == 0) && i >= 3 && (buf0 & 0xffffff00) == 0xffd8ff00 && ((buf0 & 0xFE) == 0xC0 || static_cast<uint8_t>(buf0) == 0xC4 ||
        (static_cast<uint8_t>(buf0) >= 0xDB && static_cast<uint8_t>(buf0) <= 0xFE))) {
      soi = i, app = i + 2, sos = sof = 0;
    }
    if (soi != 0) {
      if (app == i && (buf0 >> 24) == 0xff && ((buf0 >> 16) & 0xff) > 0xc1 && ((buf0 >> 16) & 0xff) < 0xff) {
        app = i + (buf0 & 0xffff) + 2;
      }
      if (app < i && (buf1 & 0xff) == 0xff && (buf0 & 0xfe0000ff) == 0xc0000008) {
        sof = i;
      }
      if ((sof != 0) && sof > soi && i - sof < 0x1000 && (buf0 & 0xffff) == 0xffda) {
        sos = i;
        if (detectionInfo.Type != BlockType::JPEG) {
          detectionInfo.Type = BlockType::JPEG;
          detectionInfo.DataStart = start + soi - 3;
        }
      }
      if (i - soi > 0x40000 && (sos == 0)) {
        soi = 0;
      }
    }
    if (detectionInfo.Type == BlockType::JPEG && (sos != 0) && i > sos && (buf0 & 0xff00) == 0xff00 && (buf0 & 0xff) != 0 && (buf0 & 0xf8) != 0xd0) {
      detectionInfo.DataLength = start + i +1 - detectionInfo.DataStart;
      return detectionInfo;
    }

    // Detect .wav file header
    if (buf0 == 0x52494646 /*RIFF*/) { 
      wavi = i;
      wavm = wavLen = 0;
    }
    if (wavi != 0) {
      uint64_t p = i - wavi;
      if (p == 4) {
        wavSize = bswap(buf0); //fileSize
      }
      else if (p == 8) {
        wavType = (buf0 == 0x57415645 /*WAVE*/) ? 1 : (buf0 == 0x7366626B /*sfbk*/) ? 2 : 0;
        if (wavType == 0) {
          wavi = 0;
        }
      }
      else if (wavType != 0) {
        if (wavType == 1) { // type: WAVE
          if (p == 16 + wavLen && (buf1 != 0x666d7420 /*"fmt "*/ || ((wavm = bswap(buf0) - 16) & 0xFFFFFFFD) != 0)) {
            wavLen = ((bswap(buf0) + 1) & (-2)) + 8, wavi *= static_cast<int>(buf1 == 0x666d7420 /*"fmt "*/ && (wavm & 0xFFFFFFFD) != 0);
          }
          else if (p == 22 + wavLen) {
            wavch = bswap(buf0) & 0xffff; // number of channels: 1 or 2
          }
          else if (p == 34 + wavLen) {
            wavbps = bswap(buf0) & 0xffff; // bits per sample: 8 or 16
          }
          else if (p == 40 + wavLen + wavm && buf1 != 0x64617461 /*"data"*/) {
            wavm += ((bswap(buf0) + 1) & (-2)) + 8, wavi = (wavm > 0xfffff ? 0 : wavi);
          }
          else if (p == 40 + wavLen + wavm) {  // The "data" subchunk contains the actual audio data
            int wavD = bswap(buf0); // size of data section
            wavLen = 0;
            if ((wavch == 1 || wavch == 2) && (wavbps == 8 || wavbps == 16) && wavD > 0 && wavSize >= wavD + 36 &&
                wavD % ((wavbps / 8) * wavch) == 0) {
              detectionInfo.AUD_DET(start, (wavbps == 8) ? BlockType::AUDIO : BlockType::AUDIO_LE, wavi - 3, 44 + wavm, wavD, wavch + wavbps / 4 - 3);
              if (detectionInfo.SizeVerificationPassed(start + n))
                return detectionInfo;
            }
            wavi = 0;
          }
        }
        else { // format: SF2
          if ((p == 16 && buf1 != 0x4C495354 /*LIST*/) || (p == 20 && buf0 != 0x494E464F /*INFO*/)) {
            wavi = 0;
          }
          else if (p > 20 && buf1 == 0x4C495354 /*LIST*/) {
            wavLen = bswap(buf0);
            if (wavLen != 0) {
              wavlist = i;
            }
            else {
              wavi = 0; // fail; bad format
            }
          }
          else if (wavlist != 0) {
            p = i - wavlist;
            if (p == 8 && (buf1 != 0x73647461 /*sdta*/ || buf0 != 0x736D706C /*smpl*/)) {
              wavi = 0; // fail; bad format
            }
            else if (p == 12) {
              int wavD = bswap(buf0); // size of data section
              if ((wavD != 0) && (wavD + 12) == wavLen) {
                detectionInfo.AUD_DET(start, BlockType::AUDIO_LE, wavi - 3, (12 + wavlist - (wavi - 3) + 1) & ~1, wavD, 1 + 16 / 4 - 3 /*mono, 16-bit*/);
                if (detectionInfo.SizeVerificationPassed(start + n))
                  return detectionInfo;
              }
              wavi = 0; // fail; bad format
            }
          }
        }
      }
    }

    // Detect .aiff file header
    if (buf0 == 0x464f524d /*FORM*/) {
      aiff = i, aiffs = 0;
    }
    if (aiff != 0) {
      const uint64_t p = i - aiff;
      if (p == 12 && (buf1 != 0x41494646 /*AIFF*/ || buf0 != 0x434f4d4d /*COMM*/)) {
        aiff = 0; // fail
      }
      else if (p == 24) {
        const int bits = buf0 & 0xffff;
        const int chn = buf1 >> 16;
        if ((bits == 8 || bits == 16) && (chn == 1 || chn == 2)) {
          aiffm = chn + bits / 4 - 3 + 4;
        }
        else {
          aiff = 0; //fail
        }
      }
      else if (p == 42 + aiffs && buf1 != 0x53534e44 /*SSND*/) {
        aiffs += (buf0 + 8) + (buf0 & 1);
        if (aiffs > 0x400) {
          aiff = 0; //fail
        }
      }
      else if (p == 42 + aiffs) { // Sound Data Chunk
        detectionInfo.AUD_DET(start, BlockType::AUDIO, aiff - 3, 54 + aiffs, buf0 - 8, aiffm);
        if (detectionInfo.SizeVerificationPassed(start + n))
          return detectionInfo;
        else
          aiff = 0; //fail
      }
    }

    // Detect .mod file header
    if ((buf0 == 0x4d2e4b2e || buf0 == 0x3643484e || buf0 == 0x3843484e // m.K. 6CHN 8CHN
        || buf0 == 0x464c5434 || buf0 == 0x464c5438) && (buf1 & 0xc0c0c0c0) == 0 && i >= 1083) {
      const uint64_t savedPos = in->curPos();
      const int chn = ((buf0 >> 24) == 0x36 ? 6 : (((buf0 >> 24) == 0x38 || (buf0 & 0xff) == 0x38) ? 8 : 4));
      int len = 0; // total length of samples
      int numPat = 1; // number of patterns
      for (int j = 0; j < 31; j++) {
        in->setpos(start + i - 1083 + 42 + j * 30);
        const int i1 = in->getchar();
        const int i2 = in->getchar();
        len += i1 * 512 + i2 * 2;
      }
      in->setpos(start + i - 131);
      for (int j = 0; j < 128; j++) {
        int x = in->getchar();
        if (x + 1 > numPat) {
          numPat = x + 1;
        }
      }
      if (numPat < 65) {
        detectionInfo.AUD_DET(start, BlockType::AUDIO, i - 1083, 1084 + numPat * 256 * chn, len, 4 /*mono, 8-bit*/);
        if (detectionInfo.SizeVerificationPassed(start + n))
          return detectionInfo;
      }
      in->setpos(savedPos);
    }

    // Detect .s3m file header
    if (buf0 == 0x1a100000) { //0x1A: signature byte, 0x10: song type, 0x0000: reserved
      s3mi = i, s3Mno = s3Mni = 0; 
    }
    if (s3mi != 0) {
      const uint64_t p = i - s3mi;
      if (p == 4) {
        s3Mno = bswap(buf0) & 0xffff; //Number of entries in the order table, should be even
        s3Mni = (bswap(buf0) >> 16); //Number of instruments in the song
      }
      else if (p == 16 && (((buf1 >> 16) & 0xff) != 0x13 || buf0 != 0x5343524d /*SCRM*/)) {
        s3mi = 0;
      }
      else if (p == 16) {
        const uint64_t savedPos = in->curPos();
        int b[31];
        int samStart = (1 << 16);
        int samEnd = 0;
        int ok = 1;
        for (int j = 0; j < s3Mni; j++) {
          in->setpos(start + s3mi - 31 + 0x60 + s3Mno + j * 2);
          int i1 = in->getchar();
          i1 += in->getchar() * 256;
          in->setpos(start + s3mi - 31 + i1 * 16);
          i1 = in->getchar();
          if (i1 == 1) { // type: sample
            for (int k = 0; k < 31; k++) {
              b[k] = in->getchar();
            }
            int len = b[15] + (b[16] << 8);
            int ofs = b[13] + (b[14] << 8);
            if (b[30] > 1) {
              ok = 0;
            }
            if (ofs * 16 < samStart) {
              samStart = ofs * 16;
            }
            if (ofs * 16 + len > samEnd) {
              samEnd = ofs * 16 + len;
            }
          }
        }
        if ((ok != 0) && samStart < (1 << 16)) {
          detectionInfo.AUD_DET(start, BlockType::AUDIO, s3mi - 31, samStart, samEnd - samStart, 0 /*mono, 8-bit*/);
          if (detectionInfo.SizeVerificationPassed(start + n))
            return detectionInfo;
        }
        s3mi = 0;
        in->setpos(savedPos);
      }
    }


    //detect uncompressed and rle encoded mrb files inside windows hlp files 506C
    //we support only single images
    if (!mrb && ((buf0 & 0xFFFF) == 0x6c70 || (buf0 & 0xFFFF) == 0x6C50) && !b64S && !cdi) { //Magic: 0x506C (SHG,lP) or 0x706C (MRB,lp)
      mrb = i; 
      mrbmulti = 0;
    }
    if (mrb != 0) {
      const int p = int(i - mrb) - mrbmulti * 4; // p: offset of first picture descriptor
      if (p == 1 && c > 1 && c < 4 && mrbmulti == 0) mrbmulti = c - 1; //c: NumberOfPictures
      else if (p == 1 && c == 0) mrb = 0; // fail
      else if (p == 7) {
        if ((c == 5 || c == 6)) mrbPictureType = c; // 5=DDB   6=DIB   8=metafile
        else mrb = 0; // fail
      }
      else if (p == 8) {
        if (c <= 3) mrbPackingMethod = c; // c: 0=uncompressed 1=RunLen 2=LZ77 3=both
        else mrb = 0; // fail
      }
      else if (p == 10) {
        if (mrbPictureType == 6) { // DIB
          uint64_t mrbTell = in->curPos() - 2; //save curPos so we can restore it
          in->setpos(mrbTell);
          uint32_t Xdpi = GetCDWord(in);
          uint32_t Ydpi = GetCDWord(in);
          uint16_t Planes = GetCWord(in);
          uint16_t BitCount = GetCWord(in);  //4: 16 colors, 8: 256 colors
          uint32_t mrbw = GetCDWord(in);
          uint32_t mrbh = GetCDWord(in);
          uint32_t ColorsUsed = GetCDWord(in);
          uint32_t ColorsImportant = GetCDWord(in);
          uint32_t mrbcsize = GetCDWord(in); // compressedSize
          uint32_t HotspotSize = GetCDWord(in);
          uint32_t CompressedOffset = bswap(in->get32());
          uint32_t HotspotOffset = bswap(in->get32());
          uint64_t mrbsize = mrbcsize + in->curPos() - mrbTell + 10 + (1 << BitCount) * 4; // ignore HotspotSize
          mrbTell = mrbTell + 2;
          in->setpos(mrbTell);
          int pixelBytes = (mrbw * mrbh * BitCount) >> 3;
          //debug:
          //printf("MRB: %d, %d, %d, %d, %d, %d, %d, %d\n", mrbPictureType, mrbPackingMethod, BitCount, ColorsUsed, mrbw, mrbh, mrbcsize, mrbmulti);
          if (!(BitCount == 1 || BitCount == 4 || BitCount == 8) || mrbw < 4 || mrbh < 4 || mrbw > 1024 || mrbh >= 4096 || mrbsize == 0 || mrbPackingMethod > 1) {
            mrb = 0; // fail
          }
          else if (mrbPackingMethod <= 1 && pixelBytes < 360) {
            //debug:
            //printf("MRB: skipping\n");
            mrb = 0; // block is too small to be worth processing as a new block
          }
          else {
            // success
            detectionInfo.MRB_DET(start, BlockType::MRB, mrbPackingMethod, BitCount, mrb - 1, mrbsize - mrbcsize, mrbcsize, mrbw, mrbh);
            if (detectionInfo.SizeVerificationPassed(start + n))
              return detectionInfo;
            else
              mrb = 0;
          }
        }
        else {
          //unsupported format
          mrb = 0; // fail
        }
      }
    }

    // Detect .bmp image
    if ((bmpi == 0) && (dibi == 0)) {
      if ((buf0 & 0xffff) == 16973) { // 'BM'
        bmpi = i; // header start: bmpi-1
        dibi = i - 1 + 18; // we expect a DIB header to come
      }
      else if (buf0 == 0x28000000) { // headerless (DIB-only)
        dibi = i + 1;
      }
    }
    else {
      const uint64_t p = i - dibi + 1 + 18;
      if (p == 10 + 4) {
        bmpof = bswap(buf0), bmpi = (bmpof < 54 || start + blockSize < bmpi - 1 + bmpof) ? (dibi = 0)
          : bmpi; //offset of pixel data (this field is still located in the BMP Header)
      }
      else if (p == 14 + 4 && buf0 != 0x28000000) {
        bmpi = dibi = 0; //BITMAPINFOHEADER (0x28)
      }
      else if (p == 18 + 4) {
        bmpx = bswap(buf0), bmpi = (((bmpx & 0xff000000) != 0 || bmpx == 0) ? (dibi = 0) : bmpi); //width
      }
      else if (p == 22 + 4) {
        bmpy = abs(int(bswap(buf0))), bmpi = (((bmpy & 0xff000000) != 0 || bmpy == 0) ? (dibi = 0) : bmpi); //height
      }
      else if (p == 26 + 2) {
        bmpi = ((bswap(buf0 << 16)) != 1) ? (dibi = 0) : bmpi; //number of color planes (must be 1)
      }
      else if (p == 28 + 2) {
        imgbpp = bswap(buf0 << 16), bmpi = ((imgbpp != 1 && imgbpp != 4 && imgbpp != 8 && imgbpp != 24 && imgbpp != 32) ? (dibi = 0)
          : bmpi); //color depth
      }
      else if (p == 30 + 4) {
        bmpi = ((buf0 != 0) ? (dibi = 0) : bmpi); //compression method must be: BI_RGB (uncompressed)
      }
      else if (p == 34 + 4) {
        bmps = bswap(buf0); //image size or 0
        //else if (p==38+4) ; // the horizontal resolution of the image (ignored)
        //else if (p==42+4) ; // the vertical resolution of the image (ignored)
      }
      else if (p == 46 + 4) {
        nColors = bswap(buf0); // the number of colors in the color palette, or 0 to default to (1<<imgbpp)
        if (nColors == 0 && imgbpp <= 8) {
          nColors = 1 << imgbpp;
        }
        if (nColors > (1 << imgbpp) || (imgbpp > 8 && nColors > 0)) {
          bmpi = dibi = 0;
        }
      }
      else if (p == 50 + 4) { //the number of important colors used
        if (bswap(buf0) <= static_cast<uint32_t>(nColors) || bswap(buf0) == 0x10000000) {
          if (bmpi == 0 /*headerless*/ && (bmpx * 2 == bmpy) && imgbpp > 1 && // possible icon/cursor?
            ((bmps > 0 && bmps == ((bmpx * bmpy * (imgbpp + 1)) >> 4)) || (((bmps == 0) || bmps < ((bmpx * bmpy * imgbpp) >> 3)) &&
              ((bmpx == 8)  || (bmpx == 10) || (bmpx == 14) || (bmpx == 16) ||
               (bmpx == 20) || (bmpx == 22) || (bmpx == 24) ||
               (bmpx == 32) || (bmpx == 40) || (bmpx == 48) ||
               (bmpx == 60) || (bmpx == 64) || (bmpx == 72) ||
               (bmpx == 80) || (bmpx == 96) || (bmpx == 128) ||
               (bmpx == 256))))) {
            bmpy = bmpx;
          }

          BlockType blockType = BlockType::DEFAULT;
          int widthInBytes = 0;
          if (imgbpp == 1) {
            blockType = BlockType::IMAGE1;
            widthInBytes = (((bmpx - 1) >> 5) + 1) * 4;
          }
          else if (imgbpp == 4) {
            blockType = BlockType::IMAGE4;
            widthInBytes = ((bmpx * 4 + 31) >> 5) * 4;
          }
          else if (imgbpp == 8) {
            blockType = BlockType::IMAGE8;
            widthInBytes = (bmpx + 3) & -4;
          }
          else if (imgbpp == 24) {
            blockType = BlockType::IMAGE24;
            widthInBytes = ((bmpx * 3) + 3) & -4;
          }
          else if (imgbpp == 32) {
            blockType = BlockType::IMAGE32;
            widthInBytes = bmpx * 4;
          }

          if (imgbpp == 8) {
            const uint64_t colorPalettePos = dibi - 18 + 54;
            const uint64_t savedPos = in->curPos();
            in->setpos(colorPalettePos);
            if (isGrayscalePalette(in, nColors, 1)) {
              blockType = BlockType::IMAGE8GRAY;
            }
            in->setpos(savedPos);
          }

          const uint64_t headerPos = bmpi > 0 ? bmpi - 1 : dibi - 4;
          const uint64_t minHeaderSize = (bmpi > 0 ? 54 : 54 - 14) + nColors * 4;
          const uint64_t headerSize = bmpi > 0 ? bmpof : minHeaderSize;

          // some final sanity checks
          if (bmps != 0 &&
            bmps < widthInBytes * bmpy) { /*printf("\nBMP guard: image is larger than reported in header\n",bmps,widthInBytes*bmpy);*/
          }
          else if (start + blockSize < headerPos + headerSize + widthInBytes * bmpy) { /*printf("\nBMP guard: cropped data\n");*/
          }
          else if (headerSize == (bmpi > 0 ? 54 : 54 - 14) && nColors > 0) { /*printf("\nBMP guard: missing palette\n");*/
          }
          else if (bmpi > 0 && bmpof < minHeaderSize) { /*printf("\nBMP guard: overlapping color palette\n");*/
          }
          else if (bmpi > 0 && uint64_t(bmpi) - 1 + bmpof + widthInBytes * bmpy >
            start + blockSize) { /*printf("\nBMP guard: reported pixel data offset is incorrect\n");*/
          }
          else if (widthInBytes * bmpy <= 64) { /*printf("\nBMP guard: too small\n");*/
          } // too small - not worthy to use the image models
          else {
            detectionInfo.IMG_DET(start, blockType, headerPos, headerSize, widthInBytes, bmpy);
            if (detectionInfo.SizeVerificationPassed(start + n))
              return detectionInfo;
          }
        }
        bmpi = dibi = 0;
      }
    }

    // Detect binary .pbm .pgm .ppm .pam images
    if ((buf0 & 0xfff0ff) == 0x50300a) { //"Px" + line feed, where "x" shall be a number
      pgmn = (buf0 & 0xf00) >> 8; // extract "x"
      if ((pgmn >= 4 && pgmn <= 6) || pgmn == 7) {
        pgm = i, pgmdata = pgmDataSize = 0; // "P4" (pbm), "P5" (pgm), "P6" (ppm), "P7" (pam)
        pgmw = pgmh = pgmc = pamd = pgmComment = 0;
        pgmPtr = 0;
        pamatr = 0;
      }
    }
    if (pgm != 0) {
      if (pgmdata == 0) { // parse header
        if (i - pgm == 1 && c == '#') {
          pgmComment = 1; // # (pgm comment)
        }
        if ((pgmComment == 0) && (pgmPtr != 0)) {
          uint64_t s = 0;
          if (pgmn == 7) {
            if ((buf1 & 0xdfdf) == 0x5749 && (buf0 & 0xdfdfdfff) == 0x44544820) {
              pgmPtr = 0, pamatr = 1; // WIDTH
            }
            if ((buf1 & 0xdfdfdf) == 0x484549 && (buf0 & 0xdfdfdfff) == 0x47485420) {
              pgmPtr = 0, pamatr = 2; // HEIGHT
            }
            if ((buf1 & 0xdfdfdf) == 0x4d4158 && (buf0 & 0xdfdfdfff) == 0x56414c20) {
              pgmPtr = 0, pamatr = 3; // MAXVAL
            }
            if ((buf1 & 0xdfdf) == 0x4445 && (buf0 & 0xdfdfdfff) == 0x50544820) {
              pgmPtr = 0, pamatr = 4; // DEPTH
            }
            if ((buf2 & 0xdf) == 0x54 && (buf1 & 0xdfdfdfdf) == 0x55504c54 && (buf0 & 0xdfdfdfff) == 0x59504520) {
              pgmPtr = 0, pamatr = 5; // TUPLTYPE
            }
            if ((buf1 & 0xdfdfdf) == 0x454e44 && (buf0 & 0xdfdfdfff) == 0x4844520a) {
              pgmPtr = 0, pamatr = 6; // ENDHDR
            }
            if (c == 0x0a) {
              if (pamatr == 0) {
                pgm = 0;
              }
              else if (pamatr < 5) {
                s = pamatr;
              }
              if (pamatr != 6) {
                pamatr = 0;
              }
            }
          }
          else if (c == 0x20 && (pgmw == 0)) {
            s = 1;
          }
          else if (c == 0x0a && (pgmh == 0)) {
            s = 2;
          }
          else if (c == 0x0a && (pgmc == 0) && pgmn != 4) {
            s = 3;
          }
          if (s != 0) {
            pgmBuf[pgmPtr++] = 0;
            int v = atoi(pgmBuf); // parse width/height/depth/maxval value
            if (s == 1) {
              pgmw = v;
            }
            else if (s == 2) {
              pgmh = v;
            }
            else if (s == 3) {
              pgmc = v;
            }
            else if (s == 4) {
              pamd = v;
            }
            if (v == 0 || (s == 3 && v > 255)) {
              pgm = 0;
            }
            else {
              pgmPtr = 0;
            }
          }
        }
        if (pgmComment == 0) {
          pgmBuf[pgmPtr++] = c;
        }
        if (pgmPtr >= 32) {
          pgm = 0;
        }
        if ((pgmComment != 0) && c == 0x0a) {
          pgmComment = 0;
        }
        if ((pgmw != 0) && (pgmh != 0) && (pgmc == 0) && pgmn == 4) {
          pgmdata = i;
          pgmDataSize = (pgmw + 7) / 8 * pgmh;
          textParserState = 0; //start monitoring pixel data
        }
        if ((pgmw != 0) && (pgmh != 0) && (pgmc != 0) && (pgmn == 5 || (pgmn == 7 && pamd == 1 && pamatr == 6))) {
          pgmdata = i;
          pgmDataSize = pgmw * pgmh;
          textParserState = 0; //start monitoring pixel data
        }
        if ((pgmw != 0) && (pgmh != 0) && (pgmc != 0) && (pgmn == 6 || (pgmn == 7 && pamd == 3 && pamatr == 6))) {
          pgmdata = i;
          pgmDataSize = pgmw * 3 * pgmh;
          textParserState = 0; //start monitoring pixel data
        }
        if ((pgmw != 0) && (pgmh != 0) && (pgmc != 0) && (pgmn == 7 && pamd == 4 && pamatr == 6)) {
          pgmdata = i;
          pgmDataSize = pgmw * 4 * pgmh;
          textParserState = 0; //start monitoring pixel data
        }
      }
      else { // pixel data
        if (textParserState == TextParserStateInfo::utf8Reject || // for any sign of non-text data in pixel area
          (pgm - 2 == 0 && n - pgmDataSize == i)) // or the image is the whole file/block -> FINISH (success)
        {
          if ((pgmw != 0) && (pgmh != 0) && (pgmc == 0) && pgmn == 4) {
            detectionInfo.IMG_DET(start, BlockType::IMAGE1, pgm - 2, pgmdata - pgm + 3, (pgmw + 7) / 8, pgmh);
            if (detectionInfo.SizeVerificationPassed(start + n))
              return detectionInfo;
          }
          if ((pgmw != 0) && (pgmh != 0) && (pgmc != 0) && (pgmn == 5 || (pgmn == 7 && pamd == 1 && pamatr == 6))) {
            detectionInfo.IMG_DET(start, BlockType::IMAGE8GRAY, pgm - 2, pgmdata - pgm + 3, pgmw, pgmh);
            if (detectionInfo.SizeVerificationPassed(start + n))
              return detectionInfo;
          }
          if ((pgmw != 0) && (pgmh != 0) && (pgmc != 0) && (pgmn == 6 || (pgmn == 7 && pamd == 3 && pamatr == 6))) {
            detectionInfo.IMG_DET(start, BlockType::IMAGE24, pgm - 2, pgmdata - pgm + 3, pgmw * 3, pgmh);
            if (detectionInfo.SizeVerificationPassed(start + n))
              return detectionInfo;
          }
          if ((pgmw != 0) && (pgmh != 0) && (pgmc != 0) && (pgmn == 7 && pamd == 4 && pamatr == 6)) {
            detectionInfo.IMG_DET(start, BlockType::IMAGE32, pgm - 2, pgmdata - pgm + 3, pgmw * 4, pgmh);
            if (detectionInfo.SizeVerificationPassed(start + n))
              return detectionInfo;
          }
          pgm = 0; //fail
        }
        else if ((--pgmDataSize) == 0) {
          pgm = 0; // all data was probably text in pixel area: fail
        }
      }
    }

    // Detect .rgb image
    if ((buf0 & 0xffff) == 0x01da) {
      rgbi = i, rgbx = rgby = 0;
    }
    if (rgbi != 0) {
      const uint64_t p = i - rgbi;
      if (p == 1 && c != 0) {
        rgbi = 0;
      }
      else if (p == 2 && c != 1) {
        rgbi = 0;
      }
      else if (p == 4 && (buf0 & 0xffff) != 1 && (buf0 & 0xffff) != 2 && (buf0 & 0xffff) != 3) {
        rgbi = 0;
      }
      else if (p == 6) {
        rgbx = buf0 & 0xffff, rgbi = (rgbx == 0 ? 0 : rgbi);
      }
      else if (p == 8) {
        rgby = buf0 & 0xffff, rgbi = (rgby == 0 ? 0 : rgbi);
      }
      else if (p == 10) {
        int z = buf0 & 0xffff;
        if ((rgbx != 0) && (rgby != 0) && (z == 1 || z == 3 || z == 4)) {
          detectionInfo.IMG_DET(start, BlockType::IMAGE8, rgbi - 1, 512, rgbx, rgby * z);
          if (detectionInfo.SizeVerificationPassed(start + n))
            return detectionInfo;
        }
        rgbi = 0;
      }
    }

    // Detect .tiff file header (2/8/24 bit color, not compressed).
    if (buf1 == 0x49492a00 && n > i + static_cast<int>(bswap(buf0))) {
      const uint64_t savedPos = in->curPos();
      in->setpos(start + i + static_cast<uint64_t>(bswap(buf0)) - 7);

      // read directory
      int dirSize = in->getchar();
      int tifX = 0;
      int tifY = 0;
      int tifZ = 0;
      int tifZb = 0;
      int tifC = 0;
      int tifofs = 0;
      int tifofval = 0;
      int tifSize = 0;
      int b[12];
      if (in->getchar() == 0) {
        for (int i = 0; i < dirSize; i++) {
          for (int j = 0; j < 12; j++) {
            b[j] = in->getchar();
          }
          if (b[11] == EOF) {
            break;
          }
          int tag = b[0] + (b[1] << 8);
          int tagFmt = b[2] + (b[3] << 8);
          int tagLen = b[4] + (b[5] << 8) + (b[6] << 16) + (b[7] << 24);
          int tagVal = b[8] + (b[9] << 8) + (b[10] << 16) + (b[11] << 24);
          if (tagFmt == 3 || tagFmt == 4) {
            if (tag == 256) {
              tifX = tagVal;
            }
            else if (tag == 257) {
              tifY = tagVal;
            }
            else if (tag == 258) {
              tifZb = tagLen == 1 ? tagVal : 8; // bits per component
            }
            else if (tag == 259) {
              tifC = tagVal; // 1 = no compression
            }
            else if (tag == 273 && tagFmt == 4) {
              tifofs = tagVal, tifofval = static_cast<int>(tagLen <= 1);
            }
            else if (tag == 277) {
              tifZ = tagVal; // components per pixel
            }
            else if (tag == 279 && tagLen == 1) {
              tifSize = tagVal;
            }
          }
        }
      }
      if ((tifX != 0) && (tifY != 0) && (tifZb != 0) && (tifZ == 1 || tifZ == 3) && ((tifC == 1) || (tifC == 5 /*LZW*/ && tifSize > 0)) &&
        ((tifofs != 0) && tifofs + i < n)) {
        if (tifofval == 0) {
          in->setpos(start + i + tifofs - 7);
          for (int j = 0; j < 4; j++) {
            b[j] = in->getchar();
          }
          tifofs = b[0] + (b[1] << 8) + (b[2] << 16) + (b[3] << 24);
        }
        if ((tifofs != 0) && tifofs < (1 << 18) && tifofs + i < n) {
          if (tifC == 1) {
            if (tifZ == 1 && tifZb == 1) {
              detectionInfo.IMG_DET(start, BlockType::IMAGE1, i - 7, tifofs, ((tifX - 1) >> 3) + 1, tifY);
              return detectionInfo;
            }
            if (tifZ == 1 && tifZb == 8) {
              detectionInfo.IMG_DET(start, BlockType::IMAGE8, i - 7, tifofs, tifX, tifY);
              return detectionInfo;
            }
            if (tifZ == 3 && tifZb == 8) {
              detectionInfo.IMG_DET(start, BlockType::IMAGE24, i - 7, tifofs, tifX * 3, tifY);
              return detectionInfo;
            }
          }
          else if (tifC == 5 && tifSize > 0) {
            tifX = ((tifX + 8 - tifZb) / (9 - tifZb)) * tifZ;
            int info = tifZ * tifZb;
            info = (((info == 1) ? BlockType::IMAGE1 : ((info == 8) ? BlockType::IMAGE8 : BlockType::IMAGE24)) << 24) | tifX;
            detectionInfo.DataInfo = info;
            detectionInfo.Type = BlockType::LZW;
            detectionInfo.DataStart = start + i - 7 + tifofs;
            detectionInfo.DataLength = tifSize;
            return detectionInfo;
          }
        }
      }
      in->setpos(savedPos);
    }

    // Detect .tga image (8-bit 256 colors or 24-bit uncompressed)
    if ((buf1 & 0xFFF7FF) == 0x00010100 && (buf0 & 0xFFFFFFC7) == 0x00000100 && (c == 16 || c == 24 || c == 32)) {
      tga = i, tgax = tgay, tgaz = 8, tgat = (buf1 >> 8) & 0xF, tgaid = buf1 >> 24, tgamap = c / 8;
    }
    else if ((buf1 & 0xFFFFFF) == 0x00000200 && buf0 == 0x00000000) {
      tga = i, tgax = tgay, tgaz = 24, tgat = 2;
    }
    else if ((buf1 & 0xFFF7FF) == 0x00000300 && buf0 == 0x00000000) {
      tga = i, tgax = tgay, tgaz = 8, tgat = (buf1 >> 8) & 0xF;
    }
    if (tga != 0) {
      if (i - tga == 8) {
        tga = (buf1 == 0 ? tga : 0), tgax = (bswap(buf0) & 0xffff), tgay = (bswap(buf0) >> 16);
      }
      else if (i - tga == 10) {
        if ((buf0 & 0xFFF7) == 32 << 8) {
          tgaz = 32;
        }
        if ((tgaz << 8) == static_cast<int>(buf0 & 0xFFD7) && (tgax != 0) && (tgay != 0) && uint32_t(tgax * tgay) < 0xFFFFFFF) {
          if (tgat == 1) {
            in->setpos(start + tga + 11 + tgaid);
            bool isGray = isGrayscalePalette(in);
            detectionInfo.IMG_DET(start, isGray ? BlockType::IMAGE8GRAY : BlockType::IMAGE8, tga - 7, 18 + tgaid + 256 * tgamap, tgax, tgay);
            if (detectionInfo.SizeVerificationPassed(start + n))
              return detectionInfo;
          }
          if (tgat == 2) {
            detectionInfo.IMG_DET(start, (tgaz == 24) ? BlockType::IMAGE24 : BlockType::IMAGE32, tga - 7, 18 + tgaid, tgax * (tgaz >> 3), tgay);
            if (detectionInfo.SizeVerificationPassed(start + n))
              return detectionInfo;
          }
          if (tgat == 3) {
            detectionInfo.IMG_DET(start, BlockType::IMAGE8GRAY, tga - 7, 18 + tgaid, tgax, tgay);
            if (detectionInfo.SizeVerificationPassed(start + n))
              return detectionInfo;
          }
          if (tgat == 9 || tgat == 11) {
            int info;
            const uint64_t savedPos = in->curPos();
            in->setpos(start + tga + 11 + tgaid);
            if (tgat == 9) {
              bool isGray = isGrayscalePalette(in);
              info = (isGray ? BlockType::IMAGE8GRAY : BlockType::IMAGE8) << 24;
              in->setpos(start + tga + 11 + tgaid + 256 * tgamap);
            }
            else {
              info = BlockType::IMAGE8GRAY << 24;
            }
            info |= tgax;
            // now detect compressed image data size
            uint32_t detd = 0;
            int c = in->getchar();
            int b = 0;
            int total = tgax * tgay;
            int line = 0;
            while (total > 0 && c >= 0 && (++detd, b = in->getchar()) >= 0) {
              if (c == 0x80) {
                c = b;
                continue;
              }
              if (c > 0x7F) {
                total -= (c = (c & 0x7F) + 1);
                line += c;
                c = in->getchar();
                detd++;
              }
              else {
                in->setpos(in->curPos() + c);
                detd += ++c;
                total -= c;
                line += c;
                c = in->getchar();
              }
              if (line > tgax) {
                break;
              }
              if (line == tgax) {
                line = 0;
              }
            }
            if (total == 0) {
              detectionInfo.Type = BlockType::RLE;
              detectionInfo.DataInfo = info;
              detectionInfo.DataStart = start + tga + 11 + tgaid + 256 * tgamap;
              detectionInfo.DataLength = detd;
              if (detectionInfo.SizeVerificationPassed(start + n))
                return detectionInfo;
            }
            in->setpos(savedPos);
          }
        }
        tga = 0;
      }
    }

    static BlockType dett = BlockType::DEFAULT;

    // Detect .gif
    if (detectionInfo.Type == BlockType::DEFAULT && dett == BlockType::GIF && i == 0) {
      dett = BlockType::DEFAULT;
      if (c == 0x2c || c == 0x21) { //image section or extension block
        gif = 2; // flag: image section or extension block
        gifi = 2; //jump 2 bytes
      }
      else {
        gifGray = 0;
      }
    }
    if ((gif == 0) && (buf1 & 0xffff) == 0x4749 /*GI*/ && (buf0 == 0x46383961 /*F89a*/ || buf0 == 0x46383761 /*F87a*/)) {
      gif = 1; // flag: found header
      gifi = i + 5; // position after the header
    }
    if (gif != 0) { //we are in a GIF file
      if (gif == 1 && i == gifi) {
        gif = 2;
        gifplt = (c & 128) != 0 ? (3 * (2 << (c & 7))) : 0;
        gifi = i + 5 + gifplt;
      }
      if (gif == 2 && (gifplt != 0) && i == gifi - gifplt - 3) {
        gifGray = isGrayscalePalette(in, gifplt / 3);
        gifplt = 0;
      }
      if (gif == 2 && i == gifi) {
        if ((buf0 & 0xff0000) == 0x210000) {
          gif = 5;
          gifi = i;
        }
        else if ((buf0 & 0xff0000) == 0x2c0000) {
          gif = 3;
          gifi = i;
        }
        else {
          gif = 0; //failed
        }
      }
      if (gif == 3 && i == gifi + 6) {
        gifw = (bswap(buf0) & 0xffff);
      }
      if (gif == 3 && i == gifi + 7) {
        gif = 4;
        gifc = gifb = 0;
        gifplt = (c & 128) != 0 ? (3 * (2 << (c & 7))) : 0;
        gifa = gifi = i + 2 + gifplt;
      }
      if (gif == 4 && (gifplt != 0)) {
        gifGray = isGrayscalePalette(in, gifplt / 3);
        gifplt = 0;
      }
      if (gif == 4 && i == gifi) {
        if (c > 0 && gifb != 0 && gifc != gifb) {
          gifw = 0;
        }
        if (c > 0) {
          gifb = gifc;
          gifc = c;
          gifi += c + 1;
        }
        else if (gifw == 0) {
          gif = 2;
          gifi = i + 3;
        }
        else {
          dett = BlockType::GIF;
          detectionInfo.Type = BlockType::GIF;
          detectionInfo.DataStart = start + gifa - 1;
          detectionInfo.DataLength = i - gifa + 2;
          detectionInfo.DataInfo = ((gifGray ? BlockType::IMAGE8GRAY : BlockType::IMAGE8) << 24) | gifw;
          return detectionInfo;
        }
      }
      if (gif == 5 && i == gifi) {
        if (c > 0) {
          gifi += c + 1;
        }
        else {
          gif = 2;
          gifi = i + 3;
        }
      }
    }

    // Detect x86/64 if the low order byte (little-endian) XX is more
    // recently seen (and within 4K) if a relative to absolute address
    // conversion is done in the context CALL/JMP (E8/E9) XX xx xx 00/FF
    // 4 times in a row.  Detect end of EXE at the last
    // place this happens when it does not happen for 64KB.

    if (((buf1 & 0xfe) == 0xe8 || (buf1 & 0xfff0) == 0x0f80) && ((buf0 + 1) & 0xfe) == 0) {
      uint64_t r = buf0 >> 24; // relative address low 8 bits
      uint64_t a = ((buf0 >> 24) + i) & 0xff; // absolute address low 8 bits
      uint64_t rDist = i - relPos[r];
      uint64_t aDist = i - absPos[a];
      if (aDist < rDist && aDist < 0x800 && absPos[a] > 5) {
        e8e9last = i;
        ++e8e9count;
        if (e8e9pos == 0 || e8e9pos > absPos[a]) {
          e8e9pos = absPos[a];
        }
      }
      else {
        e8e9count = 0;
      }
      if (detectionInfo.Type == BlockType::DEFAULT && e8e9count >= 4 && e8e9pos > 5) {
        detectionInfo.Type = BlockType::EXE;
        detectionInfo.DataStart = start + e8e9pos - 5;
      }
      absPos[a] = i;
      relPos[r] = i;
    }
    if (i + 1 == n || i - e8e9last > 0x4000) {
      // TODO: Large file support
      if (detectionInfo.Type == BlockType::EXE) {
        detectionInfo.DataInfo = static_cast<int>(detectionInfo.DataStart);
        detectionInfo.DataLength = start + e8e9last - detectionInfo.DataStart;
        return detectionInfo;
      }
      e8e9pos = 0;
      e8e9count = 0;
    }

    //DEC Alpha code section detection 
    //Based on the Tru64 Object file format
    //see: https://www3.physnet.uni-hamburg.de/physnet/Tru64-Unix/HTML/APS31DTE/DOCU_013.HTM
    //note: this document^ doesn't cover the same version as in silesia/mozilla
    //luckily the 3 sections containing DEC Alpha code are always adjacent to each other
    //so we can treat them as one continuous section.

    if (decAlpha == 0 && ((buf3 >> 16) == 0x8301 /*Target-machine magic number*/ || buf1 == 0 && buf0 == 0 /*no symbolic header*/)) {
      uint16_t numberOfSections = bswap(buf3) >> 16;
      if (numberOfSections >= 3 && numberOfSections <= 24) {
        decAlpha = 1;
        decAlphaHeaderStart = i - 15;
        decAlphaNumberOfSections = numberOfSections;
      }
    }

    if (decAlpha != 0) {
      if (decAlpha == 1) { // sanity checks
        if (i == decAlphaHeaderStart + 23) {
          //size of optional header is always 80
          uint16_t sizeOfOptionalHeader = bswap(buf0)&0xffff;
          //flags:
          //0x2005/0x2007 - Dynamically shared object (so)
          //0x3007        - Dynamic executable file
          uint16_t flags = bswap(buf0) >> 16;
          if (sizeOfOptionalHeader != 80 || !(flags == 0x2005 || flags == 0x2007 || flags == 0x3007)) {
            decAlpha = 0; //fail
          }
        }
        else if (i == decAlphaHeaderStart + 24 + 80) {
          if (c != '.') { //section names start with a dot (this is the first section)
            decAlpha = 0; //fail
          }
        }
        else if (i > decAlphaHeaderStart + 24 + 80 + decAlphaNumberOfSections * 64) {
          decAlpha = 0; //fail
        }
      }

      //we look for the 3 (adjacent) sections containing program code
      if (decAlpha == 1 && buf1 == 0x2E746578 && buf0 == 0x74000000) { // ".text   "
        decAlpha = 2;
        decAlphaNextExpectedOffset = i + 32;
      }
      else if (decAlpha == 2 && i == decAlphaNextExpectedOffset) {
        decAlphaSectionStart = ((uint64_t)bswap(buf0)) << 32 | bswap(buf1);
        decAlphaSectionLen = ((uint64_t)bswap(buf2)) << 32 | bswap(buf3);
        decAlpha = 3;
      }
      else if (decAlpha == 3 && buf1 == 0x2E696E69 && buf0 == 0x74000000) { // ".init   "
        decAlpha = 4;
        decAlphaNextExpectedOffset = i + 32;
      }
      else if (decAlpha == 4 && i == decAlphaNextExpectedOffset) {
        uint64_t initstart = ((uint64_t)bswap(buf0)) << 32 | bswap(buf1);
        if (initstart == decAlphaSectionStart + decAlphaSectionLen) {
          decAlphaSectionLen += ((uint64_t)bswap(buf2)) << 32 | bswap(buf3);
          decAlpha = 5;
        } 
        else {
          //printf("DECALPHA - found .text and .init sections are not adjacent");
          decAlpha = 0; //fail
        }
      }
      else if (decAlpha == 5 && buf1 == 0x2E66696E && buf0 == 0x69000000) { // ".fini   "
        decAlpha = 6;
        decAlphaNextExpectedOffset = i + 32;
      }
      else if (decAlpha == 6 && i == decAlphaNextExpectedOffset) {
        uint64_t finistart = ((uint64_t)bswap(buf0)) << 32 | bswap(buf1);
        if (finistart == decAlphaSectionStart + decAlphaSectionLen) {
          decAlphaSectionLen += ((uint64_t)bswap(buf2)) << 32 | bswap(buf3);
          if (decAlphaHeaderStart + decAlphaSectionStart + decAlphaSectionLen <= n) {
            detectionInfo.Type = BlockType::DEC_ALPHA;
            detectionInfo.DataStart = start + decAlphaHeaderStart + decAlphaSectionStart;
            detectionInfo.DataLength = decAlphaSectionLen;
            return detectionInfo;
          }
          else {
            //printf("DECALPHA - end of section is past end of file");
            decAlpha = 0; //fail
          }
        }
        else {
          //printf("DECALPHA - found .init and .fini sections are not adjavent");
          decAlpha = 0; //fail
        }
      }
    }

    // this is the old DEC Alpha detection logic
    // unused - kept for reference only 
#ifdef USE_OLD_DECALPHA_DETECTION

    struct {
      Array<uint64_t> absPos{ 256 * 4 };
      Array<uint64_t> relPos{ 256 * 4 };
      uint32_t opcode = 0, idx = 0, count[4] = { 0 }, branches[4] = { 0 };
      uint64_t offset[4] = { 0 }, last[4] = { 0 };
    } DEC;

    DEC.idx = i & 3;
    DEC.opcode = bswap(buf0);
    bool isValidDecAlphaInstruction = (i >= 3) && DECAlpha::IsValidInstruction(DEC.opcode);
    if (DEC.count[DEC.idx] == 0 && buf0 == 0 && buf1 == 0 /*ignore lead-in padding*/)
      isValidDecAlphaInstruction = false;
    if (isValidDecAlphaInstruction && buf0 == 0 && buf1 == 0 && buf2 == 0 && buf3 == 0 /*stop when lead-out padding*/)
      isValidDecAlphaInstruction = false;
    DEC.count[DEC.idx] = isValidDecAlphaInstruction ? DEC.count[DEC.idx] + 1 : DEC.count[DEC.idx] = 0;
    DEC.opcode >>= 21;
    //test if bsr opcode and if last 4 opcodes are valid
    if (
      (DEC.opcode == (0x34 << 5) + 26) &&
      (DEC.count[DEC.idx] > 4) &&
      ((e8e9count == 0) && !soi && !pgm && !rgbi && !bmpi && !wavi && !tga)
      ) {
      uint32_t const absAddrLSB = DEC.opcode & 0xFF; // absolute address low 8 bits
      uint32_t const relAddrLSB = ((DEC.opcode & 0x1FFFFF) + static_cast<uint32_t>(i) / 4u) & 0xff; // relative address low 8 bits
      uint64_t const absPos = DEC.absPos[absAddrLSB * 4 + DEC.idx];
      uint64_t const relPos = DEC.relPos[relAddrLSB * 4 + DEC.idx];
      uint64_t const curPos = i - 3;
      if ((absPos > relPos) && (curPos < absPos + UINT64_C(0x8000)) && (absPos > 16) && (curPos > absPos + UINT64_C(16)) && (((curPos - absPos) & UINT64_C(3)) == 0)) {
        DEC.last[DEC.idx] = curPos;
        DEC.branches[DEC.idx]++;
        if ((DEC.offset[DEC.idx] == 0) || (DEC.offset[DEC.idx] > DEC.absPos[absAddrLSB])) {
          uint64_t const addr = curPos - (DEC.count[DEC.idx] - 1) * UINT64_C(4);
          DEC.offset[DEC.idx] = std::min<uint64_t>(DEC.absPos[absAddrLSB * 4 + DEC.idx], addr);
        }
      }
      else
        DEC.branches[DEC.idx] = 0;
      DEC.absPos[absAddrLSB * 4 + DEC.idx] = DEC.relPos[relAddrLSB * 4 + DEC.idx] = curPos;
    }

    if (DEC.last[DEC.idx] != 0) {
      if ((detectionInfo.Type == BlockType::DEFAULT) && (DEC.branches[DEC.idx] >= 16) && DEC.count[DEC.idx] >= 64) {
        detectionInfo.Type = BlockType::DEC_ALPHA;
        detectionInfo.DataStart = start + DEC.offset[DEC.idx];
        for (int i = 0; i < 4; i++) {
          if (i != DEC.idx) {
            DEC.last[i] = 0;
            DEC.offset[i] = 0;
            DEC.branches[i] = 0;
          }
        }
      }

      else if ((detectionInfo.Type == BlockType::DEFAULT) && (i > DEC.last[DEC.idx] + UINT64_C(0x8000))) {
        DEC.last[DEC.idx] = 0;
        DEC.offset[DEC.idx] = 0;
        DEC.branches[DEC.idx] = 0;
      }

      else if (detectionInfo.Type == BlockType::DEC_ALPHA 
        && (start + DEC.offset[DEC.idx]-detectionInfo.DataStart) % 4 == 0
        && ((i + 4 >= n) || (i > DEC.last[DEC.idx] + UINT64_C(0x1000) && DEC.count[DEC.idx] == 0))) {
        detectionInfo.DataStart = start + DEC.offset[DEC.idx];
        detectionInfo.DataLength = (start + DEC.last[DEC.idx]) - detectionInfo.DataStart;
        return detectionInfo;
      }
    }

#endif //USE_OLD_DECALPHA_DETECTION

    // Detect base64 encoded data
    if( b64S == 0 && buf0 == 0x73653634 && ((buf1 & 0xffffff) == 0x206261 || (buf1 & 0xffffff) == 0x204261)) {
      b64S = 1; b64I = i - 6; //' base64' ' Base64'
    }
    if( b64S == 0 && ((buf1 == 0x3b626173 && buf0 == 0x6536342c) || (buf1 == 0x215b4344 && buf0 == 0x4154415b))) {
      b64S = 3; b64I = i + 1; // ';base64,' '![CDATA['
    }
    if( b64S > 0 ) {
      if( b64S == 1 && buf0 == 0x0d0a0d0a ) {
        b64I = i + 1; b64Line = 0; b64S = 2;
      } else if( b64S == 2 && (buf0 & 0xffff) == 0x0d0a && b64Line == 0 ) {
        b64Line = i + 1 - b64I; b64Nl = i;
      } else if( b64S == 2 && (buf0 & 0xffff) == 0x0d0a && b64Line > 0 && (buf0 & 0xffffff) != 0x3d0d0a ) {
        if( i - b64Nl < b64Line && buf0 != 0x0d0a0d0a ) {
          b64End = i - 1; b64S = 5;
        } else if( buf0 == 0x0d0a0d0a ) {
          b64End = i - 3 /*remove the last 0d0a*/; b64S = 5;
        } else if( i - b64Nl == b64Line ) {
          b64Nl = i;
        } else {
          b64S = 0;
        }
      } else if( b64S == 2 && (buf0 & 0xffffff) == 0x3d0d0a ) {
        b64End = i - 1; b64S = 5; // '=' or '=='
      } else if( b64S == 2 && !(isalnum(c) || c == '+' || c == '/' || c == 10 || c == 13 || c == '=')) {
        b64S = 0;
      }
      if( b64Line > 0 && (b64Line <= 4 || b64Line > 255)) {
        b64S = 0;
      }
      if (b64S == 3 && i >= b64I && !(isalnum(c) || c == '+' || c == '/' || c == '=')) {
        b64End = i;
        b64S = 4;
      }
      if((b64S == 4 && b64End - b64I > 32) || (b64S == 5 && b64End - b64I > 32 && b64End - b64I < (1 << 27))) {
        detectionInfo.Type = BlockType::BASE64;
        detectionInfo.DataStart = start + b64I;
        detectionInfo.DataLength = b64End - b64I;
        return detectionInfo;
      }
      if( b64S > 3 ) {
        b64S = 0;
      }
      if( b64S == 1 && i - b64I >= 128 ) {
        b64S = 0; // detect false positives after 128 bytes
      }
    }

    //detect base85 (ascii85) encoded data
    //headers: stream\n stream\r\n oNimage\n utimage\n \nimage\n
    if (b85state == 0 && ((buf0 == 0x65616D0A && (buf1 & 0xffffff) == 0x737472) || (buf0 == 0x616D0D0A && buf1 == 0x73747265) || (buf0 == 0x6167650A && buf1 == 0x6F4E696D) || (buf0 == 0x6167650A && buf1 == 0x7574696D) || (buf0 == 0x6167650A && (buf1 & 0xffffff) == 0x0A696D))) {
        b85state = 1;
        base85start = i;
        b85linelength = 0;
    }
    else if (b85state == 1) {
      if (c == CARRIAGE_RETURN && b85linelength == 0) {
        b85linelength = i - base85start; //capture line lenght
        if (b85linelength <= 25 || b85linelength > 255)
          b85state = 0; //fail
      }
      else if (c == '~') { //end marker
        base85end = i - 1;
        b85state = 0;
        if (((base85end - base85start) > 60) && ((base85end - base85start) < 0x8000000)) {
          detectionInfo.Type = BlockType::BASE85;
          detectionInfo.DataStart = start + base85start + 1;
          detectionInfo.DataLength = base85end - base85start;
          return detectionInfo;
        }
      }
      else if (is_base85(c)) {
        // still ok
      }
      else if (c == CARRIAGE_RETURN && b85linelength != 0) {
        if (b85linelength != i - base85start)
          b85state = 0; //fail
      }
      else {
        b85state = 0; //fail
      }
    }

    // UStar (Unix Standard TAR) detection
    // Notable uses: silesia/mozilla, silesia/samba, silesia/xml
    if (buf1 == 0x75737461 /* "usta" */ && (buf0 == 0x72202000 /* "r  \0" */ || buf0 == 0x72003030 /* "r\000" */)) {
      uint64_t posBackup = in->curPos();
      TarFilter tarFilter;
      bool success = tarFilter.detect(in, start + blockSize);
      if (success) {
        detectionInfo.Type = BlockType::TAR;
        detectionInfo.DataStart = tarFilter.detectedStartPos;
        detectionInfo.DataLength = tarFilter.detectedEndPos - tarFilter.detectedStartPos;
        return detectionInfo;
      }
      in->setpos(posBackup);
    }

  }
  
  if (detectionInfo.Type != BlockType::DEFAULT)
    quit("detect(): detection didn't finish properly.");

  //nothing detected
  detectionInfo.Type = BlockType::DEFAULT;
  detectionInfo.DataStart = start + n;
  detectionInfo.DataLength = 0;

  return detectionInfo;
}

#include "bmp.hpp"
#include "endianness16b.hpp"
#include "eol.hpp"
#include "exe.hpp"
#include "rle.hpp"

//////////////////// Compress, Decompress ////////////////////////////

static void directEncodeBlock(BlockType type, File *in, uint64_t len, Encoder &en, int info) {
  // TODO: Large file support
  Block::EncodeBlockHeader(&en, type, len, info);
  fprintf(stderr, "Compressing... ");
  for( uint64_t j = 0; j < len; ++j ) {
    if((j & 0xfff) == 0 ) {
      en.printStatus(j, len);
    }
    en.compressByte(&en.predictorMain, in->getchar());
  }
  fprintf(stderr, "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b");
}

static void compressRecursive(File *in, uint64_t blockSize, Encoder &en, String &blstr, float p1, float p2, const TransformOptions* const transformOptions);
static void compressRecursiveForTar(File* in, uint64_t blockSize, Encoder& en, String& blstr, float p1, float p2, const TransformOptions* const transformOptions);

static uint64_t decodeFunc(BlockType type, Encoder &en, File *tmp, uint64_t len, int info, File *out, FMode mode, uint64_t &diffFound, const TransformOptions* const transformOptions) {
  if( type == BlockType::IMAGE24 ) {
    auto f = BmpFilter();
    f.setWidth(info);
    f.setSkipRgb(transformOptions->skipRgb);
    f.setEncoder(en);
    return f.decode(tmp, out, mode, len, diffFound);
  }
  if( type == BlockType::IMAGE32 ) {
    auto f = BmpFilter();
    f.setWidth(info);
    f.setSkipRgb(transformOptions->skipRgb);
    f.setHasApha();
    f.setEncoder(en);
    return f.decode(tmp, out, mode, len, diffFound);
  }
  if (type == BlockType::PNG8GRAY || type == BlockType::PNG8 || type == BlockType::PNG24|| type == BlockType::PNG32) {
    auto f = PngFilter();
    f.setWidth(info);
    auto stride = type == BlockType::PNG24 ? 3 : type == BlockType::PNG32 ? 4 : 1;
    f.setStride(stride);
    f.setEncoder(en);
    return f.decode(tmp, out, mode, len, diffFound);
  }
  if( type == BlockType::AUDIO_LE ) {
    auto f = EndiannessFilter();
    f.setEncoder(en);
    return f.decode(tmp, out, mode, len, diffFound);
  } else if( type == BlockType::EXE ) {
    auto f = ExeFilter();
    f.setBegin(info); 
    f.setEncoder(en);
    return f.decode(tmp, out, mode, len, diffFound);
  } else if( type == BlockType::TEXT_EOL ) {
    auto f = EolFilter();
    f.setEncoder(en);
    return f.decode(tmp, out, mode, len, diffFound);
  } else if( type == BlockType::CD ) {
    auto f = CdFilter();
    return f.decode(tmp, out, mode, len, diffFound);
#ifndef DISABLE_ZLIB
  } else if( type == BlockType::ZLIB ) {
    return decodeZlib(tmp, len, out, mode, diffFound);
#endif //DISABLE_ZLIB
  } else if( type == BlockType::BASE64 ) {
    auto f = Base64Filter();
    return f.decode(tmp, out, mode, len, diffFound);
  } else if (type == BlockType::BASE85) {
    auto f = Base85Filter();
    return f.decode(tmp, out, mode, len, diffFound);
  }
  else if( type == BlockType::GIF ) {
    return decodeGif(tmp, len, out, mode, diffFound);
  } else if( type == BlockType::RLE ) {
    auto f = RleFilter();
    //f.setScanLineSize(info & 0xFFFFFF); //now it self-encodes this info, but eventually we need to pass it
    return f.decode(tmp, out, mode, len, diffFound);
  } else if( type == BlockType::MRB) {
    uint8_t packingMethod = (info >> 24) & 3; //0..3
    if (packingMethod == 1 /*RLE*/) {
      auto f = MrbRleFilter();
      return f.decode(tmp, out, mode, len, diffFound);
    }
    else quit("MRB: not implemented");
  } else if( type == BlockType::LZW ) {
    return decodeLzw(tmp, out, mode, diffFound);
  } else if (type == BlockType::DEC_ALPHA) {
    auto f = DECAlphaFilter();
    f.setEncoder(en);
    return f.decode(tmp, out, mode, len, diffFound);
  } else if (type == BlockType::TAR) {
    auto f = TarFilter();
    f.setEncoder(en);
    return f.decode(tmp, out, mode, len, diffFound);
  } else {
    assert(false);
  }
  return 0;
}

static uint64_t encodeFunc(BlockType type, File *in, File *tmp, uint64_t len, int info, int &hdrsize, const TransformOptions* const transformOptions) {
  if( type == BlockType::IMAGE24 ) {
    auto f= BmpFilter();
    f.setSkipRgb(transformOptions->skipRgb);
    f.encode(in, tmp, len, info, hdrsize);
  } else if( type == BlockType::IMAGE32 ) {
    auto f = BmpFilter();
    f.setSkipRgb(transformOptions->skipRgb);
    f.setHasApha();
    f.encode(in, tmp, len, info, hdrsize);
  } else if (type == BlockType::PNG8GRAY || type == BlockType::PNG8 || type == BlockType::PNG24 || type == BlockType::PNG32) {
    auto f = PngFilter();
    auto stride = type == BlockType::PNG24 ? 3 : type == BlockType::PNG32 ? 4 : 1;
    f.setStride(stride);
    f.encode(in, tmp, len, info, hdrsize);
  } else if( type == BlockType::AUDIO_LE ) {
    auto f = EndiannessFilter();
    f.encode(in, tmp, len, info, hdrsize);
  } else if( type == BlockType::EXE ) {
    auto f = ExeFilter();
    f.setBegin(info); 
    f.encode(in, tmp, len, info, hdrsize);
  } else if( type == BlockType::TEXT_EOL ) {
    auto f = EolFilter();
    f.encode(in, tmp, len, info, hdrsize);
  } else if( type == BlockType::CD ) {
    auto f = CdFilter();
    f.encode(in, tmp, len, info, hdrsize);
#ifndef DISABLE_ZLIB
  } else if( type == BlockType::ZLIB ) {
    return encodeZlib(in, tmp, len, hdrsize) ? 0 : 1;
#endif //DISABLE_ZLIB
  } else if( type == BlockType::BASE64 ) {
    auto f = Base64Filter();
    f.encode(in, tmp, len, info, hdrsize);
  } else if (type == BlockType::BASE85) {
    auto f = Base85Filter();
    f.encode(in, tmp, len, info, hdrsize);
  } else if( type == BlockType::GIF ) {
    return encodeGif(in, tmp, len, hdrsize) != 0 ? 0 : 1;
  } else if( type == BlockType::RLE ) {
    auto f = RleFilter();
    f.setScanLineSize(info & 0xFFFFFF);
    f.encode(in, tmp, len, info, hdrsize);
  } else if (type == BlockType::MRB) {
    const uint8_t packingMethod = (info >> 24) & 3; //0..3
    const uint16_t colorBits = (info >> 26); //1,4,8
    const int width = (info >> 12) & 0xfff;
    const int height = info & 0xfff;
    int widthInBytes;
    if (colorBits == 8) widthInBytes = ((width + 3) / 4) * 4;
    else if (colorBits == 4) widthInBytes = ((width + 3) / 4) * 2;
    else if (colorBits == 1) widthInBytes = ((width + 31) / 32) * 4;
    else quit("Unexpected colorBits for MRB");
    if (packingMethod == 1 /*RLE*/) {
      auto f = MrbRleFilter();
      f.setInfo(widthInBytes, height);
      f.encode(in, tmp, len, info, hdrsize);
    }
    else quit("MRB: not implemented");
  } else if( type == BlockType::LZW ) {
    return encodeLzw(in, tmp, len, hdrsize) != 0 ? 0 : 1;
  } else if (type == BlockType::DEC_ALPHA) {
    auto f = DECAlphaFilter();
    f.encode(in, tmp, len, info, hdrsize);
  } else if (type == BlockType::TAR) {
    auto f = TarFilter();
    f.encode(in, tmp, len, info, hdrsize);
  } else {
    assert(false);
  }
  return 0;
}

static void
transformEncodeBlock(BlockType type, File *in, uint64_t len, Encoder &en, int info, String &blstr, float p1, float p2, uint64_t begin, const TransformOptions* const transformOptions) {
  if( hasTransform(type, info)) {
    FileTmp tmp;
    int headerSize = 0;
    uint64_t diffFound = encodeFunc(type, in, &tmp, len, info, headerSize, transformOptions);
    const uint64_t tmpSize = tmp.curPos();
    tmp.setpos(tmpSize); //switch to read mode
    if( diffFound == 0 ) {
      tmp.setpos(0);
      en.setFile(&tmp);
      in->setpos(begin);
      decodeFunc(type, en, &tmp, tmpSize, info, in, FMode::FCOMPARE, diffFound, transformOptions);
    }
    // Test fails, compress without transform
    if( diffFound > 0 || tmp.getchar() != EOF) {
      printf("Transform fails at %" PRIu64 ", skipping...\n", diffFound - 1);
      in->setpos(begin);
      directEncodeBlock(BlockType::DEFAULT, in, len, en, -1);
    } else {
      tmp.setpos(0);
      if( hasRecursion(type)) {
        Block::EncodeBlockHeader(&en, type, tmpSize, info&0xffffff);
        BlockType type2 = static_cast<BlockType>((info >> 24) & 0xFF);
        if (isPNG(type)) {
            type2 =
                type == BlockType::PNG8GRAY ? BlockType::IMAGE8GRAY :
                type == BlockType::PNG8 ? BlockType::IMAGE8 :
                type == BlockType::PNG24 ? BlockType::IMAGE24 :
                type == BlockType::PNG32 ? BlockType::IMAGE32 : BlockType::DEFAULT;
        }
        if( type2 != BlockType::DEFAULT ) {
          String blstrSub0;
          blstrSub0 += blstr.c_str();
          blstrSub0 += "->";
          String blstrSub1;
          blstrSub1 += blstr.c_str();
          blstrSub1 += "-->";
          String blstrSub2;
          blstrSub2 += blstr.c_str();
          blstrSub2 += "-->";
          const char* exploded = 
                           "exploded    ";
          const char* addedheader =
            isPNG(type)  ? "filter data " : // PNG ->Image
                           "added header";
          const char* dataname =
            isPNG(type2) ? "png data    " : // ZLIB -> PNG
            isPNG(type)  ? "pixel data  " : // PNG  -> Image
                           "data        ";
          if(!isPNG(type))
            printf(" %-11s | ->  %s |%10d bytes [%d - %d]\n", blstrSub0.c_str(), exploded, int(tmpSize), 0, int(tmpSize - 1));
          if (headerSize != 0) {
            printf(" %-11s | --> %s |%10d bytes [%d - %d]\n", blstrSub1.c_str(), addedheader, headerSize, 0, headerSize - 1);
            directEncodeBlock(BlockType::HDR, &tmp, headerSize, en, -1);
            printf(" %-11s | --> %s |%10d bytes [%d - %d]\n", blstrSub2.c_str(), dataname, int(tmpSize - headerSize), headerSize, int(tmpSize - 1));
          }
          transformEncodeBlock(type2, &tmp, tmpSize - headerSize, en, info & 0xffffff, blstr, p1, p2, headerSize, transformOptions);
        } else {
          if (type == BlockType::TAR) {
            compressRecursiveForTar(&tmp, tmpSize, en, blstr, p1, p2, transformOptions);
          }
          else {
            compressRecursive(&tmp, tmpSize, en, blstr, p1, p2, transformOptions);
          }
        }
      } else {
        if (type == BlockType::MRB) {
          String blstrSub0;
          blstrSub0 += blstr.c_str();
          blstrSub0 += "->";
          printf(" %-11s | ->  uncompressed |%10d bytes [%d - %d]\n", blstrSub0.c_str(), int(tmpSize), 0, int(tmpSize - 1));
        }
        directEncodeBlock(type, &tmp, tmpSize, en, info);
      }
    }
    tmp.close();
  } else {
    directEncodeBlock(type, in, len, en, info);
  }
}

static void composeSubBlockStringToPrint(String& blstr, String& blstrSub, int blNum) {
  //Compose block enumeration string
  blstrSub += blstr.c_str();
  if (blstrSub.strsize() != 0) {
    blstrSub += "-";
  }
  blstrSub += uint64_t(blNum);
}

static void printBlock(const uint64_t begin, const uint64_t len, const BlockType type, const int blockInfo, String& blstrSub) {
  static const char* typeNames[30] = { "default", "jpeg", "hdr", "1b-image", "4b-image", "8b-image", "8b-img-grayscale",
                                      "24b-image", "32b-image", "audio", "audio - le", "x86/64", "cd", "zlib", "base64", "gif", "png-8b",
                                      "png-8b-grayscale", "png-24b", "png-32b", "text", "text - eol", "rle", "lzw", "dec-alpha", "mrb", 
                                      "dBase", "base85", "tar", "tar header"};
  static const char* audioTypes[4] = { "8b-mono", "8b-stereo", "16b-mono", "16b-stereo" };
  static const char* mrbTypes[4] = { "mrb-uncompressed", "mrb-rle", "mrb-lz77", "mrb-rle-lz77" };

  const char* typeName =
    type == BlockType::MRB ? mrbTypes[(blockInfo >> 24) & 3] :
    type == BlockType::ZLIB && isPNG(BlockType(blockInfo >> 24)) ? typeNames[blockInfo >> 24] :
    typeNames[(int)type];
  printf(" %-11s | %-16s |%10" PRIu64 " bytes [%" PRIu64 " - %" PRIu64 "]", blstrSub.c_str(), typeName, len, begin, (begin + len) - 1);
  if (type == BlockType::AUDIO || type == BlockType::AUDIO_LE) {
    printf(" (%s)", audioTypes[blockInfo % 4]);
  }
  else if (
    type == BlockType::IMAGE1 || 
    type == BlockType::IMAGE4 || 
    type == BlockType::IMAGE8 ||
    type == BlockType::IMAGE8GRAY || 
    type == BlockType::IMAGE24 ||
    type == BlockType::IMAGE32 ||
    (type == BlockType::ZLIB && isPNG(BlockType(blockInfo >> 24)))) {
    printf(" (width: %d)", (type == BlockType::ZLIB) ? (blockInfo & 0xFFFFFF) : blockInfo);
  }
  else if (type == BlockType::MRB) {
    const uint8_t packingMethod = (blockInfo >> 24) & 3; //0..3
    const uint16_t colorBits = (blockInfo >> 26); //1,4,8
    const int width = ((blockInfo >> 12) & 0xFFF);
    const int height = blockInfo & 0xFFF;
    printf(" (%d-bit image: %dx%d)", colorBits, width, height);
  }
  else if (hasRecursion(type) && (blockInfo >> 24) != (int)BlockType::DEFAULT) {
      printf(" (%s)", typeNames[blockInfo >> 24]);
  }
  else if (type == BlockType::CD) {
    printf(" (mode%d/form%d)", blockInfo == 1 ? 1 : 2, blockInfo != 3 ? 1 : 2);
  }
  else if (type == BlockType::DBF) {
    printf(" (record length: %d)", blockInfo);
  }
  printf("\n");
}

static void compressBlock(File* in, const uint64_t begin, const uint64_t len, int &blNum, BlockType type, int blockInfo, Encoder& en, String& blstr, float &p1, float &p2, const float pscale, const TransformOptions* const transformOptions) {
  p2 = p1 + pscale * len;
  en.setStatusRange(p1, p2);

  String blstrSub;
  composeSubBlockStringToPrint(blstr, blstrSub, blNum);
  printBlock(begin, len, type, blockInfo, blstrSub);
  transformEncodeBlock(type, in, len, en, blockInfo, blstrSub, p1, p2, begin, transformOptions);
  blNum++;

  p1 = p2;
}

static void compressRecursive(File *in, uint64_t bytesToProcess, Encoder &en, String &blstr, float p1, float p2, const TransformOptions* const transformOptions) {

  uint64_t begin = in->curPos();

  float pscale = bytesToProcess != 0 ? (p2 - p1) / bytesToProcess : 0;

  int blNum = 0;
  while(bytesToProcess > 0 ) {

    //detect a block 
    DetectionInfo detectionInfo = detect(in, bytesToProcess, transformOptions); // Special blocktypes
    in->setpos(begin);

    uint64_t blockStart = detectionInfo.HeaderLength != 0 ? detectionInfo.HeaderStart : detectionInfo.DataStart;
    while(blockStart != begin) {
      TextDetectionInfo textDetectionInfo = detectText(in, begin, blockStart - begin); // DEFAULT / TEXT / TEXT_EOL
      in->setpos(begin);
      compressBlock(in, textDetectionInfo.DataStart, textDetectionInfo.DataLength, /*ref: */ blNum, textDetectionInfo.Type, 0, en, /*in: */ blstr, /*ref: */ p1, /*ref: */ p2, pscale, transformOptions);
      begin += textDetectionInfo.DataLength;
      bytesToProcess -= textDetectionInfo.DataLength;
    }

    if (detectionInfo.HeaderLength != 0) {
      compressBlock(in, detectionInfo.HeaderStart, detectionInfo.HeaderLength, /*ref: */ blNum, BlockType::HDR, 0, en, /*in: */ blstr, /*ref: */ p1, /*ref: */ p2, pscale, transformOptions);
      begin += detectionInfo.HeaderLength;
      bytesToProcess -= detectionInfo.HeaderLength;
    }

    if (begin != detectionInfo.DataStart)
      quit("Internal error in compressRecursive");
    
    if (detectionInfo.DataLength != 0) {
      compressBlock(in, detectionInfo.DataStart, detectionInfo.DataLength, /*ref: */ blNum, detectionInfo.Type, detectionInfo.DataInfo, en, /*in: */ blstr, /*ref: */ p1, /*ref: */ p2, pscale, transformOptions);
      begin += detectionInfo.DataLength;
      bytesToProcess -= detectionInfo.DataLength;
    }
  }
}

static void compressRecursiveForTar(File* in, uint64_t bytesToProcess, Encoder& en, String& blstr, float p1, float p2, const TransformOptions* const transformOptions) {
  Array<uint64_t, 1> filePositions{ 0 };
  TarFilter tarFilter{};
  in->setpos(0);
  tarFilter.getFilePositions(in, filePositions);
  assert(filePositions[filePositions.size() - 1] == bytesToProcess);
  in->setpos(0);

  int blNum = 0;
  float pscale = bytesToProcess != 0 ? (p2 - p1) / bytesToProcess : 0;

  auto headerSize = filePositions[0];
  compressBlock(in, 0, headerSize, /*ref: */ blNum, BlockType::TARHDR, 0, en, /*in: */ blstr, /*ref: */ p1, /*ref: */ p2, pscale, transformOptions);
  for (; blNum < filePositions.size(); blNum++) {
    uint64_t blockSize = filePositions[blNum] - filePositions[blNum - 1];
    p2 = p1 + pscale * blockSize;
    en.setStatusRange(p1, p2);
    String blstrSub;
    composeSubBlockStringToPrint(blstr, blstrSub, blNum);
    compressRecursive(in, blockSize, en, blstrSub, p1, p2, transformOptions);
    p1 = p2;
  }
}

// Compress a file. Split fileSize bytes into blocks by type.
// For each block, output
// <type> <size> and call encode_X to convert to type X.
// Test transform and compress.
static void compressfile(const Shared* const shared, const char *filename, uint64_t fileSize, Encoder &en, bool verbose) {
  assert(en.getMode() == COMPRESS);
  assert(filename && filename[0]);

  uint64_t start = en.size();
  Block::EncodeBlockSize(&en, fileSize);

  FileDisk in;
  in.open(filename, true);
  printf("Block segmentation:\n");
  String blstr;
  TransformOptions transformOptions(shared);
  if (shared->GetOptionSkipBlockDetection()) {
    // skip blockType detection + compress
    const uint64_t begin = 0;
    int blNum = 0;
    const int info = -1;
    float p1 = 0.0f;
    float p2 = 1.0f;
    const float pscale = fileSize != 0 ? (p2 - p1) / fileSize : 0;
    compressBlock(&in, begin, fileSize, /*ref: */ blNum, BlockType::DEFAULT, info, en, /*in: */ blstr, /*ref: */ p1, /*ref: */ p2, pscale, &transformOptions);
  }
  else {
    // detect block types + compress
    compressRecursive(&in, fileSize, en, blstr, 0.0F, 1.0F, &transformOptions);
  }
  in.close();

  if (shared->GetOptionMultipleFileMode()) { //multiple file mode
    if( verbose ) {
      printf("File size to encode   : 4\n"); //This string must be long enough. "Compressing ..." is still on screen, we need to overwrite it.
    }
    printf("File input size       : %" PRIu64 "\n", fileSize);
    printf("File compressed size  : %" PRIu64 "\n", en.size() - start);
  }
}

static uint64_t decompressRecursive(File *out, uint64_t blockSize, Encoder &en, FMode mode, TransformOptions *transformOptions) {
  uint64_t i = 0;
  uint64_t diffFound = 0;
  while( i < blockSize ) {

    uint64_t len = Block::DecodeBlockHeader(&en);
    BlockType type = en.predictorMain.shared->State.blockType;
    int info = en.predictorMain.shared->State.blockInfo;
    if (type == BlockType::MRB) {
      FileTmp tmp;
      for (uint64_t j = 0; j < len; ++j)
          tmp.putChar(en.decompressByte(&en.predictorMain));
      if (mode != FMode::FDISCARD) {
        tmp.setpos(0);
        len = decodeFunc(type, en, &tmp, len, info, out, mode, diffFound, transformOptions);
      }
      tmp.close();
    } else if( hasRecursion(type)) {
      FileTmp tmp;
      decompressRecursive(&tmp, len, en, FMode::FDECOMPRESS, transformOptions);
      if( mode != FMode::FDISCARD ) {
        tmp.setpos(0);
        if( hasTransform(type, info)) {
          len = decodeFunc(type, en, &tmp, len, info, out, mode, diffFound, transformOptions);
        }
      }
      tmp.close();
    } else if( hasTransform(type, info)) {
      len = decodeFunc(type, en, nullptr, len, info, out, mode, diffFound, transformOptions);
    } else {
      for( uint64_t j = 0; j < len; ++j ) {
        if((j & 0xfff) == 0 ) {
          en.printStatus();
        }
        if( mode == FMode::FDECOMPRESS ) {
          out->putChar(en.decompressByte(&en.predictorMain));
        } else if( mode == FMode::FCOMPARE ) {
          if( en.decompressByte(&en.predictorMain) != out->getchar() && (diffFound == 0)) {
            mode = FMode::FDISCARD;
            diffFound = i + j + 1;
          }
        } else {
          en.decompressByte(&en.predictorMain);
        }
      }
    }
    i += len;
  }
  return diffFound;
}

// Decompress or compare a file
static void decompressFile(const Shared* const shared, const char* filename, FMode fMode, Encoder& en) {
  assert(en.getMode() == DECOMPRESS);
  assert(filename && filename[0]);

  uint64_t fileSize = Block::DecodeBlockSize(&en);

  FileDisk f;
  if( fMode == FMode::FCOMPARE ) {
    f.open(filename, true);
    printf("Comparing");
  } else { //mode==FDECOMPRESS;
    f.create(filename);
    printf("Extracting");
  }
  printf(" %s %" PRIu64 " bytes -> ", filename, fileSize);

  // Decompress/Compare
  TransformOptions transformOptions(shared);
  uint64_t r = decompressRecursive(&f, fileSize, en, fMode, &transformOptions);
  if( fMode == FMode::FCOMPARE && (r == 0) && f.getchar() != EOF) {
    printf("file is longer\n");
  } else if( fMode == FMode::FCOMPARE && (r != 0)) {
    printf("differ at %" PRIu64 "\n", r - 1);
  } else if( fMode == FMode::FCOMPARE ) {
    printf("identical\n");
  } else {
    printf("done   \n");
  }
  f.close();
}
