  /*
  PAQ8PX file compressor/archiver
  see README for information
  see DOC for technical details
  see CHANGELOG for version history
*/

//////////////////////// Versioning ////////////////////////////////////////

#define PROGNAME     "paq8px"
#define PROGVERSION  "205fix1"  //update version here before publishing your changes
#define PROGYEAR     "2021"


#include "utils.hpp"

#include <stdexcept>  //std::exception

#include "Encoder.hpp"
#include "ProgramChecker.hpp"
#include "Shared.hpp"
#include "String.hpp"
#include "file/FileName.hpp"
#include "file/ListOfFiles.hpp"
#include "file/fileUtils2.hpp"
#include "filter/Filters.hpp"
#include "simd.hpp"

typedef enum { DoNone, DoCompress, DoExtract, DoCompare, DoList } WHATTODO;

static void printHelp() {
  printf("\n"
         "Free under GPL, http://www.gnu.org/licenses/gpl.txt\n\n"
         "To compress:\n"
         "\n"
         "  " PROGNAME " -LEVEL[SWITCHES] INPUTSPEC [OUTPUTSPEC]\n"
         "\n"
         "    Examples:\n"
         "      " PROGNAME " -8 enwik8\n"
         "      " PROGNAME " -8ba b64sample.xml\n"
         "      " PROGNAME " -8 @myfolder/myfilelist.txt\n"
         "      " PROGNAME " -8a benchmark/enwik8 results/enwik8_a_" PROGNAME PROGVERSION "\n"
         "\n"
         "\n"
         "    -LEVEL:\n"
         "\n"
         "      Specifies how much memory to use. Approximately the same amount of memory\n"
         "      will be used for both compression and decompression.\n"
         "\n"
         "      -0 = no compression, only transformations when applicable (uses 415 MB)\n"
         "      -1 -2 -3 = compress using less memory (642, 657, 686 MB)\n"
         "      -4 -5 -6 -7 -8 -9 = use more memory (744, 861, 1094, 1560, 2491, 4355 MB)\n"
         "      -10  -11  -12     = use even more memory (8082, 15535, 29419 MB)\n"
         "\n"
         "      The above listed memory requirements are indicative, actual usage may vary\n"
         "      depending on several factors including need for temporary files,\n"
         "      temporary memory needs of some preprocessing (transformations), etc.\n"
         "\n"
         "\n"
         "    Optional compression SWITCHES:\n"
         "\n"
         "      b = Brute-force detection of DEFLATE streams\n"
         "      e = Pre-train x86/x64 model\n"
         "      t = Pre-train the Normal+Text+Word models with word and expression list\n"
         "          (english.dic, english.exp)\n"
         "      a = Use adaptive learning rate\n"
         "      s = For 24/32 bit images skip the color transform, just reorder the RGB channels\n"
         "      l = Use Long Short-Term Memory network as an additional model\n"
         "      r = Use repository of pre-trained LSTM models (implies option -l)\n"
         "          (english.rnn, x86_64.rnn)\n"
         "\n"
         "\n"
         "    INPUTSPEC:\n"
         "\n"
         "    The input may be a FILE or a PATH/FILE or a [PATH/]@FILELIST.\n"
         "\n"
         "    Only file content and the file size is kept in the archive. Filename,\n"
         "    path, date and any file attributes or permissions are not stored.\n"
         "    When a @FILELIST is provided the FILELIST file will be considered\n"
         "    implicitly as the very first input file. It will be compressed and upon\n"
         "    decompression it will be extracted. The FILELIST is a tab separated text\n"
         "    file where the first column contains the names and optionally the relative\n"
         "    paths of the files to be compressed. The paths should be relative to the\n"
         "    FILELIST file. In the other columns you may store any information you wish\n"
         "    to keep about the files (timestamp, owner, attributes or your own remarks).\n"
         "    These extra columns will be ignored by the compressor and the decompressor\n"
         "    but you may restore full file information using them with a 3rd party\n"
         "    utility. The FILELIST file must contain a header but will be ignored.\n"
         "\n"
         "\n"
         "    OUTPUTSPEC:\n"
         "\n"
         "    When omitted: the archive will be created in the current folder. The\n"
         "    archive filename will be constructed from the input file name by \n"
         "    appending ." PROGNAME PROGVERSION " extension to it.\n"
         "    When OUTPUTSPEC is a filename (with an optional path) it will be\n"
         "    used as the archive filename.\n"
         "    When OUTPUTSPEC is a folder the archive file will be generated from\n"
         "    the input filename and will be created in the specified folder.\n"
         "    If the archive file already exists it will be overwritten.\n"
         "\n"
         "\n"
         "To extract (decompress contents):\n"
         "\n"
         "  " PROGNAME " -d [INPUTPATH/]ARCHIVEFILE [[OUTPUTPATH/]OUTPUTFILE]\n"
         "\n"
         "    If an output folder is not provided the output file will go to the input\n"
         "    folder. If an output filename is not provided output filename will be the\n"
         "    same as ARCHIVEFILE without the last extension (e.g. without ." PROGNAME PROGVERSION")\n"
         "    When OUTPUTPATH does not exist it will be created.\n"
         "    When the archive contains multiple files, first the @LISTFILE is extracted\n"
         "    then the rest of the files. Any required folders will be created.\n"
         "\n"
         "\n"
         "To test:\n"
         "\n"
         "  " PROGNAME " -t [INPUTPATH/]ARCHIVEFILE [[OUTPUTPATH/]OUTPUTFILE]\n"
         "\n"
         "    Tests contents of the archive by decompressing it (to memory) and comparing\n"
         "    the result to the original file(s). If a file fails the test, the first\n"
         "    mismatched position will be printed to screen.\n"
         "\n"
         "\n"
         "To list archive contents:\n"
         "\n"
         "  " PROGNAME " -l [INPUTFOLDER/]ARCHIVEFILE\n"
         "\n"
         "    Extracts @FILELIST from archive (to memory) and prints its content\n"
         "    to screen. This command is only applicable to multi-file archives.\n"
         "\n"
         "\n"
         "Additional optional switches:\n"
         "\n"
         "    -v\n"
         "    Print more detailed (verbose) information to screen.\n"
         "\n"
         "    -log LOGFILE\n"
         "    Logs (appends) compression results in the specified tab separated LOGFILE.\n"
         "    Logging is only applicable for compression.\n"
         "\n"
         "    -simd [NONE|SSE2|SSSE3|AVX2|NEON]\n"
         "    Overrides detected SIMD instruction set for neural network operations\n"
         "\n"
         "\n"
         "Remark: the command line arguments may be used in any order except the input\n"
         "and output: always the input comes first then output (which may be omitted).\n"
         "\n"
         "    Example:\n"
         "      " PROGNAME " -8 enwik8 outputfolder/ -v -log logfile.txt -simd sse2\n"
         "    is equivalent to:\n"
         "      " PROGNAME " -v -simd sse2 enwik8 -log logfile.txt outputfolder/ -8\n");
}

static void printModules() {
  printf("\n");
  printf("Build: ");
#ifndef DISABLE_ZLIB
  printf("ZLIB: ENABLED, ");
#else
  printf("ZLIB: DISABLED, ");
#endif

#ifndef DISABLE_AUDIOMODEL
  printf("AUDIOMODEL: ENABLED, ");
#else
  printf("AUDIOMODEL: DISABLED, ");
#endif

#ifndef DISABLE_TEXTMODEL
  printf("TEXTMODEL: ENABLED ");
#else
  printf("TEXTMODEL: DISABLED ");
#endif

  printf("\n");
}

static void printSimdInfo(int simdIset, int detectedSimdIset) {
  printf("\nHighest SIMD vectorization support on this system: ");
  if( detectedSimdIset < 0 || detectedSimdIset > 11 ) {
    quit("Oops, sorry. Unexpected result.");
  }
  static const char *vectorizationString[12] = {"none", "MMX", "SSE", "SSE2", "SSE3", "SSSE3", "SSE4.1", "SSE4.2", "AVX", "AVX2", "AVX512", "NEON"};
  printf("%s.\n", vectorizationString[detectedSimdIset]);

  printf("Using ");
  if (simdIset == 11) {
    printf("NEON");
  } else if( simdIset >= 9 ) {
    printf("AVX2");
  } else if (simdIset >= 5) {
    printf("SSE2 & SSSE3");
  } else if( simdIset >= 3 ) {
    printf("SSE2");
  } else {
    printf("non-vectorized");
  }
  printf(" neural network and hashtable functions.\n");
}

static void printCommand(const WHATTODO &whattodo) {
  printf(" To do          = ");
  if( whattodo == DoNone ) {
    printf("-");
  }
  if( whattodo == DoCompress ) {
    printf("Compress");
  }
  if( whattodo == DoExtract ) {
    printf("Extract");
  }
  if( whattodo == DoCompare ) {
    printf("Compare");
  }
  if( whattodo == DoList ) {
    printf("List");
  }
  printf("\n");
}

static void printOptions(Shared *shared) {
  printf(" Level          = %d\n", shared->level);
  printf(" Brute      (b) = %s\n", (shared->options & OPTION_BRUTE) != 0U ? "On  (Brute-force detection of DEFLATE streams)"
                                                                          : "Off"); //this is a compression-only option, but we put/get it for reproducibility
  printf(" Train exe  (e) = %s\n", (shared->options & OPTION_TRAINEXE) != 0U ? "On  (Pre-train x86/x64 model)" : "Off");
  printf(" Train txt  (t) = %s\n",
         (shared->options & OPTION_TRAINTXT) != 0U ? "On  (Pre-train main model with word and expression list)" : "Off");
  printf(" Adaptive   (a) = %s\n", (shared->options & OPTION_ADAPTIVE) != 0U ? "On  (Adaptive learning rate)" : "Off");
  printf(" Skip RGB   (s) = %s\n",
         (shared->options & OPTION_SKIPRGB) != 0U ? "On  (Skip the color transform, just reorder the RGB channels)" : "Off");
  printf(" Use LSTM   (l) = %s\n", (shared->options & OPTION_LSTM) != 0U ? "On  (Use Long Short-Term Memory network)" : "Off");
  printf(" LSTM repo  (r) = %s\n", (shared->options & OPTION_LSTM_TRAINING) != 0U ? "On  (Use repository of pre-trained LSTM models)" : "Off");
  printf(" File mode      = %s\n", (shared->options & OPTION_MULTIPLE_FILE_MODE) != 0U ? "Multiple" : "Single");
}

auto processCommandLine(int argc, char **argv) -> int {
  ProgramChecker *programChecker = ProgramChecker::getInstance();
  Shared shared;
  try {

    if( !shared.toScreen ) { //we need a minimal feedback when redirected
      fprintf(stderr, PROGNAME " archiver v" PROGVERSION " (c) " PROGYEAR ", Matt Mahoney et al.\n");
    }
    printf(PROGNAME " archiver v" PROGVERSION " (c) " PROGYEAR ", Matt Mahoney et al.\n");

    // Print help message
    if( argc < 2 ) {
      printHelp();
      quit();
    }

    // Parse command line arguments
    WHATTODO whattodo = DoNone;
    bool verbose = false;
    int c = 0;
    int simdIset = -1; //simd instruction set to use

    FileName input;
    FileName output;
    FileName inputPath;
    FileName outputPath;
    FileName archiveName;
    FileName logfile;

    for( int i = 1; i < argc; i++ ) {
      int argLen = static_cast<int>(strlen(argv[i]));
      if( argv[i][0] == '-' ) {
        if( argLen == 1 ) {
          quit("Empty command.");
        }
        if( argv[i][1] >= '0' && argv[i][1] <= '9' ) { // first  digit of level
          if( whattodo != DoNone ) {
            quit("Only one command may be specified.");
          }
          int level = argv[i][1] - '0';
          int j = 2;
          if (argLen >= 3 && argv[i][2] >= '0' && argv[i][2] <= '9') { // second digit of level
            level = level*10 + argv[i][2] - '0';
            j++;
          }
          if (level > 12) {
            quit("Compression level must be between 0 and 12.");
          }
          shared.init(level);
          whattodo = DoCompress;
          //process optional compression switches
          for( ; j < argLen; j++ ) {
            switch( argv[i][j] & 0xDFU ) {
              case 'B':
                shared.options |= OPTION_BRUTE;
                break;
              case 'E':
                shared.options |= OPTION_TRAINEXE;
                break;
              case 'T':
                shared.options |= OPTION_TRAINTXT;
                break;
              case 'A':
                shared.options |= OPTION_ADAPTIVE;
                break;
              case 'S':
                shared.options |= OPTION_SKIPRGB;
                break;
              case 'L':
                shared.options |= OPTION_LSTM;
                break;
              case 'R':
                shared.options |= OPTION_LSTM|OPTION_LSTM_TRAINING;
                break;
              default: {
                printf("Invalid compression switch: %c", argv[1][j]);
                quit();
              }
            }
          }
        } else if( strcasecmp(argv[i], "-d") == 0 ) {
          if( whattodo != DoNone ) {
            quit("Only one command may be specified.");
          }
          whattodo = DoExtract;
        } else if( strcasecmp(argv[i], "-t") == 0 ) {
          if( whattodo != DoNone ) {
            quit("Only one command may be specified.");
          }
          whattodo = DoCompare;
        } else if( strcasecmp(argv[i], "-l") == 0 ) {
          if( whattodo != DoNone ) {
            quit("Only one command may be specified.");
          }
          whattodo = DoList;
        } else if( strcasecmp(argv[i], "-v") == 0 ) {
          verbose = true;
        } else if( strcasecmp(argv[i], "-log") == 0 ) {
          if( logfile.strsize() != 0 ) {
            quit("Only one logfile may be specified.");
          }
          if( ++i == argc ) {
            quit("The -log switch requires a filename.");
          }
          logfile += argv[i];
        }
        else if( strcasecmp(argv[i], "-simd") == 0 ) {
          if( ++i == argc ) {
            quit("The -simd switch requires an instruction set name (NONE,SSE2,SSSE3, AVX2, NEON).");
          }
          if( strcasecmp(argv[i], "NONE") == 0 ) {
            simdIset = 0;
          } else if( strcasecmp(argv[i], "SSE2") == 0 ) {
            simdIset = 3;
          } else if( strcasecmp(argv[i], "SSSE3") == 0 ) {
            simdIset = 5;
         } else if( strcasecmp(argv[i], "AVX2") == 0 ) {
            simdIset = 9;
         } else if (strcasecmp(argv[i], "NEON") == 0) {
            simdIset = 11;
          } else {
            quit("Invalid -simd option. Use -simd NONE, -simd SSE2, -simd SSSE3, -simd AVX2 or -simd NEON.");
          }
        } else {
          printf("Invalid command: %s", argv[i]);
          quit();
        }
      } else { //this parameter does not begin with a dash ("-") -> it must be a folder/filename
        if( input.strsize() == 0 ) {
          input += argv[i];
          input.replaceSlashes();
        } else if( output.strsize() == 0 ) {
          output += argv[i];
          output.replaceSlashes();
        } else {
          quit("More than two file names specified. Only an input and an output is needed.");
        }
      }
    }

    if( verbose ) {
      printModules();
    }

    // Determine CPU's (and OS) support for SIMD vectorization instruction set
    int detectedSimdIset = simdDetect();
    if( simdIset == -1 ) {
      simdIset = detectedSimdIset;
    }
    if( simdIset > detectedSimdIset ) {
      printf("\nOverriding system highest vectorization support. Expect a crash.");
    }

    // Print anything only if the user wants/needs to know
    if( verbose || simdIset != detectedSimdIset ) {
      printSimdInfo(simdIset, detectedSimdIset);
    }

    // Set highest or user selected vectorization mode
    if (simdIset == 11) {
      shared.chosenSimd = SIMDType::SIMD_NEON;
    } else if (simdIset >= 9) {
      shared.chosenSimd = SIMDType::SIMD_AVX2;
    } else if (simdIset >= 5) {
      shared.chosenSimd = SIMDType::SIMD_SSSE3;
    } else if( simdIset >= 3 ) {
      shared.chosenSimd = SIMDType::SIMD_SSE2;
    } else {
      shared.chosenSimd = SIMDType::SIMD_NONE;
    }

    if( verbose ) {
      printf("\n");
      printf(" Command line   =");
      for( int i = 0; i < argc; i++ ) {
        printf(" %s", argv[i]);
      }
      printf("\n");
    }

    // Successfully parsed command line arguments
    // Let's check their validity
    if( whattodo == DoNone ) {
      quit("A command switch is required: -0..-12 to compress, -d to decompress, -t to test, -l to list.");
    }
    if( input.strsize() == 0 ) {
      printf("\nAn %s is required %s.\n", whattodo == DoCompress ? "input file or filelist" : "archive filename",
             whattodo == DoCompress ? "for compressing" : whattodo == DoExtract ? "for decompressing" : whattodo == DoCompare
                                                                                                        ? "for testing" : whattodo == DoList
                                                                                                                          ? "to list its contents"
                                                                                                                          : "");
      quit();
    }
    if( whattodo == DoList && output.strsize() != 0 ) {
      quit("The list command needs only one file parameter.");
    }

    // File list supplied?
    if( input.beginsWith("@")) {
      if( whattodo == DoCompress ) {
        shared.options |= OPTION_MULTIPLE_FILE_MODE;
        input.stripStart(1);
      } else {
        quit("A file list (a file name prefixed by '@') may only be specified when compressing.");
      }
    }

    int pathType = 0;

    //Logfile supplied?
    if( logfile.strsize() != 0 ) {
      if( whattodo != DoCompress ) {
        quit("A log file may only be specified for compression.");
      }
      pathType = examinePath(logfile.c_str());
      if( pathType == 2 || pathType == 4 ) {
        quit("Specified log file should be a file, not a directory.");
      }
      if( pathType == 0 ) {
        printf("\nThere is a problem with the log file: %s", logfile.c_str());
        quit();
      }
    }

    // Separate paths from input filename/directory name
    pathType = examinePath(input.c_str());
    if( pathType == 2 || pathType == 4 ) {
      printf("\nSpecified input is a directory but should be a file: %s", input.c_str());
      quit();
    }
    if( pathType == 3 ) {
      printf("\nSpecified input file does not exist: %s", input.c_str());
      quit();
    }
    if( pathType == 0 ) {
      printf("\nThere is a problem with the specified input file: %s", input.c_str());
      quit();
    }
    if( input.lastSlashPos() >= 0 ) {
      inputPath += input.c_str();
      inputPath.keepPath();
      input.keepFilename();
    }

    // Separate paths from output filename/directory name
    if( output.strsize() > 0 ) {
      pathType = examinePath(output.c_str());
      if( pathType == 1 || pathType == 3 ) { //is an existing file, or looks like a file
        if( output.lastSlashPos() >= 0 ) {
          outputPath += output.c_str();
          outputPath.keepPath();
          output.keepFilename();
        }
      } else if( pathType == 2 || pathType == 4 ) {//is an existing directory, or looks like a directory
        outputPath += output.c_str();
        if( !outputPath.endsWith("/") && !outputPath.endsWith("\\")) {
          outputPath += GOODSLASH;
        }
        //output file is not specified
        output.resize(0);
        output.pushBack(0);
      } else {
        printf("\nThere is a problem with the specified output: %s", output.c_str());
        quit();
      }
    }

    //determine archive name
    if( whattodo == DoCompress ) {
      archiveName += outputPath.c_str();
      if( output.strsize() == 0 ) { // If no archive name is provided, construct it from input (append PROGNAME extension to input filename)
        archiveName += input.c_str();
        archiveName += "." PROGNAME PROGVERSION;
      } else {
        archiveName += output.c_str();
      }
    } else { // extract/compare/list: archivename is simply the input
      archiveName += inputPath.c_str();
      archiveName += input.c_str();
    }
    if( verbose ) {
      printf(" Archive        = %s\n", archiveName.c_str());
      printf(" Input folder   = %s\n", inputPath.strsize() == 0 ? "." : inputPath.c_str());
      printf(" Output folder  = %s\n", outputPath.strsize() == 0 ? "." : outputPath.c_str());
    }

    Mode mode = whattodo == DoCompress ? COMPRESS : DECOMPRESS;

    ListOfFiles listoffiles;

    // set basePath for file list
    listoffiles.setBasePath(whattodo == DoCompress ? inputPath.c_str() : outputPath.c_str());

    // Process file list (in multiple file mode)
    if((shared.options & OPTION_MULTIPLE_FILE_MODE) != 0U ) { //multiple file mode
      assert(whattodo == DoCompress);
      // Read and parse file list file
      FileDisk f;
      FileName fn(inputPath.c_str());
      fn += input.c_str();
      f.open(fn.c_str(), true);
      while( true ) {
        c = f.getchar();
        listoffiles.addChar(c);
        if( c == EOF) {
          break;
        }
      }
      f.close();
      //Verify input files
      for( int i = 0; i < listoffiles.getCount(); i++ ) {
        getFileSize(listoffiles.getfilename(i)); // Does file exist? Is it readable? (we don't actually need the file size now)
      }
    } else { //single file mode or extract/compare/list
      FileName fn(inputPath.c_str());
      fn += input.c_str();
      getFileSize(fn.c_str()); // Does file exist? Is it readable? (we don't actually need the file size now)
    }

    FileDisk archive;  // compressed file

    if( mode == DECOMPRESS ) {
      archive.open(archiveName.c_str(), true);
      // Verify archive header, get level and options
      int len = static_cast<int>(strlen(PROGNAME));
      for( int i = 0; i < len; i++ ) {
        if( archive.getchar() != PROGNAME[i] ) {
          printf("%s: not a valid %s file.", archiveName.c_str(), PROGNAME);
          quit();
        }
      }
      
      c = archive.getchar();
      uint8_t level = static_cast<uint8_t>(c);
      if (level > 12) {
        quit("Unexpected compression level setting in archive");
      }
      shared.init(level);
      c = archive.getchar();
      if( c == EOF) {
        printf("Unexpected end of archive file.\n");
      }
      shared.options = static_cast<uint8_t>(c);
    }

    if( verbose ) {
      printCommand(whattodo);
      printOptions(&shared);
    }
    printf("\n");

    int numberOfFiles = 1; //default for single file mode

    // Write archive header to archive file
    if( mode == COMPRESS ) {
      if((shared.options & OPTION_MULTIPLE_FILE_MODE) != 0U ) { //multiple file mode
        numberOfFiles = listoffiles.getCount();
        printf("Creating archive %s in multiple file mode with %d file%s...\n", archiveName.c_str(), numberOfFiles,
               numberOfFiles > 1 ? "s" : "");
      } else { //single file mode
        printf("Creating archive %s in single file mode...\n", archiveName.c_str());
      }
      archive.create(archiveName.c_str());
      archive.append(PROGNAME);
      archive.putChar(shared.level);
      archive.putChar(shared.options);
    }

    // In single file mode with no output filename specified we must construct it from the supplied archive filename
    if((shared.options & OPTION_MULTIPLE_FILE_MODE) == 0 ) { //single file mode
      if((whattodo == DoExtract || whattodo == DoCompare) && output.strsize() == 0 ) {
        output += input.c_str();
        const char *fileExtension = "." PROGNAME PROGVERSION;
        if( output.endsWith(fileExtension)) {
          output.stripEnd(static_cast<int>(strlen(fileExtension)));
        } else {
          printf("Can't construct output filename from archive filename.\nArchive file extension must be: '%s'", fileExtension);
          quit();
        }
      }
    }

    Encoder en(&shared, mode, &archive);
    uint64_t contentSize = 0;
    uint64_t totalSize = 0;

    // Compress list of files
    if( mode == COMPRESS ) {
      uint64_t start = en.size(); //header size (=8)
      if( verbose ) {
        printf("Writing header : %" PRIu64 " bytes\n", start);
      }
      totalSize += start;
      if((shared.options & OPTION_MULTIPLE_FILE_MODE) != 0 ) { //multiple file mode

        en.encodeBlockType(BlockType::TEXT);
        uint64_t len1 = input.size(); //ASCIIZ filename of listfile - with ending zero
        const String *const s = listoffiles.getString();
        uint64_t len2 = s->size(); //ASCIIZ filenames of files to compress - with ending zero
        en.encodeBlockSize(len1 + len2);

        for( uint64_t i = 0; i < len1; i++ ) {
          en.compressByte(input[i]); //ASCIIZ filename of listfile
        }
        for( uint64_t i = 0; i < len2; i++ ) {
          en.compressByte((*s)[i]); //ASCIIZ filenames of files to compress
        }

        printf("1/2 - Filename of listfile : %" PRIu64 " bytes\n", len1);
        printf("2/2 - Content of listfile  : %" PRIu64 " bytes\n", len2);
        printf("----- Compressed to        : %" PRIu64 " bytes\n", en.size() - start);
        totalSize += len1 + len2;
      }
    }

    // Decompress list of files
    if( mode == DECOMPRESS && (shared.options & OPTION_MULTIPLE_FILE_MODE) != 0 ) {
      const char *errmsgInvalidChar = "Invalid character or unexpected end of archive file.";
      // name of listfile
      FileName listFilename(outputPath.c_str());
      if( output.strsize() != 0 ) {
        quit("Output filename must not be specified when extracting multiple files.");
      }
      if((en.decodeBlockType()) != BlockType::TEXT ) {
        quit(errmsgInvalidChar);
      }
      en.decodeBlockSize(); //we don't really need it
      while((c = en.decompressByte()) != 0 ) {
        if( c == 255 ) {
          quit(errmsgInvalidChar);
        }
        listFilename += static_cast<char>(c);
      }
      while((c = en.decompressByte()) != 0 ) {
        if( c == 255 ) {
          quit(errmsgInvalidChar);
        }
        listoffiles.addChar(static_cast<char>(c));
      }
      if( whattodo == DoList ) {
        printf("File list of %s archive:\n", archiveName.c_str());
      }

      numberOfFiles = listoffiles.getCount();

      //write filenames to screen or listfile or verify (compare) contents
      if( whattodo == DoList ) {
        printf("%s\n", listoffiles.getString()->c_str());
      } else if( whattodo == DoExtract ) {
        FileDisk f;
        f.create(listFilename.c_str());
        String *s = listoffiles.getString();
        f.blockWrite(reinterpret_cast<uint8_t *>(&(*s)[0]), s->strsize());
        f.close();
      } else if( whattodo == DoCompare ) {
        FileDisk f;
        f.open(listFilename.c_str(), true);
        String *s = listoffiles.getString();
        for( uint64_t i = 0; i < s->strsize(); i++ ) {
          if( f.getchar() != static_cast<uint8_t>((*s)[i])) {
            quit("Mismatch in list of files.");
          }
        }
        if( f.getchar() != EOF) {
          printf("Filelist on disk is larger than in archive.\n");
        }
        f.close();
      }
    }

    if( whattodo == DoList && (shared.options & OPTION_MULTIPLE_FILE_MODE) == 0 ) {
      quit("Can't list. Filenames are not stored in single file mode.\n");
    }

    // Compress or decompress files
    if( mode == COMPRESS ) {
      if( !shared.toScreen ) { //we need a minimal feedback when redirected
        fprintf(stderr, "Output is redirected - only minimal feedback is on screen\n");
      }
      if((shared.options & OPTION_MULTIPLE_FILE_MODE) != 0 ) { //multiple file mode
        for( int i = 0; i < numberOfFiles; i++ ) {
          const char *fName = listoffiles.getfilename(i);
          uint64_t fSize = getFileSize(fName);
          if( !shared.toScreen ) { //we need a minimal feedback when redirected
            fprintf(stderr, "\n%d/%d - Filename: %s (%" PRIu64 " bytes)\n", i + 1, numberOfFiles, fName, fSize);
          }
          printf("\n%d/%d - Filename: %s (%" PRIu64 " bytes)\n", i + 1, numberOfFiles, fName, fSize);
          compressfile(&shared, fName, fSize, en, verbose);
          totalSize += fSize + 4; //4: file size information
          contentSize += fSize;
        }
      } else { //single file mode
        FileName fn;
        fn += inputPath.c_str();
        fn += input.c_str();
        const char *fName = fn.c_str();
        uint64_t fSize = getFileSize(fName);
        if( !shared.toScreen ) { //we need a minimal feedback when redirected
          fprintf(stderr, "\nFilename: %s (%" PRIu64 " bytes)\n", fName, fSize);
        }
        printf("\nFilename: %s (%" PRIu64 " bytes)\n", fName, fSize);
        compressfile(&shared, fName, fSize, en, verbose);
        totalSize += fSize + 4; //4: file size information
        contentSize += fSize;
      }

      auto preFlush = en.size();
      en.flush();
      totalSize += en.size() - preFlush; //we consider padding bytes as auxiliary bytes
      printf("-----------------------\n");
      printf("Total input size     : %" PRIu64 "\n", contentSize);
      if( verbose ) {
        printf("Total metadata bytes : %" PRIu64 "\n", totalSize - contentSize);
      }
      printf("Total archive size   : %" PRIu64 "\n", en.size());
      printf("\n");
      // Log compression results
      if( logfile.strsize() != 0 ) {
        String results;
        pathType = examinePath(logfile.c_str());
        //Write header if needed
        if( pathType == 3 /*does not exist*/ ||
            (pathType == 1 && getFileSize(logfile.c_str()) == 0)/*exists but does not contain a header*/) {
          results += "PROG_NAME\tPROG_VERSION\tCOMMAND_LINE\tLEVEL\tINPUT_FILENAME\tORIGINAL_SIZE_BYTES\tCOMPRESSED_SIZE_BYTES\tRUNTIME_MS\n";
        }
        //Write results to logfile
        results += PROGNAME "\t" PROGVERSION "\t";
        for( int i = 1; i < argc; i++ ) {
          if( i != 0 ) {
            results += ' ';
          }
          results += argv[i];
        }
        results += "\t";
        results += uint64_t(shared.level);
        results += "\t";
        results += input.c_str();
        results += "\t";
        results += contentSize;
        results += "\t";
        results += en.size();
        results += "\t";
        results += uint64_t(programChecker->getRuntime() * 1000.0);
        results += "\t";
        results += "\n";
        appendToFile(logfile.c_str(), results.c_str());
        printf("Results logged to file '%s'\n", logfile.c_str());
        printf("\n");
      }
    } else { //decompress
      if( whattodo == DoExtract || whattodo == DoCompare ) {
        FMode fMode = whattodo == DoExtract ? FDECOMPRESS : FCOMPARE;
        if((shared.options & OPTION_MULTIPLE_FILE_MODE) != 0 ) { //multiple file mode
          for( int i = 0; i < numberOfFiles; i++ ) {
            const char *fName = listoffiles.getfilename(i);
            decompressFile(&shared, fName, fMode, en);
          }
        } else { //single file mode
          FileName fn;
          fn += outputPath.c_str();
          fn += output.c_str();
          const char *fName = fn.c_str();
          decompressFile(&shared, fName, fMode, en);
        }
      }
    }

    archive.close();
    if( whattodo != DoList ) {
      programChecker->print();
    }
  }
    // we catch only the intentional exceptions from quit() to exit gracefully
    // any other exception should result in a crash and must be investigated
  catch( IntentionalException const & ) {
  }

  return 0;
}

#ifdef WINDOWS
#include "shellapi.h"
#pragma comment(lib,"shell32.lib")
#endif

auto main(int argc, char **argv) -> int {
#ifdef WINDOWS
  // On Windows, argv is encoded in the effective codepage, therefore unsuitable for acquiring command line arguments (file names
  // in our case) not representable in that particular codepage.
  // -> We will recreate argv as UTF8 (like in Linux)
  uint32_t oldcp = GetConsoleOutputCP();
  SetConsoleOutputCP(CP_UTF8);
  wchar_t **szArglist;
  int argc_utf8;
  char** argv_utf8;
  if( (szArglist = CommandLineToArgvW(GetCommandLineW(), &argc_utf8)) == NULL) {
    printf("CommandLineToArgvW failed\n");
    return 0;
  } else {
    if(argc!=argc_utf8)quit("Error preparing command line arguments.");
    argv_utf8=new char*[argc_utf8+1];
    for(int i=0; i<argc_utf8; i++) {
      wchar_t *s=szArglist[i];
      int buffersize = WideCharToMultiByte(CP_UTF8,0,s,-1,NULL,0,NULL,NULL);
      argv_utf8[i] = new char[buffersize];
      WideCharToMultiByte(CP_UTF8,0,s,-1,argv_utf8[i],buffersize,NULL,NULL);
      //printf("%d: %s\n", i, argv_utf8[i]); //debug: see if conversion is successful
    }
    argv_utf8[argc_utf8]=nullptr;
    int retval=processCommandLine(argc_utf8, argv_utf8);
    for(int i=0; i<argc_utf8; i++)
      delete[] argv_utf8[i];
    delete[] argv_utf8;
    SetConsoleOutputCP(oldcp);
    return retval;
  }
#else
  return processCommandLine(argc, argv);
#endif
}
