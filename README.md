# PAQ8PX – Experimental Lossless Data Compressor & Entropy Estimator

## About

**PAQ** is a family of experimental, high-end lossless data compression programs.
`paq8px` is one of the longest-running branches of PAQ, started by Jan Ondrus in 2009 with major contributions 
from Márcio Pais and Zoltán Gotthardt (see [Contribution Timeline](#timeline)).

`paq8px` consistently achieves state-of-the-art compression ratios on various data compression benchmarks (see [Benchmark Results](#benchmarks)).
This performance comes at the cost of speed and memory usage, which makes it impractical for production use or long-term storage.
However, it is particularly well-suited for file entropy estimation and as a reference for compression research.

For detailed history and ongoing development discussions, see the
[paq8px thread on encode.su](https://encode.su/threads/342-paq8px).

## Quick start

`paq8px` is **portable** software – no installation required. 

Get the latest binary for Windows (x64) from the [Releases](../../releases) page
or from the [paq8px thread on encode.su](https://encode.su/threads/342-paq8px),
or build it from source for your platform – see [below](#compile).

### Command line interface

`paq8px` does not include a graphical user interface (GUI). All operations are performed from the command line.

Open a terminal and run `paq8px` with the desired options to compress your file (such as `paq8px -8 file.txt`).  
Start with a small file – compression takes time.

Example output (on Windows):
```
c:\>paq8px.exe -8 file.txt
paq8px archiver v211 (c) 2026, Matt Mahoney et al.

Creating archive file.txt.paq8px211 in single file mode...

Filename: file.txt (111261 bytes)
Block segmentation:
 0           | text             |    111261 bytes [0 - 111260]
-----------------------
Total input size     : 111261
Total archive size   : 19595

Time 19.58 sec, used 2163 MB (2268982538 bytes) of memory
```

> [!NOTE]
> The output archive extension is versioned (e.g., .paq8px211).

> [!NOTE]
> You can place the binary anywhere and reference inputs/outputs by path.

### Some examples

Compress a file at level 8 (balanced speed and compression ratio):
```
paq8px.exe -8 filename_to_compress 
```

Compress at the maximum level with LSTM modeling included (`-12L`): 
```
paq8px.exe -12L filename_to_compress 
```
> [!WARNING]
> This mode is extremely slow and memory-intensive. Make sure you have 32 GB+ RAM.

## Getting help

To view available options, run `paq8px` without arguments.
To view available options + detailed help pages, run `paq8px -help`.

<details>
<summary>Click to expand: full <code>paq8px</code> help</summary>

```
paq8px archiver v211 (c) 2026, Matt Mahoney et al.
Free under GPL, http://www.gnu.org/licenses/gpl.txt

Usage:
  to compress       ->   paq8px -LEVEL[FLAGS] [OPTIONS] INPUT [OUTPUT]
  to decompress     ->   paq8px -d INPUT.paq8px211 [OUTPUT]
  to test           ->   paq8px -t INPUT.paq8px211 [OUTPUT]
  to list contents  ->   paq8px -l INPUT.paq8px211

LEVEL:
  -1 -2 -3 -4          | Compress using less memory (529, 543, 572, 630 MB)
  -5 -6 -7 -8          | Use more memory (747, 980, 1446, 2377 MB)
  -9 -10 -11 -12       | Use even more memory (4241, 7968, 15421, 29305 MB)
  -0                   | Segment and transform only, no compression
  -0L                  | Segment and transform then LSTM-only compression (alternative: -lstmonly)

FLAGS:
  L                    | Enable LSTM model (+24 MB per block type)
  A                    | Use adaptive learning rate in CM mixer
  S                    | Skip RGB color transform (images)
  B                    | Brute-force DEFLATE detection
  E                    | Pre-train x86/x64 model
  T                    | Pre-train text models (dictionary-based)

  Example: paq8px -8LA file.txt   <- Level 8 + LSTM + adaptive learning rate

Block detection control (compression-only):
  -forcebinary         | Force generic (binary) mode
  -forcetext           | Force text mode

LSTM-specific options (expert-only):
  -lstmlayers=N        | Set the number of LSTM layers to N (1..5, default is 2)
  -savelstm:text FILE  | Save learned LSTM model weights after compression
  -loadlstm:text FILE  | Load LSTM model weights before compression/decompression

Misc options:
  -v                   | Verbose output
  -log FILE            | Append compression results to log file
  -simd MODE           | Override SIMD detection - expert only (NONE|SSE2|AVX2|AVX512|NEON)

Notes:
  INPUT may be FILE, PATH/FILE, or @FILELIST
  OUTPUT is optional: FILE, PATH, PATH/FILE
  The archive is created in the current folder with .paq8px211 extension if OUTPUT omitted
  FLAGS are case-insensitive and only needed for compression; they may appear in any order
  INPUT must precede OUTPUT; all other OPTIONS may appear anywhere

=============
Detailed Help
=============

---------------
 1. Compression
---------------

  Compression levels control the amount of memory used during both compression and decompression.
  Higher levels generally improve compression ratio at the cost of higher memory usage and slower speed.
  Specifying the compression level is needed only for compression - no need to specify it for decompression.
  Approximately the same amount of memory will be used during compression and decompression.

  The listed memory usage for each LEVEL (-1 = 529 MB .. -12 = 29305 MB) is typical/indicative for compressing binary
  files with no preprocessing. Actual memory use is lower for text files and higher when a preprocessing step
  (segmentation and transformations) requires temporary memory. When special file types are detected, special models
  (image, jpg, audio) will be used and thus will require extra RAM.

------------------
 2. Special Levels
------------------

  -0   Only block type detection (segmentation) and block transformations are performed.
       The data is copied (verbatim or transformed); no compression happens.
       This mode is similar to a preprocessing-only tool like precomp.
       Uses approximately 3-7 MB total.

  -0L  Uses only a single LSTM model for prediction which is shared across all block types.
       Uses approximately 20-24 MB total RAM.
       Alternative: -lstmonly

---------------------
 3. Compression Flags
---------------------

  Compression flags are single-letter, case-insensitive, and appended directly to the level.
  They are valid only during compression. No need to specify them for decompression.

  L   Enable the LSTM (Long Short-Term Memory) model.
      Uses a fixed-size model, independent of compression level.

      At level -0L (also: -lstmonly) a single LSTM model is used for prediction for all detected block types.
      Block detection and segmentation are still performed, but no context mixing or Secondary Symbol
      Estimation (SSE) stage is used.

      At higher levels (-1L .. -12L) the LSTM model is included as a submodel in Context Mixing and its predictions
      are mixed with the other models.
      When special block types are detected, for each block type an individual LSTM model is created dynamically and
      used within that block type. Each such LSTM model adds approximately 24 MB to the total memory use.

  A   Enable adaptive learning rate in the CM mixer.
      May improve compression for some files.

  S   Skip RGB color transform for 24/32-bit images.
      Useful when the transform worsens compression.
      This flag has no effect when no image block types are detected.

  B   Enable brute-force DEFLATE stream detection.
      Slower but may improve detection of compressed streams.

  E   Pre-train the x86/x64 executable model.
      This option pre-trains the EXE model using the paq8px.exe binary itself.
      Archives created with a different paq8px.exe executable (even when built from the same source and build options)
      will differ. To decompress an archive created with -E, you must use the exact same executable that created it.

  T   Pre-train text-oriented models using a dictionary and expression list.
      The word list (english.dic) and expression list (english.exp) are used only to pre-train models before
      compression and they are not stored in the archive.
      You must have these same files available to decompress archives created with -T.

---------------------------
 4. Block Detection Control
---------------------------

  Block detection and segmentation always happen regardless of the memory level or other options - except when forced:

  -forcebinary

      Disable block detection; the whole file is considered as a single binary block and only the generic (binary)
      model set will be used.
      Useful when block detection produces false positives.

  -forcetext

      Disable block detection; consider the whole file as a single text block and use the text model set only.
      Useful when text data is misclassified as binary or fragments in a text file are incorrectly detected as some
      other block type.

---------------------------------------
 5. LSTM-Specific Options (expert-only)
---------------------------------------

  -lstmlayers=N

      Set the number of LSTM layers to N. Using more layers generally leads to better compression, but memory use
      will be higher (scales linearly with N) and compression time will be significantly slower. The default is N=2.

  -savelstm:text FILE

      Saves the LSTM model's learned parameters as a lossless snapshot to the specified file when compression finishes.
      Only the model used for text block(s) will be saved.
      It's not possible to save a snapshot from other block types. This is an experimental feature.

  -loadlstm:text FILE

      Loads the LSTM model's learned parameters from the specified file (which was saved earlier
      by the -savelstm:text option) before compression starts. The LSTM model will use this loaded
      snapshot to bootstrap its predictions.
      At levels -1L .. -12L only text blocks are affected.
      At level -0L all blocks are affected (because a single LSTM model is used for all block types).
      Critical: The same snapshot file MUST be used during decompression or the original content cannot be recovered.

----------------------
 6. Archive Operations
----------------------

  -d  Decompress an archive.
      In single-file mode the content is decompressed, the name of the output is the name of the archive without
      the .paq8px211 extension.
      In multi-file mode first the @LISTFILE is extracted then the rest of the files. Any required folders will
      be created recursively, all files will be extracted with their original names.
      If the output file or files already exist they will be overwritten.

      Example: to decompress file.txt to the current folder:
      paq8px -d file.txt.paq8px211

  -t  Test archive contents by decompressing to memory and comparing with the original data on-the-fly.
      If a file fails the test, the first mismatched position will be printed to screen.

      Example: to test archive contents:
      paq8px -t file.txt.paq8px211

  -l  List archive contents.
      Extracts the embedded @FILELIST (if present) and prints it.
      Applicable only to multi-file archives.

      Example: to list the file list (when the archive was created using @files):
      paq8px -l files.paq8px211

----------------------------------
 7. INPUT and OUTPUT Specification
----------------------------------

  INPUT may be:

  * A single file
  * A path/file
  * A [path/]@FILELIST

  In multi-file mode (i.e. when @FILELIST is provided) only file names, file contents and file sizes are stored
  in the archive. Timestamps, permissions, attributes or any other metadata are not preserved unless stored
  separately and manually by the user in the FILELIST.

  OUTPUT is optional:

    For compression:

    * If omitted, the archive is created in the current directory.
      The name of the archive: INPUT + paq8px211 extension appended.
    * If a filename is given, it is used as the archive name.
    * If a directory is given, the archive is created inside it.
    * If the archive file already exists, it will be overwritten.

    For decompression:

    * If an output filename is not provided, the output will be named the same as the archive without
      the paq8px211 extension.
    * If a filename is given, it is used as the output name.
    * If a directory is given, the restored file will be created inside it (the directory must exist).
    * If the output file(s) already exist, they will be overwritten.

  Examples:

  To create data.txt.paq8px211 in current directory:
  paq8px -8 data.txt

  To create archive.paq8px211 in current directory:
  paq8px -8 data.txt archive.paq8px211

  To create data.txt.paq8px211 in results/ directory:
  paq8px -8 data.txt results/

---------------------------------
 8. @FILELIST Format and Behavior
---------------------------------

  When a @FILELIST is provided, the FILELIST file itself is compressed as the first file in the archive and
  automatically extracted during decompression.

  The FILELIST is a tab-separated text file with this structure:

    Column 1:  Filenames and optional relative paths (required, used by compressor)
    Column 2+: Arbitrary metadata - timestamps, ownership, etc. (optional, preserved but ignored)

    First line: Header (preserved but ignored during processing the file list)

  Only the first column is used by the compressor and decompressor.
  All other columns are preserved but ignored.
  Paths must be relative to the FILELIST location.

  Using this mechanism allows full restoration of file metadata with third-party tools after decompression.


-------------------------
 9. Miscellaneous Options
-------------------------

  -v

    Enable verbose output.

  -log FILE

    Append compression results to a tab-separated log file.
    Logging applies only to compression.

  -simd MODE

    Normally, the highest usable SIMD instruction set is detected and used automatically

    - for the CM mixer - supported: SSE2, AVX2, AVX512, ARM NEON
    - for neural network operations in the LSTM model - supported: SSE2, AVX2
    - for the LSM and OLS predictors (used mainly in image and audio models) - supported: SSE2.
     
    This option overrides the detected SIMD instruction set. Intended for expert use and benchmarking.
    Supported values (case-insensitive):
       NONE
       SSE2, AVX2, AVX512 (on x64)
       NEON (on ARM)

    Note that when paq8px is compiled for a specific CPU architecture, the compiler may automatically
    vectorize some parts of the code. While selecting 'NONE' disables all manually optimized SIMD
    implementations, the remaining scalar code may still be auto-vectorized by the compiler and
    therefore may not be entirely free of vector instructions.

----------------------
 10. Argument Ordering
----------------------

  Command-line arguments may appear in any order with the following exception:
  INPUT must always precede OUTPUT.

  Example: the following two are equivalent:

    paq8px -v -simd sse2 enwik8 -log results.txt output/ -8
    paq8px -8 enwik8 -log results.txt output/ -v -simd sse2

  Further examples:

    paq8px -8 file.txt         | Compress using ~2.3 GB RAM
    paq8px -12L enwik8         | Compress 'enwik8' with maximum compression (~29 GB RAM), use the LSTM model as well
    paq8px -4 image.jpg        | Compress the 'image.jpg' file - using less memory, even faster
    paq8px -8ba b64sample.xml  | Compress 'b64sample.xml' faster and using less memory
                                 Put more effort into finding and transforming DEFLATE blocks
                                 Use adaptive learning rate.
    paq8px -8s rafale.bmp      | Compress the 'rafale.bmp' image file
                                 Skip color transform - this file compresses better without it
```
</details>

## Compatibility & archive basics

A `paq8px` archive stores one or more files in a highly compressed format.

> [!NOTE]
> Files and archives larger than 2 GB are not supported.

> [!NOTE]
> `paq8px` archives are not compatible across different `paq8px` releases (past or future).

> [!NOTE]
> A `paq8px` archive may contain multiple files, but once created, you cannot add to or remove files from the archive.

### How to recognize it

The file extension reflects the exact `paq8px` version that created it (e.g., `.paq8px211`).  
You can also check the header: if the first bytes read "paq8px", it is likely a `paq8px` archive.  
Exact version information cannot be inferred from the archive content: the archive header does not encode the specific `paq8px` version used. Only the file extension reflects the version.

### Single file vs multiple file modes

In **single-file mode**, only file contents are stored – no paths, names, timestamps, attributes, permissions, or other metadata.

In **multi-file mode**, you may preserve such metadata via the @FILELIST mechanism (see the help screen for details).

### Notes on pre-training

> [!WARNING]  
> Archives made with pre-training-like options (`-E`, `-T`, `-R`) are fragile — decompression requires the same binary and/or external files.

1. **The exe pre-training (`-E`)**  
This option pre-trains the EXE model using the paq8px.exe binary itself.  
Archives created with a different paq8px.exe binary (even when built from the same source and build options) will differ.  
To decompress an archive created with `-E`, you must use the exact same executable that created it.

2. **Text pre-training (`-T`)**  
The word list (`english.dic`) and expression list (`english.exp`) are used only to pre-train models before compression and they are not stored in the archive.  
You must have these same files available to decompress archives created with `-T`.

3. **LSTM pre-trained weight repositories (`-R`)**  
If you use pre-trained LSTM repositories, ensure the same RNN weight files (`english.rnn`, `x86_64.rnn`) are available during decompression.

> [!WARNING]  
> The LSTM repositories are temporarily unavailable in the latest release due to the refactoring of the model.
> The latest version supporting this feature was v209.

<a id="compile"></a>
## How to compile

Building `paq8px` requires a `C++17` capable `C++` compiler:  
[https://en.cppreference.com/w/cpp/compiler_support#cpp17](https://en.cppreference.com/w/cpp/compiler_support#cpp17)

**Windows:**  
On Windows, you can download a prebuilt executable instead of compiling. Just grab the latest executable from the [https://encode.su/threads/342-paq8px](https://encode.su/threads/342-paq8px) thread.  
If you would like to build an executable yourself you may use the Visual Studio solution file or in case of Mingw-w64 see the `build-mingw-w64-generic-publish.cmd` batch file in the build subfolder.

**Linux/macOS:**  
The ./build folder already contains helper scripts.  
You may use the following commands to build with cmake:

```
sudo apt-get install build-essential zlib1g-dev cmake make
cd build
./build-linux-with-cmake.sh
```

### Testing in a Linux VM

- Get a Linux VM (such as Lubuntu 25.04 Plucky Puffin)
- Install the required compilers and tools with the following commands:

```
sudo apt update
sudo apt install gcc clang gcc-aarch64-linux-gnu g++-aarch64-linux-gnu build-essential cmake zlib1g-dev
```

Sample build scripts are provided in the build/ folder:
- `build/build-linux-with-cmake.sh`
- `build/build-linux-with-gcc.sh`
- `build/build-linux-with-clang.sh`
- `build/build-linux-cross-compile-aarch64.sh`

### Tested toolchains

The following compiler/OS combinations have been tested successfully:

| Version | OS                             | Compiler/IDE                                                  |
|---------|--------------------------------|---------------------------------------------------------------|
| v211    | Windows                        | Visual Studio 2022 Community Edition 17.14.14                 |
| v211    | Windows                        | Microsoft (R) C/C++ Optimizing Compiler Version 19.44.35216   |
| v211    | Windows                        | MinGW-w64 13.0.0 (gcc-15.2.0)                                 |
| v211    | Lubuntu 25.04 Plucky Puffin    | gcc (Ubuntu 14.2.0-19ubuntu2) 14.2.0                          |
| v211    | Lubuntu 25.04 Plucky Puffin    | Ubuntu clang version 20.1.2 (0ubuntu1), Target: x86_64-pc-linux-gnu |
| v211    | Lubuntu 25.04 Plucky Puffin    | aarch64-linux-gnu-gcc (Ubuntu 14.2.0-19ubuntu2) 14.2.0        |

Other modern C++17 compilers may also work but are not routinely tested.

> [!NOTE]
> We build and test 64-bit releases. 32-bit releases are seldom built or tested.  
> A known limitation of 32-bit releases is the 2 GB memory barrier. As a consequence, compression and decompression with 32-bit releases may not work ("out of memory") on level 8 and above.

## Release checklist

When you make a new release:

- Please update the version number in the "Versioning" section in the `paq8px.cpp` source file.
- Please append a short description of your modifications to the [CHANGELOG](CHANGELOG) file.
- Please carry out some basic tests. Run your tests with asserts on (remove the `NDEBUG` preprocessor directive).
- Please verify if paq8px can be propely built on different platforms (i.e. test all the build scripts)
- Update README.md, especially the Benchmark results.

### References

- Get Visual Studio 2022 Community Edition from: [https://visualstudio.microsoft.com/vs/community/](https://visualstudio.microsoft.com/vs/community/)
- Get MinGW-w64 for Windows from: [https://winlibs.com/](https://winlibs.com/)
- zlib source files in the zlib folder originate from: [https://github.com/madler/zlib](https://github.com/madler/zlib)
- Get Lubuntu 25.04 Plucky Puffin for testing the build from: [https://www.osboxes.org/lubuntu/](https://www.osboxes.org/lubuntu/)

## How it works

`paq8px` compresses files **bit by bit** using a technique called **context mixing**: 
multiple models make probabilistic predictions for the next bit, and a mixer combines them into a single,
more accurate probability, which is then encoded with an arithmetic coder.

This approach is computationally intensive but highly adaptive, making paq8px especially effective
for **entropy estimation**, **compressibility testing** and **research** purposes.

For an in-depth technical explanation, see the [DOC](DOC) file.

<a id="benchmarks"></a>
## Benchmark results

Benchmark results are provided on various corpora for comparison with other compressors.  
Rankings are based solely on compression ratio, not speed or memory usage to show reference compressed sizes achievable on these datasets.  
Results are drawn from official listings where available, or from community testing when benchmarks are no longer maintained.  
Results last verified: Sept 21, 2025.

Summary:

| Corpus / Benchmark                               | Version | Rank |
|:-------------------------------------------------|:--------|-----:|
| Calgary                                          | v210    | #2   |
| Canterbury                                       | v210    | #2   |
| Silesia                                          | v210    | #1   |
| Kodak Lossless True Color Image Suite            | v211    | #1   |
| Lossless Photo Compression Benchmark (LPCB)      | v206    | #1   |
| Large Text Compression Benchmark (LTCB)          | v206    | #10  |
| Darek's corpus (DBA)                             | v207fix1| #1   |
| Maximumcompression benchmark                     | v207fix1| #1   |
| fenwik9 benchmark by Sportman                    | v210    | #1   |
| World English Bible benchmark by Sportman        | v208fix1| #1   |

For the Calgary, Canterbury, Silesia and MaximumCompression benchmarks, see paq8px evolution up to paq8px_v207fix1, run by Darek in his [post in the paq8px thread](https://encode.su/threads/1925-cmix?p=71001&viewfull=1#post71001)

### Calgary corpus

The Calgary corpus does not have an official maintained ranking, and most published results do not include modern experimental compressors.

Below are compressed sizes for `paq8px v210` under various options, compared with `cmix v21`.

| File   |         -8 |        -12L |       -12LT |(v209) -12RT | cmix v21 (reference)|
|:-------|-----------:|------------:|------------:|------------:|--------------------:|
| bib    |      19595 |       19520 |       17492 |       17376 |              17180  |
| book1  |     183318 |      181492 |      175722 |      163431 |             173709  |
| book2  |     113979 |      113143 |      108844 |      106668 |             105918  |
| geo    |      42475 |       42255 |       42265 |       42367 |              42760  |
| news   |      83023 |       82681 |       78490 |       77166 |              76389  |
| obj1   |       7063 |        6982 |        6841 |        6892 |               7053  |
| obj2   |      40934 |       40129 |       39820 |       39950 |              40139  |
| paper1 |      12360 |       12317 |       11041 |       10749 |              10831  |
| paper2 |      19538 |       19467 |       17478 |       16589 |              17169  |
| pic    |      19624 |       19666 |       19669 |       19677 |              21883  |
| progc  |       8870 |        8804 |        8206 |        8189 |               8193  |
| progl  |       9512 |        9449 |        8876 |        8864 |               8788  |
| progp  |       6378 |        6296 |        6061 |        6097 |               6126  |
| trans  |      10977 |       10939 |       10056 |       10045 |               9990  |
|**Total compressed size**|   **577'646** |     **573'140** |   **550'861** | **534'060** | **546'128** |
|**Compression time (approx. sec)**| **307** |   **864** | **1231** | **1567**| **n/a**|

With fair options (`-12LT`), `paq8px v210` achieves results close to `cmix v21`.  
With unfair options (`-12RT`), results surpass cmix, but these should be excluded (see [Benchmarking Notes](#benchmarking-notes)).

At the time of writing, `paq8px v210` likely ranks #2 on Calgary behind `cmix v21`.

### Canterbury corpus

The same general notes apply to the Canterbury corpus as to the Calgary corpus.  

Below are compressed sizes for `paq8px v210` under various options, compared with `cmix v21`.

| File          |        -8 |       -12L |      -12LT | (v209) -12RT | cmix v21 (reference)|
|:--------------|----------:|-----------:|-----------:|-------------:|--------------------:|
| alice29.txt   |     33065 |      32851 |      31138 |      28317   |               31076 |
| asyoulik.txt  |     31512 |      31423 |      29601 |      28062   |               29434 |
| cp.html       |      5405 |       5389 |       4740 |       4720   |                4746 |
| fields.c      |      2027 |       2017 |       1856 |       1848   |                1909 |
| grammar.lsp   |       861 |        862 |        750 |        732   |                 771 |
| kennedy.xls   |      8137 |       7849 |       7850 |       7972   |                7955 |
| lcet10.txt    |     79119 |      78807 |      74655 |      72594   |               73365 |
| plrabn12.txt  |    117451 |     116694 |     112546 |     108648   |              112263 |
| ptt5          |     19624 |      19666 |      19669 |      19677   |               21883 |
| sum           |      6825 |       6798 |       6657 |       6679   |                6870 |
| xargs.1       |      1295 |       1293 |       1097 |       1061   |                1123 |
|**Total compressed size**      |**305'321**|**303'649** |**290'559** |**280'310**   |         **291'395** |
|**Compression time (approx. sec)**| **256** |   **707** | **1015** | **1352**   | **n/a**|

At the time of writing, `paq8px v210` likely ranks #2 on Canterbury behind `cmix v21`.

### Silesia corpus

`paq8px v210` **ranked #1** in [The Silesia Open Source Compression Benchmark](https://mattmahoney.net/dc/silesia.html) at the time of writing.

Results for `paq8px v210` together with `cmix v21` as a reference:

| File      |       -12L | precomp v0.4.7 -cn + cmix v21 (reference) |
|:----------|-----------:|---------------------:|
| dickens   |  1'860'023 |            1'802'071 |
| mozilla   |  6'129'742 |            6'634'210 |
| mr        |  1'852'494 |            1'828'423 |
| nci       |    776'723 |              781'325 |
| ooffice   |  1'218'806 |            1'221'977 |
| osdb      |  1'968'252 |            1'963'597 |
| reymont   |    699'456 |              704'817 |
| samba     |  1'589'315 |            1'588'875 |
| sao       |  3'723'922 |            3'726'502 |
| webster   |  4'402'064 |            4'271'915 |
| xml       |    245'824 |              233'696 |
| x-ray     |  3'521'286 |            3'503'686 |
|**Total compressed size**      |**27'987'907**|**28'261'094** |
|**Compression time (approx. sec)**| **68'837** | **n/a**|

Here `paq8px` outperformed `cmix v21` overall, though performance varies per file.

### Kodak Lossless True Color Image Suite

The [Kodak Lossless True Color Image Suite](https://r0k.us/graphics/kodak/) has no official benchmarking for lossless image compression.
The images were converted from PNG to PPM before compression.

| File        | -8 (v211) | -8L (v211) |
|:------------|----------:|-----------:|
| kodim01.ppm |   322'743 |    318'033 |
| kodim02.ppm |   266'212 |    262'726 |
| kodim03.ppm |   208'063 |    206'330 |
| kodim04.ppm |   273'983 |    270'569 |
| kodim05.ppm |   350'224 |    345'048 |
| kodim06.ppm |   296'388 |    292'696 |
| kodim07.ppm |   229'395 |    226'944 |
| kodim08.ppm |   361'408 |    355'403 |
| kodim09.ppm |   252'594 |    250'051 |
| kodim10.ppm |   259'685 |    257'040 |
| kodim11.ppm |   285'699 |    282'074 |
| kodim12.ppm |   238'039 |    235'167 |
| kodim13.ppm |   406'000 |    398'454 |
| kodim14.ppm |   321'954 |    318'182 |
| kodim15.ppm |   260'866 |    257'842 |
| kodim16.ppm |   243'852 |    241'137 |
| kodim17.ppm |   259'714 |    257'286 |
| kodim18.ppm |   371'378 |    364'566 |
| kodim19.ppm |   299'803 |    296'129 |
| kodim20.ppm |   243'599 |    241'381 |
| kodim21.ppm |   304'384 |    300'814 |
| kodim22.ppm |   337'390 |    331'449 |
| kodim23.ppm |   258'592 |    255'585 |
| kodim24.ppm |   306'685 |    301'034 |
|**Total compressed size**         | **6'958'650** | **6'865'940** |
|**Compression time (approx. sec)**|   **1'125**   | **5'121**     |

At the time of writing, `paq8px v211` likely ranks #1 on the Kodak test set among lossless compressors with no pre-trained models.

Other compressors for reference:
[GitHub - WangXuan95/Image-Compression-Benchmark: A comparison of many lossless image compression formats.](https://github.com/WangXuan95/Image-Compression-Benchmark)

### Lossless Photo Compression Benchmark (LPCB)

`paq8px v206` **ranked #1** at [Lossless Photo Compression Benchmark](http://qlic.altervista.org/).

The benchmark has not been rerun for later versions.

### Large Text Compression Benchmark (LTCB)

`paq8px v206` **ranked #10** at [Large Text Compression Benchmark](https://www.mattmahoney.net/dc/text.html) at the time of writing.  
Note, that unlike paq8px, most higher-ranked compressors are tuned specifically for enwik8/enwik9, and often apply enwik-specific preprocessing (e.g., word replacement, article reordering).  

The benchmark has not been rerun for later versions.

### Darek's corpus (DBA)

Darek's benchmark is no longer actively maintained.  
This is not an exhaustive benchmark – it targets only high-end compressors.

See the last results targeting only high-end compressors in [Darek's post to the encode.su forum](https://encode.su/threads/342-paq8px?p=75549&viewfull=1#post75549) from 2022 including results for v207fix1.

`paq8px v207fix1` **ranked #1** at that time.

### MaximumCompression benchmark

The MaximumCompression benchmark is no longer actively maintained and has no up-to-date official listing.  
The official site was last updated in 2011. At that time paq8px was **ranked #1**.

See `paq8px` evolution on the MaximumCompression benchmark up until paq8px v207fix1 in [Darek's post to the encode.su forum](https://encode.su/threads/342-paq8px?p=75636&viewfull=1#post75636) from 2022.

Compressed sizes for v210 with compression option `-12L` (`-12Ls` for rafale.bmp).

| File          |      -12L |
|:--------------|----------:|
|A10.jpg        |    624023 |
|acrord32.exe   |    786553 |
|english_mc.dic |    333089 |
|FlashMX.pdf    |   1289571 |
|fp.log         |    199933 |
|mso97.dll      |   1121228 |
|ohs.doc        |    452209 |
|rafale.bmp     |    463390 |
|vcfiu.hlp      |    245448 |
|world95.txt    |    309236 |
|**Total compressed size**  | **5'824'680** |
|**Compression time (sec)**| **19'384** |

To the best of our knowledge, `paq8px`'s latest version, `v211`, would still **rank #1** at the time of writing.

### fenwik9 benchmark

`paq8px v210` **ranks #1** in the [fenwik9 benchmark](https://encode.su/threads/3873-fenwik9-benchmark-results).  
This is a non-standard but exhaustive single-file benchmark maintained by Sportman.

### World English Bible benchmark (WEB)

`paq8px v208fix1` **ranked #1** in the [World English Bible benchmark](https://encode.su/threads/4314-World-English-Bible-benchmark-results).  
This is a non-standard but exhaustive single-file benchmark maintained by Sportman.


### Benchmarking Notes

> [!WARNING]
> 1) Using `-R` to load pre-trained LSTM weight repositories is unfair if the target file to be compressed was part of the training data.  
> 2) Benchmarks and leaderboards change over time – rankings may shift.
> 3) Hardware does not affect compression ratio and memory use, but it does affect runtime; reported times are approximate and for context only.

<a id="timeline"></a>
## PAQ8PX contribution timeline

`paq8px` is a branch of the PAQ compressor series, descended from earlier versions such as PAQ7 and the PAQ8 variants (e.g., PAQ8A-PAQ8P).

Development began in 2009 and remains active, supported by a global community of contributors. 

Work has focused on expanding model coverage (images, audio, executables, text) with emphasis on compression ratio.

The table below highlights milestones, contributors, and notable changes over the years.

| Year         | Versions   | Contributors & Highlights |
|--------------|------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Pre-2009** | PAQ roots  | **Matt Mahoney**: Original PAQ author. Early branches (`paq8hp*`, `paq8fthis*`, `paq8p3`, `lpaq1`) introduced context maps with 16-bit checksums, probabilistic state tables, specialized models (JPEG, sparse, DMC, distance-based), exe model/filter. Added directory compression and drag-and-drop (PAQ8A), BMP/PGM/JPEG/WAV support, APM/StateMap optimizations. |
| **2009**     | v0–v67     | **Jan Ondrus**: Founded `paq8px`, adding TGA/TIFF/AIFF/MOD/S3M models, PPM/PBM compression, CD sector transform, exe filters, recursive sub-blocks, WAV-model improvements. <br> **Simon Berger**: TGA 24/8-bit, TIFF/AIFF improvements, MSVC fixes, compression pipeline rewrite. <br> **LovePimple**: Portability fixes. |
| **2010**     | v68–v69    | **Jan Ondrus**: Added `-l` listing option, fix for multi-path file compression.  |
| **2016**     | v70–v75    | **Jan Ondrus**: Add zlib recompression (initially unstable), PDF image support, Base64 transform, GIF recompression, and paq8pxd model updates (incl. im8bitModel), plus multiple bugfixes (zlib header/progress display, Base64, GIF). |
| **2017**     | v76–v127   | **Márcio Pais**: JPEG upgrades (subsampling, thumbnails, MJPEG), record/BMP models, grayscale detection, XML model, x86/x64 pre-training, PNG recompression, DEFLATE MTF + brute force, dBASE parsing, adaptive learning rate, English stemmer. <br> **Jan Ondrus**: JPEG tweaks, PAM format detection, block handling, PDF 4-bit fix. <br> **Zoltán Gotthardt**: Fixes, MSVC/Array/`ilog2` fixes, faster JPEG learning rate, IO improvements. <br> **Mauro Vezzosi**: Bug reports, dmcModel patch. |
| **2018**     | v128–v173  | **Márcio Pais**: Extended text modeling (English/French/German stemmers, language detection, SparseMatchModel, SSE refinements, RLE/EOL transforms), 8bpp/24–32bpp image model improvements, JPEG tweaks, pre-training refinements. <br> **Zoltán Gotthardt**: New CLI and file handling, DMC enhancements, hashing improvements, charGroupModel, compiler/portability fixes. <br> **Andrew Epstein**: AVX2 optimizations, macOS build fixes. |
| **2019**     | v174–v183  | **Márcio Pais**: Added linearPredictionModel, audio8bModel, audio16bModel, new image/GIF/TIFF handling, text model with word embeddings. <br> **Zoltán Gotthardt**: refactoring (global scope cleanup, model factory, Shared struct), improved WordModel (PDF text extraction, pre-training), enhancements to StateMap, ContextMap2, MatchModel, and NormalModel. |
| **2020**     | v184–v200  | **Andrew Epstein**: Code cleanup, modularization, Doxygen docs. <br> **Moisés Cardona**: ARM/NEON support, base64 fix, SIMD work. <br> **Zoltán Gotthardt**: Refactoring (predictor separation, RNG, ContextMap), Sparse/SparseBit/Indirect model improvements, fixes, cleanup. <br> **Márcio Pais**: LSTM model (pre-training, retraining, x86/64 optimizations), DEC Alpha transform/model, new SSE stages. <br> **Surya Kandau**: JPEG model refinements. |
| **2021**     | v201–v206  | **Zoltán Gotthardt**: Improved IndirectContext/MatchModel, added high-precision arithmetic encoder & APMPost, introduced ChartModel, MRB detection, metadata modeling, separate mixers per block type, refined text detection, and `-skipdetection` option. |
| **2022**     | v207       | **Zoltán Gotthardt**: PNG filtering moved to transform layer; DEC-Alpha detection via object signature; TAR detection/transform; base85 filter (from paq8pxd); structured-text WordModel (linemodel) enhancements; separate LSTM per main context. |
| **2023**     | v208       | **Zoltán Gotthardt**: TAR detection fixes; new -forcetext option; enhanced 1-bit image model; shifted contexts (fewer in IndirectModel, added to WordModel for TEXT); refactors; Pavel Rosický: AVX512 detection |
| **2025**     | v209       | **Zoltán Gotthardt**: Model tweaks (initialized mixer weights; corrected matchmodel context); TEXT detection fixes; build/toolchain updates |
| **2026**     | v210-v211  | **Zoltán Gotthardt**: LSTM model enhancements, speed improvements in image and audio compression |

This timeline is not exhaustive, for details, see [CHANGELOG](CHANGELOG).

## Notable borrows

`paq8px` incorporates ideas and code from a range of sources, often adapted and extended to fit the project’s design:

- **UTF-8 detection** – based on Bjoern Hoehrmann's [UTF decoder DFA](http://bjoern.hoehrmann.de/utf-8/decoder/dfa/); integrated by Zoltán Gotthardt
- **Base64 transform** – from paq8pxd by Kaido Orav; integrated by Jan Ondrus
- **Base85 transform** – from paq8pxd by Kaido Orav; integrated by Zoltán Gotthardt
- **MRB detection** – from paq8pxd by Kaido Orav; integrated with enhancements by Zoltán Gotthardt
- **zlib recompression** – from AntiZ; integrated by Jan Ondrus
- **Text modeling with stemming** – based on the Porter/Porter2 stemmers; integrated by Márcio Pais
- **Audio modeling ideas** – based on 'An asymptotically Optimal Predictor for Stereo Lossless Audio Compression' by Florin Ghido; integrated with enhancements by Márcio Pais
- **Image modeling ideas** – from Emma by Márcio Pais
- **EXE model** – incorporates ideas from [DisFilter](http://www.farbrausch.de/~fg/code/disfilter/) by Fabian Giesen; integrated with enhancements by Márcio Pais
- **ChartModel** – from paq8kx7; integrated with enhancements by Zoltán Gotthardt
- **MatchModel** – ideas from Emma; integrated by Márcio Pais
- **MatchModel** – improvements from paq8gen; integrated by Zoltán Gotthardt
- **LSTM model** – adapted from cmix by Byron Knoll; integrated with enhancements by Márcio Pais, further enhancements based on ligru-compress by Zoltán Gotthardt
- **OLS predictor** – by Sebastian Lehmann; integrated by Márcio Pais
- **LMS predictor** – by Sebastian Lehmann; integrated by Márcio Pais

## Similar compressors

- [paq8pdx](https://github.com/kaitz/paq8pxd) by Kaido Orav
- [cmix](https://www.byronknoll.com/cmix.html) by Byron Knoll

## Copyright

Copyright (C) 2009-2026 Matt Mahoney, Serge Osnach, Alexander Ratushnyak, Bill Pettis, Przemyslaw Skibinski, Matthew Fite, wowtiger, Andrew Paterson, 
Jan Ondrus, Andreas Morphis, Pavel L. Holoborodko, Kaido Orav, Simon Berger, Neill Corlett, Márcio Pais, Andrew Epstein, Mauro Vezzosi, Zoltán Gotthardt, Moisés Cardona and others.

We would like to express our gratitude for the endless support of many contributors who encouraged `paq8px` development with ideas, testing, compiling, debugging: 
LovePimple, Skymmer, Darek, Stephan Busch, m^2, Christian Schneider, pat357, Rugxulo, Gonzalo, a902cd23, pinguin2, Luca Biondi,
and the broader community at [encode.su](https://encode.su/threads/342-paq8px).

## License

> This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
> This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  

See the GNU General Public License for more details at [http://www.gnu.org/copyleft/gpl.html](http://www.gnu.org/copyleft/gpl.html).  

A summary in plain language is available at [https://tldrlegal.com/license/gnu-general-public-license-v2](https://tldrlegal.com/license/gnu-general-public-license-v2).
