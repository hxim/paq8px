# PAQ8PX - Experimental Lossless Data Compressor & Entropy Estimator

## About

**PAQ** is a family of experimental, high-end lossless data compression programs.
`paq8px` is one of the longest-running branches of PAQ, started by **Jan Ondrus** in 2009 with major contributions 
from **Márcio Pais** and **Zoltán Gotthardt** (see [Contribution Timeline](#timeline)).

`paq8px` consistently achieves state-of-the-art compression ratios on various data compression benchmarks (see [Benchmark Results](#benchmarks)).
This performance comes at the cost of speed and memory usage, which makes it impractical for production use or long-term storage.
However, it is particularly well-suited for file entropy estimation and as a reference for compression research.

For detailed history and ongoing development discussions, see the
[paq8px thread on encode.su](https://encode.su/threads/342-paq8px).

## Quick start

`paq8px` is **portable** software – no installation required. 

Get the latest binary for Windows (x64) from the [paq8px thread on encode.su](https://encode.su/threads/342-paq8px),
or build it from source for your platform - see [below](#compile).

### Command line interface

`paq8px` does not include a graphical user interface (GUI). All operations are performed from the command line.

Open a terminal and run `paq8px` with the desired options to compress your file (such as `paq8px -8 file.txt`).  
Start with a small file - compression takes time.

Example output (on Windows):
```
c:\>paq8px.exe -8 file.txt
paq8px archiver v209 (c) 2025, Matt Mahoney et al.

Creating archive file.txt.paq8px209 in single file mode...

Filename: file.txt (111261 bytes)
Block segmentation:
 0           | text             |    111261 bytes [0 - 111260]
-----------------------
Total input size     : 111261
Total archive size   : 19595

Time 19.58 sec, used 2163 MB (2268982538 bytes) of memory
```

> [!NOTE]
> The output archive extension is versioned (e.g., .paq8px209).

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

To view all available options, run `paq8px` without arguments:

<details>
<summary>Click to expand: full <code>paq8px</code> help</summary>

```
paq8px archiver v209 (c) 2025, Matt Mahoney et al.

Free under GPL, http://www.gnu.org/licenses/gpl.txt

To compress:

  paq8px -LEVEL[SWITCHES] INPUTSPEC [OUTPUTSPEC]

    Examples:
      paq8px -8 enwik8
      paq8px -8ba b64sample.xml
      paq8px -8 @myfolder/myfilelist.txt
      paq8px -8a benchmark/enwik8 results/enwik8_a_paq8px209


    -LEVEL:

      Specifies how much memory to use. Approximately the same amount of memory
      will be used for both compression and decompression.

      -0 = no compression, only transformations when applicable (uses 146 MB)
      -1 -2 -3 = compress using less memory (529, 543, 572 MB)
      -4 -5 -6 -7 -8 -9 = use more memory (630, 747, 980, 1446, 2377, 4241 MB)
      -10  -11  -12     = use even more memory (7968, 15421, 29305 MB)

      The above listed memory requirements are indicative, actual usage may vary
      depending on several factors including need for temporary files,
      temporary memory needs of some preprocessing (transformations),
      and whether special models (audio, image, jpeg, LSTM) are in use or not.
      Note: memory use of the LSTM model is not included/reported.


    Optional compression SWITCHES:

      b = Brute-force detection of DEFLATE streams
      e = Pre-train x86/x64 model
      t = Pre-train the Normal+Text+Word models with word and expression list
          (english.dic, english.exp)
      a = Use adaptive learning rate
      s = For 24/32 bit images skip the color transform, just reorder the RGB channels
      l = Use Long Short-Term Memory network as an additional model
      r = Use repository of pre-trained LSTM models (implies option -l)
          (english.rnn, x86_64.rnn)


    INPUTSPEC:

    The input may be a FILE or a PATH/FILE or a [PATH/]@FILELIST.

    Only file content and the file size is kept in the archive. Filename,
    path, date and any file attributes or permissions are not stored.
    When a @FILELIST is provided the FILELIST file will be considered
    implicitly as the very first input file. It will be compressed and upon
    decompression it will be extracted. The FILELIST is a tab separated text
    file where the first column contains the names and optionally the relative
    paths of the files to be compressed. The paths should be relative to the
    FILELIST file. In the other columns you may store any information you wish
    to keep about the files (timestamp, owner, attributes or your own remarks).
    These extra columns will be ignored by the compressor and the decompressor
    but you may restore full file information using them with a 3rd party
    utility. The FILELIST file must contain a header but will be ignored.


    OUTPUTSPEC:

    When omitted: the archive will be created in the current folder. The
    archive filename will be constructed from the input file name by
    appending .paq8px209 extension to it.
    When OUTPUTSPEC is a filename (with an optional path) it will be
    used as the archive filename.
    When OUTPUTSPEC is a folder the archive file will be generated from
    the input filename and will be created in the specified folder.
    If the archive file already exists it will be overwritten.


To extract (decompress contents):

  paq8px -d [INPUTPATH/]ARCHIVEFILE [[OUTPUTPATH/]OUTPUTFILE]

    If an output folder is not provided the output file will go to the input
    folder. If an output filename is not provided output filename will be the
    same as ARCHIVEFILE without the last extension (e.g. without .paq8px209)
    When OUTPUTPATH does not exist it will be created.
    When the archive contains multiple files, first the @LISTFILE is extracted
    then the rest of the files. Any required folders will be created.


To test:

  paq8px -t [INPUTPATH/]ARCHIVEFILE [[OUTPUTPATH/]OUTPUTFILE]

    Tests contents of the archive by decompressing it (to memory) and comparing
    the result to the original file(s). If a file fails the test, the first
    mismatched position will be printed to screen.


To list archive contents:

  paq8px -l [INPUTFOLDER/]ARCHIVEFILE

    Extracts @FILELIST from archive (to memory) and prints its content
    to screen. This command is only applicable to multi-file archives.


Additional optional switches:

    -forcebinary
    Skip block detection, use the DEFAULT (binary aka generic) model set only.
    It helps when block detection would find false positives in a file with purely binary content.


    -forcetext
    Skip block detection, use the TEXT model set only.
    It helps when block detection would detect the file as DEFAULT with text-like content.


    -v
    Print more detailed (verbose) information to screen.

    -log LOGFILE
    Logs (appends) compression results in the specified tab separated LOGFILE.
    Logging is only applicable for compression.

    -simd [NONE|SSE2|AVX2|AVX512|NEON]
    Overrides detected SIMD instruction set for neural network operations


Remark: the command line arguments may be used in any order except the input
and output: always the input comes first then output (which may be omitted).

    Example:
      paq8px -8 enwik8 outputfolder/ -v -log logfile.txt -simd sse2
    is equivalent to:
      paq8px -v -simd sse2 enwik8 -log logfile.txt outputfolder/ -8
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

The file extension reflects the exact `paq8px` version that created it (e.g., `.paq8px209`).  
You can also check the header: if the first bytes read "paq8px", it is likely a `paq8px` archive.  
Exact version information cannot be inferred from the archive content: the archive header does not encode the specific `paq8px` version used. Only the file extension reflects the version.

### Single file vs multiple file modes

In **single-file mode**, only file contents are stored - no paths, names, timestamps, attributes, permissions, or other metadata.

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
- `build/build-linux-with-clang.sh`
- `build/build-linux-cross-compile-aarch64.sh`

### Tested toolchains

The following compiler/OS combinations have been tested successfully:

| Version | OS                             | Compiler/IDE                                                  |
|---------|--------------------------------|---------------------------------------------------------------|
| v209    | Windows                        | Visual Studio 2022 Community Edition 17.14.14                 |
| v209    | Windows                        | Microsoft (R) C/C++ Optimizing Compiler Version 19.44.35216   |
| v209    | Windows                        | MinGW-w64 13.0.0 (gcc-15.2.0)                                 |
| v209    | Lubuntu 25.04 Plucky Puffin    | gcc (Ubuntu 14.2.0-19ubuntu2) 14.2.0                          |
| v209    | Lubuntu 25.04 Plucky Puffin    | Ubuntu clang version 20.1.2 (0ubuntu1), Target: x86_64-pc-linux-gnu |
| v209    | Lubuntu 25.04 Plucky Puffin    | aarch64-linux-gnu-gcc (Ubuntu 14.2.0-19ubuntu2) 14.2.0        |

Other modern C++17 compilers may also work but are not routinely tested.

> [!NOTE]
> We build and test 64-bit releases. 32-bit releases are seldom built or tested.  
> A known limitation of 32-bit releases is the 2 GB memory barrier. As a consequence, compression and decompression with 32-bit releases may not work ("out of memory") on level 8 and above.

## Release checklist

When you make a new release:

- Please update the version number in the "Versioning" section in the `paq8px.cpp` source file.
- Please append a short description of your modifications to the `CHANGELOG` file.
- If you publish an executable for Windows, always publish a static build (link the required library files statically).
- Always publish a build for the general public (e.g. don't use `-march=native`).
- Before publishing your build, please carry out some basic tests. Run your tests with asserts on (remove the `NDEBUG` preprocessor directive).
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
| Calgary                                          | v209    | #2   |
| Canterbury                                       | v209    | #2   |
| Silesia                                          | v209    | #1   |
| Lossless Photo Compression Benchmark (LPCB)      | v206    | #1   |
| Large Text Compression Benchmark (LTCB)          | v206    | #10  |
| Darek's corpus (DBA)                             | v207fix1| #1   |
| Maximumcompression benchmark                     | v207fix1| #1   |
| fenwik9 benchmark by Sportman                    | v206fix1| #1   |
| World English Bible benchmark by Sportman        | v208fix1| #1   |

For the Calgary, Canterbury, Silesia and MaximumCompression benchmarks, see paq8px evolution up to paq8px_v207fix1, run by Darek in his [post in the paq8px thread](https://encode.su/threads/1925-cmix?p=71001&viewfull=1#post71001)

### Calgary corpus

The Calgary corpus does not have an official maintained ranking, and most published results do not include modern experimental compressors.

Below are compressed sizes for `paq8px v209` under various options, compared with `cmix v21`.

| File   |         -8 |        -12L |       -12LT |       -12RT | cmix v21 (reference)|
|:-------|-----------:|------------:|------------:|------------:|--------------------:|
| bib    |      19595 |       19550 |       17505 |       17376 |              17180  |
| book1  |     183318 |      182175 |      176322 |      163431 |             173709  |
| book2  |     113979 |      113516 |      109153 |      106668 |             105918  |
| geo    |      42476 |       42357 |       42367 |       42367 |              42760  |
| news   |      83023 |       82813 |       78588 |       77166 |              76389  |
| obj1   |       7063 |        7037 |        6892 |        6892 |               7053  |
| obj2   |      40934 |       40258 |       39950 |       39950 |              40139  |
| paper1 |      12360 |       12340 |       11060 |       10749 |              10831  |
| paper2 |      19538 |       19515 |       17501 |       16589 |              17169  |
| pic    |      19624 |       19678 |       19677 |       19677 |              21883  |
| progc  |       8870 |        8851 |        8247 |        8189 |               8193  |
| progl  |       9512 |        9479 |        8900 |        8864 |               8788  |
| progp  |       6378 |        6343 |        6105 |        6097 |               6126  |
| trans  |      10977 |       10956 |       10070 |       10045 |               9990  |
|**Total compressed size**|   **577'647** |     **574'868** |   **552'337** | **534'060** | **546'128** |
|**Compression time (approx. sec)**| **310** |  **1187** | **1562** | **1567**| **n/a**|

With fair options (`-12LT`), `paq8px v209` achieves results close to `cmix v21`.  
With unfair options (`-12RT`), results surpass cmix, but these should be excluded (see [Benchmarking Notes](#benchmarking-notes)).

At the time of writing, `paq8px v209` likely ranks #2 on Calgary behind `cmix v21`.

### Canterbury corpus

The same general notes apply to the Canterbury corpus as to the Calgary corpus.  

Below are compressed sizes for `paq8px v209` under various options, compared with `cmix v21`.

| File          |        -8 |       -12L |      -12LT |      -12RT | cmix v21 (reference)|
|:--------------|----------:|-----------:|-----------:|-----------:|--------------------:|
| alice29.txt   |     33065 |      32979 |      31242 |      28317 |               31076 |
| asyoulik.txt  |     31512 |      31476 |      29630 |      28062 |               29434 |
| cp.html       |      5405 |       5397 |       4745 |       4720 |                4746 |
| fields.c      |      2027 |       2026 |       1864 |       1848 |                1909 |
| grammar.lsp   |       861 |        861 |        749 |        732 |                 771 |
| kennedy.xls   |      8137 |       7972 |       7972 |       7972 |                7955 |
| lcet10.txt    |     79119 |      78972 |      74770 |      72594 |               73365 |
| plrabn12.txt  |    117451 |     116808 |     112625 |     108648 |              112263 |
| ptt5          |     19624 |      19678 |      19677 |      19677 |               21883 |
| sum           |      6825 |       6822 |       6681 |       6679 |                6870 |
| xargs.1       |      1295 |       1298 |       1099 |       1061 |                1123 |
|**Total compressed size**      |**305'321**|**304'286** |**291'054** |**280'310** |         **291'395** |
|**Compression time (approx. sec)**| **263** |  **1070** | **1348** | **1352** | **n/a**|

At the time of writing, `paq8px v209` likely ranks #2 on Canterbury behind `cmix v21`.

### Silesia corpus

`paq8px v209` **ranked #1** in [The Silesia Open Source Compression Benchmark](https://mattmahoney.net/dc/silesia.html) at the time of writing.

Detailed results for `paq8px v209` together with `cmix v21` as a reference:

| File      |      -12L | cmix v21 (reference) |
|:----------|----------:|---------------------:|
| dickens   |   1865439 |              1802071 |
| mozilla   |   6165693 |              6634210 |
| mr        |   1835756 |              1828423 |
| nci       |    775907 |               781325 |
| ooffice   |   1221715 |              1221977 |
| osdb      |   1968064 |              1963597 |
| reymont   |    699630 |               704817 |
| samba     |   1593115 |              1588875 |
| sao       |   3724799 |              3726502 |
| webster   |   4416122 |              4271915 |
| xml       |    245899 |               233696 |
| x-ray     |   3513402 |              3503686 |
|**Total compressed size**      |**28'025'541**|**28'261'094** |
|**Compression time (approx. sec)**| **105'424** | **n/a**|

Here `paq8px` outperformed `cmix v21` overall, though performance varies per file.

### Lossless Photo Compression Benchmark (LPCB)

`paq8px v206` **ranked #1** at [Lossless Photo Compression Benchmark](http://qlic.altervista.org/).

The benchmark has not been rerun for later versions.

### Large Text Compression Benchmark (LTCB)

`paq8px v206` **ranked #10** at [Large Text Compression Benchmark](https://www.mattmahoney.net/dc/text.html) at the time of writing.  
Note, that unlike paq8px, most higher-ranked compressors are tuned specifically for enwik8/enwik9, and often apply enwik-specific preprocessing (e.g., word replacement, article reordering).  

The benchmark has not been rerun for later versions.

### Darek's corpus (DBA)

Darek's benchmark is no longer actively maintained.  
This is not an exhaustive benchmark - it targets only high-end compressors.

See the last results targeting only high-end compressors in [Darek's post to the encode.su forum](https://encode.su/threads/342-paq8px?p=75549&viewfull=1#post75549) from 2022 including results for v207fix1.

`paq8px v207fix1` **ranked #1** at that time.

### MaximumCompression benchmark

The MaximumCompression benchmark is no longer actively maintained and has no up-to-date official listing.  
The official site was last updated in 2011. At that time paq8px was **ranked #1**.

See `paq8px` evolution on the MaximumCompression benchmark up until paq8px v207fix1 in [Darek's post to the encode.su forum](https://encode.su/threads/342-paq8px?p=75636&viewfull=1#post75636) from 2022.

Compressed sizes for v209 with compression option `-12L` (`-12Ls` for rafale.bmp).

| File          |      -12L |
|:--------------|----------:|
|A10.jpg        |    624039 |
|acrord32.exe   |    788727 |
|english_mc.dic |    331499 |
|FlashMX.pdf    |   1289611 |
|fp.log         |    200031 |
|mso97.dll      |   1125345 |
|ohs.doc        |    452266 |
|rafale.bmp     |    463612 |
|vcfiu.hlp      |    246167 |
|world95.txt    |    309748 |
|**Total compressed size**  | **6'455'101** |
|**Compression time (sec)**| **30'074** |

To the best of our knowledge, `paq8px`'s latest version, `v209`, would still **rank #1** at the time of writing.

### fenwik9 benchmark

`paq8px v206fix1` **ranked #1** in the [fenwik9 benchmark](https://encode.su/threads/3873-fenwik9-benchmark-results).  
This is a non-standard but exhaustive single-file benchmark maintained by Sportman.

### World English Bible benchmark (WEB)

`paq8px v208fix1` **ranked #1** in the [World English Bible benchmark](https://encode.su/threads/4314-World-English-Bible-benchmark-results).  
This is a non-standard but exhaustive single-file benchmark maintained by Sportman.


### Benchmarking Notes

> [!WARNING]
> 1) Using `-R` to load pre-trained LSTM weight repositories is unfair if the target file to be compressed was part of the training data.  
> 2) Benchmarks and leaderboards change over time - rankings may shift.
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

This timeline is not exhaustive, for details, see [CHANGELOG](CHANGELOG).

## Notable borrows

`paq8px` incorporates ideas and code from a range of sources, often adapted and extended to fit the project’s design:

- **UTF-8 detection** - based on Bjoern Hoehrmann's [UTF decoder DFA](http://bjoern.hoehrmann.de/utf-8/decoder/dfa/); integrated by Zoltán Gotthardt
- **Base64 transform** - from paq8pxd by Kaido Orav; integrated by Jan Ondrus
- **Base85 transform** - from paq8pxd by Kaido Orav; integrated by Zoltán Gotthardt
- **MRB detection** - from paq8pxd by Kaido Orav; integrated with enhancements by Zoltán Gotthardt
- **zlib recompression** - from AntiZ; integrated by Jan Ondrus
- **Text modeling with stemming** - based on the Porter/Porter2 stemmers; integrated by Márcio Pais
- **Audio modeling ideas** - based on 'An asymptotically Optimal Predictor for Stereo Lossless Audio Compression' by Florin Ghido; integrated with enhancements by Márcio Pais
- **Image modeling ideas** - from Emma by Márcio Pais
- **EXE model** - incorporates ideas from [DisFilter](http://www.farbrausch.de/~fg/code/disfilter/) by Fabian Giesen; integrated with enhancements by Márcio Pais
- **ChartModel** - from paq8kx7; integrated with enhancements by Zoltán Gotthardt
- **MatchModel** - ideas from Emma; integrated by Márcio Pais
- **MatchModel** - improvements from paq8gen; integrated by Zoltán Gotthardt
- **LSTM model** - adapted from cmix by Byron Knoll; integrated with enhancements by Márcio Pais
- **OLS predictor** - by Sebastian Lehmann; integrated by Márcio Pais

## Similar compressors

- [paq8pdx](https://github.com/kaitz/paq8pxd) by Kaido Orav
- [cmix](https://www.byronknoll.com/cmix.html) by Byron Knoll

## Copyright

Copyright (C) 2009-2025 Matt Mahoney, Serge Osnach, Alexander Ratushnyak, Bill Pettis, Przemyslaw Skibinski, Matthew Fite, wowtiger, Andrew Paterson, 
Jan Ondrus, Andreas Morphis, Pavel L. Holoborodko, Kaido Orav, Simon Berger, Neill Corlett, Márcio Pais, Andrew Epstein, Mauro Vezzosi, Zoltán Gotthardt, Moisés Cardona and others.

We would like to express our gratitude for the endless support of many contributors who encouraged `paq8px` development with ideas, testing, compiling, debugging: 
LovePimple, Skymmer, Darek, Stephan Busch, m^2, Christian Schneider, pat357, Rugxulo, Gonzalo, a902cd23, pinguin2, Luca Biondi,
and the broader community at [encode.su](https://encode.su/threads/342-paq8px).

## License

> This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.
> This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  

See the GNU General Public License for more details at [http://www.gnu.org/copyleft/gpl.html](http://www.gnu.org/copyleft/gpl.html).  

A summary in plain language is available at [https://tldrlegal.com/license/gnu-general-public-license-v2](https://tldrlegal.com/license/gnu-general-public-license-v2).
