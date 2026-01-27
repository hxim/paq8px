@echo off
setlocal EnableDelayedExpansion

rem * This script builds a paq8px executable for all x86/x64 CPUs with runtime SIMD dispatch.
rem * Requirements: MinGW-w64 (set path below)
rem * Output: paq8px.exe, with errors and/or warnings in _error1_zlib.txt and/or in _error2_paq.txt if compilation fails.
rem * Usage: %0 [diag] (optional: 'diag' enables warnings during build and asserts during runtime).

rem * Set your mingw-w64 path below
set path=%path%;c:/mingw/winlibs-x86_64-posix-seh-gcc-15.2.0-mingw-w64ucrt-13.0.0-r1/bin

rem * Define base paths
set "parent_dir=%~dp0.."

rem * Define zlib sources and objects
set zpath=../zlib/
set zsrc=%zpath%adler32.c %zpath%crc32.c %zpath%deflate.c %zpath%gzlib.c %zpath%inffast.c %zpath%inflate.c %zpath%inftrees.c %zpath%trees.c %zpath%zutil.c
set zobj=adler32.o crc32.o deflate.o gzlib.o inffast.o inflate.o inftrees.o trees.o zutil.o

rem * Check for MinGW-w64
where gcc.exe >nul 2>&1 || (
  echo ERROR: gcc.exe not found. 
  echo Suggested path: "c:\mingw\winlibs-x86_64-posix-seh-gcc-15.2.0-mingw-w64ucrt-13.0.0-r1"
  echo If your MinGW-w64 installation is in a different folder update the path in this script.
  pause
  exit /b 1
)

rem * Set MAKE for parallel LTO
rem * Why: GCC's LTO wrapper needs a 'make' executable to coordinate parallel compilation jobs, and it can't find it automatically. 
rem * By explicitly setting MAKE=mingw32-make, we tell it where to look.
set MAKE=mingw32-make

rem * Force floating point reproducibility explicitly
set safefp=-fno-fast-math -ffp-contract=off

rem * Set compiler options (release by default)
set options=-DNDEBUG -I%zpath% -O3 %safefp% -m64 -march=nocona -mtune=generic -flto=auto -floop-strip-mine -funroll-loops -ftree-vectorize -fgcse-sm -falign-loops=16

rem * Override for debug build if specified
if /i "%1"=="diag" (
  set options=-Wall -I%zpath% -O3 %safefp% -m64 -march=nocona -mtune=generic -flto=auto -floop-strip-mine -funroll-loops -ftree-vectorize -fgcse-sm -falign-loops=16
  echo Building with warnings and activating asserts during runtime.
  echo When done, see warnings in _error1_zlib.txt and _error2_paq.txt
)

rem * Clean previous build artifacts
del /q _error1_zlib.txt _error2_paq.txt paq8px.exe *.o _sources.txt 2>nul

rem * Compile zlib sources
echo Compiling zlib sources...
gcc.exe -c %options% -fexceptions %zsrc% 2>_error1_zlib.txt
if %ERRORLEVEL% neq 0 (
  echo ERROR: zlib compilation failed. See _error1_zlib.txt for details.
  pause
  exit /b 1
)

rem * Build response file with quoted full paths to *.cpp files
echo Building source file list...
> _sources.txt (
  rem Process .cpp files in parent directory and specified subfolders
  for %%D in (. file filter model text lstm) do (
      for /f "delims=" %%F in ('dir /b /a:-d "%parent_dir%\%%D\*.cpp" 2^>nul') do (
      rem Convert relative path to full path and quote it
      pushd "%parent_dir%\%%D"
      echo "!CD!\%%F"
      popd
    )
  )
)


echo Compiling and linking paq8px...
g++.exe -s -static -fno-rtti -std=gnu++17 %options% %zobj% @_sources.txt -o paq8px.exe 2>_error2_paq.txt
if %ERRORLEVEL% neq 0 (
  echo ERROR: paq8px compilation failed. See _error2_paq.txt for details.
  pause
  exit /b 1
)

echo Build successful: paq8px.exe created.

if exist _error1_zlib.txt findstr /v "^$" _error1_zlib.txt >nul && (
  echo NOTE: Issues found in _error1_zlib.txt.
)

if exist _error1_zlib.txt findstr /v "^$" _error2_paq.txt >nul && (
  echo NOTE: Issues found in _error2_paq.txt.
)

del /q _sources.txt *.o 2>nul
pause
exit /b 0