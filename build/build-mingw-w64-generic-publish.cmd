@echo off
setlocal enabledelayedexpansion

rem * This script builds a paq8px executable for all x86/x64 CPUs - no questions asked
rem * Remark: SIMD-specific functions are dispatched at runtime to match the runtime processor. (See -march=nocona -mtune=generic below.)
rem * If the build fails see compiler errors in _error1_zlib.txt and/or in _error2_paq.txt

rem * Set your mingw-w64 path below
set path=%path%;c:/mingw/winlibs-x86_64-posix-seh-gcc-15.2.0-mingw-w64ucrt-13.0.0-r1/bin

set zpath=../zlib/
set zsrc=%zpath%adler32.c %zpath%crc32.c %zpath%deflate.c %zpath%gzlib.c %zpath%inffast.c %zpath%inflate.c %zpath%inftrees.c %zpath%trees.c %zpath%zutil.c
set zobj=adler32.o crc32.o deflate.o gzlib.o inffast.o inflate.o inftrees.o trees.o zutil.o

rem * The following settings are for a release build.
rem * For a debug build remove -DNDEBUG to enable asserts and array bound checks and add -Wall to show compiler warnings.
set options=-DNDEBUG -I%zpath%  -O3 -m64 -march=nocona -mtune=generic -flto -fwhole-program -floop-strip-mine -funroll-loops -ftree-vectorize -fgcse-sm -falign-loops=16

del _error1_zlib.txt >nul 2>&1
del _error2_paq.txt  >nul 2>&1
del paq8px.exe       >nul 2>&1


gcc.exe -c %options% -fexceptions %zsrc%     2>_error1_zlib.txt
IF %ERRORLEVEL% NEQ 0 goto end

rem * This double for loop builds a response file (_sources.txt) with quoted full paths to *.cpp files
> _sources.txt (
  rem Process .cpp files in parent directory and specified subfolders
  for %%D in (. file filter model text) do (
    for /f "delims=" %%F in ('dir /b /a:-d "%~dp0..\%%D\*.cpp" 2^>nul') do (
      rem Convert relative path to full path and quote it
      pushd "%~dp0..\%%D"
      echo "!CD!\%%F"
      popd
    )
  )
)

g++.exe -s -static -fno-rtti -std=gnu++1z %options% %zobj% @_sources.txt -opaq8px.exe 2>_error2_paq.txt
IF %ERRORLEVEL% NEQ 0 goto end


:end
del _sources.txt     >nul 2>&1
del *.o              >nul 2>&1
pause
