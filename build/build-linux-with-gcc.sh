# compile paq8px on Lubuntu (x64) with gcc
# prepare with:
# sudo apt update
# sudo apt install gcc g++
#------------------------
# if you're on arch, use these instead:
# sudo pacman -Syu
# sudo pacman -S gcc g++

gcc -c -DNDEBUG -I../zlib/ -O3 -fno-fast-math -ffp-contract=off -flto=auto -funroll-loops -ftree-vectorize -fexceptions -include unistd.h ../zlib/adler32.c ../zlib/crc32.c ../zlib/deflate.c ../zlib/gzlib.c ../zlib/inffast.c ../zlib/inflate.c ../zlib/inftrees.c ../zlib/trees.c ../zlib/zutil.c 

g++ -DNDEBUG -I../zlib/ -O3 -fno-fast-math -ffp-contract=off -flto=auto -funroll-loops -ftree-vectorize -s -static -fno-rtti -std=gnu++17 adler32.o crc32.o deflate.o gzlib.o inffast.o inflate.o inftrees.o trees.o zutil.o ../file/*.cpp ../filter/*.cpp ../model/*.cpp ../text/*.cpp ../lstm/*.cpp ../*.cpp -opaq8px

rm -rf *.o
