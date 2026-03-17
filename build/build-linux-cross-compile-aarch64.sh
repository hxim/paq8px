# cross compile paq8px for cortex-a73 (ARM NEON) on Lubuntu
# prepare with:
# sudo apt update
# sudo apt install gcc-aarch64-linux-gnu
# sudo apt install g++-aarch64-linux-gnu

aarch64-linux-gnu-gcc -mcpu=cortex-a73 -c -DNDEBUG -I../zlib/ -O3 -fno-fast-math -ffp-contract=off -flto=auto -fwhole-program -floop-strip-mine -funroll-loops -ftree-vectorize -fgcse-sm -falign-loops=16 -fexceptions -include unistd.h ../zlib/adler32.c ../zlib/crc32.c ../zlib/deflate.c ../zlib/gzlib.c ../zlib/inffast.c ../zlib/inflate.c ../zlib/inftrees.c ../zlib/trees.c ../zlib/zutil.c 

aarch64-linux-gnu-g++ -mcpu=cortex-a73 -DNDEBUG -I../zlib/ -O3 -fno-fast-math -ffp-contract=off -flto=auto -floop-strip-mine -funroll-loops -ftree-vectorize -fgcse-sm -falign-loops=16 -s -static -fno-rtti -std=gnu++17 adler32.o crc32.o deflate.o gzlib.o inffast.o inflate.o inftrees.o trees.o zutil.o ../file/*.cpp ../filter/*.cpp ../model/*.cpp ../text/*.cpp ../lstm/*.cpp ../*.cpp -opaq8px

rm -rf *.o
